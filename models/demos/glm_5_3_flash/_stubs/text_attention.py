# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn, tensor-parallel port of `Glm5NextTextAttention` (GLM-5.3-Flash DSA layer).

Reference math (HF `Glm5NextTextAttention.forward`, NoPE: qk_rope_head_dim == 0):

    q_resid = RMSNorm(q_a_proj(x))                 # [B,S,q_lora]
    q       = q_b_proj(q_resid)  -> [B,H,S,dqk]
    kv      = RMSNorm(kv_a_proj_with_mqa(x))       # [B,S,kv_lora]
    k, v    = split_per_head(kv_b_proj(kv))        # [B,H,S,dqk], [B,H,S,dv]
    mask    = sparse mask from the DSA indexer's top-k pool selection
    out     = o_proj(softmax(q k^T * scale + mask) v)

DSA selection for S <= index_topk: the indexer scores ceil(S / kpool) pools and keeps
select_k = min(index_topk // kpool, n_pools) of them, i.e. EVERY pool. Masking drops the pools that
are not fully visible, and `append_visible_tail` adds back the visible tokens of the open pool, so
the selected set for query i is exactly {j <= i, j a valid key}: the causal (+ key padding) mask.
The indexer's scores therefore cannot change the output in that regime, and this port computes the
mask directly. Longer sequences (where top-k genuinely drops pools) raise instead of silently
diverging.

Tensor parallelism over the N chips of the mesh (H heads -> H/N heads per chip):
  * q_a_proj, kv_a_proj_with_mqa and both RMSNorms are REPLICATED (low-rank, feed full-width norms).
  * q_b_proj, kv_b_proj (K and V halves) are COLUMN-parallel by head: ShardTensorToMesh(dim=-1).
  * attention runs on the chip-local heads only.
  * o_proj is ROW-parallel (its input features are the head-major attention output):
    ShardTensorToMesh(dim=0), then an all-reduce (all_gather of the partials + on-device sum)
    -> replicated output.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.glm_5_3_flash._stubs.text_indexer import row_chunks


def _num_devices(device):
    try:
        return int(device.get_num_devices())
    except Exception:
        return 1


def _cluster_axis(device):
    try:
        shape = tuple(device.shape)
    except Exception:
        return None
    axes = [i for i, s in enumerate(shape) if s > 1]
    return axes[0] if len(axes) == 1 else None


class TtTextAttention:
    def __init__(self, device, *, cfg, weights, n_dev):
        self.device = device
        self.n_dev = n_dev
        self.cluster_axis = _cluster_axis(device)
        self.__dict__.update(cfg)
        self.w = weights
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, device, torch_module):
        m = torch_module
        n_dev = _num_devices(device)
        is_mesh = n_dev > 1

        H = int(m.num_heads)
        d_nope = int(m.qk_nope_head_dim)
        d_rope = int(m.qk_rope_head_dim)
        d_qk = int(m.qk_head_dim)
        d_v = int(m.v_head_dim)
        kv_lora = int(m.kv_lora_rank)
        if d_rope != 0:
            raise NotImplementedError("GLM-5.3-Flash DSA attention is NoPE (qk_rope_head_dim == 0)")
        if H % n_dev:
            raise ValueError(f"num_heads={H} not divisible by TP={n_dev}")
        if m.q_a_proj.bias is not None or m.kv_a_proj_with_mqa.bias is not None or m.o_proj.bias is not None:
            raise NotImplementedError("attention_bias=True is not used by this checkpoint")

        indexer = getattr(m, "indexer", None)
        cfg_src = indexer if indexer is not None else m.config
        index_topk = int(getattr(cfg_src, "index_topk"))
        from models.demos.glm_5_3_flash._stubs.text_indexer import TtTextIndexer

        tt_indexer = TtTextIndexer.build(device, indexer) if indexer is not None else None

        def _t(w):  # nn.Linear weight [out, in] -> ttnn linear operand [in, out]
            return w.detach().to(torch.float32).t().contiguous().to(torch.bfloat16)

        def _put(t, shard_dim=None, layout=ttnn.TILE_LAYOUT):
            kw = dict(dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if is_mesh:
                kw["mesh_mapper"] = (
                    ttnn.ShardTensorToMesh(device, dim=shard_dim)
                    if shard_dim is not None
                    else ttnn.ReplicateTensorToMesh(device)
                )
            return ttnn.from_torch(t, **kw)

        def _norm_w(norm):
            w = norm.weight.detach().to(torch.bfloat16)
            return _put(w.reshape(-1, 32), layout=ttnn.ROW_MAJOR_LAYOUT)

        # kv_b_proj output is head-major [H, d_nope + d_v]; split into separate K / V weights that stay
        # head-major so a contiguous dim=-1 shard hands each chip whole heads.
        kvb = m.kv_b_proj.weight.detach().to(torch.float32).view(H, d_nope + d_v, kv_lora)
        wk_b = kvb[:, :d_nope, :].reshape(H * d_nope, kv_lora)
        wv_b = kvb[:, d_nope:, :].reshape(H * d_v, kv_lora)

        weights = dict(
            wq_a=_put(_t(m.q_a_proj.weight)),
            q_norm=_norm_w(m.q_a_layernorm),
            wq_b=_put(_t(m.q_b_proj.weight), shard_dim=-1),
            wkv_a=_put(_t(m.kv_a_proj_with_mqa.weight)),
            kv_norm=_norm_w(m.kv_a_layernorm),
            wk_b=_put(_t(wk_b), shard_dim=-1),
            wv_b=_put(_t(wv_b), shard_dim=-1),
            wo=_put(_t(m.o_proj.weight), shard_dim=0),
        )
        cfg = dict(
            H=H,
            H_local=H // n_dev,
            d_qk=d_qk,
            d_v=d_v,
            kv_lora=kv_lora,
            scale=float(m.scaling),
            q_eps=float(m.q_a_layernorm.variance_epsilon),
            kv_eps=float(m.kv_a_layernorm.variance_epsilon),
            index_topk=index_topk,
            kpool=int(indexer.index_kpool) if indexer is not None else 1,
        )
        obj = cls(device, cfg=cfg, weights=weights, n_dev=n_dev)
        obj.indexer = tt_indexer
        obj._consts = {}
        return obj

    # ---------------------------------------------------------------- helpers
    def _linear(self, x, w):
        return ttnn.linear(x, w, compute_kernel_config=self.compute_kernel_config)

    def _rms(self, x, w, eps):
        return ttnn.rms_norm(x, epsilon=eps, weight=w, compute_kernel_config=self.compute_kernel_config)

    def _split_heads(self, x, d):
        # [B,1,S,Hl*d] -> [B,Hl,S,d] via tile-aligned slices (Hl is small under TP).
        B, S = x.shape[0], x.shape[-2]
        heads = [ttnn.slice(x, [0, 0, 0, h * d], [B, 1, S, (h + 1) * d]) for h in range(self.H_local)]
        return heads[0] if len(heads) == 1 else ttnn.concat(heads, dim=1)

    def _merge_heads(self, x, d):
        # [B,Hl,S,d] -> [B,1,S,Hl*d], head-major (matches o_proj's input layout).
        B, S = x.shape[0], x.shape[-2]
        heads = [ttnn.slice(x, [0, h, 0, 0], [B, h + 1, S, d]) for h in range(self.H_local)]
        return heads[0] if len(heads) == 1 else ttnn.concat(heads, dim=-1)

    @staticmethod
    def _rs(x, shape):
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.to_layout(ttnn.reshape(x, shape), ttnn.TILE_LAYOUT)

    def _f32(self, t):
        kw = dict(dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        if self.n_dev > 1:
            kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self.device)
        return ttnn.from_torch(t.to(torch.float32).contiguous(), **kw)

    def prepare(self, S, capacity):
        """Shape constants for a prefill of S tokens and a resident cache of `capacity` positions."""
        key = (S, capacity)
        if key not in self._consts:
            self._consts[key] = dict(
                key_ids=self._f32(torch.arange(S, dtype=torch.float32).reshape(1, 1, 1, S)),
                cap_col=self._f32(torch.arange(capacity, dtype=torch.float32).reshape(1, 1, capacity, 1)),
                cap_row=self._f32(torch.arange(capacity, dtype=torch.float32).reshape(1, 1, 1, capacity)),
            )
        return self._consts[key]

    def _topk_mask(self, topk, B, S, consts):
        """Indexer output [B,S,W] int32 (token ids, -1 = none) -> additive mask [B,1,S,S] bf16:
        0 where key j is among query i's selected tokens, -1e9 elsewhere (HF build_attention_mask_from_topk).
        For S <= index_topk every selected id lies in the first S + kpool - 1 columns (the pools + tail)."""
        W = min(int(topk.shape[-1]), S + self.kpool - 1)
        t = ttnn.typecast(ttnn.to_layout(topk, ttnn.ROW_MAJOR_LAYOUT), ttnn.float32)
        t = ttnn.slice(t, [0, 0, 0], [B, S, W])
        t = ttnn.to_layout(ttnn.reshape(t, [B, S, W, 1]), ttnn.TILE_LAYOUT)
        pad = lambda n: -(-n // 32) * 32
        hits = [  # [B,rows,1,S] count of selections; the [B,rows,W,S] comparison is formed per row chunk
            ttnn.sum(
                ttnn.eq(t if (s0, s1) == (0, S) else ttnn.slice(t, [0, s0, 0, 0], [B, s1, W, 1]), consts["key_ids"]),
                dim=2,
                keepdim=True,
            )
            for s0, s1 in row_chunks(S, B * pad(W) * pad(S))
        ]
        hit = hits[0] if len(hits) == 1 else ttnn.concat(hits, dim=1)
        sel = ttnn.gt(hit, 0.0)
        mask = ttnn.multiply(ttnn.subtract(sel, 1.0), 1e9)
        return ttnn.typecast(self._rs(mask, [B, 1, S, S]), ttnn.bfloat16)

    def _project(self, x, B, S):
        """x [1,1,B*S,D] -> q_resid [1,1,B*S,q_lora], per-head q/k/v [B,Hl,S,d]."""
        q_resid = self._rms(self._linear(x, self.w["wq_a"]), self.w["q_norm"], self.q_eps)
        kv = self._rms(self._linear(x, self.w["wkv_a"]), self.w["kv_norm"], self.kv_eps)
        up = lambda t, w, d: self._split_heads(self._rs(self._linear(t, w), [B, 1, S, self.H_local * d]), d)
        return (
            q_resid,
            up(q_resid, self.w["wq_b"], self.d_qk),
            up(kv, self.w["wk_b"], self.d_qk),
            up(kv, self.w["wv_b"], self.d_v),
        )

    def _o_proj(self, attn, B, S):
        out = self._linear(
            self._rs(self._merge_heads(attn, self.d_v), [1, 1, B * S, self.H_local * self.d_v]), self.w["wo"]
        )
        if self.n_dev > 1:  # all_gather(partials) + sum (ttnn.all_reduce deadlocks on the 1x32 line)
            gathered = ttnn.all_gather(out, dim=1, cluster_axis=self.cluster_axis, topology=ttnn.Topology.Linear)
            out = ttnn.sum(gathered, dim=1, keepdim=True)
        return out  # [1,1,B*S,D] replicated

    def prefill(self, x, B, S, real_len=None, capacity=None):
        """Batched prefill with the DSA indexer selecting each query's keys.
        x [1,1,B*S,D] bf16 (sample-major rows, right-padded to S). Returns (out [1,1,B*S,D],
        (k_cache, v_cache) [B,Hl,capacity,d] bf16 with positions [0, S) filled)."""
        capacity = S if capacity is None else int(capacity)
        if S > self.index_topk or capacity > self.index_topk:
            raise NotImplementedError(f"length > index_topk={self.index_topk}: sparse selection drops pools")
        consts = self.prepare(S, capacity)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        q_resid, q, k, v = self._project(x, B, S)
        D = x.shape[-1]
        topk = self.indexer(ttnn.reshape(x, [B, S, D]), ttnn.reshape(q_resid, [B, S, q_resid.shape[-1]]))
        mask = self._topk_mask(topk, B, S, consts)
        scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1), compute_kernel_config=self.compute_kernel_config)
        scores = ttnn.add(ttnn.multiply(scores, self.scale), mask)
        probs = ttnn.softmax(scores, dim=-1, numeric_stable=True)
        attn = ttnn.matmul(probs, v, compute_kernel_config=self.compute_kernel_config)  # [B,Hl,S,dv]
        pad = capacity - S
        kc = ttnn.pad(k, [(0, 0), (0, 0), (0, pad), (0, 0)], 0.0) if pad else k
        vc = ttnn.pad(v, [(0, 0), (0, 0), (0, pad), (0, 0)], 0.0) if pad else v
        return self._o_proj(attn, B, S), (kc, vc)

    def decode(self, x, cache, pos):
        """One token per sample at position ``pos`` (device fp32 [1,1,1,1], same for every sample).
        For a context <= index_topk the indexer keeps every visible pool + the open tail, i.e. the
        causal set, so the selection is the causal mask over [0, pos]."""
        kc, vc = cache
        B, cap = kc.shape[0], kc.shape[2]
        consts = self.prepare(cap, cap)
        q_resid, q, k, v = self._project(x, B, 1)
        oh = ttnn.typecast(ttnn.eq(consts["cap_col"], pos), ttnn.bfloat16)  # [1,1,cap,1] write slot
        keep = ttnn.rsub(oh, 1.0)
        kc = ttnn.add(ttnn.multiply(kc, keep), ttnn.multiply(oh, k))
        vc = ttnn.add(ttnn.multiply(vc, keep), ttnn.multiply(oh, v))
        future = ttnn.typecast(ttnn.multiply(ttnn.gt(consts["cap_row"], pos), -1e9), ttnn.bfloat16)  # [1,1,1,cap]
        scores = ttnn.matmul(q, ttnn.transpose(kc, -2, -1), compute_kernel_config=self.compute_kernel_config)
        scores = ttnn.add(ttnn.multiply(scores, self.scale), future)
        probs = ttnn.softmax(scores, dim=-1, numeric_stable=True)
        attn = ttnn.matmul(probs, vc, compute_kernel_config=self.compute_kernel_config)  # [B,Hl,1,dv]
        return self._o_proj(attn, B, 1), (kc, vc)

    def _mask(self, S, attention_mask):
        causal = torch.tril(torch.ones(S, S, dtype=torch.bool))
        if attention_mask is not None:
            am = attention_mask
            if not isinstance(am, torch.Tensor):
                am = ttnn.to_torch(am)
            am = am.reshape(-1, am.shape[-1]).bool()
            if am.shape[0] != 1:
                raise NotImplementedError("batch > 1 not supported by this port")
            causal = causal & am[0][None, :S]
        add = torch.where(causal, 0.0, -1e9).to(torch.bfloat16).reshape(1, 1, S, S)
        kw = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        if self.n_dev > 1:
            kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self.device)
        return ttnn.from_torch(add, **kw)

    # ---------------------------------------------------------------- forward
    def __call__(self, hidden_states, attention_mask=None, past_key_values=None, prev_topk_indices=None, **kwargs):
        if past_key_values is not None:
            raise NotImplementedError("KV-cache path not ported in this component")
        x = hidden_states
        in_shape = list(x.shape)
        S, D = in_shape[-2], in_shape[-1]
        if S > self.index_topk:
            raise NotImplementedError(
                f"S={S} > index_topk={self.index_topk}: DSA top-k drops pools; sparse selection not ported"
            )
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = ttnn.reshape(x, [1, 1, S, D])
        if attention_mask is not None:
            am = attention_mask if isinstance(attention_mask, torch.Tensor) else ttnn.to_torch(attention_mask)
            if not bool(am.bool().all()):
                raise NotImplementedError("padded single-call path not ported; use prefill(real_len=...)")
        out, _ = self.prefill(ttnn.typecast(x, ttnn.bfloat16), 1, S)
        return ttnn.reshape(out, in_shape)


def build(device, torch_module=None):
    return TtTextAttention.build(device, torch_module)


def text_attention(device, torch_module=None):
    return TtTextAttention.build(device, torch_module)

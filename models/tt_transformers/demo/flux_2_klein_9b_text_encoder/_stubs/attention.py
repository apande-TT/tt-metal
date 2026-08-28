# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN, tensor-parallel port of the Qwen3 GQA attention block.

Component `attention` of `/tmp/tt_hw_planner_components/flux_2_klein_9b_text_encoder`
(`model.layers.0.self_attn`, `Qwen3Attention`).

Shapes for this model: hidden=4096, head_dim=128, n_heads=32, n_kv_heads=8
(GQA group size 4), no projection biases, per-head q_norm / k_norm RMSNorm
over head_dim only.

Tensor-parallel scheme (TP = number of mesh devices; 8 here)
------------------------------------------------------------
q/k/v_proj are COLUMN-parallel. Their outputs feed per-head ops (RMSNorm,
RoPE, softmax), so each chip owns a contiguous slice of the head axis:
n_heads/TP = 4 query heads and n_kv_heads/TP = 1 key/value head per chip.
Query head `j` attends to kv head `j // 4`, so the contiguous split lands
query heads 4i..4i+3 and kv head i together on chip i -- every chip holds a
self-contained GQA group and the whole attention core (RoPE, scores,
softmax, context) runs with NO cross-chip traffic.

o_proj is ROW-parallel: its input axis is that same head axis, so chip i
owns input rows [512i, 512i+512) and produces a PARTIAL sum over the full
hidden dim. A single `ttnn.all_reduce` sums the partials and leaves the
full-width result replicated on every chip -- which is what the harness
gathers and compares against the single-device golden.

Replicated (never sharded): q_norm / k_norm gammas (they act on head_dim,
which is not split), the RoPE cos/sin tables, and the activations entering
q/k/v_proj.

The k and v projections are packed into ONE per-chip weight laid out as
[K_i | V_i] so `nlp_create_qkv_heads` can do the head split tile-natively
(this build's row-major reshape kernel does not compile, so reshapes that
split the last dim are avoided entirely).

Placement changes; the math does not.
"""
from __future__ import annotations

import torch

import ttnn

TILE = 32


def _num_devices(device) -> int:
    fn = getattr(device, "get_num_devices", None)
    if callable(fn):
        try:
            return int(fn())
        except Exception:
            pass
    ids = getattr(device, "get_device_ids", None)
    if callable(ids):
        try:
            return max(1, len(ids()))
        except Exception:
            pass
    return 1


def _is_mesh(device) -> bool:
    try:
        if isinstance(device, ttnn.MeshDevice):
            return True
    except AttributeError:
        pass
    return hasattr(device, "get_device_ids") or hasattr(device, "get_devices")


class TtAttention:
    """Native ttnn Qwen3 GQA attention, column/row-parallel over the mesh."""

    def __init__(self, device, torch_module) -> None:
        self.device = device
        self.mesh = _is_mesh(device)
        self.tp = _num_devices(device) if self.mesh else 1

        cfg = getattr(torch_module, "config", None)
        self.head_dim = int(getattr(torch_module, "head_dim", None) or getattr(cfg, "head_dim", 0) or 128)
        sd = {k: v.detach().float() for k, v in torch_module.state_dict().items()}

        wq = sd["q_proj.weight"]  # [n_heads*hd, hidden]
        wk = sd["k_proj.weight"]  # [n_kv*hd,    hidden]
        wv = sd["v_proj.weight"]
        wo = sd["o_proj.weight"]  # [hidden, n_heads*hd]

        self.hidden = int(wq.shape[1])
        self.n_heads = int(wq.shape[0]) // self.head_dim
        self.n_kv_heads = int(wk.shape[0]) // self.head_dim
        self.scaling = float(getattr(torch_module, "scaling", self.head_dim**-0.5))

        if self.n_heads % self.tp or self.n_kv_heads % self.tp:
            # TP must divide BOTH head counts, else a chip holds a partial GQA
            # group and the attention core would need cross-chip traffic.
            self.tp = 1
        self.n_local_heads = self.n_heads // self.tp
        self.n_local_kv_heads = self.n_kv_heads // self.tp
        self.n_rep = self.n_local_heads // self.n_local_kv_heads

        def _shard(t, dim):
            mapper = ttnn.ShardTensorToMesh(device, dim=dim) if (self.mesh and self.tp > 1) else None
            return ttnn.from_torch(
                t.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        def _replicate(t, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                t.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=layout,
                device=device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device) if self.mesh else None,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # ttnn.linear wants [in_features, out_features]; nn.Linear stores [out, in].
        wq_t, wk_t, wv_t = wq.t().contiguous(), wk.t().contiguous(), wv.t().contiguous()

        # Column-parallel: split the OUTPUT (head) axis.
        self.wq = _shard(wq_t, -1)
        # K and V are packed per chip as [K_i | V_i] so that a plain dim=-1
        # shard hands each chip its own kv head pair -- a naive cat([K, V])
        # would instead give chip 0 two K heads and no V.
        kvw = self.n_local_kv_heads * self.head_dim
        self.wkv = _shard(
            torch.cat(
                [
                    torch.cat([wk_t[:, i * kvw : (i + 1) * kvw], wv_t[:, i * kvw : (i + 1) * kvw]], dim=-1)
                    for i in range(self.tp)
                ],
                dim=-1,
            ).contiguous(),
            -1,
        )
        # Row-parallel: split the INPUT (head) axis.
        self.wo = _shard(wo.t().contiguous(), 0)

        # Norm gammas act on head_dim (never sharded) -> replicated, in the
        # (1, 1, dim//32, 32) ROW_MAJOR form ttnn.rms_norm expects.
        qn, kn = sd.get("q_norm.weight"), sd.get("k_norm.weight")
        self.eps = float(
            getattr(getattr(torch_module, "q_norm", None), "variance_epsilon", 0.0)
            or getattr(cfg, "rms_norm_eps", 1e-6)
        )
        self.q_norm = (
            _replicate(qn.reshape(1, 1, self.head_dim // TILE, TILE), ttnn.ROW_MAJOR_LAYOUT) if qn is not None else None
        )
        self.k_norm = (
            _replicate(kn.reshape(1, 1, self.head_dim // TILE, TILE), ttnn.ROW_MAJOR_LAYOUT) if kn is not None else None
        )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._rope_cache = {}
        self._mask_cache = {}

    # ---------------------------------------------------------------- helpers
    def _stage(self, t, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=layout,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if self.mesh else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _rope_tables(self, position_embeddings):
        """Stage (cos, sin) once per call site, pre-broadcast to the local head counts."""
        key = id(position_embeddings)
        hit = self._rope_cache.get(key)
        if hit is not None:
            return hit
        cos, sin = position_embeddings

        def _b(t, n_local):
            t = t.reshape(1, 1, t.shape[-2], t.shape[-1]).expand(1, n_local, -1, -1).contiguous()
            return self._stage(t)

        tables = (
            _b(cos, self.n_local_heads),
            _b(sin, self.n_local_heads),
            _b(cos, self.n_local_kv_heads),
            _b(sin, self.n_local_kv_heads),
        )
        self._rope_cache[key] = tables
        return tables

    def _rotate_half(self, x):
        half = self.head_dim // 2
        s = list(x.shape)
        x1 = ttnn.slice(x, [0, 0, 0, 0], [s[0], s[1], s[2], half])
        x2 = ttnn.slice(x, [0, 0, 0, half], [s[0], s[1], s[2], self.head_dim])
        return ttnn.concat([ttnn.neg(x2), x1], dim=-1)

    def _apply_rope(self, x, cos, sin):
        return ttnn.add(ttnn.mul(x, cos), ttnn.mul(self._rotate_half(x), sin))

    def _mask_bias(self, attention_mask, seq_len):
        """HF adds `attention_mask` straight onto the (B, H, S, S) score matrix.

        A mask that is constant along the score axis (the all-ones mask this
        harness builds) cancels inside softmax, so it is dropped rather than
        materialised; anything else is broadcast to the local head count and
        added on device.
        """
        if attention_mask is None or not torch.is_tensor(attention_mask):
            return None
        key = id(attention_mask)
        if key in self._mask_cache:
            return self._mask_cache[key]
        bias = torch.zeros(1, 1, seq_len, seq_len) + attention_mask.to(torch.float32)
        out = None
        if not torch.allclose(bias, bias[..., :1].expand_as(bias)):
            out = self._stage(bias.expand(bias.shape[0], self.n_local_heads, seq_len, seq_len).contiguous())
        self._mask_cache[key] = out
        return out

    def _repeat_kv(self, x):
        """GQA: expand this chip's kv heads to its query-head count."""
        if self.n_rep == 1:
            return x
        if self.n_local_kv_heads == 1:
            return ttnn.repeat(x, ttnn.Shape([1, self.n_rep, 1, 1]))
        return ttnn.repeat_interleave(x, self.n_rep, 1)

    # ---------------------------------------------------------------- forward
    def __call__(self, *args, **kwargs):
        hidden_states = kwargs.pop("hidden_states", None)
        position_embeddings = kwargs.pop("position_embeddings", None)
        attention_mask = kwargs.pop("attention_mask", None)

        # The harness can elect a non-primary arg as the positional one, so
        # classify leftovers by type instead of trusting position.
        for a in args:
            if isinstance(a, tuple) and len(a) == 2 and position_embeddings is None:
                position_embeddings = a
            elif hidden_states is None:
                hidden_states = a
            elif attention_mask is None and a is not None:
                attention_mask = a
        if hidden_states is None:
            raise ValueError("attention stub: no hidden_states supplied")

        x = self._stage(hidden_states) if torch.is_tensor(hidden_states) else hidden_states
        in_shape = list(x.shape)
        seq_len = int(in_shape[-2])
        if len(in_shape) != 4:
            x = ttnn.reshape(x, (x.shape[0], 1, seq_len, in_shape[-1]))

        xq = ttnn.linear(x, self.wq, compute_kernel_config=self.compute_kernel_config)
        xkv = ttnn.linear(x, self.wkv, compute_kernel_config=self.compute_kernel_config)

        # Tile-native head split: [1, 1, S, nq*hd] + [1, 1, S, 2*nkv*hd] ->
        # q [1, nq, S, hd], k/v [1, nkv, S, hd].
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xq,
            xkv,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False,
        )

        if self.q_norm is not None:
            q = ttnn.rms_norm(q, epsilon=self.eps, weight=self.q_norm, compute_kernel_config=self.compute_kernel_config)
        if self.k_norm is not None:
            k = ttnn.rms_norm(k, epsilon=self.eps, weight=self.k_norm, compute_kernel_config=self.compute_kernel_config)

        if position_embeddings is not None:
            cos_q, sin_q, cos_k, sin_k = self._rope_tables(position_embeddings)
            q = self._apply_rope(q, cos_q, sin_q)
            k = self._apply_rope(k, cos_k, sin_k)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1), compute_kernel_config=self.compute_kernel_config)
        scores = ttnn.mul(scores, self.scaling)
        bias = self._mask_bias(attention_mask, seq_len)
        if bias is not None:
            scores = ttnn.add(scores, bias)
        scores = ttnn.softmax(scores, dim=-1, compute_kernel_config=self.compute_kernel_config)

        context = ttnn.matmul(scores, v, compute_kernel_config=self.compute_kernel_config)
        context = ttnn.experimental.nlp_concat_heads(context)  # [1, 1, S, nq*hd]

        # Row-parallel o_proj -> per-chip PARTIAL sums over the full hidden dim.
        out = ttnn.linear(context, self.wo, compute_kernel_config=self.compute_kernel_config)
        if self.tp > 1:
            out = ttnn.all_reduce(out)
        if len(in_shape) != 4:
            out = ttnn.reshape(out, tuple(in_shape[:-1]) + (self.hidden,))
        return out

    @classmethod
    def build(cls, device, torch_module):
        if torch_module is None:
            raise RuntimeError("attention stub needs the torch reference module to source its weights")
        return cls(device, torch_module)


def build(device, torch_module=None):
    return TtAttention.build(device, torch_module)


def attention(device, torch_module=None):
    return TtAttention.build(device, torch_module)

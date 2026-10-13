# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native tensor-parallel TTNN port of `ministral3_attention` (Ministral3Attention,
`language_model.layers.0.self_attn`) of `mistralai/Devstral-Small-2-24B-Instruct-2512`.

    q, k, v = x Wq, x Wk, x Wv                       (32 q heads, 8 kv heads, head_dim 128 -> GQA 4:1)
    q, k    = rope(q), rope(k)
    q       = q * (1 + beta * log(1 + floor(pos / orig_max)))      (llama-4 attention scaling)
    out     = sdpa(q, k, v, mask, scale=head_dim**-0.5) Wo

The canonical tt_transformers Attention is not used: it has no llama-4 query scaling and its
ModelArgs path does not load this multimodal checkpoint.

Two modes, one weight set:
  * prefill  x (B, 1, S, D): causal SDPA over the prompt. With `kv_cache=(k, v)` the prompt's K/V
             (B, kv_heads/TP, S, head_dim) are written into the resident cache rows 0..S-1.
  * decode   x (1, 1, B, D): one token per batch row. K/V are appended at `cur_pos` (int32 [B]) with
             paged_update_cache and attention reads the resident cache with SDPA-decode.

Tensor-parallel over the mesh (TP = number of devices; TP = 1 on a single device):
  * q/k/v  column-parallel by HEAD: device d holds q heads [d*Hq/TP, (d+1)*Hq/TP) and the kv heads
           those q heads attend with ([d*Hkv/TP, (d+1)*Hkv/TP)), so every GQA group stays local.
           Fused per device as [q | k | v] into one wqkv.
  * o_proj row-parallel (input features = that device's q heads), then all_reduce.
  * rotary tables, the mask and the llama-4 scale are replicated; the KV cache holds the local heads.
"""
from __future__ import annotations

import math

import torch

import ttnn


def tp_ccl_args():
    """Topology / links for the text decoder's TP collectives: Ring over 2 links when the ring fabric is active
    (QB2, see mesh.py), else Linear on the default link count."""
    try:
        if ttnn.get_fabric_config() == ttnn.FabricConfig.FABRIC_1D_RING:
            return {"topology": ttnn.Topology.Ring, "num_links": 2}
    except Exception:  # noqa: BLE001
        pass
    return {"topology": ttnn.Topology.Linear}


def _mesh_size(device):
    try:
        return int(device.get_num_devices())
    except Exception:
        return 1


class TtMinistral3Attention:
    # dtype of o_proj's output and its all_reduce: fp32, added unrounded into the decoder's fp32 residual stream.
    OUT_DTYPE = ttnn.float32
    # Widest SDPA key chunk prefill may use (see __call__).
    PREFILL_MAX_K_CHUNK = 512

    def __init__(self, device, torch_module) -> None:
        self.device = device
        self.tp = tp = _mesh_size(device)
        cfg = torch_module.config
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = hd = torch_module.head_dim
        self.scale = float(torch_module.scaling)
        rope = getattr(cfg, "rope_parameters", None) or {}
        self.l4_beta = rope.get("llama_4_scaling_beta")
        self.l4_max = rope.get("original_max_position_embeddings")
        assert (
            self.num_heads % tp == 0 and self.num_kv_heads % tp == 0
        ), f"heads {self.num_heads}/{self.num_kv_heads} not divisible by TP={tp}"
        self.local_heads = lh = self.num_heads // tp
        self.local_kv_heads = lkv = self.num_kv_heads // tp

        is_mesh = tp > 1
        shard = lambda dim: ttnn.ShardTensorToMesh(device, dim=dim) if is_mesh else None
        self._repl = repl = ttnn.ReplicateTensorToMesh(device) if is_mesh else None

        def _put(t, mapper):
            return ttnn.from_torch(
                t.detach().to(torch.bfloat16).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        wq = torch_module.q_proj.weight.T  # (in, Hq*hd)
        wk = torch_module.k_proj.weight.T  # (in, Hkv*hd)
        wv = torch_module.v_proj.weight.T
        chunks = []
        for d in range(tp):
            q_sl = slice(d * lh * hd, (d + 1) * lh * hd)
            kv_sl = slice(d * lkv * hd, (d + 1) * lkv * hd)
            chunks += [wq[:, q_sl], wk[:, kv_sl], wv[:, kv_sl]]
        self.wqkv = _put(torch.concat(chunks, dim=-1), shard(-1))
        self.wo = _put(torch_module.o_proj.weight.T, shard(-2))

        # rotate_half(x) == x @ R  (HF convention: cat(-x2, x1)).
        half = hd // 2
        rot = torch.zeros(hd, hd)
        rot[torch.arange(half) + half, torch.arange(half)] = -1.0
        rot[torch.arange(half), torch.arange(half) + half] = 1.0
        self.rot = _put(rot.reshape(1, 1, hd, hd), repl)

        self.compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,  # L1 accumulation made batched prefill matmuls non-repeatable
        )
        self.sdpa_decode_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            q_chunk_size=0,
            k_chunk_size=0,
            exp_approx_mode=False,
        )
        self._head_cfgs = {}

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def _to_tt(self, t, dtype=ttnn.bfloat16):
        return ttnn.from_torch(
            t.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=self._repl,
        )

    def _head_cfg(self, batch):
        """HEIGHT-sharded L1 config holding one batch row's [heads (tile-padded), head_dim] per core."""
        if batch not in self._head_cfgs:
            grid = self.device.compute_with_storage_grid_size()
            cores = ttnn.num_cores_to_corerangeset(batch, grid, row_wise=True)
            self._head_cfgs[batch] = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, self.head_dim),
                core_grid=cores,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
        return self._head_cfgs[batch]

    def _reduce(self, t):
        """all_reduce of o_proj's fp32 partials: reduce-scatter in fp32 (the sum is exact to fp32), round the reduced
        1/TP shard to bf16 once and all_gather that -- half the gather bytes. The decoder adds the bf16 result into
        its fp32 residual unrounded (mixed-dtype ADD runs on the SFPU in fp32)."""
        if self.tp == 1:
            return t
        ccl = tp_ccl_args()
        t = ttnn.reduce_scatter(t, dim=3, **ccl)
        return ttnn.all_gather(ttnn.typecast(t, ttnn.bfloat16), dim=3, **ccl)

    def _rope(self, x, cos, sin):
        return ttnn.add(
            ttnn.multiply(x, cos),
            ttnn.multiply(ttnn.matmul(x, self.rot, compute_kernel_config=self.compute_cfg), sin),
        )

    def _llama4_scale(self, position_ids, shape):
        """Per-position query scale 1 + beta * log(1 + floor(pos / orig_max)), on device, in `shape`."""
        pos = position_ids
        if isinstance(pos, torch.Tensor):
            pos = self._to_tt(pos.reshape(shape).to(torch.float32), dtype=ttnn.float32)
        t = ttnn.floor(ttnn.multiply(pos, 1.0 / float(self.l4_max)))
        t = ttnn.log(ttnn.add(t, 1.0))
        scale = ttnn.add(ttnn.multiply(t, float(self.l4_beta)), 1.0)
        return ttnn.typecast(ttnn.reshape(scale, shape), ttnn.bfloat16)

    @classmethod
    def sdpa_len(cls, s):
        """Rows prefill SDPA runs on for an s-row prompt: one power-of-two key chunk (32..PREFILL_MAX_K_CHUNK) that
        holds it, else s rounded up to a multiple of PREFILL_MAX_K_CHUNK."""
        if s > cls.PREFILL_MAX_K_CHUNK:
            return int(math.ceil(s / cls.PREFILL_MAX_K_CHUNK) * cls.PREFILL_MAX_K_CHUNK)
        n = 32
        while n < s:
            n *= 2
        return n

    def query_rope_tables(self, cos, sin, position_ids, shape):
        """(cos, sin) for q with the llama-4 query scale folded in: rope(q) * s == q * (cos * s) + rot(q) * (sin * s).

        The scale depends only on the position, so the model builds these ONCE per forward and every layer shares
        them, instead of each layer re-deriving the scale (floor/log/...) and multiplying its q by it. Below
        original_max_position_embeddings the scale is exactly 1, so the shared tables are bit-identical."""
        if self.l4_beta is None or not self.l4_max or position_ids is None:
            return cos, sin
        scale = self._llama4_scale(position_ids, shape)
        return ttnn.multiply(cos, scale), ttnn.multiply(sin, scale)

    def _fill_cache(self, cache, kv):
        """Write prompt K or V (B, kv_heads, S, hd) into rows 0..S-1 of the resident (B, kv_heads, C, hd) cache."""
        s, cap = kv.shape[2], cache.shape[2]
        if s < cap:
            kv = ttnn.pad(kv, padding=((0, 0), (0, 0), (0, cap - s), (0, 0)), value=0.0)
        ttnn.copy(kv, cache)

    def __call__(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        position_ids=None,
        kv_cache=None,
        cur_pos=None,
        mode="prefill",
        **kwargs,
    ):
        if mode == "decode":
            return self._decode(hidden_states, position_embeddings, position_ids, kv_cache, cur_pos)
        x = hidden_states
        in_rank = len(x.shape)
        if in_rank == 3:
            x = ttnn.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
        b, _, s, _ = x.shape
        hd = self.head_dim

        cos, sin = position_embeddings[:2]
        q_tables = tuple(position_embeddings[2:4]) if len(position_embeddings) == 4 else None
        if isinstance(cos, torch.Tensor):
            cos = self._to_tt(cos.to(torch.bfloat16).reshape(1, 1, s, hd))
        if isinstance(sin, torch.Tensor):
            sin = self._to_tt(sin.to(torch.bfloat16).reshape(1, 1, s, hd))
        mask = attention_mask
        if isinstance(mask, torch.Tensor):
            mask = self._to_tt(mask.reshape(1, 1, s, s).to(torch.bfloat16))

        # Prefill QKV on the full grid (ttnn.linear's auto config left this many-row projection on a partial one).
        qkv = ttnn.experimental.minimal_matmul(
            x,
            self.wqkv,
            compute_kernel_config=self.compute_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=self.local_heads, num_kv_heads=self.local_kv_heads, transpose_k_heads=False
        )
        k = self._rope(k, cos, sin)
        if q_tables is not None:
            q = self._rope(q, *q_tables)  # llama-4 scale already folded into the shared q tables
        else:
            q = self._rope(q, cos, sin)
            if self.l4_beta is not None and self.l4_max and position_ids is not None:
                q = ttnn.multiply(q, self._llama4_scale(position_ids, (1, 1, s, 1)))
        if kv_cache is not None:
            self._fill_cache(kv_cache[0], k)
            self._fill_cache(kv_cache[1], v)

        # SDPA's online softmax rounds its running state once per KEY chunk, which biases the output level
        # upward (layer-8 Q/K/V of the image prompt: +0.23% at k_chunk 32, +0.10% at 128, +0.02% at 512).
        # Image-token rows amplify that bias over 40 layers (prefill hidden PCC 0.95 on those rows at k_chunk
        # 32), so prefill attends over the widest key chunk: q/k/v are padded HERE (and only here -- no other
        # layer computes pad rows) to one chunk when the prompt fits in one, else to a multiple of the widest.
        # The pad keys sit after every real row, so causal attention leaves the real rows unchanged; the q chunk
        # does not affect the level.
        s_attn = self.sdpa_len(s)
        if s_attn != s:
            pad = ((0, 0), (0, 0), (0, s_attn - s), (0, 0))
            q, k, v = (ttnn.pad(t, padding=pad, value=0.0) for t in (q, k, v))
        k_chunk = next(c for c in (512, 256, 128, 64, 32) if s_attn % c == 0 and c <= self.PREFILL_MAX_K_CHUNK)
        q_chunk = 64 if s_attn % 64 == 0 else 32
        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=mask is None,
            attn_mask=mask,
            scale=self.scale,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
                q_chunk_size=q_chunk,
                k_chunk_size=k_chunk,
                exp_approx_mode=False,
            ),
            compute_kernel_config=self.compute_cfg,
        )
        if s_attn != s:
            attn = ttnn.slice(attn, (0, 0, 0, 0), (b, self.local_heads, s, hd))
        attn = ttnn.experimental.nlp_concat_heads(attn)
        out = self._reduce(ttnn.linear(attn, self.wo, dtype=self.OUT_DTYPE, compute_kernel_config=self.compute_cfg))
        if in_rank == 3:
            return ttnn.reshape(out, (b, s, out.shape[-1]))
        return out

    def _decode(self, x, position_embeddings, position_ids, kv_cache, cur_pos):
        """x (1, 1, B, D) -> (1, 1, B, D); cos/sin (1, B, 1, hd); cur_pos int32 [B]; position_ids (B, 1, 1, 1) fp32."""
        b = x.shape[-2]
        head_cfg = self._head_cfg(b)
        qkv = ttnn.linear(x, self.wqkv, compute_kernel_config=self.compute_cfg)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv, num_heads=self.local_heads, num_kv_heads=self.local_kv_heads, memory_config=head_cfg
        )
        cos, sin = position_embeddings[:2]
        k = self._rope(ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG), cos, sin)
        if len(position_embeddings) == 4:
            q = self._rope(ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG), *position_embeddings[2:4])
        else:
            q = self._rope(ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG), cos, sin)
            if self.l4_beta is not None and self.l4_max and position_ids is not None:
                q = ttnn.multiply(q, self._llama4_scale(position_ids, (1, b, 1, 1)))
        k = ttnn.to_memory_config(k, head_cfg)
        k_cache, v_cache = kv_cache
        ttnn.experimental.paged_update_cache(k_cache, k, update_idxs_tensor=cur_pos)
        ttnn.experimental.paged_update_cache(v_cache, v, update_idxs_tensor=cur_pos)
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            k_cache,
            v_cache,
            cur_pos_tensor=cur_pos,
            scale=self.scale,
            program_config=self.sdpa_decode_cfg,
            compute_kernel_config=self.compute_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attn = ttnn.reshape(attn, (1, 1, b, self.local_heads * self.head_dim))  # concat heads per row
        return self._reduce(ttnn.linear(attn, self.wo, dtype=self.OUT_DTYPE, compute_kernel_config=self.compute_cfg))


def build(device, torch_module=None):
    return TtMinistral3Attention.build(device, torch_module)


def ministral3_attention(device, torch_module=None):
    return TtMinistral3Attention.build(device, torch_module)

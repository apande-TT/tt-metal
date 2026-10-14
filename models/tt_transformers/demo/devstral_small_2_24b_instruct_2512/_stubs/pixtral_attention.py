# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native tensor-parallel TTNN port of `pixtral_attention` (PixtralAttention,
`vision_tower.transformer.layers.0.attention`) of `mistralai/Devstral-Small-2-24B-Instruct-2512`.

    q, k, v = x Wq, x Wk, x Wv                     (16 heads, head_dim 64, MHA)
    out     = sdpa(rope(q), rope(k), v, mask, scale=head_dim**-0.5, non-causal) Wo

Tensor-parallel over the mesh (TP = number of devices; TP = 1 on a single device):
  * q/k/v   column-parallel by HEAD: each device holds num_heads/TP heads, fused into one wqkv.
  * o_proj  row-parallel (input features = that device's heads), then all_reduce.
  * rotary tables and the mask are replicated.
"""
from __future__ import annotations

import torch

import ttnn
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs.ministral3_attention import fold_input_scale


def _mesh_size(device):
    try:
        return int(device.get_num_devices())
    except Exception:
        return 1


class TtPixtralAttention:
    def __init__(self, device, torch_module, in_scale=None) -> None:
        """in_scale: the attention_norm gamma, folded into q/k/v (see TtAttention.FOLD_NORM_GAMMA)."""
        self.device = device
        self.tp = tp = _mesh_size(device)
        self.num_heads = torch_module.num_heads
        self.head_dim = hd = torch_module.head_dim
        self.scale = float(torch_module.scaling)
        assert self.num_heads % tp == 0, f"{self.num_heads} heads not divisible by TP={tp}"
        self.local_heads = lh = self.num_heads // tp

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

        # Fused per-device [q_heads | k_heads | v_heads] so a dim=-1 shard hands each device its own heads.
        wq, wk, wv = (
            fold_input_scale(w.weight.T, in_scale)
            for w in (torch_module.q_proj, torch_module.k_proj, torch_module.v_proj)
        )
        chunks = []
        for d in range(tp):
            sl = slice(d * lh * hd, (d + 1) * lh * hd)
            chunks += [wq[:, sl], wk[:, sl], wv[:, sl]]
        self.wqkv = _put(torch.hstack(chunks), shard(-1))  # 2-D slices: hstack == concat on dim -1
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

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def _to_tt(self, t):
        return ttnn.from_torch(
            t.to(torch.bfloat16).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=self._repl,
        )

    def _rope(self, x, cos, sin):
        # rotate_half as x @ R on minimal_matmul, which folds the (images, heads) batch into its rows over the full
        # grid (ttnn.matmul ran the 32 x 4 x 256-row batch on 16 cores: 252 us); R is a +-1 permutation, so exact.
        rot = ttnn.experimental.minimal_matmul(
            x, self.rot, compute_kernel_config=self.compute_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        return ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(rot, sin))

    def __call__(self, hidden_states, attention_mask=None, position_embeddings=None, **kwargs):
        x = hidden_states
        in_shape = tuple(x.shape)
        if len(in_shape) == 3:  # HF layout (B, S, D)
            x = ttnn.reshape(x, (in_shape[0], 1, in_shape[1], in_shape[2]))
        # Batch is the leading axis (one image per row); take it from the tensor, never assume 1.
        b, s = x.shape[0], x.shape[2]
        hd = self.head_dim

        cos, sin = position_embeddings
        cos = self._to_tt(cos.reshape(1, 1, s, hd)) if isinstance(cos, torch.Tensor) else cos
        sin = self._to_tt(sin.reshape(1, 1, s, hd)) if isinstance(sin, torch.Tensor) else sin
        mask = attention_mask
        if isinstance(mask, torch.Tensor):
            mask = self._to_tt(mask.reshape(1, 1, s, s).float())

        # minimal_matmul folds the images into its rows over the full grid (ttnn.linear ran this batch of 32 x 256
        # rows as a batched matmul on 64 cores: 816 us).
        qkv = ttnn.experimental.minimal_matmul(
            x, self.wqkv, compute_kernel_config=self.compute_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=self.local_heads, num_kv_heads=self.local_heads, transpose_k_heads=False
        )
        q = self._rope(q, cos, sin)
        k = self._rope(k, cos, sin)
        if q.dtype != ttnn.bfloat16:  # fp32 activations upstream; SDPA reads bf16
            q, k, v = (ttnn.typecast(t, ttnn.bfloat16) for t in (q, k, v))
        # Widest key chunk the sequence allows: SDPA's per-key-chunk rescale rounding biases its output level
        # upward (+0.23% at k_chunk 32, +0.02% at 512 on the text model's attention).
        k_chunk = next(c for c in (512, 256, 128, 64, 32) if s % c == 0)
        q_chunk = 64 if s % 64 == 0 else 32
        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
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
        attn = ttnn.experimental.nlp_concat_heads(attn)
        # fp32 output: it is added straight into the vision layer's fp32 residual stream.
        out = ttnn.linear(attn, self.wo, dtype=ttnn.float32, compute_kernel_config=self.compute_cfg)
        if self.tp > 1:
            out = ttnn.all_reduce(out, topology=ttnn.Topology.Linear)
        if len(in_shape) == 3:
            return ttnn.reshape(out, (b, s, out.shape[-1]))
        return out


def build(device, torch_module=None):
    return TtPixtralAttention.build(device, torch_module)


def pixtral_attention(device, torch_module=None):
    return TtPixtralAttention.build(device, torch_module)

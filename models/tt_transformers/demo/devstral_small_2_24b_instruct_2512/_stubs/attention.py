# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native tensor-parallel TTNN port of `attention` (PixtralAttentionLayer, vision_tower layer 0)
of `mistralai/Devstral-Small-2-24B-Instruct-2512`.

    h   = x + attention(attention_norm(x))             attention = o_proj(sdpa(rope(q), rope(k), v, mask))
    out = h + feed_forward(ffn_norm(h))                feed_forward = down(silu(gate(n)) * up(n))

Composed from the graduated child ports, each holding its own weights:
  * attention_norm / ffn_norm  `pixtral_r_m_s_norm.py` (replicated weight)
  * attention     `pixtral_attention.py` (q/k/v column-parallel by head, o_proj row-parallel + all_reduce)
  * feed_forward  a SwiGLU port: `pixtral_m_l_p.py` by default, or `mlp.py` (same PixtralMLP module) when
                  the owning stack passes `mlp_cls` (gate/up column-parallel, down row-parallel + all_reduce)

Layout: x (B, S, D) or (B, 1, S, D) -- B images of one size side by side, or B == 1 with images packed
along S under a block-diagonal mask. The output keeps the input's rank.
"""
from __future__ import annotations

import torch

import ttnn
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs.pixtral_attention import TtPixtralAttention
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs.pixtral_m_l_p import TtPixtralMLP
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs.pixtral_r_m_s_norm import TtPixtralRMSNorm


class TtAttention:
    # dtype the norm outputs enter the attention / MLP matmuls in (the residual itself is always fp32). fp32: the
    # QKV and gate/up minimal_matmuls read the fp32 norm output themselves (and emit bf16), so no separate cast of
    # each norm output to bf16 (113 us per norm on 32 images x 256 patches).
    MATMUL_INPUT_DTYPE = ttnn.float32
    # Each norm's gamma is folded into the projections that read it (q/k/v; gate/up), so the norms apply none: the
    # separate fp32 x bf16 broadcast gamma multiply cost 456 us per norm on 32 images x 256 patches.
    FOLD_NORM_GAMMA = True
    # The ffn (and attention) norm's output (read once, by the gate/up or the QKV, which frees it) lands in L1 while it
    # takes at most this many bytes per core (fp32 32 x 256 x 1024: 298 KB).
    FFN_NORM_L1_BYTES = 320 * 1024
    # dtype the L1 ffn norm output enters the fused gate/up in (cast in L1): bf16 halves the activation blocks the
    # LoFi gate/up unpacks and multicasts, and leaves room for full-depth K blocks.
    GATE_UP_INPUT_DTYPE = ttnn.bfloat16
    # dtype the attention norm output enters the QKV in when it fits in L1 (norm output to L1, cast there, freed by
    # the QKV): bf16 halves the activation blocks the HiFi4 QKV unpacks. None: the fp32 norm output from DRAM.
    QKV_INPUT_DTYPE = ttnn.bfloat16

    def __init__(self, device, torch_module, mlp_cls=None) -> None:
        self.device = device
        fold = self.FOLD_NORM_GAMMA
        an, fn = torch_module.attention_norm, torch_module.ffn_norm
        self.attention_norm = TtPixtralRMSNorm.build(device, an, fold_gamma=fold)
        self.attention = TtPixtralAttention(device, torch_module.attention, in_scale=an.weight if fold else None)
        self.ffn_norm = TtPixtralRMSNorm.build(device, fn, fold_gamma=fold)
        self.feed_forward = (mlp_cls or TtPixtralMLP)(
            device, torch_module.feed_forward, in_scale=fn.weight if fold else None
        )
        self.tp = self.attention.tp
        self.head_dim = self.attention.head_dim
        self._repl = ttnn.ReplicateTensorToMesh(device) if isinstance(device, ttnn.MeshDevice) else None

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

    def __call__(self, hidden_states, attention_mask=None, position_embeddings=None, **kwargs):
        x = hidden_states
        in_shape = tuple(x.shape)
        if len(in_shape) == 3:
            x = ttnn.reshape(x, (in_shape[0], 1, in_shape[1], in_shape[2]))
        s, hd = x.shape[2], self.head_dim

        cos, sin = position_embeddings
        cos = self._to_tt(cos.reshape(1, 1, s, hd)) if isinstance(cos, torch.Tensor) else cos
        sin = self._to_tt(sin.reshape(1, 1, s, hd)) if isinstance(sin, torch.Tensor) else sin
        mask = attention_mask
        if isinstance(mask, torch.Tensor):
            mask = self._to_tt(mask.reshape(1, 1, s, s).float())

        # The residual stream is carried in fp32. The vision encoder grows a few massive activations
        # (row norms ~1e4) around layer 13, and whether a patch row crosses that onset depends on low bits a
        # bf16 residual drops: a bf16 stream (TT's or the CPU reference's alike) moves rows away from the
        # fp32 result there. Norm outputs go to the matmuls in MATMUL_INPUT_DTYPE; the sub-blocks return fp32.
        if x.dtype != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        grid = self.device.compute_with_storage_grid_size()
        n_bytes = 4 * int(x.shape[0]) * int(x.shape[-2]) * int(x.shape[-1]) // (grid.x * grid.y)
        l1 = self.FOLD_NORM_GAMMA and self.MATMUL_INPUT_DTYPE == ttnn.float32 and n_bytes <= self.FFN_NORM_L1_BYTES
        if l1 and self.QKV_INPUT_DTYPE not in (None, ttnn.float32):
            n32 = self.attention_norm(x, memory_config=ttnn.L1_MEMORY_CONFIG)
            n = ttnn.typecast(n32, self.QKV_INPUT_DTYPE, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(n32)
        else:
            n = self._mm_in(self.attention_norm(x))
        h = ttnn.add(x, self.attention(n, attention_mask=mask, position_embeddings=(cos, sin)))
        n = self.ffn_norm(h, memory_config=ttnn.L1_MEMORY_CONFIG if l1 else None)
        if l1 and self.GATE_UP_INPUT_DTYPE not in (None, n.dtype):
            n32, n = n, ttnn.typecast(n, self.GATE_UP_INPUT_DTYPE, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(n32)
            out = ttnn.add(h, self.feed_forward(n))
        else:
            out = ttnn.add(h, self.feed_forward(self._mm_in(n)))
        if len(in_shape) == 3:
            return ttnn.reshape(out, in_shape)
        return out

    def _mm_in(self, t):
        return t if t.dtype == self.MATMUL_INPUT_DTYPE else ttnn.typecast(t, self.MATMUL_INPUT_DTYPE)


def build(device, torch_module=None):
    return TtAttention.build(device, torch_module)


def attention(device, torch_module=None):
    return TtAttention.build(device, torch_module)

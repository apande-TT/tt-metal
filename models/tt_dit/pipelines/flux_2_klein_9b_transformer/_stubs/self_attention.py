# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `self_attention` of FLUX.2-klein-9B.

HF reference: `Flux2ParallelSelfAttention` + `Flux2ParallelSelfAttnProcessor`,
reached as `Flux2Transformer2DModel.single_transformer_blocks[0].attn`. This is
a ViT-22B-style *parallel* block: the attention QKV projection and the MLP
input projection are one fused matmul, and the attention output projection and
the MLP output projection are another::

    proj      = to_qkv_mlp_proj(x)        # 4096 -> 3*4096 | 2*12288 = 36864
    qkv, mlp  = split(proj, by the global QKV:MLP ratio)
    q,k,v     = qkv.chunk(3, dim=-1); QK-RMSNorm over head_dim; interleaved RoPE
    attn      = sdpa(q, k, v)             # non-causal
    mlp       = Flux2SwiGLU(mlp)          # 24576 -> 12288
    return      to_out(cat([attn, mlp], dim=-1))    # 16384 -> 4096

This is the same module as the `flux2_parallel_self_attention` component,
reached under the scaffold's generic role name (the capture step recorded the
same `single_transformer_blocks.0.attn` path for both); the two ports are
deliberately identical so neither can drift.

The scaffold seeded this file with an adapter around
`models/tt_transformers/tt/attention.py` — a causal decoder attention with a KV
cache and no fused MLP path, which cannot express a parallel block. Ported
directly instead.

Tensor-parallel scheme (TP=8)
-----------------------------
Both fused projections shard PER PACKED BLOCK — diffusers records the intent in
`_tp_packed_col_blocks` / `_tp_packed_row_blocks` on the Linears:

* `to_qkv_mlp_proj` is COLUMN-parallel with blocks `[q, k, v, gate, up]` =
  `[4096, 4096, 4096, 12288, 12288]`, so chip *i* holds
  `[q_i(512) | k_i(512) | v_i(512) | gate_i(1536) | up_i(1536)]` = 4608 wide:
  4 whole attention heads and a whole SwiGLU slice. The reference splits qkv
  from mlp by the GLOBAL ratio precisely so this local layout works unchanged
  (4608 * 12288/36864 = 1536).
* `to_out` is ROW-parallel with blocks `[attn(4096), mlp(12288)]`, matching the
  chip-local `[attn_i(512) | mlp_i(1536)]` activation, and one `all_reduce`
  sums the eight partials. Summing the two packed row blocks is exactly what a
  matmul over the concatenated input does.
* QK-norms stay replicated — they normalize over head_dim, after the split.
"""

from __future__ import annotations

import importlib.util as _ilu
import os as _os

import ttnn

HF_MODEL_ID = "/tmp/tt_hw_planner_components/flux_2_klein_9b_transformer"

_CANDIDATE_SUBMODULE_PATHS = ["single_transformer_blocks[0].attn"]


def _load_helpers():
    path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "_flux2_ttnn.py")
    spec = _ilu.spec_from_file_location("_flux2_ttnn", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _load_helpers()


class TtFlux2ParallelSelfAttention:
    def __init__(self, device, torch_module):
        self.device = device
        self.cfg = H.compute_config()

        self.head_dim = int(torch_module.head_dim)
        total_heads = int(torch_module.heads)
        inner = int(torch_module.inner_dim)
        mlp_hidden = int(torch_module.mlp_hidden_dim)
        mlp_blocks = int(torch_module.mlp_mult_factor)

        tp = H.mesh_width(device)
        if tp > 1 and (total_heads % tp or mlp_hidden % tp):
            tp = 1
        self.tp = tp
        self.heads = total_heads // tp
        self.qk_eps = float(torch_module.norm_q.eps)

        wi = torch_module.to_qkv_mlp_proj.weight.detach().float().t().contiguous()
        col_blocks = [wi[:, i * inner : (i + 1) * inner] for i in range(3)]
        col_blocks += [wi[:, 3 * inner + j * mlp_hidden : 3 * inner + (j + 1) * mlp_hidden] for j in range(mlp_blocks)]
        self.w_in = H.stage(H.pack_col_blocks(col_blocks, tp), device, shard_dim=1 if tp > 1 else None)
        self.b_in = H.bias_vector(torch_module.to_qkv_mlp_proj, device)

        wo = torch_module.to_out.weight.detach().float().t().contiguous()  # (inner + mlp_hidden, out)
        self.w_out = H.stage(H.pack_row_blocks([wo[:inner], wo[inner:]], tp), device, shard_dim=0 if tp > 1 else None)
        self.b_out = H.bias_vector(torch_module.to_out, device)

        def qk_gamma(norm):
            return H.stage(norm.weight.detach().float().reshape(1, 1, 1, -1), device)

        self.g_q = qk_gamma(torch_module.norm_q)
        self.g_k = qk_gamma(torch_module.norm_k)

        self.rot = H.rotate_matrix(self.head_dim, device)
        # Chip-local widths of the fused projection's two halves.
        self.local_qkv = 3 * self.heads * self.head_dim
        self.local_mlp = (mlp_hidden // tp) * mlp_blocks

    def __call__(self, hidden_states, attention_mask=None, image_rotary_emb=None, **kwargs):
        device = self.device
        x = H.as_device(hidden_states, device)
        rope = H.rope_pair(image_rotary_emb, device)

        proj = ttnn.linear(x, self.w_in, bias=self.b_in, compute_kernel_config=self.cfg)
        rows = proj.shape[-2]
        split = self.local_qkv
        qkv = ttnn.slice(proj, [0, 0, 0], [proj.shape[0], rows, split])
        mlp = ttnn.slice(proj, [0, 0, split], [proj.shape[0], rows, split + self.local_mlp])

        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            ttnn.unsqueeze(qkv, 1),
            num_heads=self.heads,
            num_kv_heads=self.heads,
            transpose_k_heads=False,
        )
        q = H.apply_rope(ttnn.rms_norm(q, weight=self.g_q, epsilon=self.qk_eps), rope, self.rot)
        k = H.apply_rope(ttnn.rms_norm(k, weight=self.g_k, epsilon=self.qk_eps), rope, self.rot)

        attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False, compute_kernel_config=self.cfg)
        attn = ttnn.squeeze(ttnn.experimental.nlp_concat_heads(attn), 1)

        half = self.local_mlp // 2
        gate = ttnn.slice(mlp, [0, 0, 0], [mlp.shape[0], rows, half])
        up = ttnn.slice(mlp, [0, 0, half], [mlp.shape[0], rows, self.local_mlp])
        mlp = ttnn.multiply(ttnn.silu(gate), up)

        out = ttnn.linear(ttnn.concat([attn, mlp], dim=-1), self.w_out, compute_kernel_config=self.cfg)
        if self.tp > 1:
            out = ttnn.all_reduce(out)
        # A row-parallel bias is added ONCE, after the reduction.
        if self.b_out is not None:
            out = ttnn.add(out, self.b_out)
        return out


def build(device, torch_module):
    return TtFlux2ParallelSelfAttention(device, torch_module)


def self_attention(device, torch_module, hidden_states, **kwargs):
    """Module-level entry point for callers that do not hold a built port."""
    return build(device, torch_module)(hidden_states, **kwargs)

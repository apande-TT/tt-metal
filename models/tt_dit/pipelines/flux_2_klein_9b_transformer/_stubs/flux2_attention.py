# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `flux2_attention` of FLUX.2-klein-9B.

HF reference: `Flux2Attention` + `Flux2AttnProcessor`, reached as
`Flux2Transformer2DModel.transformer_blocks[0].attn` — the joint attention of
the double-stream MMDiT block::

    q,k,v       = to_q/to_k/to_v(hidden_states)                      # image
    eq,ek,ev    = add_q/add_k/add_v_proj(encoder_hidden_states)      # text
    per-stream QK-RMSNorm over head_dim (norm_q/k, norm_added_q/k)
    q,k,v       = cat([text, image], dim=sequence)
    q,k         = interleaved RoPE
    joint       = sdpa(q, k, v)                                      # non-causal
    text, image = split back apart
    return to_out[0](image), to_add_out(text)                        # IMAGE first

`added_kv_proj_dim` is set on this module, so the processor projects the text
stream unconditionally — the text stream is not optional here.

The scaffold seeded this file with an adapter around
`models/tt_transformers/tt/attention.py`. That class is a single-stream causal
decoder attention with a KV cache and rotary applied to one sequence; it has no
`add_*_proj` text stream, no joint sequence and no non-causal path, so there is
no configuration of it that computes this module. Ported directly instead.

Tensor-parallel scheme (TP=8)
-----------------------------
COLUMN-parallel BY HEAD on both q/k/v triples: 32 heads / 8 = 4 local heads =
512 features per chip. Attention is independent per head, so each chip runs a
complete 4-head attention over the whole sequence and nothing is exchanged
inside the attention. `to_out[0]` and `to_add_out` are ROW-parallel — chip *i*'s
result is exactly input features `[512i, 512i + 512)` of those projections — so
each chip matmuls its own rows and one `all_reduce` per stream sums the
partials. The QK-norms stay replicated: they normalize over head_dim, which is
intact after the split.
"""

from __future__ import annotations

import importlib.util as _ilu
import os as _os

import ttnn

HF_MODEL_ID = "/tmp/tt_hw_planner_components/flux_2_klein_9b_transformer"

_CANDIDATE_SUBMODULE_PATHS = ["transformer_blocks[0].attn"]


def _load_helpers():
    path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "_flux2_ttnn.py")
    spec = _ilu.spec_from_file_location("_flux2_ttnn", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _load_helpers()


class TtFlux2Attention:
    def __init__(self, device, torch_module):
        self.device = device
        self.cfg = H.compute_config()

        self.head_dim = int(torch_module.head_dim)
        total_heads = int(torch_module.heads)

        tp = H.mesh_width(device)
        # Heads are the shard unit; anything that does not divide them cleanly
        # falls back to a replicated port rather than to a wrong split.
        if tp > 1 and total_heads % tp:
            tp = 1
        self.tp = tp
        self.heads = total_heads // tp
        self.qk_eps = float(torch_module.norm_q.eps)

        shard_col = 1 if tp > 1 else None

        def w(linear):
            return linear.weight.detach().float().t().contiguous()

        self.w_qkv_img = H.stage(
            H.pack_col_blocks([w(torch_module.to_q), w(torch_module.to_k), w(torch_module.to_v)], tp),
            device,
            shard_dim=shard_col,
        )
        self.w_qkv_txt = H.stage(
            H.pack_col_blocks([w(torch_module.add_q_proj), w(torch_module.add_k_proj), w(torch_module.add_v_proj)], tp),
            device,
            shard_dim=shard_col,
        )
        self.w_out = H.matmul_weight(torch_module.to_out[0], device, tp=tp, mode="row")
        self.b_out = H.bias_vector(torch_module.to_out[0], device)
        self.w_add_out = H.matmul_weight(torch_module.to_add_out, device, tp=tp, mode="row")
        self.b_add_out = H.bias_vector(torch_module.to_add_out, device)

        def qk_gamma(norm):
            return H.stage(norm.weight.detach().float().reshape(1, 1, 1, -1), device)

        self.g_q = qk_gamma(torch_module.norm_q)
        self.g_k = qk_gamma(torch_module.norm_k)
        self.g_added_q = qk_gamma(torch_module.norm_added_q)
        self.g_added_k = qk_gamma(torch_module.norm_added_k)

        self.rot = H.rotate_matrix(self.head_dim, device)

    def _reduce(self, t, bias):
        """Sum the row-parallel partials, THEN add the bias once — folding it
        into the matmul would add it `tp` times."""
        if self.tp > 1:
            t = ttnn.all_reduce(t)
        if bias is not None:
            t = ttnn.add(t, bias)
        return t

    def _heads(self, x, fused_qkv):
        proj = ttnn.linear(x, fused_qkv, compute_kernel_config=self.cfg)
        return ttnn.experimental.nlp_create_qkv_heads(
            ttnn.unsqueeze(proj, 1),
            num_heads=self.heads,
            num_kv_heads=self.heads,
            transpose_k_heads=False,
        )

    def __call__(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
        **kwargs,
    ):
        device = self.device
        x = H.as_device(hidden_states, device)
        c = H.as_device(encoder_hidden_states, device)
        rope = H.rope_pair(image_rotary_emb, device)

        q_i, k_i, v_i = self._heads(x, self.w_qkv_img)
        q_t, k_t, v_t = self._heads(c, self.w_qkv_txt)

        q_i = ttnn.rms_norm(q_i, weight=self.g_q, epsilon=self.qk_eps)
        k_i = ttnn.rms_norm(k_i, weight=self.g_k, epsilon=self.qk_eps)
        q_t = ttnn.rms_norm(q_t, weight=self.g_added_q, epsilon=self.qk_eps)
        k_t = ttnn.rms_norm(k_t, weight=self.g_added_k, epsilon=self.qk_eps)

        # Text leads the joint sequence, matching the reference's
        # `cat([encoder_*, *], dim=1)` and the rope table the caller built.
        q = H.apply_rope(ttnn.concat([q_t, q_i], dim=2), rope, self.rot)
        k = H.apply_rope(ttnn.concat([k_t, k_i], dim=2), rope, self.rot)
        v = ttnn.concat([v_t, v_i], dim=2)

        joint = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False, compute_kernel_config=self.cfg)
        joint = ttnn.squeeze(ttnn.experimental.nlp_concat_heads(joint), 1)

        txt_len = c.shape[-2]
        total = joint.shape[-2]
        width = joint.shape[-1]
        ctx = ttnn.slice(joint, [0, 0, 0], [joint.shape[0], txt_len, width])
        img = ttnn.slice(joint, [0, txt_len, 0], [joint.shape[0], total, width])

        out_img = self._reduce(ttnn.linear(img, self.w_out, compute_kernel_config=self.cfg), self.b_out)
        out_txt = self._reduce(ttnn.linear(ctx, self.w_add_out, compute_kernel_config=self.cfg), self.b_add_out)

        # The processor returns the IMAGE stream first.
        return out_img, out_txt


def build(device, torch_module):
    return TtFlux2Attention(device, torch_module)


def flux2_attention(device, torch_module, hidden_states, **kwargs):
    """Module-level entry point for callers that do not hold a built port."""
    return build(device, torch_module)(hidden_states, **kwargs)

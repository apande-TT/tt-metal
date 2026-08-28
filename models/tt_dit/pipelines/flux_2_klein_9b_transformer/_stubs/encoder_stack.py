# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `encoder_stack` of FLUX.2-klein-9B.

HF reference: `Flux2Transformer2DModel.transformer_blocks[0]`
(`Flux2TransformerBlock`) — the double-stream MMDiT block. It carries two
parallel residual streams, image (`hidden_states`) and text
(`encoder_hidden_states`), each with its own affine-free LayerNorms, its own
AdaLN modulation vector and its own SwiGLU feed-forward, joined by one
attention over the concatenated `[text, image]` sequence::

    (shift, scale, gate) x2 = Flux2Modulation.split(temb_mod_*, 2)
    attention over cat([txt, img], dim=seq), per-stream QK-RMSNorm and RoPE
    residual + gate * attn, then residual + gate * swiglu_ff, per stream
    return (encoder_hidden_states, hidden_states)   # TEXT stream first

(The scaffold seeded this file with a copy of the Llama vision encoder; nothing
of that ViT stack applies to a double-stream MMDiT block.)

Tensor-parallel scheme (TP=8)
-----------------------------
Textbook column-then-row, two collectives per stream:

* `to_q/k/v` and `add_q/k/v` are COLUMN-parallel BY HEAD — 32 heads / 8 = 4
  local heads = 512 features per chip. Attention is independent per head, so
  each chip runs a complete 4-head attention over the full sequence and
  nothing is exchanged inside the attention itself.
* `to_out[0]` and `to_add_out` are ROW-parallel: chip *i*'s attention result is
  exactly input features `[512i, 512i + 512)` of the output projection, so each
  chip matmuls its own rows and one `all_reduce` sums the partials.
* `ff.linear_in` is the packed `[gate | up]` SwiGLU projection, column-parallel
  PER PACKED BLOCK (see `_flux2_ttnn.pack_col_blocks`: an even split of the
  packed matrix would hand one chip the tail of `gate` and the head of `up`);
  `ff.linear_out` is row-parallel. Same for `ff_context`.
* LayerNorms, QK-norms and the modulation vectors stay REPLICATED — the
  LayerNorm reduction needs the full hidden dim, and the QK-norm runs over
  head_dim after the heads are already split.

Four `all_reduce`s per block (attn out, attn context out, ff, ff_context). The
maths is unchanged — a matmul over a concatenated contraction axis IS the sum
of the per-block matmuls — so the gathered output matches the golden.
"""

from __future__ import annotations

import importlib.util as _ilu
import os as _os

import ttnn

HF_MODEL_ID = "/tmp/tt_hw_planner_components/flux_2_klein_9b_transformer"

_CANDIDATE_SUBMODULE_PATHS = ["transformer_blocks[0]"]


def _load_helpers():
    path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "_flux2_ttnn.py")
    spec = _ilu.spec_from_file_location("_flux2_ttnn", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _load_helpers()


class TtFlux2TransformerBlock:
    def __init__(self, device, torch_module):
        self.device = device
        self.cfg = H.compute_config()

        attn = torch_module.attn
        self.head_dim = int(attn.head_dim)
        total_heads = int(attn.heads)

        tp = H.mesh_width(device)
        # Heads are the shard unit; anything that does not divide them cleanly
        # falls back to a replicated port rather than to a wrong split.
        if tp > 1 and total_heads % tp:
            tp = 1
        self.tp = tp
        self.heads = total_heads // tp

        self.eps = float(torch_module.norm1.eps)
        self.qk_eps = float(attn.norm_q.eps)

        shard_col = 1 if tp > 1 else None

        def w(linear):
            return linear.weight.detach().float().t().contiguous()

        # Column-parallel BY HEAD, packed [q | k | v] so one matmul feeds
        # nlp_create_qkv_heads.
        self.w_qkv_img = H.stage(
            H.pack_col_blocks([w(attn.to_q), w(attn.to_k), w(attn.to_v)], tp),
            device,
            shard_dim=shard_col,
        )
        self.w_qkv_txt = H.stage(
            H.pack_col_blocks([w(attn.add_q_proj), w(attn.add_k_proj), w(attn.add_v_proj)], tp),
            device,
            shard_dim=shard_col,
        )
        self.w_out = H.matmul_weight(attn.to_out[0], device, tp=tp, mode="row")
        self.b_out = H.bias_vector(attn.to_out[0], device)
        self.w_add_out = H.matmul_weight(attn.to_add_out, device, tp=tp, mode="row")
        self.b_add_out = H.bias_vector(attn.to_add_out, device)

        def qk_gamma(norm):
            return H.stage(norm.weight.detach().float().reshape(1, 1, 1, -1), device)

        self.g_q = qk_gamma(attn.norm_q)
        self.g_k = qk_gamma(attn.norm_k)
        self.g_added_q = qk_gamma(attn.norm_added_q)
        self.g_added_k = qk_gamma(attn.norm_added_k)

        self.rot = H.rotate_matrix(self.head_dim, device)

        def feed_forward(ff):
            wi = w(ff.linear_in)
            inner = wi.shape[1] // 2  # packed [gate | up]
            return (
                H.stage(H.pack_col_blocks([wi[:, :inner], wi[:, inner:]], tp), device, shard_dim=shard_col),
                H.matmul_weight(ff.linear_out, device, tp=tp, mode="row"),
                H.bias_vector(ff.linear_out, device),
            )

        self.w_ff_in, self.w_ff_out, self.b_ff_out = feed_forward(torch_module.ff)
        self.w_ffc_in, self.w_ffc_out, self.b_ffc_out = feed_forward(torch_module.ff_context)

    def _reduce(self, t, bias):
        """Finish a row-parallel matmul: sum the per-chip partials, THEN add the
        bias once — folding it into the matmul would add it `tp` times."""
        if self.tp > 1:
            t = ttnn.all_reduce(t)
        if bias is not None:
            t = ttnn.add(t, bias)
        return t

    def _modulated_norm(self, x, scale, shift):
        """`LayerNorm(x) * (1 + scale) + shift`, folded into the norm's affine."""
        return ttnn.layer_norm(
            x,
            epsilon=self.eps,
            weight=ttnn.add(scale, 1.0),
            bias=shift,
            compute_kernel_config=self.cfg,
        )

    def _heads(self, x, fused_qkv):
        proj = ttnn.linear(x, fused_qkv, compute_kernel_config=self.cfg)
        return ttnn.experimental.nlp_create_qkv_heads(
            ttnn.unsqueeze(proj, 1),
            num_heads=self.heads,
            num_kv_heads=self.heads,
            transpose_k_heads=False,
        )

    def _attention(self, norm_img, norm_txt, rope):
        q_i, k_i, v_i = self._heads(norm_img, self.w_qkv_img)
        q_t, k_t, v_t = self._heads(norm_txt, self.w_qkv_txt)

        q_i = ttnn.rms_norm(q_i, weight=self.g_q, epsilon=self.qk_eps)
        k_i = ttnn.rms_norm(k_i, weight=self.g_k, epsilon=self.qk_eps)
        q_t = ttnn.rms_norm(q_t, weight=self.g_added_q, epsilon=self.qk_eps)
        k_t = ttnn.rms_norm(k_t, weight=self.g_added_k, epsilon=self.qk_eps)

        # Text tokens lead the joint sequence, matching the reference's
        # `cat([encoder_*, *], dim=1)` and the rope table the caller built.
        q = H.apply_rope(ttnn.concat([q_t, q_i], dim=2), rope, self.rot)
        k = H.apply_rope(ttnn.concat([k_t, k_i], dim=2), rope, self.rot)
        v = ttnn.concat([v_t, v_i], dim=2)

        joint = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False, compute_kernel_config=self.cfg)
        joint = ttnn.squeeze(ttnn.experimental.nlp_concat_heads(joint), 1)

        txt_len = norm_txt.shape[-2]
        total = joint.shape[-2]
        width = joint.shape[-1]
        ctx = ttnn.slice(joint, [0, 0, 0], [joint.shape[0], txt_len, width])
        img = ttnn.slice(joint, [0, txt_len, 0], [joint.shape[0], total, width])

        return (
            self._reduce(ttnn.linear(img, self.w_out, compute_kernel_config=self.cfg), self.b_out),
            self._reduce(ttnn.linear(ctx, self.w_add_out, compute_kernel_config=self.cfg), self.b_add_out),
        )

    def _feed_forward(self, x, w_in, w_out, bias):
        packed = ttnn.linear(x, w_in, compute_kernel_config=self.cfg)
        rows = packed.shape[-2]
        half = packed.shape[-1] // 2
        gate = ttnn.slice(packed, [0, 0, 0], [packed.shape[0], rows, half])
        up = ttnn.slice(packed, [0, 0, half], [packed.shape[0], rows, packed.shape[-1]])
        hidden = ttnn.multiply(ttnn.silu(gate), up)
        return self._reduce(ttnn.linear(hidden, w_out, compute_kernel_config=self.cfg), bias)

    def __call__(
        self,
        hidden_states,
        encoder_hidden_states=None,
        temb_mod_img=None,
        temb_mod_txt=None,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
        **kwargs,
    ):
        device = self.device
        x = H.as_device(hidden_states, device)
        c = H.as_device(encoder_hidden_states, device)
        rope = H.rope_pair(image_rotary_emb, device)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = H.mod_chunks(
            H.as_device(temb_mod_img, device), 6
        )
        c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = H.mod_chunks(
            H.as_device(temb_mod_txt, device), 6
        )

        attn_img, attn_txt = self._attention(
            self._modulated_norm(x, scale_msa, shift_msa),
            self._modulated_norm(c, c_scale_msa, c_shift_msa),
            rope,
        )

        x = ttnn.add(x, ttnn.multiply(attn_img, gate_msa))
        ff_img = self._feed_forward(
            self._modulated_norm(x, scale_mlp, shift_mlp), self.w_ff_in, self.w_ff_out, self.b_ff_out
        )
        x = ttnn.add(x, ttnn.multiply(ff_img, gate_mlp))

        c = ttnn.add(c, ttnn.multiply(attn_txt, c_gate_msa))
        ff_txt = self._feed_forward(
            self._modulated_norm(c, c_scale_mlp, c_shift_mlp), self.w_ffc_in, self.w_ffc_out, self.b_ffc_out
        )
        c = ttnn.add(c, ttnn.multiply(ff_txt, c_gate_mlp))

        return c, x


def build(device, torch_module):
    return TtFlux2TransformerBlock(device, torch_module)


def encoder_stack(device, torch_module, hidden_states, **kwargs):
    """Module-level entry point for callers that do not hold a built port."""
    return build(device, torch_module)(hidden_states, **kwargs)

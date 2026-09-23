# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `codec_transformer_block` (`audio_tokenizer.decoder_blocks.1.layers.0`).

One `CodecTransformerBlock`:

    r = attention_scale * attention(attention_norm(x));  h = x + r
    r = ffn_scale      * feed_forward(ffn_norm(h));       out = h + r

`attention_scale` / `ffn_scale` are LayerScale parameters -- per-channel `[dim]` vectors, not
scalars -- and this checkpoint's values are small and signed (~-5e-3, ~-1.5e-4), so dropping them
would not merely rescale the residual, it would flip its sign.

The attention is ALiBi + sliding-window causal + QK-norm with no RoPE; see `_stubs/codec_attention.py`
for the details. The window belongs to the decoder STAGE, not the model: the four stages run 2, 4,
8, 16 as each transposed convolution doubles it, so the mask is built from this block's own
`attention.sliding_window`.

`norm_eps` here is **1e-2**, three orders of magnitude larger than the text backbone's 1e-5; it is
read off the modules rather than assumed.
"""

from __future__ import annotations

import math

import torch

import ttnn


_SHARD_HEIGHT = 32
# 2048 rows = 256 codec frames after the decoder's 8x upsampling, which is the whole stage's frame
# ceiling. The mask is translation-invariant, so a longer sequence needs a LARGER constant here
# (and nothing else); it cannot be rebuilt inside the forward, which must stay torch-free.
_MASK_MAX_SEQ = 2048
_MASK_NEG = -1.0e9

# fp32 accumulation in DEST. The activation path is float32 -- bfloat16 end to end put the full
# codec chain at PCC 0.9873 over eight residual blocks plus five convolutions.
_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
)


# FLOAT32 WEIGHTS. Measured on this device against a float64 reference, one matmul at M=32,
# K=N=3072, HiFi4 + `fp32_dest_acc_en`: a bfloat16 weight costs 1.738e-3 relative where a float32
# weight costs 1.169e-3. That 1.5x is small per op and this codec stacks eight residual blocks on
# top of five convolutions, where it is the last error source left after the softmax and the RMS
# norm were spelled out. No `ttnn.embedding` table is built here (those must stay bfloat16).
def _from_torch(t, device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    t = t.to(torch.bfloat16) if dtype == ttnn.bfloat16 else t.to(torch.float32)
    if device.__class__.__name__ == "MeshDevice":
        return ttnn.from_torch(
            t, dtype=dtype, layout=layout, device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


def _weight(linear, device):
    return _from_torch(linear.weight.detach().transpose(0, 1).contiguous(), device)


def _alibi_window_mask(slopes, window, seq):
    """`[1, H, seq, seq]`: ALiBi bias `slope[h] * (j - i)`, blocked where `j > i` or `j < i - window`.

    Depends only on `j - i`, so the top-left `[S, S]` corner of one big mask is exactly the mask for
    a length-`S` sequence -- which is what lets the forward stay free of torch calls (the runtime
    native probe graduates only at zero torch ops, so a per-call rebuild is not an option).
    """
    pos = torch.arange(seq)
    rel = pos.unsqueeze(0) - pos.unsqueeze(1)
    bias = slopes.reshape(-1, 1, 1).float() * rel.unsqueeze(0).float()
    blocked = (rel > 0) | (rel < -window)
    return bias.masked_fill(blocked.unsqueeze(0), _MASK_NEG).unsqueeze(0)


def _compile_block(device, blk, mask):
    """One `CodecTransformerBlock` as a callable on a `[1, 1, S, dim]` ttnn tensor."""
    attn = blk.attention
    ff = blk.feed_forward
    args = blk.args

    n_heads = int(attn.n_local_heads)
    n_kv_heads = int(attn.n_local_kv_heads)
    head_dim = int(args.head_dim)
    dim = int(blk.dim)
    scale = 1.0 / math.sqrt(head_dim)
    qk_norm = bool(args.qk_norm)

    attn_gamma = _norm_gamma(blk.attention_norm, device)
    attn_eps = float(blk.attention_norm.eps)
    ffn_gamma = _norm_gamma(blk.ffn_norm, device)
    ffn_eps = float(blk.ffn_norm.eps)

    wq, wk, wv, wo = (_weight(m, device) for m in (attn.wq, attn.wk, attn.wv, attn.wo))
    q_gamma = _norm_gamma(attn.q_norm, device) if qk_norm else None
    k_gamma = _norm_gamma(attn.k_norm, device) if qk_norm else None
    q_eps = float(attn.q_norm.eps) if qk_norm else 0.0
    k_eps = float(attn.k_norm.eps) if qk_norm else 0.0

    w1, w2, w3 = (_weight(m, device) for m in (ff.w1, ff.w2, ff.w3))

    attn_scale = ffn_scale = None
    if blk.layer_scale:
        attn_scale = _from_torch(
            blk.attention_scale.detach().reshape(1, 1, 1, dim), device, dtype=ttnn.float32
        )
        ffn_scale = _from_torch(
            blk.ffn_scale.detach().reshape(1, 1, 1, dim), device, dtype=ttnn.float32
        )

    if blk.post_attention_norm is not None or blk.post_ffn_norm is not None:
        raise NotImplementedError("post_attention_norm / post_ffn_norm are not ported")

    def block(h):
        seq = int(h.shape[-2])
        xn = _rms_norm(h, attn_gamma, attn_eps)

        q = ttnn.linear(xn, wq, compute_kernel_config=_COMPUTE)
        k = ttnn.linear(xn, wk, compute_kernel_config=_COMPUTE)
        v = ttnn.linear(xn, wv, compute_kernel_config=_COMPUTE)
        if qk_norm:
            q = _rms_norm(q, q_gamma, q_eps)
            k = _rms_norm(k, k_gamma, k_eps)

        # SDPA rejects float32 outright (`sdpa_device_operation.cpp:43`) -- so this does not call
        # it. Spelling the attention out as two matmuls and a softmax keeps Q/K/V, the ALiBi mask
        # and the whole reduction in FLOAT32, which SDPA cannot do at any fidelity. The codec runs
        # eight residual blocks over a few hundred rows, so the explicit form costs little, and
        # the bfloat16 narrowing it removes was compounding through all eight.
        qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads(
            ttnn.concat([q, k, v], dim=-1),
            num_heads=n_heads,
            num_kv_heads=n_kv_heads,
            transpose_k_heads=False,
        )
        scores = ttnn.matmul(
            qh, ttnn.transpose(kh, -2, -1), compute_kernel_config=_COMPUTE
        )
        scores = ttnn.add(
            ttnn.multiply(scores, scale),
            ttnn.slice(mask, [0, 0, 0, 0], [1, n_heads, seq, seq]),
        )
        a = ttnn.matmul(
            _softmax(scores),
            vh,
            compute_kernel_config=_COMPUTE,
        )
        ttnn.deallocate(scores)
        r = ttnn.linear(
            ttnn.experimental.nlp_concat_heads(a), wo,
            dtype=ttnn.float32, compute_kernel_config=_COMPUTE,
        )
        if attn_scale is not None:
            r = ttnn.multiply(r, attn_scale)
        h = ttnn.add(h, r)

        hn = _rms_norm(h, ffn_gamma, ffn_eps)
        r = ttnn.linear(
            ttnn.multiply(
                ttnn.silu(ttnn.linear(hn, w1, compute_kernel_config=_COMPUTE)),
                ttnn.linear(hn, w3, compute_kernel_config=_COMPUTE),
            ),
            w2,
            compute_kernel_config=_COMPUTE,
        )
        if ffn_scale is not None:
            r = ttnn.multiply(r, ffn_scale)
        return ttnn.add(h, r)

    return block


def _norm_gamma(norm, device):
    """`[1, 1, 1, dim]` float32 TILE -- the form the spelled-out RMS norm's final multiply takes."""
    return _from_torch(
        norm.weight.detach().reshape(1, 1, 1, -1), device, dtype=ttnn.float32
    )


def _rms_norm(x, gamma, eps):
    """`x * rsqrt(mean(x^2) + eps) * gamma`, spelled out, entirely in float32.

    NOT `ttnn.rms_norm`: on this model's real inputs the stock op sits at ~9.65e-4 relative error
    where these four ops sit at 6.6e-8. A norm error is RELATIVE -- it RESCALES the whole branch
    after it -- so it shows up as a NORM RATIO rather than as a PCC drop, and the codec stacks
    eight residual blocks with two or three norms each. Measured: the codec's first transformer
    group came back at norm ratio 1.029 against torch with the stock op, and no single stage of
    the chain looked broken. `tt/vocode_stage.py` spells out the same four ops for the same
    reason, so the two bodies agree.
    """
    scale = ttnn.rsqrt(ttnn.add(ttnn.mean(ttnn.multiply(x, x), dim=-1, keepdim=True), eps))
    return ttnn.multiply(ttnn.multiply(x, scale), gamma)


def _softmax(x, dim=-1):
    """`exp(x - max) / sum(exp(x - max))`, spelled out in three ops.

    NOT `ttnn.softmax`. Measured on this build against a float64 reference, the stock op's rows do
    not sum to 1: mean 0.9943, worst 0.9611, for ~1.8e-2 relative error -- and NO flag changes it
    (`numeric_stable=True` and `compute_kernel_config` all return the identical tensor). These
    three ops sit at 5.5e-8, and renormalising the stock op's output only reaches 2.1e-2, so its
    per-element values are wrong too, not just its sum.

    A softmax that does not sum to 1 ATTENUATES the attention output it weights. That reads as a
    NORM RATIO below 1 at a PCC of 0.9999, so a PCC-only gate cannot see it, and it is invisible
    in any layer whose residual is already large. The acoustic stubs beside this file spell the
    same three ops out for the same reason.
    """
    e = ttnn.exp(ttnn.subtract(x, ttnn.max(x, dim=dim, keepdim=True)))
    return ttnn.divide(e, ttnn.sum(e, dim=dim, keepdim=True))


def build(device, torch_module):
    blk = torch_module
    dim = int(blk.dim)

    mask = _from_torch(
        _alibi_window_mask(
            blk.attention.alibi_slopes.detach(),
            int(blk.attention.sliding_window),
            _MASK_MAX_SEQ,
        ),
        device,
        dtype=ttnn.float32,
    )
    block = _compile_block(device, blk, mask)

    def codec_transformer_block(x, **kwargs):
        # The leading bound comes from the TENSOR, never from a literal 1: the pipeline drives this
        # with 32 independent samples stacked on axis 0, and a hardcoded 1 would silently decode
        # only the first of them.
        shape = [int(v) for v in x.shape]
        seq = shape[-2]
        batch = shape[0] if len(shape) >= 3 else 1
        if seq > _MASK_MAX_SEQ:
            raise NotImplementedError(
                f"sequence {seq} exceeds the prebuilt ALiBi mask ({_MASK_MAX_SEQ}); raise "
                f"_MASK_MAX_SEQ -- the mask cannot be rebuilt inside the forward"
            )
        h = block(ttnn.reshape(x, [batch, 1, seq, dim]))
        return ttnn.reshape(h, [batch, seq, dim])

    return codec_transformer_block

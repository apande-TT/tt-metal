# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `flow_matching_audio_transformer` (`acoustic_transformer`).

The deterministic core of the flow-matching sampler -- the velocity field plus the semantic head:

    t_proj   = time_projection(time_embedding(t))
    llm_proj = llm_projection(llm_hidden)
    seq      = [input_projection(x_t), t_proj, llm_proj]     # THREE tokens
    velocity = acoustic_codebook_output(norm(blocks(seq))[:, 0, :])
    semantic = semantic_codebook_output(llm_hidden)          # reads the RAW hidden state

The Euler loop around this is not ported here because it is not a function of its inputs: it draws
`x_0 = torch.randn(...)` inside the module. `x_t` is supplied instead, which leaves every weight
under test (see the note in `tests/pcc/test_flow_matching_audio_transformer.py`).

**The sequence is padded to one tile.** The real sequence is 3 tokens, which in TILE layout lives
inside a 32-row tile with 29 rows of padding. Rather than hope an op respects a sub-tile logical
length, the sequence is explicitly built 32 rows long and attention is given an additive mask that
blocks columns 3..31 -- so rows 0..2 attend to exactly the three real tokens, which is what the
reference computes. Only row 0 is ever read out.

The blocks are `AcousticTransformerBlock`s: bidirectional, RoPE-free, GQA 32/8, head_dim 128,
dim 3072, hidden 9216, `norm_eps` 1e-5, no biases.

The activation path runs in float32 (weights stay bfloat16 -- `ttnn.linear` takes a float32
activation against a bfloat16 weight). All-bfloat16 cleared the 0.99 target by only 0.0007, and
that margin is not worth holding: the residual accumulates over three blocks and the read-out is a
single row. The attention is written out rather than calling SDPA, which rejects float32
(`sdpa_device_operation.cpp:43`) -- see `_attention` -- because the sampler around this field
rounds onto 21 levels and SDPA's bfloat16 softmax was worth ~7% of the output codes.

THE 29-ROW TILE PAD IS A BUILD-TIME BUFFER. It used to be a `ttnn.zeros(...)` on every call,
which cannot live inside a captured trace (a trace replays kernels, it cannot allocate). It is
now a persistent device buffer, created once per distinct batch and reused; `build(..., batch=N)`
pre-creates the one the caller will actually use so nothing is allocated on the first traced
call. The numerics are unchanged -- same shape, same zeros, same concat.
"""

from __future__ import annotations

import math

import torch

import ttnn


_TILE = 32
_MASK_NEG = -1.0e9


def _from_torch(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    t = t.to(torch.bfloat16) if dtype == ttnn.bfloat16 else t.to(torch.float32)
    if device.__class__.__name__ == "MeshDevice":
        return ttnn.from_torch(
            t, dtype=dtype, layout=layout, device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


def _weight(linear, device):
    return _from_torch(linear.weight.detach().transpose(0, 1).contiguous(), device)


def _gamma(norm, device):
    """A norm's gamma as a `[1, 1, 1, dim]` float32 tile tensor, for `_rms_norm`."""
    return _from_torch(
        norm.weight.detach().reshape(1, 1, 1, -1).contiguous(), device, dtype=ttnn.float32
    )


def _rms_norm(x, gamma, eps):
    """RMSNorm in float32: `x * rsqrt(mean(x^2) + eps) * gamma`.

    NOT the `ttnn` layernorm op, which carries 2.7e-3 of RELATIVE error -- measured on this chip
    against a float64 reference, at either gamma dtype, with a float32 input and a float32
    output. Written out with `mean / rsqrt / multiply` the same normalization holds 1.2e-7. The
    sampler downstream rounds onto 21 levels 0.1 apart in x, so 2.7e-3 through seven norms is
    worth ~1% of the output codes and 1.2e-7 is worth none of them.
    """
    inv = ttnn.rsqrt(ttnn.add(ttnn.mean(ttnn.multiply(x, x), dim=-1, keepdim=True), eps))
    return ttnn.multiply(ttnn.multiply(x, inv), gamma)


_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
)


def _attention(h, wqkv, wo, n_heads, n_kv_heads, scale, attn_mask):
    """GQA attention, bidirectional and non-causal, entirely in float32.

    NOT `ttnn.transformer.scaled_dot_product_attention`. SDPA rejects float32
    (`sdpa_device_operation.cpp:43`), so it forces Q/K/V down to bfloat16, and its softmax carries
    ~6e-3 of ABSOLUTE error on the attention probabilities -- measured on this chip against a
    float64 reference, and `ttnn.softmax` on its own is just as bad (6.2e-3) at either dtype and
    with `numeric_stable` either way. The same softmax written out as
    `max / exp / sum / divide` in float32 holds 6e-8. The sampler this feeds rounds onto 21
    levels 0.1 apart in x, so 6e-3 on a probability is worth roughly 7% of the output codes;
    6e-8 is worth none of them. The sequence here is one 32-row tile, so the explicit form costs
    four small ops per block and nothing measurable in time.

    K/V are repeated to the query head count exactly as the reference's `repeat_kv` does
    (`_repeat_interleave`: query head h reads KV head h // repeats).
    """
    qkv = ttnn.linear(h, wqkv, compute_kernel_config=_COMPUTE)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        qkv, num_heads=n_heads, num_kv_heads=n_kv_heads, transpose_k_heads=False
    )
    repeats = n_heads // n_kv_heads
    if repeats > 1:
        k = ttnn.repeat_interleave(k, repeats, dim=1)
        v = ttnn.repeat_interleave(v, repeats, dim=1)

    # The reference scales the QUERY before the product, not the scores after it.
    scores = ttnn.matmul(
        ttnn.multiply(q, scale), ttnn.transpose(k, -2, -1), compute_kernel_config=_COMPUTE
    )
    if attn_mask is not None:
        scores = ttnn.add(scores, attn_mask)
    scores = ttnn.subtract(scores, ttnn.max(scores, dim=-1, keepdim=True))
    weights = ttnn.exp(scores)
    weights = ttnn.divide(weights, ttnn.sum(weights, dim=-1, keepdim=True))

    out = ttnn.matmul(weights, v, compute_kernel_config=_COMPUTE)
    return ttnn.linear(
        ttnn.experimental.nlp_concat_heads(out), wo, dtype=ttnn.float32,
        compute_kernel_config=_COMPUTE,
    )


def _compile_block(device, blk, mask):
    """One `AcousticTransformerBlock` as a callable on `[B, 1, TILE, dim]`."""
    attn = blk.attention
    ff = blk.feed_forward
    n_heads = int(attn.n_local_heads)
    n_kv_heads = int(attn.n_local_kv_heads)
    scale = 1.0 / math.sqrt(int(attn.head_dim))

    wqkv = _from_torch(
        torch.cat(
            [
                attn.wq.weight.detach().transpose(0, 1),
                attn.wk.weight.detach().transpose(0, 1),
                attn.wv.weight.detach().transpose(0, 1),
            ],
            dim=-1,
        ).contiguous(),
        device,
    )
    wo = _weight(attn.wo, device)
    w1, w2, w3 = (_weight(m, device) for m in (ff.w1, ff.w2, ff.w3))
    g_attn = _gamma(blk.attention_norm, device)
    g_ffn = _gamma(blk.ffn_norm, device)
    eps = float(blk.attention_norm.eps)

    def run(h):
        xn = _rms_norm(h, g_attn, eps)
        h = ttnn.add(h, _attention(xn, wqkv, wo, n_heads, n_kv_heads, scale, mask))

        hn = _rms_norm(h, g_ffn, eps)
        gated = ttnn.multiply(
            ttnn.silu(ttnn.linear(hn, w1, compute_kernel_config=_COMPUTE)),
            ttnn.linear(hn, w3, compute_kernel_config=_COMPUTE),
        )
        return ttnn.add(h, ttnn.linear(gated, w2, compute_kernel_config=_COMPUTE))

    return run


def build(device, torch_module, batch=None):
    at = getattr(torch_module, "inner", torch_module)
    args = at.acoustic_transformer_args
    dim = int(args.dim)
    n_real_tokens = 3

    # float32, not bfloat16: `inv_freq` is COMPUTED by the module (exp of an arange), not a
    # checkpoint tensor, so bfloat16 would put 0.4% of error into the sinusoidal phase that
    # nothing downstream can recover.
    inv_freq = _from_torch(
        at.time_embedding.inv_freq.detach().reshape(1, -1).contiguous(), device, dtype=ttnn.float32
    )
    w_time = _weight(at.time_projection, device)
    w_llm = _weight(at.llm_projection, device)
    w_input = _weight(at.input_projection, device)
    w_acoustic = _weight(at.acoustic_codebook_output, device)
    w_semantic = _weight(at.semantic_codebook_output, device)
    semantic_bias = None
    if at.semantic_codebook_output.bias is not None:
        semantic_bias = _from_torch(
            at.semantic_codebook_output.bias.detach().reshape(1, 1, 1, -1), device
        )

    # Columns 3..31 of the padded tile are not real tokens; block them so rows 0..2 attend to
    # exactly the three the reference builds.
    mask_torch = torch.zeros(1, 1, _TILE, _TILE)
    mask_torch[:, :, :, n_real_tokens:] = _MASK_NEG
    mask = _from_torch(mask_torch, device, dtype=ttnn.float32)

    # The 29 pad rows of the one-tile sequence, as a PERSISTENT buffer rather than a per-call
    # `ttnn.zeros` (which a trace cannot replay). One buffer per distinct batch.
    pad_rows = _TILE - n_real_tokens
    _pads = {}

    def _pad_for(rows):
        buf = _pads.get(rows)
        if buf is None:
            buf = _from_torch(
                torch.zeros(rows, 1, pad_rows, dim), device, dtype=ttnn.float32
            )
            _pads[rows] = buf
        return buf

    if batch is not None:
        _pad_for(int(batch))

    blocks = [
        _compile_block(device, at.layers[str(i)], mask) for i in at.layers_ids
    ]
    g_final = _gamma(at.norm, device)
    eps_final = float(at.norm.eps)

    acoustic_out = int(at.acoustic_codebook_output.out_features)
    semantic_out = int(at.semantic_codebook_output.out_features)

    def flow_matching_audio_transformer(llm_hidden, x_t=None, t=None, **kwargs):
        batch = int(llm_hidden.shape[0])
        h_in = ttnn.reshape(llm_hidden, [batch, 1, 1, dim])

        h_in = ttnn.typecast(h_in, ttnn.float32)
        semantic = ttnn.linear(h_in, w_semantic, compute_kernel_config=_COMPUTE)
        if semantic_bias is not None:
            semantic = ttnn.add(semantic, semantic_bias)
        semantic = ttnn.reshape(semantic, [batch, semantic_out])

        # TimeEmbedding: outer product t (x) inv_freq, then cat(cos, sin).
        freqs = ttnn.matmul(
            ttnn.typecast(ttnn.reshape(t, [batch, 1, 1, int(t.shape[-1])]), ttnn.float32),
            ttnn.typecast(inv_freq, ttnn.float32),
            compute_kernel_config=_COMPUTE,
        )
        t_emb = ttnn.concat([ttnn.cos(freqs), ttnn.sin(freqs)], dim=-1)
        t_proj = ttnn.linear(t_emb, w_time, compute_kernel_config=_COMPUTE)
        llm_proj = ttnn.linear(h_in, w_llm, compute_kernel_config=_COMPUTE)
        x_proj = ttnn.linear(
            ttnn.typecast(ttnn.reshape(x_t, [batch, 1, 1, int(x_t.shape[-1])]), ttnn.float32),
            w_input,
            compute_kernel_config=_COMPUTE,
        )

        h = ttnn.concat([x_proj, t_proj, llm_proj, _pad_for(batch)], dim=2)

        for block in blocks:
            h = block(h)
        h = _rms_norm(h, g_final, eps_final)

        first = ttnn.slice(h, [0, 0, 0, 0], [batch, 1, 1, dim])
        velocity = ttnn.reshape(
            ttnn.linear(first, w_acoustic, compute_kernel_config=_COMPUTE),
            [batch, acoustic_out],
        )
        return velocity, semantic

    return flow_matching_audio_transformer

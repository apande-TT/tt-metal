# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `acoustic_transformer_block` (`AcousticTransformerBlock`).

    r = attention(attention_norm(x)); h = x + r
    r = feed_forward(ffn_norm(h));    out = h + r

The attention inside is `BidirectionalAttention`: NO positional encoding and NO causal mask, so
SDPA runs with `is_causal=False`. Shapes come from `params.json`'s `acoustic_transformer_args`:
dim 3072, 32 heads / 8 KV heads, head_dim 128, hidden_dim 9216, no biases.

BATCH AXIS. The leading bound is read from the tensor, never assumed to be 1: a `[B, 1, S, dim]`
input keeps all B samples and comes back as `[B, 1, S, dim]`, while the rank-<=3 input the
component test feeds (`[1, S, dim]`) is unchanged. A hardcoded leading 1 would have silently
dropped samples 1..B-1 once the sampler ran a real batch.

ADDITIVE MASK. `attn_mask=` is added to the attention scores. The flow-matching sampler's
sequence is 3 real tokens living in one 32-row tile, so it hands in a mask that blocks columns
3..31; the component test passes no mask and runs unmasked as before.

FIDELITY. Everything -- residual stream, Q/K/V, softmax -- runs in float32 against bfloat16
weights with HiFi4 + `fp32_dest_acc_en`. The sampler's output is rounded onto 21 levels 0.1 apart,
so a code flips on a few-1e-3 error, and SDPA's bfloat16-only interface plus `ttnn.softmax`'s
~6e-3 absolute error on the probabilities cost ~7% of the codes. See `_attention`.
"""

from __future__ import annotations

import math

import torch

import ttnn

_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
)


def _lin(x, w, **kwargs):
    """`ttnn.linear` with the leading batch folded into M, so the weight streams ONCE.

    A `[B, 1, S, K]` activation against a 2-D weight runs as B separate `S x K x N` matmuls that
    each re-read the whole weight from DRAM; `[1, 1, B*S, K]` is one matmul that reads it once.
    """
    shape = [int(d) for d in x.shape]
    lead = 1
    for d in shape[:-2]:
        lead *= d
    if lead == 1:
        return ttnn.linear(x, w, **kwargs)
    y = ttnn.linear(ttnn.reshape(x, [1, 1, lead * shape[-2], shape[-1]]), w, **kwargs)
    return ttnn.reshape(y, shape[:-1] + [int(y.shape[-1])])


def _from_torch(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    t = t.to(torch.bfloat16) if dtype == ttnn.bfloat16 else t.to(torch.float32)
    if device.__class__.__name__ == "MeshDevice":
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=layout,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


def _weight(linear, device):
    """A `[in, out]` device tensor for a torch `nn.Linear` (whose weight is `[out, in]`)."""
    return _from_torch(linear.weight.detach().transpose(0, 1).contiguous(), device)


def _gamma(norm, device):
    """A norm's gamma as a `[1, 1, 1, dim]` float32 tile tensor, for `_rms_norm`."""
    return _from_torch(norm.weight.detach().reshape(1, 1, 1, -1).contiguous(), device, dtype=ttnn.float32)


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
    qkv = _lin(h, wqkv, compute_kernel_config=_COMPUTE)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        qkv, num_heads=n_heads, num_kv_heads=n_kv_heads, transpose_k_heads=False
    )
    repeats = n_heads // n_kv_heads
    if repeats > 1:
        k = ttnn.repeat_interleave(k, repeats, dim=1)
        v = ttnn.repeat_interleave(v, repeats, dim=1)

    # The reference scales the QUERY before the product, not the scores after it.
    scores = ttnn.matmul(ttnn.multiply(q, scale), ttnn.transpose(k, -2, -1), compute_kernel_config=_COMPUTE)
    if attn_mask is not None:
        scores = ttnn.add(scores, attn_mask)
    scores = ttnn.subtract(scores, ttnn.max(scores, dim=-1, keepdim=True))
    weights = ttnn.exp(scores)
    weights = ttnn.divide(weights, ttnn.sum(weights, dim=-1, keepdim=True))

    out = ttnn.matmul(weights, v, compute_kernel_config=_COMPUTE)
    return _lin(
        ttnn.experimental.nlp_concat_heads(out),
        wo,
        dtype=ttnn.float32,
        compute_kernel_config=_COMPUTE,
    )


def _leading(shape) -> int:
    """The product of every axis before `[seq, dim]` -- the real batch, from the tensor."""
    dims = list(shape)[:-2]
    batch = 1
    for d in dims:
        batch *= int(d)
    return batch


def build(device, torch_module):
    blk = torch_module
    attn = blk.attention
    ff = blk.feed_forward

    n_heads = int(attn.n_local_heads)
    n_kv_heads = int(attn.n_local_kv_heads)
    head_dim = int(attn.head_dim)
    dim = int(blk.dim)
    scale = 1.0 / math.sqrt(head_dim)
    eps = float(blk.attention_norm.eps)

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
    w1 = _weight(ff.w1, device)
    w2 = _weight(ff.w2, device)
    w3 = _weight(ff.w3, device)
    g_attn = _gamma(blk.attention_norm, device)
    g_ffn = _gamma(blk.ffn_norm, device)

    def acoustic_transformer_block(x, attn_mask=None, **kwargs):
        seq = int(x.shape[-2])
        batch = _leading(x.shape)
        rank = len(list(x.shape))

        h4 = ttnn.reshape(x, [batch, 1, seq, dim])
        if h4.dtype != ttnn.float32:
            h4 = ttnn.typecast(h4, ttnn.float32)

        xn = _rms_norm(h4, g_attn, eps)
        h4 = ttnn.add(h4, _attention(xn, wqkv, wo, n_heads, n_kv_heads, scale, attn_mask))

        hn = _rms_norm(h4, g_ffn, eps)
        gate = ttnn.silu(_lin(hn, w1, compute_kernel_config=_COMPUTE))
        up = _lin(hn, w3, compute_kernel_config=_COMPUTE)
        h4 = ttnn.add(h4, _lin(ttnn.multiply(gate, up), w2, compute_kernel_config=_COMPUTE))

        if rank >= 4:
            return h4
        return ttnn.reshape(h4, [batch, seq, dim])

    return acoustic_transformer_block

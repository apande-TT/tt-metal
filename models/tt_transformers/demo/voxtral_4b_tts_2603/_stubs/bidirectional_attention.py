# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `bidirectional_attention` (`acoustic_transformer.layers.0.attention`).

The canonical `models/tt_transformers/tt/attention.py` is not usable here: it builds itself from
`ModelArgs`, which resolves the model through `AutoConfig`, and this checkpoint is a native Mistral
`consolidated.safetensors` with no `config.json` / `model_type` -- `ModelArgs` raises before
reading a weight. Its math is also the wrong math (see below). So this is a direct ttnn forward
over the resolved submodule's own weights.

Despite tensor names identical to the text backbone (`wq/wk/wv/wo`), this attention is
**bidirectional and RoPE-free**: no positional encoding and no causal mask, so SDPA runs with
`is_causal=False` and nothing is rotated. The `rope_theta: 10000.0` under
`acoustic_transformer_args` in `params.json` is dead config -- `AcousticTransformerArgs` has no
such field.

GQA 32 query heads over 8 KV heads, head_dim 128, dim 3072, no biases. The reference returns
`wo(...).squeeze(0)`, i.e. `[S, dim]`; a leading-1 reshape is metadata only, so keeping rank
`[1, S, dim]` compares the same elements in the same order.

BATCH AXIS. The leading bound is read from the tensor, never assumed to be 1: a `[B, 1, S, dim]`
input keeps all B samples and comes back at rank 4, while the rank-<=3 input the component test
feeds is unchanged. A hardcoded leading 1 would silently drop samples 1..B-1.

`attn_mask=` is an additive mask applied to the scores (the flow-matching sampler pads its
3-token sequence to one 32-row tile and blocks columns 3..31). The whole attention runs in
float32 against bfloat16 weights at HiFi4 + `fp32_dest_acc_en` -- see `_attention` for why it is
written out instead of calling SDPA.
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


def _leading(shape) -> int:
    """The product of every axis before `[seq, dim]` -- the real batch, from the tensor."""
    batch = 1
    for d in list(shape)[:-2]:
        batch *= int(d)
    return batch


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


def build(device, torch_module):
    attn = torch_module
    n_heads = int(attn.n_local_heads)
    n_kv_heads = int(attn.n_local_kv_heads)
    head_dim = int(attn.head_dim)
    dim = int(attn.wq.in_features)
    out_dim = int(attn.wo.out_features)
    scale = 1.0 / math.sqrt(head_dim)

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
    wo = _from_torch(attn.wo.weight.detach().transpose(0, 1).contiguous(), device)

    def bidirectional_attention(x, attn_mask=None, **kwargs):
        seq = int(x.shape[-2])
        batch = _leading(x.shape)
        rank = len(list(x.shape))

        h = ttnn.reshape(x, [batch, 1, seq, dim])
        if h.dtype != ttnn.float32:
            h = ttnn.typecast(h, ttnn.float32)
        out = _attention(h, wqkv, wo, n_heads, n_kv_heads, scale, attn_mask)
        if rank >= 4:
            return ttnn.reshape(out, [batch, 1, seq, out_dim])
        return ttnn.reshape(out, [batch, seq, out_dim])

    return bidirectional_attention

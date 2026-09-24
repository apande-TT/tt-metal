# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `feed_forward` (`acoustic_transformer.layers.0.feed_forward`).

SwiGLU: `w2(silu(w1(x)) * w3(x))`, dim 3072 -> hidden 9216 -> 3072. Only `w2` can carry a bias
(`use_biases`, false in this checkpoint), so it is applied conditionally rather than assumed away.

Note the naming: this module's `w1` is the GATE and `w3` is the up-projection, the opposite of the
`{gate,up}_proj` ordering the checkpoint's key map implies -- `feed_forward.w1/w2/w3` maps to
`mlp.{gate,down,up}_proj`.

BATCH AXIS. The leading bound is read from the tensor: a `[B, 1, S, dim]` input keeps all B
samples and comes back at rank 4, while the rank-<=3 input the component test feeds is unchanged.
HiFi4 + `fp32_dest_acc_en` on a float32 activation against bfloat16 weights, because the
flow-matching sampler this feeds rounds its output onto 21 levels.
"""

from __future__ import annotations

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


def _weight(linear, device):
    """A `[in, out]` device tensor for a torch `nn.Linear` (whose weight is `[out, in]`)."""
    return _from_torch(linear.weight.detach().transpose(0, 1).contiguous(), device)


def build(device, torch_module):
    ff = torch_module
    dim = int(ff.w1.in_features)
    out_dim = int(ff.w2.out_features)

    w1 = _weight(ff.w1, device)
    w2 = _weight(ff.w2, device)
    w3 = _weight(ff.w3, device)
    bias = None
    if ff.w2.bias is not None:
        bias = _from_torch(ff.w2.bias.detach().reshape(1, 1, 1, out_dim), device)

    def feed_forward(x, **kwargs):
        seq = int(x.shape[-2])
        batch = _leading(x.shape)
        rank = len(list(x.shape))

        h = ttnn.reshape(x, [batch, 1, seq, dim])
        gated = ttnn.multiply(
            ttnn.silu(_lin(h, w1, compute_kernel_config=_COMPUTE)),
            _lin(h, w3, compute_kernel_config=_COMPUTE),
        )
        out = _lin(gated, w2, compute_kernel_config=_COMPUTE)
        if bias is not None:
            out = ttnn.add(out, bias)
        if rank >= 4:
            return ttnn.reshape(out, [batch, 1, seq, out_dim])
        return ttnn.reshape(out, [batch, seq, out_dim])

    return feed_forward

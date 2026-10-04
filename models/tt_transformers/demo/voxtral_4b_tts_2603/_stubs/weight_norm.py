# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `weight_norm`
(`audio_tokenizer.decoder_blocks.0.conv.parametrizations.weight.0`).

`torch._weight_norm(v, g, dim=0)`:

    w[o] = g[o] * v[o] / ||v[o]||_2

i.e. the L2 norm is per OUTPUT CHANNEL, over every other axis. `g` is `[out, 1, 1]` and `v` is
`[out, in, k]`.

Reduced in two stages (over `k`, then over `in`) rather than by flattening the tail axes into one.
`v` arrives as `[1024, 292, 3]` in TILE layout, where the trailing 3 is padded to a tile width of 32
and the 292 to 320, so a `[1024, 876]` reshape would have to move data across that padding; two
reductions plus an `[out, 1, 1]` broadcast stay inside the existing tiles.

Widened to float32 for the reduction: 876 squares are summed per channel. `g` is **signed** here --
23 of this layer's 1024 channels are negative -- so the verification identity is `||w|| == |g|`,
not `== g`; checked the wrong way it reads as a relative error of 2.0.
"""

from __future__ import annotations

import ttnn


def build(device, torch_module):
    if int(torch_module.dim) != 0:
        raise NotImplementedError(
            f"weight_norm over dim {torch_module.dim} is not ported; only dim 0 " f"(per-output-channel)"
        )

    def weight_norm(weight_g, weight_v=None, **kwargs):
        if weight_v is None:
            raise ValueError("weight_norm needs both weight_g and weight_v")
        if kwargs.get("taps") is not None:
            return _wide(weight_g, weight_v, int(kwargs["taps"]), kwargs["pad"], kwargs.get("dtype"))
        # Widened only when not float32 already: a float32 -> float32 typecast is a full copy of the
        # tile-padded v (k = 7 padded to 32: a 31 MB pass for the codec output projection).
        v = weight_v if weight_v.dtype == ttnn.float32 else ttnn.typecast(weight_v, ttnn.float32)
        g = weight_g if weight_g.dtype == ttnn.float32 else ttnn.typecast(weight_g, ttnn.float32)
        # One reduction over both trailing axes: each separate pass fills the tile padding and re-reads the
        # whole padded tensor (k = 7 is padded to a tile width), so two passes cost two of each.
        sq_sum = ttnn.sum(ttnn.multiply(v, v), dim=[-2, -1], keepdim=True)
        return ttnn.multiply(v, ttnn.multiply(ttnn.rsqrt(sq_sum), g))

    return weight_norm


def _wide(g, v, taps, pad, dtype=None):
    """The same reconstruction on a conv's WIDE layout: `v` is `[.., C_in, taps * C']` -- tap t's output channels
    at columns `t * C' ..` (C' the tile-padded C_out, padding columns zero) -- and `g` / `pad` are `[.., 1, C']`
    (`pad` 1.0 on the padding channels so their zero norm stays finite; their `g` is 0).

    structural: the conv takes this layout as is, so the k-last `[C_out, C_in, k]` reconstruction (k padded to a
    whole tile) and its permutes into it are gone. One row reduction over C_in, the taps' partial sums added per
    output channel (tile-aligned column blocks), one scale row broadcast down the rows."""
    cp = int(g.shape[-1])
    cols = ttnn.sum(ttnn.multiply(v, v), dim=-2, keepdim=True)
    dims = [int(d) for d in cols.shape]  # (a ttnn.Shape does not slice)
    rank, lead = len(dims), dims[:-1]
    sq = None
    for t in range(taps):
        part = ttnn.slice(cols, [0] * (rank - 1) + [t * cp], lead + [(t + 1) * cp])
        sq = part if sq is None else ttnn.add(sq, part)
    scale = ttnn.multiply(ttnn.rsqrt(ttnn.add(sq, pad)), g)
    out = {"dtype": dtype} if dtype is not None else {}
    return ttnn.multiply(v, ttnn.concat([scale] * taps, dim=-1), **out)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `parametrized_conv1d` (`audio_tokenizer.decoder_blocks.0.conv`).

The bare `weight_norm`-parametrized `Conv1d` INSIDE `CausalConv1d`: 292 -> 1024 channels, kernel 3,
stride 1, **no padding of its own** (the causal wrapper does the padding), no bias. Reading
`conv.weight` here runs the parametrization on the host, which is where weight prep belongs.

Done as shifted matmuls: with the sequence channels-last as `[B, 1, L, C]`, output position `t` is
`sum_k x[t + k] @ W_k`, so each kernel tap is one `[C_in, C_out]` matmul over a row-shifted slice.
That keeps the forward to ttnn matmul / slice / add -- no conv sharding config to get wrong -- and
the row-shifted slices are exact even at non-tile-aligned offsets.

The `weight=` keyword takes the weight as a DEVICE TENSOR and splits the taps out of it per call.
That is what lets the `weight_norm` parametrization stay in the forward, where torch runs it: the
`parametrization_list` port reconstructs `g * v / ||v||` on the device and hands the result
straight here, so the tensor those two stubs compute is the tensor this convolution consumes.
Without it the taps are read once in `build` (which also runs the parametrization, on the host).
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import cpp_shift_add

# A PERSISTENT ZERO BUFFER, NOT A PER-CALL `ttnn.zeros`.
# `ttnn.zeros` builds its tensor on the host and enqueues a WRITE to land it on the device, and a
# captured trace cannot replay a write (`TT_FATAL: Writes are not supported during trace capture`).
# The shape follows the input shape, which a trace pins, so it is created once per
# (device, shape, dtype) and reused. Read-only, so it is never deallocated by a caller.
_ZEROS = {}


def _zeros_like_buf(device, shape, dtype):
    key = (id(device), tuple(int(s) for s in shape), str(dtype))
    buf = _ZEROS.get(key)
    if buf is None:
        buf = ttnn.zeros(list(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        _ZEROS[key] = buf
    return buf


# fp32 accumulation in DEST. The codec's activation path is float32 -- bfloat16 end to end put the
# full chain at PCC 0.9873 over eight residual blocks plus five convolutions.
_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
)


# FLOAT32 WEIGHTS. Measured on this device against a float64 reference, one matmul at M=32,
# K=N=3072, HiFi4 + `fp32_dest_acc_en`: a bfloat16 weight costs 1.738e-3 relative where a float32
# weight costs 1.169e-3. That 1.5x is small per op and this codec stacks eight residual blocks on
# top of five convolutions, where it is the last error source left after the softmax and the RMS
# norm were spelled out. Nothing here is an `ttnn.embedding` table (which would have to stay
# bfloat16); those live in the codebook stubs.
def _from_torch(t, device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    t = t.to(torch.bfloat16) if dtype == ttnn.bfloat16 else t.to(torch.float32)
    if layout == ttnn.TILE_LAYOUT and t.numel() <= 262144:
        # A small tensor (a gamma, a bias, a layer scale) tilizes on the HOST: tilized on the device it is
        # a whole single-core op (~80 us for a 1 x 1024 float32 row) at load time.
        mapper = ttnn.ReplicateTensorToMesh(device) if device.__class__.__name__ == "MeshDevice" else None
        return ttnn.to_device(ttnn.from_torch(t, dtype=dtype, layout=layout, mesh_mapper=mapper), device)
    if device.__class__.__name__ == "MeshDevice":
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=layout,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


def _tap_linear(x, w, **kwargs):
    """`ttnn.linear` for one tap with the leading batch folded into M, so the tap streams ONCE.

    A `[B, 1, L, C]` slice against a 2-D tap runs as B separate `L x C x C_out` matmuls that each
    re-read the whole tap; `[1, 1, B*L, C]` is one matmul. An L that is not tile-aligned makes the
    fold a real relayout each way, still far cheaper than re-reading the tap B times.
    """
    shape = [int(d) for d in x.shape]
    lead = 1
    for d in shape[:-2]:
        lead *= d
    if lead == 1:
        return ttnn.linear(x, w, **kwargs)
    y = ttnn.linear(ttnn.reshape(x, [1, 1, lead * shape[-2], shape[-1]]), w, **kwargs)
    return ttnn.reshape(y, shape[:-1] + [int(y.shape[-1])])


def _wide_linear(x, w, **kwargs):
    """The wide conv product (`[1, 1, B*L, C_in] x [C_in, K * C_out]`) on the codec linears' hand-sized
    full-grid 2D-mcast config (tt/vocode_stage._mcast_cfg) -- left to the default, its float32 rows pick
    in0_block_w=1 (every K step re-packs the whole float32 out block: 4096 x 1024 x 1792 ran 287 us)."""
    from models.demos.voxtral_4b_tts_2603.tt.vocode_stage import _mcast_cfg

    rows = int(x.shape[-2])
    # grid: the full-grid config for the tall products (the fidelity stays the caller's), sized by the
    # product's real output dtype.
    out_dtype = kwargs.get("dtype") or ttnn.float32
    cfg = _mcast_cfg(x, w, rows, out_dtype) if rows >= 128 and rows % 32 == 0 else None
    if cfg is not None:
        kwargs["program_config"] = cfg
        # fidelity: the tall wide product at HiFi2 (fp32 DEST kept) -- what the tokenizer half's same output
        # conv runs; HiFi4 doubled its math passes.
        kwargs["compute_kernel_config"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
        )
    return ttnn.linear(x, w, **kwargs)


def build(device, torch_module):
    conv = torch_module

    weight = conv.weight.detach()
    out_channels, in_channels, kernel = (int(v) for v in weight.shape)
    stride = int(conv.stride[0])
    dilation = int(conv.dilation[0])
    pad = int(conv.padding[0]) if not isinstance(conv.padding, str) else 0
    effective_kernel = (kernel - 1) * dilation + 1

    taps = [_from_torch(weight[:, :, i].transpose(0, 1).contiguous(), device) for i in range(kernel)]
    bias = None
    if conv.bias is not None:
        bias = _from_torch(conv.bias.detach().reshape(1, 1, 1, out_channels), device)

    def _taps_from(weight):
        """The `k` `[C_in, C_out]` taps of a device-resident `[C_out, C_in, k]` weight.

        `permute(2, 1, 0)` puts the tap axis first, so each tap is a leading-axis slice -- no
        sub-tile slice on the length-`k` axis. Narrowed to bfloat16 to match the taps `build`
        prepares: the reconstruction is computed wide, but the weight the matmul consumes is the
        same width either way.
        """
        wp = ttnn.permute(weight, (2, 1, 0))
        return [
            ttnn.typecast(
                ttnn.reshape(
                    ttnn.slice(wp, [i, 0, 0], [i + 1, in_channels, out_channels]),
                    [in_channels, out_channels],
                ),
                ttnn.bfloat16,
            )
            for i in range(kernel)
        ]

    def _wide_from(weight):
        """`cpp_shift_add.wide_weight(_taps_from(weight))` without the (2, 1, 0) permute of the k-last weight.

        The device weight is `[C_out, C_in, k]` with k (7) padded to a whole tile, so permuting k to the front
        moves a mostly-padding tensor element by element. Instead: swap the last two axes (a tile transpose),
        swap the two leading axes (whole rows move), pad C_out to a tile multiple, view the `[k, C_out', C_in]`
        block as `[k * C_out', C_in]` and transpose it -- the k taps side by side, `[C_in, k * C_out']`. The
        same values as the per-tap path (pure data movement, then the same bf16 narrowing).
        """
        cp = -(-out_channels // 32) * 32
        wk = ttnn.permute(ttnn.transpose(weight, -2, -1), (1, 0, 2))
        if cp != out_channels:
            wk = ttnn.pad(wk, [(0, 0), (0, cp - out_channels), (0, 0)], 0.0)
        flat = ttnn.reshape(wk, [kernel * cp, in_channels])
        return ttnn.typecast(ttnn.transpose(flat, -2, -1), ttnn.bfloat16), kernel, out_channels

    def parametrized_conv1d(x, weight=None, **kwargs):
        # `[B, C, L]` in, `[B, C_out, L']` out. The leading bound comes from the TENSOR, never from
        # a literal 1: the pipeline stacks 32 independent samples on axis 0.
        shape = [int(v) for v in x.shape]
        length = shape[-1]
        batch = shape[0] if len(shape) >= 3 else 1
        if pad == 0 and kwargs.get("x_rm") is not None:
            # The rows the caller hands over ROW_MAJOR (`[B, 1, L, C]`) are the conv's input; `x` is not read.
            length = int(kwargs["x_rm"].shape[-2])

        x_cl = kwargs.get("x_cl") if pad == 0 else None
        if x_cl is not None and cpp_shift_add.supports_unpadded(x_cl, kwargs.get("pad_mode")) and kernel > 1:
            # structural: the caller's UNPADDED channels-last TILE rows, its padding (reflect / replicate)
            # resolved in the shift-add -- no padded copy of the rows is ever built (tt/cpp_shift_add).
            real = int(x_cl.shape[-2])
            front, back = int(kwargs.get("pad_front", 0)), int(kwargs.get("pad_back", 0))
            out_len = (real + front + back - effective_kernel) // stride + 1
            if kwargs.get("wide_weight") is not None:
                # A weight already reconstructed in the wide `[C_in, k * C']` layout (parametrization_list(wide=True)).
                wide = (kwargs["wide_weight"], kernel, out_channels)
            else:
                wide = cpp_shift_add.wide_weight(taps) if weight is None else _wide_from(weight)
            acc = cpp_shift_add.conv_wide_unpadded(
                x_cl, wide, kwargs["pad_mode"], front, out_len, _COMPUTE, linear=_wide_linear
            )
            if bias is not None:
                acc = ttnn.add(acc, bias)
            if kwargs.get("cl_out"):
                # The caller takes the result channels-LAST, `[B, L', C_out]`.
                return ttnn.reshape(acc, [batch, out_len, out_channels])
            return ttnn.reshape(ttnn.transpose(acc, -2, -1), [batch, out_channels, out_len])

        padded_len = length + 2 * pad
        out_len = (padded_len - effective_kernel) // stride + 1

        # `x_rm`: a caller holding the same rows ROW_MAJOR (`[B, 1, L, C]`) lets the taps be cut there --
        # a row-offset slice of a TILE tensor untilizes it whole for every tap. With it, `x` only gives the
        # shape: its channels-last transpose is never made.
        x_rm = kwargs.get("x_rm") if pad == 0 else None
        if x_rm is not None and cpp_shift_add.supports(x_rm, taps, stride, dilation):
            # structural: ONE tilize of the rows and ONE product against the taps side by side; the row shift
            # happens on the narrow tap outputs (tt/cpp_shift_add), not as K copies of the wide input.
            if kwargs.get("wide_weight") is not None:
                # A weight already reconstructed in the wide `[C_in, k * C']` layout (parametrization_list(wide=True)).
                wide = (kwargs["wide_weight"], kernel, out_channels)
            else:
                wide = cpp_shift_add.wide_weight(taps) if weight is None else _wide_from(weight)
            acc = cpp_shift_add.conv_wide(x_rm, wide, out_len, _COMPUTE, linear=_wide_linear)
            if bias is not None:
                acc = ttnn.add(acc, bias)
            if kwargs.get("cl_out"):
                # The caller takes the result channels-LAST, `[B, L', C_out]`.
                return ttnn.reshape(acc, [batch, out_len, out_channels])
            return ttnn.reshape(ttnn.transpose(acc, -2, -1), [batch, out_channels, out_len])
        active_taps = taps if weight is None else _taps_from(weight)
        x4 = None
        if x_rm is None:
            x4 = ttnn.reshape(ttnn.transpose(x, -2, -1), [batch, 1, length, in_channels])
            if pad > 0:
                zeros = _zeros_like_buf(device, [batch, 1, pad, in_channels], x4.dtype)
                x4 = ttnn.concat([zeros, x4, zeros], dim=2)
        acc = None
        for i, tap in enumerate(active_taps):
            begin = i * dilation
            end = begin + (out_len - 1) * stride + 1
            step = [1, 1, stride, 1] if stride > 1 else None
            if x_rm is not None:
                seg = ttnn.to_layout(
                    ttnn.slice(x_rm, [0, 0, begin, 0], [batch, 1, end, in_channels], step), ttnn.TILE_LAYOUT
                )
            else:
                seg = ttnn.slice(x4, [0, 0, begin, 0], [batch, 1, end, in_channels], step)
            term = _tap_linear(seg, tap, compute_kernel_config=_COMPUTE)
            acc = term if acc is None else ttnn.add(acc, term)
        if bias is not None:
            acc = ttnn.add(acc, bias)

        return ttnn.reshape(ttnn.transpose(acc, -2, -1), [batch, out_channels, out_len])

    return parametrized_conv1d

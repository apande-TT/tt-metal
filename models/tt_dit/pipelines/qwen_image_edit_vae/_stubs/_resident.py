# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resident composition of the VAE ports.

Each block port (causal conv, RMS norm, residual / mid / up / attention block, resample) is a thin
adapter around a tt_dit Wan body: its __call__ takes a replicated BCTHW tensor, partitions it over the
mesh, runs the body and gathers the result. Inside the encoder / decoder the activation is ALREADY
partitioned (BTHWC shards), so the port is entered through `forward_sharded`, which runs the same
body on the shards and skips the adapter.

`around(port_cls, body, ...)` builds a port instance that owns an existing body (no second copy of
the weights) and routes the body's forward through the port. The encoder3d / decoder3d ports call
`attach_block_ports()` once at build time, so every child of their Wan stack runs as the matching
graduated port.
"""

from __future__ import annotations

import ttnn


class ResidentPort:
    """Mixin for a port whose body lives inside an encoder/decoder stack."""

    BODY_ATTR = "block"

    @classmethod
    def around(cls, body, device, parallel_config=None, ccl_manager=None, **extra):
        self = cls.__new__(cls)
        self.device = device
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        setattr(self, cls.BODY_ATTR, body)
        for k, v in extra.items():
            setattr(self, k, v)
        self._body_forward = type(body).forward
        # late-bound so a subclass swapped in after attach (e.g. an invocation counter) is honoured
        body.forward = lambda *a, **k: self.forward_sharded(*a, **k)  # the Wan stack now calls the port
        return self

    def forward_sharded(self, *args, **kwargs):
        """The graduated body on the already-partitioned activation (BTHWC shards)."""
        return self._body_forward(getattr(self, self.BODY_ATTR), *args, **kwargs)


def ports_of(ports, bodies):
    """The ports wrapping `bodies` (e.g. a Wan ModuleList), in the bodies' order.

    A tt_dit ModuleList keeps its children in `_children`, which a generic walk does not open; this
    plain list of the same port objects is how the stack stays visible (it owns nothing new)."""
    by_body = {id(getattr(p, p.BODY_ATTR)): p for p in ports}
    return [by_body[id(b)] for b in bodies if id(b) in by_body]


def walk(module):
    """Depth-first over a tt_dit Module tree (the root included)."""
    yield module
    for _, child in module.named_children():
        yield from walk(child)


def attach_block_ports(stack, device, parallel_config, ccl_manager, torch_stack=None, sink=None):
    """Route every child of a Wan encoder/decoder `stack` through its graduated port.

    torch_stack: the HF QwenImageEncoder3d / QwenImageDecoder3d, used to build the op-level ports
    (ZeroPad2d, QwenImageUpsample) that sit inside each resample. sink: optional list collecting the
    ports that were created (in stack order).
    """
    from models.tt_dit.layers.normalization import RMSNorm
    from models.tt_dit.models.vae.vae_wan2_1 import (
        WanAttentionBlock,
        WanCausalConv3d,
        WanMidBlock,
        WanResample,
        WanResidualBlock,
        WanUpBlock,
    )
    from models.tt_dit.pipelines.qwen_image_edit_vae._stubs.qwen_image_attention_block import TtQwenImageAttentionBlock
    from models.tt_dit.pipelines.qwen_image_edit_vae._stubs.qwen_image_causal_conv3d import TtQwenImageCausalConv3d
    from models.tt_dit.pipelines.qwen_image_edit_vae._stubs.qwen_image_mid_block import TtQwenImageMidBlock
    from models.tt_dit.pipelines.qwen_image_edit_vae._stubs.qwen_image_r_m_s import TtQwenImageRMSNorm
    from models.tt_dit.pipelines.qwen_image_edit_vae._stubs.qwen_image_resample import TtQwenImageResample
    from models.tt_dit.pipelines.qwen_image_edit_vae._stubs.qwen_image_residual_block import TtQwenImageResidualBlock
    from models.tt_dit.pipelines.qwen_image_edit_vae._stubs.qwen_image_up_block import TtQwenImageUpBlock

    # op-level ports for the resamples, in stack order (HF: resample = Sequential(op, Conv2d))
    torch_ops = []
    if torch_stack is not None:
        for m in torch_stack.modules():
            if type(m).__name__ == "QwenImageResample" and m.mode in (
                "downsample2d",
                "downsample3d",
                "upsample2d",
                "upsample3d",
            ):
                torch_ops.append(m.resample[0])

    table = [
        (WanUpBlock, TtQwenImageUpBlock),
        (WanMidBlock, TtQwenImageMidBlock),
        (WanResidualBlock, TtQwenImageResidualBlock),
        (WanAttentionBlock, TtQwenImageAttentionBlock),
        (WanCausalConv3d, TtQwenImageCausalConv3d),
        (RMSNorm, TtQwenImageRMSNorm),
    ]
    ports = sink if sink is not None else []
    op_i = 0
    for mod in list(walk(stack))[1:]:
        if isinstance(mod, WanResample):
            op = torch_ops[op_i] if op_i < len(torch_ops) else None
            op_i += 1
            ports.append(TtQwenImageResample.around_resample(mod, device, parallel_config, ccl_manager, op))
            continue
        for body_cls, port_cls in table:
            if type(mod) is body_cls:
                ports.append(port_cls.around(mod, device, parallel_config, ccl_manager))
                break
    return ports


# ---- precise evaluation of an affine op (the ports' float32 precise modes) ------------------------------
# Two measured floors of the float32 conv / matmul kernels (encoder conv_in on HF's own input, vs float64):
#   * dense accumulation: relative L2 error 4.2e-4 (hi + lo bf16 input limbs: 2.2e-4). A tile dot product
#     is exact when each 32-term group holds <= 4 nonzeros spaced 8 apart, so the input channels are split
#     into 8 lanes (c % 8) x 2 bf16 limbs: 16 runs summed in float32 -> 3.0e-6;
#   * a rare output off by an exact power of two (decoder last up block: 3 of 6.3M outputs, |err| up to
#     0.25 of a max 0.85; conv_in: one output off by 2.0), deterministic in the data, which the VAE's
#     channel norms amplify. The same op on different contents does not glitch at the same place
#     (conv_in: plain glitches, -x and hi + lo do not).
# mode "median": elementwise median of op(x), 2 b - op(-x), op(x_hi) + op(x_lo) - b  (b = op(0), exact).
# mode "exact":  the 16-run exact-lane sum, kept where it agrees with 2 b - op(-x) and replaced by the
#                median of (exact, plain, negated) where it does not.


# Optional fused kernel for mode "exact"'s vote tail: VOTE_TAIL(acc, bn, f0, e_neg, tol) -> the res below
# (bit-identical), consuming (deallocating) its operands; bn the (n - 1) b bias to take off acc (output-shaped,
# or a (1, 1, C) row). None for a shape it does not take -- then the ttnn spelling runs.
VOTE_TAIL = None
# Optional fused kernel for the bf16 rounding of a float32 tile tensor, as float32 (the hi limb):
# ROUND_BF16(t) -> typecast(typecast(t, bf16), float32) in one pass, t untouched; None for a shape it does not
# take.
ROUND_BF16 = None


def _flat_rows(t):
    """Whether t folds to (1, rows, C) as a view: row-major with tile-aligned C (its pages stay C wide;
    the (n/32, 32) view would re-page it, a full copy)."""
    s = list(t.shape)
    return t.layout != ttnn.TILE_LAYOUT and s[-1] % 32 == 0


def _flat(t):
    n = 1
    for d in t.shape:
        n *= d
    if _flat_rows(t):
        return ttnn.to_layout(ttnn.reshape(t, (1, n // t.shape[-1], t.shape[-1])), ttnn.TILE_LAYOUT)
    return ttnn.to_layout(ttnn.reshape(t, (n // 32, 32)), ttnn.TILE_LAYOUT) if n % 32 == 0 else None


def _bias_flat(bias, shape, rows_layout=False):
    """_flat of the [*, C] tensor whose every row is `bias` (None when that layout does not apply). In the
    (1, rows, C) fold it is bias itself as (1, 1, C), broadcast by the ops that use it. In the (n/32, 32)
    view row r holds channels (32 r) mod C .. +31, so for C % 32 == 0 it is bias as (C/32, 32) repeated
    n/C times: one output-sized buffer, no copy of the output kept alive."""
    if bias is None:
        return None
    c, n = shape[-1], 1
    for d in shape:
        n *= d
    if c % 32 or tuple(bias.shape) != (1, c) or n % c:
        return None
    if rows_layout:
        return ttnn.reshape(bias, (1, 1, c))
    rows = ttnn.reshape(ttnn.to_layout(bias, ttnn.ROW_MAJOR_LAYOUT), (c // 32, 32))
    return ttnn.to_layout(ttnn.repeat(rows, (n // c, 1)), ttnn.TILE_LAYOUT)


_LANE_MASKS = {}


def _lane_masks(device, c):
    key = (id(device), c)
    if key not in _LANE_MASKS:
        _LANE_MASKS[key] = [
            ttnn.Tensor(
                [1.0 if i % 8 == r else 0.0 for i in range(c)], [1, 1, c], ttnn.float32, ttnn.TILE_LAYOUT, device
            )
            for r in range(8)
        ]
    return _LANE_MASKS[key]


_WEIGHT_LANE_MASKS = {}


def _weight_lane_masks(device, rows):
    """[rows, 1] 0/1 masks, mask r keeping the rows j % 8 == r: a prepared conv3d weight's rows run
    [C_in block][kD][kH][kW][channel in block] with C_in_block % 8 == 0, so row j's input channel is j mod 8 too."""
    key = (id(device), rows)
    if key not in _WEIGHT_LANE_MASKS:
        _WEIGHT_LANE_MASKS[key] = [
            ttnn.Tensor(
                [1.0 if j % 8 == r else 0.0 for j in range(rows)], [rows, 1], ttnn.float32, ttnn.TILE_LAYOUT, device
            )
            for r in range(8)
        ]
    return _WEIGHT_LANE_MASKS[key]


def precise_affine(run, x, mode, device, extra=None, out=None, bias=None, prepad=None, weight=None, run_w=None):
    """run(x, extra) -> output, affine in (x, extra) (a conv / linear with bias); channels on the last dim.
    Returns the output in run's own layout and shape, evaluated per `mode` (see above). `out`: run(x, extra)
    if the caller already has it. `bias`: run's [1, C_out] bias when run(0, 0) is exactly it (a float32
    conv3d adds its bias unrounded); the zero-input run is then built from it instead of convolved.
    `prepad(x, extra) -> x_padded`: the zero padding / halo exchange / masking run applies before its core,
    with `run` then the core alone. Every input variant below maps 0 to 0, so padding commutes with it
    and is done once instead of once per run.
    `weight`, `run_w(x, w)`: run's prepared conv3d weight and run with a stand-in weight. Given both, an exact
    lane masks the WEIGHT's input-channel rows instead of the input: conv(x * m_r, W) = conv(x, W * m_r^T) with
    the very same products (the zeros now come from the weight), so the hi limb is untilized once for all the
    lanes and each lane costs a weight-sized multiply, not an input-sized multiply plus an untilize."""
    if prepad is not None:
        x, extra = prepad(x, extra), None

    def to3(t):
        s = list(t.shape)
        if t.layout != ttnn.TILE_LAYOUT:
            # every op on the copy is elementwise: fold all rows into one dim so the tiles are dense (a narrow
            # W, e.g. a 10-wide halo-padded shard, would otherwise pad each 32-row tile to a third full)
            rows = 1
            for d in s[:-1]:
                rows *= d
            return ttnn.to_layout(ttnn.reshape(t, (1, rows, s[-1])), ttnn.TILE_LAYOUT)
        lead = 1
        for d in s[:-2]:
            lead *= d
        return ttnn.to_layout(ttnn.reshape(t, (lead, s[-2], s[-1])), ttnn.TILE_LAYOUT)

    def back(u, like):
        return ttnn.reshape(ttnn.to_layout(u, like.layout), list(like.shape))

    # each run's limb / lane is a transient of the operand's tile copy
    def f32(t):  # t rounded to bf16, as float32
        y = None if ROUND_BF16 is None else ROUND_BF16(t)
        return ttnn.typecast(ttnn.typecast(t, ttnn.bfloat16), ttnn.float32) if y is None else y

    hi, lo = f32, lambda t: ttnn.subtract(t, f32(t))
    if mode == "split":
        # the exact-product split alone, op(x_hi) + op(x_lo) - b: two runs, no plain run and no vote
        ops = [to3(x)] + ([] if extra is None else [to3(extra)])

        def raw(fn):
            return run(back(fn(ops[0]), x), None if extra is None else back(fn(ops[1]), extra))

        o_hi = raw(hi)
        shape, layout = list(o_hi.shape), o_hi.layout
        f_hi = _flat(o_hi)
        if f_hi is None:  # no flat view of this output: the plain run
            return run(x, extra)
        y = ttnn.add(f_hi, _flat(raw(lo)))
        b = _bias_flat(bias, shape, rows_layout=_flat_rows(o_hi))
        b = b if b is not None else _flat(raw(lambda t: ttnn.multiply(t, 0.0)))
        return ttnn.reshape(ttnn.to_layout(ttnn.subtract(y, b), layout), shape)

    out = run(x, extra) if out is None else out
    shape, layout = list(out.shape), out.layout
    f0 = _flat(out)
    if f0 is None or mode not in ("median", "exact"):
        return out

    # with prepad the tile copy is shared by every run; without it each run tilizes its own
    shared = [to3(x)] + ([] if extra is None else [to3(extra)]) if prepad is not None else None

    def run_on(fn, tiles=None):
        tiles = tiles or shared or [to3(x)] + ([] if extra is None else [to3(extra)])
        xv = back(fn(tiles[0]), x)
        ev = None if extra is None else back(fn(tiles[1]), extra)
        return _flat(run(xv, ev))

    def free(ts, srcs=None):
        for t, src in zip(ts, srcs or ts):
            if src.layout != ttnn.TILE_LAYOUT or srcs is None:  # a tile src: to3 may have returned a view of it
                ttnn.deallocate(t)

    b = _bias_flat(bias, shape, rows_layout=_flat_rows(out))
    if b is None:
        b = run_on(lambda t: ttnn.multiply(t, 0.0))
    # 2 b - op(-x), written (-op(-x)) + 2 b so that only the right operand broadcasts (b may be (1, 1, C))
    neg = [ttnn.UnaryWithParam(ttnn.UnaryOpType.NEG)]
    e_neg = ttnn.add(run_on(ttnn.neg), ttnn.multiply(b, 2.0), input_tensor_a_activations=neg)
    if mode == "exact":
        from models.demos.qwen_image_edit_text_encoder._stubs.attention import EXACT_MODE

        acc, n = None, 0
        c = x.shape[-1]
        # "guarded" (the text-encoder ports' switch): exact lanes on the hi limb only -- the lo limb is
        # ~2^-8 of it, so its dense accumulation error is at float32's level -- and no run for a lane that
        # holds no channel (conv_in's 3 input channels leave 5 of the 8 empty); the guard below stays
        guarded = EXACT_MODE == "guarded"
        masks = _lane_masks(device, c)[: min(8, c)] if guarded else _lane_masks(device, c)
        if guarded and shared:
            # the shared copy's hi limb is formed once for all the lanes, its lo limb once after them
            his = [hi(t) for t in shared]
            if weight is not None and run_w is not None and len(his) == 1:
                h_rm = back(his[0], x)
                for _m, wm in zip(masks, _weight_lane_masks(device, weight.shape[0])):
                    w_r = ttnn.multiply(weight, wm)
                    y = _flat(run_w(h_rm, w_r))
                    ttnn.deallocate(w_r)
                    acc, n = (y if acc is None else ttnn.add(acc, y)), n + 1
                ttnn.deallocate(h_rm)
            else:
                for m in masks:
                    y = run_on(lambda t, m=m: ttnn.multiply(t, m), his)
                    acc, n = (y if acc is None else ttnn.add(acc, y)), n + 1
            los = [ttnn.subtract(t, h) for t, h in zip(shared, his)]
            free(his)
            free(shared, [x, extra])
            shared = None
            acc, n = ttnn.add(acc, run_on(lambda t: t, los)), n + 1
            free(los)
        else:
            for m in masks:
                for limb in (hi,) if guarded else (hi, lo):
                    y = run_on(lambda t, m=m, limb=limb: ttnn.multiply(limb(t), m))
                    acc, n = (y if acc is None else ttnn.add(acc, y)), n + 1
            if guarded:
                acc, n = ttnn.add(acc, run_on(lo)), n + 1
        bn = ttnn.multiply(b, float(n - 1))  # each run added the bias once
        tol = ttnn.add(ttnn.multiply(ttnn.abs(e_neg), 3e-3), 1e-3)
        res = None if VOTE_TAIL is None else VOTE_TAIL(acc, bn, f0, e_neg, tol)
        if res is None:
            e_x = ttnn.subtract(acc, bn)
            med = ttnn.maximum(ttnn.minimum(e_x, f0), ttnn.minimum(ttnn.maximum(e_x, f0), e_neg))
            res = ttnn.where(ttnn.le(ttnn.abs(ttnn.subtract(e_x, e_neg)), tol), e_x, med)
    else:
        e_split = ttnn.subtract(ttnn.add(run_on(hi), run_on(lo)), b)
        res = ttnn.maximum(ttnn.minimum(f0, e_neg), ttnn.minimum(ttnn.maximum(f0, e_neg), e_split))
    if shared:
        free(shared, [x, extra])
    return ttnn.reshape(ttnn.to_layout(res, layout), shape)


def precise_forward(module, mode, device):
    """Route a float32 affine tt_dit module (conv / linear, channels last) through precise_affine."""
    if getattr(module, "dtype", ttnn.float32) != ttnn.float32 or not mode:
        return False
    orig = type(module).forward
    # the callers' matmul config for these linears runs without float32 dest accumulation (16-bit
    # partials: the 1x1 conv_shortcut measured 4.7e-4 relative even on exact-lane inputs)
    cfg = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    from models.tt_dit.layers.linear import Linear

    if isinstance(module, Linear):
        # minimal_matmul keeps a 4.7e-4 relative error on this 1x1 shortcut even with exact-lane float32
        # inputs, a float32 dest and a float32 output (measured); the exact-lane split linear of the
        # text-encoder ports is exact to float32 there, so the linear runs as that.
        from models.demos.qwen_image_edit_text_encoder._stubs.attention import split_linear

        w = ttnn.typecast(module.weight.data, ttnn.bfloat16)  # the checkpoint is bf16: exact
        bias = None if module.bias is None else ttnn.typecast(module.bias.data, ttnn.float32)

        def forward(x, *args, **kwargs):
            s = list(x.shape)
            lead = 1
            for d in s[:-2]:
                lead *= d
            x3 = ttnn.reshape(x, (lead, s[-2], s[-1]))
            x3 = x3 if x3.dtype == ttnn.float32 else ttnn.typecast(x3, ttnn.float32)
            y = split_linear(x3, w, bias=bias, compute_kernel_config=cfg, exact=True, limbs=2)
            return ttnn.reshape(y, s[:-1] + [y.shape[-1]])

        module.forward = forward
        return True

    def forward(x, *args, **kwargs):
        if "compute_kernel_config" in kwargs:
            kwargs["compute_kernel_config"] = cfg
        return precise_affine(lambda t, _e: orig(module, t, *args, **kwargs), x, mode, device)

    module.forward = forward
    return True

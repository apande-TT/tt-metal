# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""A hand-written Metalium SwiGLU product (silu(gate) * up) driven through ttnn.generic_op.

WHY, AND WHAT IS ACTUALLY CHEAPER -- see _kernels/silu_mul_compute.cpp.  Short version: ttnn's
binary_ng already runs this in one op and it is COMPUTE bound (98.2 us/call bare against 307.2
with the silu on operand A's unpack, on 86.9 MB of interleaved L1), the exact sigmoid has to stay
because BOTH of the SFPU's LUT sigmoids are too coarse for this model's 0.0076 of PCC margin, and
what is left to remove is the reciprocal's unreachable NaN guard and one of two bf16 roundings.

THE PROGRAM HASH IS THE PART THAT IS NOT OBVIOUS.  generic_op's `compute_program_hash` hashes a
kernel's `runtime_args.size()`, NOT its values (generic_op_device_operation.cpp), so a descriptor
rebuilt per call with fresh `buffer_address()` values would HIT the cached program and run against
the addresses of whatever tensors the first call happened to allocate.  cpp_argmax avoids this by
being resident -- it owns its buffers and builds once -- but the SwiGLU's three tensors are
per-call matmul outputs, so residency is not available here: making them resident would cost two
full 29 MB copies to save 40 us of SFPU work.

`ProgramDescriptor.custom_program_hash` is the way out and it is exposed in Python: setting it to a
hash of (kernel identity, the three buffer addresses) makes a change of address a cache MISS -- so
the program is rebuilt exactly when it must be, and the 32 layers of a prefill that reuse the same
allocator slots (same sizes, same alloc/free order every layer) still hit.

TRACE.  A trace records commands rather than re-running this build, so what is captured is the
addresses the capture pass saw.  Replay reproduces the same allocation sequence, so those are the
right ones -- but that makes this a build-time-invariant, not a runtime one, which is why the
fallback below is unconditional: any shape, dtype, layout or build this cannot serve exactly keeps
ttnn.multiply.
"""
from __future__ import annotations

import os

import ttnn

_KERNEL_DIR = "models/tt_transformers/demo/voxtral_mini_3b_2507/_kernels"
_READER = f"{_KERNEL_DIR}/silu_mul_reader.cpp"
_COMPUTE = f"{_KERNEL_DIR}/silu_mul_compute.cpp"
_WRITER = f"{_KERNEL_DIR}/silu_mul_writer.cpp"

# Tiles per DEST acquire, AND IT HAS TO MATCH binary_ng's OR THE FRAMING IS TWICE AS EXPENSIVE.
# binary_ng picks `num_tiles_per_cycle = 8` for every 16-bit type (binary_ng_program_factory.cpp),
# so a per-tile pipeline pays its handshake 8x as often -- which is most of why the ttl attempt at
# this op measured 33% SLOWER than the stock one.
#
# BUT EIGHT IS NOT REACHABLE HERE AND FOUR BEATS IT ANYWAY -- MEASURED, 2026-09-08.  This kernel
# holds BOTH operands in DEST (the multiply is an SFPU binary on two DEST indices) where binary_ng
# holds only the output there (its multiply is an FPU op reading SrcA/SrcB), so a block of 8 needs
# 16 DEST tiles, i.e. dst_full_sync_en, i.e. no DEST ping-pong against the packer.  That trade is
# the wrong way round on this op: block 8 + full sync measured 310.4 us/call against block 4 +
# half sync at 279.0 and ttnn's own 307.2.  The SFPU pass is long enough that losing the pack
# overlap costs more than halving the handshake count saves.
_BLOCK = 4

_CB_A, _CB_B, _CB_Y = 0, 1, 16


def enabled() -> bool:
    return os.environ.get("VOXTRAL_CPP_SILU_MUL", "1") == "1"


# A DIAGNOSTIC SINK, OFF UNLESS ASKED FOR.  It exists because the harness surfaces only the last
# ~2 kB of a failing run's output and that window is nanobind teardown noise, so a kernel that
# declines, raises or returns NaN is indistinguishable from a crash from the outside -- this file is
# how the NaN in the reciprocal was found.  Set VOXTRAL_SILU_MUL_LOG=<path> to re-arm it.
# Best-effort and never raises: a kernel that cannot write its own log must still let the model
# finish on the ttnn fallback.
_LOG = os.environ.get("VOXTRAL_SILU_MUL_LOG")
_NOTED: set = set()


def note(msg: str) -> None:
    if _LOG is None or msg in _NOTED:
        return
    _NOTED.add(msg)
    try:
        with open(_LOG, "a") as fh:
            fh.write(msg + "\n")
    except OSError:
        pass


def _tiles(tensor) -> int:
    padded = tensor.padded_shape
    n = 1
    for d in padded:
        n *= int(d)
    return n // (32 * 32)


def _accessor_args(tensor):
    acc = ttnn.TensorAccessorArgs(tensor)
    if list(acc.get_common_runtime_args()):
        # These kernels wire only the compile-time half of an accessor's description, so refuse a
        # layout that splits it rather than build a descriptor that reads garbage.
        raise RuntimeError("silu_mul: tensor needs common runtime accessor args")
    return list(acc.get_compile_time_args())


class SiluMul:
    """Plan the per-core tile split once per (device, tile-count) and build descriptors per call."""

    def __init__(self, device, n_tiles: int):
        grid = device.compute_with_storage_grid_size()
        self.grid_x, self.grid_y = int(grid.x), int(grid.y)
        ncores = self.grid_x * self.grid_y
        blocks = n_tiles // _BLOCK
        if blocks * _BLOCK != n_tiles or blocks == 0:
            raise RuntimeError(f"silu_mul: {n_tiles} tiles is not a whole number of {_BLOCK}-tile blocks")
        # EVERY CORE GETS A WHOLE NUMBER OF BLOCKS, so the kernels need no tail guard: the reader,
        # the compute and the writer all step by exactly `block` and the counts add up by
        # construction.  Cores are used only if there is a block for them.
        self.ncores = min(ncores, blocks)
        base, rem = divmod(blocks, self.ncores)
        self.runs = []
        at = 0
        for c in range(self.ncores):
            nb = base + (1 if c < rem else 0)
            self.runs.append((at, nb * _BLOCK))
            at += nb * _BLOCK
        assert at == n_tiles, (at, n_tiles)
        self.cores = _core_ranges(self.grid_x, self.ncores)
        self.n_tiles = n_tiles

    def _cbs(self, tile_bytes, fmt):
        size = _BLOCK * 2 * tile_bytes
        return [
            ttnn.CBDescriptor(
                total_size=size,
                core_ranges=self.cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=tile_bytes)
                ],
            )
            for idx in (_CB_A, _CB_B, _CB_Y)
        ]

    def descriptor(self, a, b, y):
        a_addr, b_addr, y_addr = a.buffer_address(), b.buffer_address(), y.buffer_address()
        a_acc, b_acc, y_acc = _accessor_args(a), _accessor_args(b), _accessor_args(y)

        rd = ttnn.RuntimeArgs()
        cp = ttnn.RuntimeArgs()
        wr = ttnn.RuntimeArgs()
        for c, (start, count) in enumerate(self.runs):
            y_, x_ = divmod(c, self.grid_x)
            rd[x_][y_] = [a_addr, b_addr, start, count]
            cp[x_][y_] = [count]
            wr[x_][y_] = [y_addr, start, count]

        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=_READER,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.cores,
                compile_time_args=[_BLOCK] + a_acc + b_acc,
                runtime_args=rd,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=_WRITER,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.cores,
                compile_time_args=[_BLOCK] + y_acc,
                runtime_args=wr,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=_COMPUTE,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.cores,
                compile_time_args=[_BLOCK],
                runtime_args=cp,
                # The compute flags are set below, where the reasoning for each one lives.
                config=ttnn.ComputeConfigDescriptor(),
            ),
        ]
        kernels[2].config.math_fidelity = ttnn.MathFidelity.LoFi
        kernels[2].config.fp32_dest_acc_en = False
        kernels[2].config.math_approx_mode = False
        # MATCH binary_ng's PACK, DO NOT IMPROVE ON IT.  binary_ng sets only fp32_dest_acc_en and
        # unpack_to_dest_mode on its compute config -- it never asks for bfp8_pack_precise -- so
        # requesting it here would encode the same values into different bf8_b LSBs than the op this
        # replaces, and every such difference is spent out of a 0.0076 e2e PCC margin.
        kernels[2].config.bfp8_pack_precise = False

        tile_bytes = a.buffer_page_size()
        desc = ttnn.ProgramDescriptor(
            kernels=kernels,
            semaphores=[],
            cbs=self._cbs(tile_bytes, a.dtype),
        )
        # SEE THE MODULE DOCSTRING: generic_op hashes the runtime-arg COUNT, not its values, so the
        # addresses have to enter the hash here or a cache hit would run against stale buffers.
        desc.custom_program_hash = (
            hash(("voxtral_silu_mul", self.n_tiles, self.ncores, _BLOCK, a_addr, b_addr, y_addr)) & 0xFFFFFFFFFFFFFFFF
        )
        return desc


def _core_ranges(grid_x: int, n: int):
    """The first `n` cores of the grid in row-major (x fastest) order."""
    full, rem = divmod(n, grid_x)
    ranges = []
    if full:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, full - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rem - 1, full)))
    return ttnn.CoreRangeSet(ranges)


_PLANS: dict = {}

# THE PRODUCT BUFFER IS REBUILT EVERY LAYER, AND ITS ADDRESS IS WHAT THE CONSUMER PAYS FOR.  This
# kernel owns its output, so a fresh `allocate_tensor_on_device` per call hands the allocator a
# 29 MB request thirty-two times a prefill and takes whatever region is free at that moment -- and
# `down`, which reads it as a 256-tile-deep in0, is the one op in the layer whose cost tracks that
# choice.  The capture shows it: `3328 x 8192 x 3072` runs 417.5 us on the six calls before the
# decode trace region is resident and 553.0 us on the six after, with a BIT-IDENTICAL program
# config, placement and dtype, and the whole 32% lands on NCRISC -- the reader goes 373 -> 537 us
# while the compute simply waits on it.  Its siblings do not move (gate/up 362.0 -> 365.4, o_proj
# 211.9 -> 211.9), and `down` is the only op reading a 29 MB operand, so it is the only one whose
# bank spread the late-region DRAM pressure can spoil.
#
# A CACHED BUFFER FIXES THE ADDRESS instead of re-bidding for one: the first call allocates while
# DRAM is still unfragmented and every later call -- including every later PROMPT -- reuses that
# exact region.  Keyed like _PLANS, on (grid, tiles, dtype, placement) rather than `id(device)`,
# and validated by reading the address back so a stale tensor from a closed device reallocates
# instead of being handed to a kernel.
#
# SAFE BECAUSE THE PRODUCT HAS EXACTLY ONE CONSUMER, ONE OP LATER.  `_swiglu_body` hands it straight
# to the down projection and never holds it across another SwiGLU, so no two live products share a
# key; the decode shape has its own tile count and therefore its own buffer.  And a FIXED address is
# strictly better for trace than a floating one: generic_op's custom_program_hash keys on the three
# buffer addresses, so pinning this one turns a rebuild into a cache hit.
_OUTS: dict = {}


def _cached_out(key, shape, dtype, device, mem_cfg):
    """The resident product buffer for `key`, allocated once."""
    out = _OUTS.get(key)
    if out is not None:
        try:
            out.buffer_address()
            return out
        except (AttributeError, RuntimeError, TypeError, ValueError):
            _OUTS.pop(key, None)
    out = ttnn.allocate_tensor_on_device(shape, dtype, ttnn.TILE_LAYOUT, device, mem_cfg)
    _OUTS[key] = out
    return out


def serves(gate, up, memory_config) -> bool:
    """Whether this kernel can run THIS call exactly -- otherwise the caller keeps ttnn.multiply."""
    if not enabled():
        return False
    try:
        if gate.dtype != up.dtype or gate.layout != ttnn.TILE_LAYOUT or up.layout != ttnn.TILE_LAYOUT:
            note(f"decline dtype/layout {gate.dtype}/{up.dtype} {gate.layout}/{up.layout}")
            return False
        if gate.dtype not in (ttnn.bfloat8_b, ttnn.bfloat16):
            note(f"decline dtype {gate.dtype}")
            return False
        if gate.is_sharded() or up.is_sharded():
            note("decline sharded operand")
            return False
        if list(gate.padded_shape) != list(up.padded_shape):
            note(f"decline shape {list(gate.padded_shape)} vs {list(up.padded_shape)}")
            return False
        if memory_config is not None and memory_config.shard_spec is not None:
            note("decline sharded output config")
            return False
        n = _tiles(gate)
        if n % _BLOCK or n < _BLOCK:
            note(f"decline tiles {n}")
            return False
        note(f"serves shape={list(gate.padded_shape)} tiles={n} dtype={gate.dtype} out={memory_config}")
        return True
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        note(f"decline probe raised {type(exc).__name__}: {exc}")
        return False


def silu_mul(gate, up, memory_config):
    """silu(gate) * up through the custom kernel.  Raises on anything unexpected; caller falls back."""
    device = gate.device()
    n = _tiles(gate)
    # KEYED ON THE GRID, NOT `id(device)`.  A plan is a pure function of (grid, tile count) -- it
    # holds no device handle -- and `id()` is only unique while the object it named is alive, so a
    # closed device whose address a later one reuses would serve a plan built for the wrong grid.
    grid = device.compute_with_storage_grid_size()
    key = (int(grid.x), int(grid.y), n)
    plan = _PLANS.get(key)
    if plan is None:
        plan = SiluMul(device, n)
        _PLANS[key] = plan
    wanted = ttnn.DRAM_MEMORY_CONFIG if memory_config is None else memory_config
    out = _cached_out(
        key + (str(gate.dtype), str(wanted)),
        ttnn.Shape(list(gate.padded_shape)),
        gate.dtype,
        device,
        wanted,
    )
    if _LOG is not None:
        # Three pybind round-trips on the hot path, for a sink that is off by default.
        note(
            f"run tiles={n} ncores={plan.ncores} a={gate.buffer_address()} "
            f"b={up.buffer_address()} y={out.buffer_address()}"
        )
    ttnn.generic_op([gate, up, out], plan.descriptor(gate, up, out))
    note(f"ran ok tiles={n}")
    _selfcheck(gate, up, out)
    shp = list(gate.shape)
    if list(out.shape) != shp:
        out = ttnn.reshape(out, tuple(shp))
    return out


# ---------------------------------------------------------------- diagnostics
_SELFCHECKED = [False]


def _selfcheck(gate, up, out):
    """ONCE per process, compare this kernel against the ttnn call it replaces, on device.

    Exists because the harness surfaces only the last ~2 kB of a failing run's output and that
    window is nanobind teardown noise, so a numerically wrong kernel is indistinguishable from a
    crash from the outside.  Pure ttnn plus two scalar readbacks, gated OFF by default so it never
    reaches a perf measurement.
    """
    if _SELFCHECKED[0] or os.environ.get("VOXTRAL_SILU_MUL_SELFCHECK", "0") != "1":
        return
    _SELFCHECKED[0] = True
    try:
        act = [ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)]
        ref = ttnn.multiply(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG, input_tensor_a_activations=act)
        d = ttnn.abs(ttnn.subtract(out, ref))
        stats = {
            "maxdiff": ttnn.max(d),
            "maxref": ttnn.max(ttnn.abs(ref)),
            "meanref": ttnn.mean(ttnn.abs(ref)),
            "meanout": ttnn.mean(ttnn.abs(out)),
            "meandiff": ttnn.mean(d),
        }
        vals = {k: ttnn.to_torch(v).flatten()[0] for k, v in stats.items()}
        note("selfcheck " + " ".join(f"{k}={float(v):.6g}" for k, v in vals.items()))
    except Exception as exc:  # noqa: BLE001 - a diagnostic must never be the reason a run fails
        note(f"selfcheck raised {type(exc).__name__}: {exc}")

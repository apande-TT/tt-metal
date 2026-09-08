# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The tt-lang rung on the SwiGLU's `silu(gate) * up`, authored, RUN, measured and NOT wired in.

THE VERDICT FIRST, because that is why nothing imports this.  Measured 2026-09-08 with this kernel
substituted for `_multiply_with_silu`'s ttnn call at the prefill height, on the full 11x10 grid:

    ttnn.multiply, silu on operand A   307.2 us/call    e2e PCC 0.9576212
    this kernel                        409.8 us/call    e2e PCC 0.9502707

33% slower, and the margin over the 0.95 gate drops from 0.0076 to 0.00027.  Reverted.  The file
is kept because the rung will be asked for again and this is what the answer cost: the kernel
lowers, it runs on device, it is correct, and it is simply slower than the op it replaces.

WHY IT LOSES, FROM THE EMITTED C++.  ttl builds a ONE-TILE-AT-A-TIME pipeline, so every tile pays
`init_sfpu`, `tile_regs_acquire/commit/wait/release`, a `cb_wait_front`/`cb_pop_front` pair per
operand and its own `noc_async_write_barrier`.  binary_ng amortises all of that over blocks.  The
compute body is otherwise IDENTICAL -- `silu_tile()` then `mul_binary_tile()` -- which is the
independent confirmation of the floor argument in `_multiply_with_silu`: a hand-written kernel has
the same non-approximate sigmoid available to it and there is no third option to reach for.  The
PCC loss is the PACK, not the math: binary_ng sets `bfp8_pack_precise` for a bf8_b output and a
ttl operation has no way to ask for it.

AND ONE MEASUREMENT DISAGREEMENT, LEFT ON RECORD RATHER THAN EXPLAINED AWAY.  The tracy capture
puts this at +102 us/call, i.e. +3.1 ms over 30 layers, but the trace+1cq prefill stage read
106.32 ms against a 106.37-106.47 ms HEAD -- unchanged -- and the kernel provably ran in that run
too.  Either the 409.8 includes tracy marker cost that a three-kernel generic_op pays and a
binary_ng does not, or prefill's stage time is insensitive to op-level deltas of this size.

AND TWO ttl 1.0.1 LIMITS THAT CLOSE THE tt-lang RUNG ON OTHER OPS OUTRIGHT, verified while
looking for a second target:

* **ROW_MAJOR TENSORS ARE NOT SUPPORTED AT ALL.**  `ttl.operation` takes a `tiled=` parameter, but
  passing `tiled=False` with row-major operands raises "Only tiled tensors supported".  That closes
  the rung on everything in the sampling tail -- the lm_head vocab `concat`, the untilized pieces
  it joins, and the argmax itself all work on ROW_MAJOR bf16 because the scan reads sticks, not
  tiles.  A C++ generic_op kernel has no such limit (tt/cpp_argmax.py already reads a ROW_MAJOR
  sharded tensor), so for those ops the cpp rung is the FIRST reachable one, which is the escape
  GUIDELINES/12 names as "tt-lang provably cannot express the op".
* **`num_outs` MUST BE 1.**  The decorator rejects anything else, so no multi-output op is
  expressible.  That closes the rung on `nlp_create_qkv_heads`, which returns q, k and v from one
  launch; splitting it into three ttl operations would triple the launch count of an op that is
  already at the ~757 GB/s L1 datamove floor.

FOUR ttl 1.0.1 GOTCHAS, which cost most of the authoring time:

1. `grid=` IS (x, y), NOT (rows, cols).  GUIDELINES/11 writes it `grid=(R, C)`; ttl compares the
   tuple against the device grid as (cols, rows), so (10, 11) on an 11x10 Blackhole raises
   "Kernel grid (10, 11) exceeds device compute grid (11, 10)".  TTLANG_COMPILE_ONLY=1 does NOT
   catch it -- that check needs a real device -- so a kernel that lowers cleanly still refuses at
   its first call.
2. THE DOCUMENTED MULTI-CORE ROUTE IS BROKEN.  `indexing_maps=` + `iterator_types=` is unusable:
   the validator computes `num_dims = list(tuple(inspect.signature(indexing_map).parameters))` --
   a list of parameter NAMES -- and compares it to `len(iterator_types)`, so `['i'] != 1` raises
   for every well-formed map.  `indexing_maps` alone is accepted, but then `copy()` rejects an
   unsubscripted tensor.  What works is the explicit form below: declare the full grid and read
   the core's own coordinate with `ttl.node(dims=2)`.
3. `ttl.node(dims=2)` HAS TO BE CALLED INSIDE A THREAD BODY.  Hoisting it (or any arithmetic on
   it) to the operation body raises "An MLIR function requires a Location", because there is no
   insertion context out there.  `ttl.grid_size(dims=2)` is worse -- it is not `@syntax`-decorated,
   so its result is not a traced value at all; use plain Python constants for the grid extents.
4. THE OPERAND RANK IS THE MODEL'S, NOT THE KERNEL'S.  A `[r, c]` tile subscript needs rank 2, and
   this model's prefill activation is `[B, C, width]` -- a leading BATCH, not a unit dim.  Refusing
   it silently sent every prefill call back to ttnn and the substitution measured as a no-op.  Fold
   instead: when the row dim is tile-aligned, `[B, C, W]` and `[B*C, W]` have the identical
   physical tile order, so it is a metadata view.  See `_rank2`.

A ttl operation lowers to `ttnn.generic_op`, and `ttl/kernel_runner.py` REBUILDS the whole
ProgramDescriptor on every call (fresh `buffer_address()` in `common_runtime_args`).  That means
addresses are never stale -- unlike a hand-built descriptor -- but it also means the build is host
work on every call, and under trace only the captured addresses are replayed.  It happened to be
safe here because this model's allocator is deterministic per step; a call site whose operands move
between steps would need the resident-buffer treatment in tt/cpp_argmax.py.
"""
from __future__ import annotations

import os

import ttnn

try:
    import ttl

    _HAVE_TTL = True
except ImportError:  # pragma: no cover - depends on the environment
    _HAVE_TTL = False

TILE = 32
# Grid extents as plain constants -- see gotcha 3.
_GX, _GY = 11, 10


def enabled() -> bool:
    """Off unless explicitly asked for: this kernel is 33% slower than the op it replaces."""
    return _HAVE_TTL and os.environ.get("VOXTRAL_TTL_SILU_MUL") == "1"


if _HAVE_TTL:

    @ttl.operation(grid=(_GX, _GY))
    def silu_mul(a: ttnn.Tensor, b: ttnn.Tensor, y: ttnn.Tensor) -> None:
        """y = silu(a) * b, one tile at a time, tiles dealt out flat across the grid."""
        rt = a.shape[-2] // TILE
        ct = a.shape[-1] // TILE
        nt = rt * ct
        per = (nt + _GX * _GY - 1) // (_GX * _GY)
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
        y_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)

        @ttl.datamovement()
        def read():
            x, cy = ttl.node(dims=2)
            start = (cy * _GX + x) * per
            for i in range(per):
                t = start + i
                if t < nt:
                    r = t // ct
                    c = t - r * ct
                    with a_dfb.reserve() as ab, b_dfb.reserve() as bb:
                        ta = ttl.copy(a[r, c], ab)
                        tb = ttl.copy(b[r, c], bb)
                        ta.wait()
                        tb.wait()

        @ttl.compute()
        def compute():
            x, cy = ttl.node(dims=2)
            start = (cy * _GX + x) * per
            for i in range(per):
                if start + i < nt:
                    with a_dfb.wait() as ab, b_dfb.wait() as bb:
                        with y_dfb.reserve() as yb:
                            yb.store(ttl.math.silu(ab) * bb)

        @ttl.datamovement()
        def write():
            x, cy = ttl.node(dims=2)
            start = (cy * _GX + x) * per
            for i in range(per):
                t = start + i
                if t < nt:
                    r = t // ct
                    c = t - r * ct
                    with y_dfb.wait() as yb:
                        ttl.copy(yb, y[r, c]).wait()


def _rank2(t):
    """Fold every leading dim into the row dim so the kernel's [r, c] subscript is well formed.

    See gotcha 4: the fold is exact and free while the row dim is tile-aligned, because [B, C, W]
    and [B*C, W] have the identical physical tile order.
    """
    dims = [int(d) for d in t.shape]
    if len(dims) == 2:
        return t
    if dims[-2] % TILE:
        raise ValueError("silu_mul needs a tile-aligned row dim to fold a leading batch")
    rows = 1
    for d in dims[:-1]:
        rows *= d
    return ttnn.reshape(t, (rows, dims[-1]))


def apply(gate, up, memory_config):
    """Run the kernel, or raise so the caller keeps ttnn's own multiply.

    The I/O contract is ttnn's: same shape, same dtype, and the output placement the caller asked
    for.  Every precondition is CHECKED rather than assumed -- the fallback is the op this replaces,
    and a silent mismatch would be worse than not trying.
    """
    if not enabled():
        raise RuntimeError("ttl silu_mul disabled")
    if gate.dtype != up.dtype or tuple(gate.shape) != tuple(up.shape):
        raise ValueError("silu_mul needs matching operands")
    if gate.layout != ttnn.TILE_LAYOUT or up.layout != ttnn.TILE_LAYOUT:
        raise ValueError("silu_mul needs TILE operands")
    if gate.memory_config().is_sharded() or up.memory_config().is_sharded():
        # The accessor args these three threads wire are the compile-time half only; a sharded
        # operand also needs the runtime half.
        raise ValueError("silu_mul needs interleaved operands")
    a = _rank2(gate)
    b = _rank2(up)
    out_cfg = memory_config if memory_config is not None else a.memory_config()
    if out_cfg.is_sharded():
        raise ValueError("silu_mul needs an interleaved output")
    y = ttnn.allocate_tensor_on_device(a.shape, a.dtype, ttnn.TILE_LAYOUT, a.device(), out_cfg)
    silu_mul(a, b, y)
    dims = [int(d) for d in gate.shape]
    return y if len(dims) == 2 else ttnn.reshape(y, tuple(dims))

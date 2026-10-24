# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Prefill norm scale y = bf16(x * r) on the residual's height-sharded L1 copy, as ONE program (ttnn.generic_op,
kernels/norm_scale_*.cpp): ttnn.multiply(x_l1, r, dtype=bfloat16) for an fp32 x and an fp32 per-row factor r [..., S,
1] -- binary_ng's SFPU column-broadcast program call for call (r's tile column-broadcast to DEST and packed, x and
that tile unpacked straight to fp32 DEST, mul_binary_tile, the fp32 -> bf16 typecast_tile post activation, pack) --
with CHUNK x tiles a DEST acquire instead of one, so the per-tile copies of the broadcast tile, srcA reconfigs and
inits are paid once per chunk. Each core reads its own factor tiles straight from r; the output lands in x's shards."""

from __future__ import annotations

from pathlib import Path

import ttnn

_KDIR = Path(__file__).resolve().parent / "kernels"
CB_X, CB_R, CB_OUT, CB_RB = 0, 1, 2, 6
CHUNK = 3  # x tiles a DEST acquire (fp32 half-sync DEST holds 4: CHUNK + the broadcast factor)


def _cores(grid):
    """The cores of a ROW_MAJOR shard grid in shard order."""
    out = []
    for rng in grid.ranges():
        for y in range(rng.start.y, rng.end.y + 1):
            for x in range(rng.start.x, rng.end.x + 1):
                out.append((x, y))
    return out


def supports(x_l1, r) -> bool:
    try:
        mc = x_l1.memory_config()
        spec = mc.shard_spec
        return (
            x_l1.dtype == ttnn.float32
            and r.dtype == ttnn.float32
            and x_l1.layout == r.layout == ttnn.TILE_LAYOUT
            and mc.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            and spec is not None
            and spec.orientation == ttnn.ShardOrientation.ROW_MAJOR
            and int(spec.shape[1]) == int(x_l1.shape[-1])
            and int(spec.shape[0]) % ttnn.TILE_SIZE == 0
            and not r.is_sharded()
            and int(r.shape[-1]) == 1
            and int(r.shape[-2]) == int(x_l1.shape[-2])
        )
    except Exception:  # noqa: BLE001
        return False


def scale(x_l1, r):
    """bf16(x_l1 * r) in x_l1's height-sharded layout; x_l1 fp32 [..., S, W], r fp32 [..., S, 1] interleaved."""
    device = x_l1.device()
    mc = x_l1.memory_config()
    spec = mc.shard_spec
    rows, wt = int(spec.shape[0]) // ttnn.TILE_SIZE, int(spec.shape[1]) // ttnn.TILE_SIZE
    out = ttnn.allocate_tensor_on_device(x_l1.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
    total_rows = 1
    for i in range(len(x_l1.shape) - 1):  # (ttnn.Shape does not slice)
        total_rows *= int(x_l1.shape[i])
    ncores = total_rows // ttnn.TILE_SIZE // rows
    cores = _cores(spec.grid)[:ncores]
    crs = spec.grid
    if len(cores) != len(_cores(spec.grid)):  # (a grid with more cores than shards: only the shard holders run)
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in cores])

    def cb(index, pages, page, dtype):
        fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)
        return ttnn.CBDescriptor(total_size=pages * page, core_ranges=crs, format_descriptors=[fmt])

    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_X, x_l1),
        cb(CB_R, rows, 4096, ttnn.float32),
        cb(CB_RB, 2, 4096, ttnn.float32),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, out),
    ]
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for c in (CB_X, CB_R, CB_RB):
        modes[c] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    acc = list(ttnn.TensorAccessorArgs(r).get_compile_time_args())
    ra = r.buffer_address()
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / "norm_scale_reader.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs,
            compile_time_args=[rows, wt] + acc,
            runtime_args=[(c, [ra, k * rows]) for k, c in enumerate(cores)],
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / "norm_scale_compute.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs,
            compile_time_args=[rows, wt, CHUNK],
            runtime_args=[(c, []) for c in cores],
            config=cfg,
        ),
    ]
    ttnn.generic_op([x_l1, r, out], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
    return out

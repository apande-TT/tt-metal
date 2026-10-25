# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The prefill final norm's gamma multiply bf16(y * gamma) as ONE program (ttnn.generic_op, kernels/gamma_mul_*.cpp):
ttnn.multiply(y, gamma) for a bf16 y [..., S, W] and a bf16 gamma row [1, W] runs binary_ng's SFPU row-broadcast
program, which per output tile re-reads gamma's tile, row-broadcasts it into a CB (unary_bcast<ROW>) and multiplies
(copy_tile both into a 16-bit DEST, mul_binary_tile: fp32 product, RNE to bf16, zero inputs give +0), one tile a DEST
acquire. Here each core owns CHUNK columns over a band of rows, reads and broadcasts its CHUNK gamma tiles ONCE and multiplies
a row's CHUNK tiles a DEST acquire --
the same unpacks, broadcast and SFPU multiply (fp32 DEST off, approx off): bit-identical."""

from __future__ import annotations

from pathlib import Path

import ttnn

_KDIR = Path(__file__).resolve().parent / "kernels"
CB_Y, CB_G, CB_OUT, CB_GB = 0, 1, 2, 6
CHUNK = 4  # columns a core (= output tiles a DEST acquire; 16-bit half-sync DEST: 8 slots = 4 x (y, gamma))


def supports(y, gamma) -> bool:
    try:
        return (
            y.dtype == ttnn.bfloat16
            and gamma.dtype == ttnn.bfloat16
            and y.layout == gamma.layout == ttnn.TILE_LAYOUT
            and not y.is_sharded()
            and not gamma.is_sharded()
            and int(gamma.shape[-1]) == int(y.shape[-1])
            and int(gamma.shape[-2]) == 1
            and int(y.shape[-1]) % ttnn.TILE_SIZE == 0
            and int(y.shape[-2]) % ttnn.TILE_SIZE == 0
        )
    except Exception:  # noqa: BLE001
        return False


def scale(y, gamma, memory_config):
    """bf16(y * gamma) (gamma's one row broadcast down every row of y), interleaved in memory_config. Each core takes
    a GW-column group (WT / GW groups) over a band of tile rows, so it broadcasts only its GW gamma tiles."""
    device = y.device()
    rows = 1
    for i in range(len(y.shape) - 1):  # (ttnn.Shape does not slice)
        rows *= int(y.shape[i])
    wt, ht = int(y.shape[-1]) // ttnn.TILE_SIZE, rows // ttnn.TILE_SIZE
    gw = CHUNK if wt % CHUNK == 0 else 1
    ng = wt // gw
    grid = device.compute_with_storage_grid_size()
    nr = max(1, min(int(grid.x) * int(grid.y) // ng, ht))
    ncores = ng * nr
    cores = ttnn.num_cores_to_corerangeset(ncores, grid, row_wise=True)
    out = ttnn.allocate_tensor_on_device(y.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, memory_config)

    def cb(index, pages):
        fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=2048)
        return ttnn.CBDescriptor(total_size=pages * 2048, core_ranges=cores, format_descriptors=[fmt])

    cbs = [cb(CB_Y, 2 * gw), cb(CB_G, 2), cb(CB_OUT, 2 * gw), cb(CB_GB, gw)]
    ya, ga, oa = y.buffer_address(), gamma.buffer_address(), out.buffer_address()
    acc_y = list(ttnn.TensorAccessorArgs(y).get_compile_time_args())
    acc_g = list(ttnn.TensorAccessorArgs(gamma).get_compile_time_args())
    acc_o = list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    base, extra = divmod(ht, nr)
    rr, rw, rc = [], [], []
    c = 0
    th0 = 0
    for r in range(nr):
        nrows = base + (1 if r < extra else 0)
        for g in range(ng):
            cy, cx = divmod(c, int(grid.x))
            rr.append(((cx, cy), [ya, ga, th0, nrows, g * gw]))
            rw.append(((cx, cy), [oa, th0, nrows, g * gw]))
            rc.append(((cx, cy), [nrows]))
            c += 1
        th0 += nrows
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    cfg.dst_full_sync_en = False
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / "gamma_mul_reader.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[wt, gw] + acc_y + acc_g,
            runtime_args=rr,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / "gamma_mul_writer.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[wt, gw] + acc_o,
            runtime_args=rw,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(_KDIR / "gamma_mul_compute.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[gw],
            runtime_args=rc,
            config=cfg,
        ),
    ]
    ttnn.generic_op([y, gamma, out], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
    return out

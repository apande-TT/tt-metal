# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The C++ Metalium rung on the prefill k/v join, through ttnn.generic_op (kernels in tt/cpp_kv_join_kernels).

The compact prefill tail attends through `[1, H, P + T, D]` k and v: the shared prefix k/v in front of the
tail's own. The stubs built each with a `ttnn.concat` -- two dispatches a layer for ~3.5 us of copying each.
Here ONE generic_op joins both: unit (operand, head, joined tile row) reads the row's tiles from the prefix or
the tail and writes them to the joined tensor, tile for tile (the same bits).

On unless VOXTRAL_CPP_KV_JOIN=0.
"""

from __future__ import annotations

import os
import pathlib

import ttnn

_DIR = pathlib.Path(__file__).resolve().parent / "cpp_kv_join_kernels"
_READER = str(_DIR / "reader.cpp")
_WRITER = str(_DIR / "writer.cpp")

_TILE = 32
_TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.float32: 4096}


def enabled() -> bool:
    return os.environ.get("VOXTRAL_CPP_KV_JOIN", "1") == "1"


def _accessor_args(tensor):
    acc = ttnn.TensorAccessorArgs(tensor)
    if list(acc.get_common_runtime_args()):
        raise RuntimeError("cpp_kv_join: tensor needs common runtime accessor args")
    return list(acc.get_compile_time_args())


def supports(pk, k, pv, v) -> bool:
    if not enabled():
        return False
    try:
        tensors = (pk, k, pv, v)
        p_shape = [int(s) for s in pk.shape]
        t_shape = [int(s) for s in k.shape]
        return (
            all(t.layout == ttnn.TILE_LAYOUT and t.dtype == pk.dtype and not t.is_sharded() for t in tensors)
            and pk.dtype in _TILE_BYTES
            and [int(s) for s in pv.shape] == p_shape
            and [int(s) for s in v.shape] == t_shape
            and p_shape[0] == 1
            and t_shape[0] == 1
            and p_shape[1] == t_shape[1]
            and p_shape[3] == t_shape[3]
            and p_shape[2] % _TILE == 0
            and t_shape[2] % _TILE == 0
            and p_shape[3] % _TILE == 0
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def join(pk, k, pv, v, memory_config=None):
    """`(concat([pk, k], dim=2), concat([pv, v], dim=2))` for `[1, H, P, D]` prefixes and `[1, H, T, D]` tails."""
    device = pk.device()
    _, h, p, d = (int(s) for s in pk.shape)
    t = int(k.shape[2])
    pt, tt, wt = p // _TILE, t // _TILE, d // _TILE
    units = 2 * h * (pt + tt)
    grid = device.compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    ncores = min(units, gx * gy)
    base, extra = divmod(units, ncores)
    cores = ttnn.num_cores_to_corerangeset(ncores, grid, row_wise=True)
    mem = memory_config or ttnn.DRAM_MEMORY_CONFIG
    outs = [
        ttnn.allocate_tensor_on_device(ttnn.Shape([1, h, p + t, d]), pk.dtype, ttnn.TILE_LAYOUT, device, mem)
        for _ in range(2)
    ]
    srcs = [pk.buffer_address(), k.buffer_address(), pv.buffer_address(), v.buffer_address()]
    dsts = [o.buffer_address() for o in outs]
    rr, rw = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    u0 = 0
    for c in range(ncores):
        cy, cx = divmod(c, gx)
        nu = base + (1 if c < extra else 0)
        rr[cx][cy] = srcs + [u0, nu]
        rw[cx][cy] = dsts + [u0, nu]
        u0 += nu
    page = _TILE_BYTES[pk.dtype]
    dims = [h, pt, tt, wt, page]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_READER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=dims + sum((_accessor_args(x) for x in (pk, k, pv, v)), []),
            runtime_args=rr,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=dims + _accessor_args(outs[0]) + _accessor_args(outs[1]),
            runtime_args=rw,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=2 * wt * page,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=pk.dtype, page_size=page)],
        )
    ]
    desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    desc.custom_program_hash = (
        hash(("voxtral_cpp_kv_join", h, pt, tt, wt, str(pk.dtype), *srcs, *dsts, str(mem))) & 0xFFFFFFFFFFFFFFFF
    )
    ttnn.generic_op([pk, k, pv, v] + outs, desc)
    return outs[0], outs[1]

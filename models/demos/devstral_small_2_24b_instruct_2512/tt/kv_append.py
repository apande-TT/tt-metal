# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Decode K / V append into the bfloat8_b caches as ONE program (ttnn.generic_op, kernels/kv_append_*.cpp), on the
cores that hold k's and v's head shards: the same bytes ttnn.experimental.paged_fused_update_cache writes (non-paged,
update_idxs_tensor), without its per-head untilize -> row patch -> tilize of the whole cache tile row.

Each core packs its (32, head_dim) bf16 head rows (row h = kv head h) to bfloat8_b with the packer the stock tilize
uses -- a bfp8 face row (shared exponent + 16 mantissas) depends only on its own 16 values, so input row h's face rows
are the stock op's bytes for cache row u -- and each data-movement RISC writes its heads' rows straight into the
cache tiles: per tile the 64-byte exponent section is read, the row's two exponent bytes spliced in, and the two
16-byte exponent groups plus the two 16-byte mantissa face rows written back (Blackhole's DRAM write alignment is 16
bytes). The stock untilize -> tilize round trip rewrites every other byte of the tile unchanged."""

from __future__ import annotations

from pathlib import Path

import ttnn

_KDIR = Path(__file__).resolve().parent / "kernels"
_BF8_TILE = 1088
CB_K, CB_V, CB_BF8, CB_SCR = 0, 1, 2, 3


def _cores(grid):
    """The cores of a ROW_MAJOR shard grid in shard order."""
    out = []
    for r in grid.ranges():
        for y in range(r.start.y, r.end.y + 1):
            for x in range(r.start.x, r.end.x + 1):
                out.append((x, y))
    return out


def supports(k_cache, k, v_cache, v, cur_pos) -> bool:
    try:
        for cache, t in ((k_cache, k), (v_cache, v)):
            if cache.dtype != ttnn.bfloat8_b or cache.layout != ttnn.TILE_LAYOUT or cache.is_sharded():
                return False
            if cache.memory_config().buffer_type != ttnn.BufferType.DRAM:
                return False
            mc = t.memory_config()
            spec = mc.shard_spec
            if t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT or not t.is_sharded() or spec is None:
                return False
            if mc.memory_layout != ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
                return False
            if spec.orientation != ttnn.ShardOrientation.ROW_MAJOR or int(spec.shape[0]) != ttnn.TILE_SIZE:
                return False
            b, nkv, cap, hd = (int(cache.padded_shape[i]) for i in range(4))
            if int(spec.shape[1]) != hd or spec.grid.num_cores() != b or nkv > ttnn.TILE_SIZE // 2:
                return False
            if cap % ttnn.TILE_SIZE or hd % ttnn.TILE_SIZE:
                return False
        cores_k, cores_v = set(_cores(k.memory_config().shard_spec.grid)), set(_cores(v.memory_config().shard_spec.grid))
        if cores_k & cores_v or tuple(k_cache.padded_shape) != tuple(v_cache.padded_shape):
            return False
        return cur_pos.dtype in (ttnn.int32, ttnn.uint32) and cur_pos.layout == ttnn.ROW_MAJOR_LAYOUT
    except Exception:  # noqa: BLE001
        return False


def append(k_cache, k, v_cache, v, cur_pos, pos_src=None):
    """ttnn.experimental.paged_fused_update_cache(k_cache, k, v_cache, v, update_idxs_tensor=cur_pos), non-paged:
    row cur_pos[b] of every kv head of user b takes k's / v's row h of the core holding user b.
    pos_src: the step's (1, 2B, 1, head_dim) height-sharded cos table from tt/rope_gather.py, whose core B + b holds
    cur_pos[b] in row 1 of its first tile -- read from there (L1) instead of the DRAM cur_pos."""
    b, nkv, cap, hd = (int(k_cache.padded_shape[i]) for i in range(4))
    wt, st = hd // ttnn.TILE_SIZE, cap // ttnn.TILE_SIZE
    idx_bytes = b * 4
    nrisc = min(2, nkv)
    scr_risc = 64 + ((idx_bytes + 63) // 64) * 64 + ((nkv + nrisc - 1) // nrisc) * wt * 64
    ccfg = ttnn.ComputeConfigDescriptor()  # the stock op's default: 16-bit DEST
    ccfg.fp32_dest_acc_en = False
    kernels, cbs = [], []
    for cb_in, cache, t in ((CB_K, k_cache, k), (CB_V, v_cache, v)):
        cores = _cores(t.memory_config().shard_spec.grid)
        crs = t.memory_config().shard_spec.grid
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(cb_in, t))
        for cb, size, fmt in ((CB_BF8, wt * _BF8_TILE, ttnn.bfloat8_b), (CB_SCR, nrisc * scr_risc, ttnn.uint32)):
            f = ttnn.CBFormatDescriptor(buffer_index=cb, data_format=fmt, page_size=_BF8_TILE if cb == CB_BF8 else size)
            cbs.append(ttnn.CBDescriptor(total_size=size, core_ranges=crs, format_descriptors=[f]))
        acc = [a for x in (cache, cur_pos) for a in ttnn.TensorAccessorArgs(x).get_compile_time_args()]
        rt = [(c, [cache.buffer_address(), cur_pos.buffer_address(), i]) for i, c in enumerate(cores)]
        if pos_src is not None:
            pos_cores = _cores(pos_src.memory_config().shard_spec.grid)
            dev = pos_src.device()
            for i, (c, args) in enumerate(rt):
                pc = dev.worker_core_from_logical_core(ttnn.CoreCoord(*pos_cores[b + i]))
                args += [int(pc.x), int(pc.y), pos_src.buffer_address() + 32]
        for r in range(nrisc):
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(_KDIR / "kv_append_dm.cpp"),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=crs,
                    compile_time_args=[wt, r, nrisc, nkv, st, cb_in, CB_BF8, CB_SCR, idx_bytes, r * scr_risc] + acc,
                    runtime_args=rt,
                    defines=[("POS_L1", "1")] if pos_src is not None else [],
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0 if r == 0 else ttnn.DataMovementProcessor.RISCV_1,
                        noc=ttnn.NOC.NOC_0 if r == 0 else ttnn.NOC.NOC_1,
                    ),
                )
            )
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(_KDIR / "kv_append_compute.cpp"),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=crs,
                compile_time_args=[wt, cb_in, CB_BF8],
                runtime_args=[(c, []) for c in cores],
                config=ccfg,
            )
        )
    ttnn.generic_op([k, v, cur_pos, k_cache, v_cache], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))

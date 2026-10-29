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


def rope_ok(q, k, v, cos, sin, trans) -> bool:
    """Whether rope_append can stand in for rotary_embedding_llama_fused_qk(q, k, cos, sin, trans) + append: q and v
    one (32, hd) bf16 shard a user on the same B cores, k on B others, and the tables / pair-swap tile one shard a core
    over exactly q's cores then k's (decode_rope_tables with DECODE_FUSED_QK)."""
    try:
        hd = int(q.padded_shape[-1])
        grid = lambda t: t.memory_config().shard_spec.grid
        for t in (q, k, v, cos, sin, trans):
            spec = t.memory_config().shard_spec
            if t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT or not t.is_sharded() or spec is None:
                return False
            if spec.orientation != ttnn.ShardOrientation.ROW_MAJOR or int(spec.shape[0]) != ttnn.TILE_SIZE:
                return False
            if int(spec.shape[1]) != (ttnn.TILE_SIZE if t is trans else hd):
                return False
        if _cores(grid(q)) != _cores(grid(v)) or set(_cores(grid(q))) & set(_cores(grid(k))):
            return False
        both = _cores(grid(q)) + _cores(grid(k))
        return all(_cores(grid(t)) == both for t in (cos, sin, trans))
    except Exception:  # noqa: BLE001
        return False


def rope_append(q, k, v, cos, sin, trans, k_cache, v_cache, cur_pos, pos_src=None):
    """q, k = ttnn.experimental.rotary_embedding_llama_fused_qk(q, k, cos, sin, trans) (HiFi4, fp32 DEST) followed by
    append(k_cache, k, v_cache, v, cur_pos, pos_src) as ONE program (kernels/rope_append_compute.cpp: the stock
    RoPE's arithmetic, then the append's bfloat8_b pack; the append's data movement unchanged). Returns the rotated q,
    allocated like the stock op's q output; the rotated k only lives in a CB on k's cores."""
    q_out = ttnn.allocate_tensor_on_device(q.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, q.device(), q.memory_config())
    append(k_cache, k, v_cache, v, cur_pos, pos_src=pos_src, rope=(q, q_out, cos, sin, trans))
    return q_out


CB_Q, CB_COS, CB_SIN, CB_TRANS, CB_ROT, CB_COSI, CB_SINI, CB_QOUT, CB_KROT = 4, 5, 6, 7, 8, 9, 10, 11, 12


def append(k_cache, k, v_cache, v, cur_pos, pos_src=None, rope=None):
    """ttnn.experimental.paged_fused_update_cache(k_cache, k, v_cache, v, update_idxs_tensor=cur_pos), non-paged:
    row cur_pos[b] of every kv head of user b takes k's / v's row h of the core holding user b.
    pos_src: the step's (1, 2B, 1, head_dim) height-sharded cos table from tt/rope_gather.py, whose core B + b holds
    cur_pos[b] in row 1 of its first tile -- read from there (L1) instead of the DRAM cur_pos.
    rope = (q, q_out, cos, sin, trans): rope_append's fused RoPE (k is rotated before it is appended)."""
    b, nkv, cap, hd = (int(k_cache.padded_shape[i]) for i in range(4))
    wt, st = hd // ttnn.TILE_SIZE, cap // ttnn.TILE_SIZE
    idx_bytes = b * 4
    nrisc = min(2, nkv)
    scr_risc = 64 + ((idx_bytes + 63) // 64) * 64 + ((nkv + nrisc - 1) // nrisc) * wt * 64
    ccfg = ttnn.ComputeConfigDescriptor()  # the stock op's default: 16-bit DEST
    ccfg.fp32_dest_acc_en = False
    kernels, cbs = [], []
    if rope is not None:
        # the stock RoPE's compute config (the attention's compute_cfg: HiFi4, fp32 DEST)
        ccfg = ttnn.ComputeConfigDescriptor()
        ccfg.math_fidelity = ttnn.MathFidelity.HiFi4
        ccfg.fp32_dest_acc_en = True
        q, q_out, cos, sin, trans = rope
        for cb, t in ((CB_Q, q), (CB_QOUT, q_out), (CB_COS, cos), (CB_SIN, sin), (CB_TRANS, trans)):
            cbs.append(ttnn.cb_descriptor_from_sharded_tensor(cb, t))
        bf16_tile = 2048
        plain = lambda cb, crs: ttnn.CBDescriptor(
            total_size=wt * bf16_tile,
            core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb, data_format=ttnn.bfloat16, page_size=bf16_tile)],
        )
        all_crs = cos.memory_config().shard_spec.grid
        cbs += [plain(cb, all_crs) for cb in (CB_ROT, CB_COSI, CB_SINI)]
        cbs.append(plain(CB_KROT, k.memory_config().shard_spec.grid))
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
        if rope is None:
            src, ct = "kv_append_compute.cpp", [wt, cb_in, CB_BF8]
        elif cb_in == CB_K:  # k's cores: rotate k (pushed by the data movement) into CB_KROT, append that
            src, ct = "rope_append_compute.cpp", [0, wt, CB_K, CB_KROT, CB_COS, CB_SIN, CB_TRANS]
            ct += [CB_ROT, CB_COSI, CB_SINI, CB_KROT, CB_BF8]
        else:  # q's (and v's) cores: append v, rotate q into q_out
            src, ct = "rope_append_compute.cpp", [1, wt, CB_Q, CB_QOUT, CB_COS, CB_SIN, CB_TRANS]
            ct += [CB_ROT, CB_COSI, CB_SINI, CB_V, CB_BF8]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(_KDIR / src),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=crs,
                compile_time_args=ct,
                runtime_args=[(c, []) for c in cores],
                config=ccfg,
            )
        )
    io = [k, v, cur_pos, k_cache, v_cache] + ([] if rope is None else [rope[0]] + list(rope[2:]) + [rope[1]])
    ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.transformer.scaled_dot_product_attention (prefill) as a generic_op program whose compute kernel the model owns.

The program is the stock op's (transformer/sdpa/device/sdpa_program_factory.cpp create_descriptor, standard fp32-DEST
path): the same global Q-chunk schedule, circular buffers, compile-time and runtime arguments, and the stock reader /
writer kernels themselves. Two differences:
  * the compute kernel (kernels/sdpa_compute.cpp) is the stock sdpa.cpp with its softmax exponent's two SFPU passes
    (scale multiply, accurate exp) run as one pass with the exp's constants held in registers -- bit-identical;
  * no K/V chain forwarding between cores sharing a head: every core reads its own K/V (the stock non-causal chains
    only change who fetches the same tiles, not the arithmetic).

Covers what the model runs: no mask (non-causal) or the lightweight causal mask, no chunking / sink / sliding window,
fp32 DEST (see supported()). Call sdpa(q, k, v, is_causal, scale, program_config, compute_kernel_config, memory_config)."""
from __future__ import annotations

import struct
from pathlib import Path

import ttnn

_KDIR = Path(__file__).resolve().parent / "kernels"
_COMPUTE = str(_KDIR / "sdpa_compute.cpp")
_DM_DIR = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/"
_READER = _DM_DIR + "reader_interleaved.cpp"
# cpp attempt (run 8): with CAUSAL_SKIP on one K chunk, kernels/sdpa_reader.cpp (the stock reader + SDPA_CAUSAL_READ_SKIP)
# reads only each Q chunk's K / V rows up to its diagonal -- the tiles the compute skips are never fetched
_CAUSAL_READER = str(_KDIR / "sdpa_reader.cpp")
CAUSAL_READ_SKIP = True
_WRITER = _DM_DIR + "writer_interleaved.cpp"

_TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.float32: 4096, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}
_INACTIVE = 0xFFFFFFFF
_INVALID, _VALID = 0, 1
T = 32

# The SDPA programs ride generic_op (off: the stock op).
ENABLED = True
# A causal prefill whose keys are one K chunk computes each Q chunk's key columns only up to its diagonal (kernels/
# sdpa_compute.cpp SDPA_CAUSAL_SKIP): the skipped tiles are all -inf before the softmax, exp to exact zeros, and leave
# every output bit as it was.
CAUSAL_SKIP = True


def _subblock(h, w, dst_size, max_h=None):
    """sdpa_subblock_utils.hpp determine_largest_subblock_size."""
    for sh, sw in (
        (2, 4), (4, 2), (1, 8), (8, 1), (1, 7), (7, 1), (2, 3), (3, 2), (1, 6), (6, 1),
        (1, 5), (5, 1), (2, 2), (1, 4), (4, 1), (1, 3), (3, 1), (1, 2), (2, 1), (1, 1),
    ):  # fmt: skip
        if sh * sw > dst_size or (max_h is not None and sh > max_h):
            continue
        if h % sh == 0 and w % sw == 0:
            return sh, sw
    raise ValueError("no subblock")


def _granularity(count, max_g):
    """sdpa_subblock_utils.hpp find_valid_granularity."""
    g = min(count, max_g)
    while g > 1 and count % g:
        g -= 1
    return g


def supported(q, k, v, is_causal, attn_mask, compute_kernel_config) -> bool:
    if not ENABLED or attn_mask is not None:
        return False
    if compute_kernel_config is None or not compute_kernel_config.fp32_dest_acc_en:
        return False  # the stock op takes its streaming compute without fp32 DEST
    for t in (q, k, v):
        if t.layout != ttnn.TILE_LAYOUT or t.is_sharded() or t.dtype not in (ttnn.bfloat16, ttnn.bfloat8_b):
            return False
        if len(t.shape) != 4:
            return False
    if q.dtype != ttnn.bfloat16:
        return False
    b, nqh, sq, dh = (int(d) for d in q.shape)
    if dh % T or int(k.shape[3]) != dh or int(v.shape[3]) != dh:
        return False
    return int(k.shape[0]) == b and int(v.shape[0]) == b


def sdpa(q, k, v, is_causal, scale, program_config, compute_kernel_config, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    device = q.device()
    B, NQH, Sq, DH = (int(d) for d in q.shape)
    NKH, NVH, Sk = int(k.shape[1]), int(v.shape[1]), int(k.shape[2])
    q_chunk, k_chunk = int(program_config.q_chunk_size), int(program_config.k_chunk_size)
    padded_Sq = -(-Sq // q_chunk) * q_chunk
    padded_Sk = -(-Sk // k_chunk) * k_chunk
    Skt, DHt = padded_Sk // T, DH // T
    vDHt = DHt
    valid_Sqt, valid_Skt = -(-Sq // T), -(-Sk // T)
    assert is_causal or (padded_Sk == Sk and padded_Sq == Sq), "the padded-mask path is not replicated"
    Sq_chunk_t, Sk_chunk_t = q_chunk // T, k_chunk // T
    q_num_chunks, k_num_chunks = padded_Sq // q_chunk, padded_Sk // k_chunk

    grid = program_config.compute_with_storage_grid_size
    gx, gy = int(grid.x), int(grid.y)
    num_cores = gx * gy
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])

    # global Q scheduling: the flat B * NQH * q_num_chunks space split evenly (pairs when causal + even chunks)
    total_q_chunks = B * NQH * q_num_chunks
    if is_causal and q_num_chunks % 2 == 0:
        pairs = total_q_chunks // 2
        base, extra_cores, extra_per = (pairs // num_cores) * 2, pairs % num_cores, 2
    else:
        base, extra_cores, extra_per = total_q_chunks // num_cores, total_q_chunks % num_cores, 1
    max_chunks = base + (extra_per if extra_cores > 0 else 0)
    q_buffer_factor = 2 if max_chunks > 1 else 1

    fp32_acc = bool(compute_kernel_config.fp32_dest_acc_en)
    dst_size = 4 if fp32_acc else 8
    qk_in0_block_w = DHt
    qk_sh, qk_sw = _subblock(Sq_chunk_t, Sk_chunk_t, dst_size)
    lightweight_mask = bool(is_causal)

    q_tiles = Sq_chunk_t * DHt * q_buffer_factor
    k_tiles = Sk_chunk_t * DHt * 2
    v_tiles = Sk_chunk_t * vDHt * 2
    mask_tiles = 2  # lightweight causal: neginf + diagonal
    qk_tiles = Sq_chunk_t * Sk_chunk_t
    out_im_tiles = Sq_chunk_t * vDHt
    out0_t = Sq_chunk_t * vDHt
    stats_tiles = Sq_chunk_t

    qk_in0_nsb, qk_in1_nsb = Sq_chunk_t // qk_sh, Sk_chunk_t // qk_sw
    out_in0_block_w = Sk_chunk_t
    out_sh, out_sw = _subblock(Sq_chunk_t, vDHt, dst_size)
    out_in0_nsb, out_in1_nsb = Sq_chunk_t // out_sh, vDHt // out_sw

    g = min(DHt, vDHt, dst_size)
    while g > 1 and (DHt % g or vDHt % g):
        g -= 1
    defines = [
        ("STATS_GRANULARITY", str(_granularity(Sq_chunk_t, dst_size))),
        ("SUB_EXP_GRANULARITY", str(_granularity(Sk_chunk_t, dst_size))),
        ("DHT_GRANULARITY", str(g)),
        ("REDUCE_GRANULARITY", str(_granularity(Sq_chunk_t, dst_size // 2))),
        ("EXP_APPROX_MODE", "1" if program_config.exp_approx_mode else "0"),
    ]
    compute_defines = defines + (
        [("SDPA_CAUSAL_SKIP", "1"), ("SDPA_VALID_SQT", str(valid_Sqt))] if is_causal and CAUSAL_SKIP else []
    )
    packed_identity = 0x3F803F80  # two bf16 1.0
    scale_packed = struct.unpack("<I", struct.pack("<f", float(scale)))[0]
    zigzag = int(bool(is_causal))

    # ---- circular buffers (the stock allocation order = indices) ----
    out_dtype = q.dtype
    scalar_dtype = ttnn.float32 if q.dtype == ttnn.float32 else ttnn.bfloat16
    im_dtype, stats_dtype = ttnn.bfloat16, ttnn.bfloat16
    qk_im_dtype = sum_dtype = ttnn.float32 if fp32_acc else ttnn.bfloat16
    cbs = []

    def cb(tiles, dtype):
        idx = len(cbs)
        page = _TILE_BYTES[dtype]
        f = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=page)
        cbs.append(ttnn.CBDescriptor(total_size=tiles * page, core_ranges=core_grid, format_descriptors=[f]))
        return idx

    c_q, c_k, c_v = cb(q_tiles, q.dtype), cb(k_tiles, k.dtype), cb(v_tiles, v.dtype)
    c_mask = cb(mask_tiles, ttnn.bfloat16) if is_causal else _INACTIVE
    c_identity, c_col_identity = cb(1, scalar_dtype), cb(1, scalar_dtype)
    c_qk_im = cb(qk_tiles, qk_im_dtype)
    c_out_im_a, c_out_im_b = cb(out_im_tiles, im_dtype), cb(out_im_tiles, im_dtype)
    c_max_a, c_max_b = cb(stats_tiles, stats_dtype), cb(stats_tiles, stats_dtype)
    c_sum_a, c_sum_b = cb(stats_tiles, sum_dtype), cb(stats_tiles, sum_dtype)
    c_exp_max_diff = cb(stats_tiles, stats_dtype)
    c_out = cb(out0_t, out_dtype)
    # windowed-mode ids fall back to q_in; the rest of the optional CBs stay inactive
    c_win = c_q

    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([B, NQH, Sq, DH]), out_dtype, ttnn.TILE_LAYOUT, device, memory_config
    )
    acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    null_acc = [0, 0]

    sems = []
    sem_ids = (0, 0, 0)
    if not is_causal:
        sem_ids = (0, 1, 2)
        sems = [
            ttnn.SemaphoreDescriptor(id=i, core_ranges=core_grid, initial_value=v)
            for i, v in enumerate((_INVALID, _INVALID, _VALID))
        ]

    reader_ct = [B, NQH, NKH, NVH, Skt, valid_Sqt, valid_Skt, DHt, vDHt, Sq_chunk_t, q_num_chunks, Sk_chunk_t]
    reader_ct += [k_num_chunks, num_cores, int(is_causal), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, qk_sh, 0, 0]
    reader_ct += [*sem_ids, 0, zigzag, 0]  # semaphores, mcast off, zigzag, not windowed
    reader_ct += acc(q) + acc(k) + acc(v) + null_acc * 6  # mask, page table, sink, chunk start, cu window, q offset
    reader_ct += [c_q, c_k, c_v, c_mask, _INACTIVE, _INACTIVE, _INACTIVE, _INACTIVE, c_win, c_win]

    writer_ct = [B, NQH, valid_Sqt, Sk, vDHt, Sq_chunk_t, q_num_chunks, Sk_chunk_t, k_num_chunks, packed_identity]
    writer_ct += [num_cores, int(is_causal), 0, 0, 0, 0, int(lightweight_mask), 0, out_sh, 0, zigzag, 0, 0]
    writer_ct += acc(out) + null_acc + null_acc
    writer_ct += [c_mask, c_identity, c_col_identity, _INACTIVE, c_out, c_win, c_win]

    compute_ct = [Skt, DHt, vDHt, Sq_chunk_t, q_num_chunks, Sk_chunk_t, k_num_chunks, qk_in0_block_w, qk_sw, qk_sh]
    compute_ct += [qk_in0_nsb, qk_in1_nsb, out_in0_block_w, out_sw, out_sh, out_in0_nsb, out_in1_nsb]
    compute_ct += [int(is_causal), 0, 0, 0, scale_packed, 0, 0, 0, valid_Skt, 0, zigzag, 0]
    compute_ct += [c_q, c_k, c_v, c_mask, _INACTIVE, c_identity, c_col_identity, _INACTIVE, _INACTIVE, c_out]
    compute_ct += [c_qk_im, c_out_im_a, c_out_im_b, c_max_a, c_max_b, c_sum_a, c_sum_b, c_exp_max_diff, c_win]
    compute_ct += acc(out)

    q_addr, k_addr, v_addr, o_addr = q.buffer_address(), k.buffer_address(), v.buffer_address(), out.buffer_address()
    rt_r, rt_w, rt_c = [], [], []
    for i in range(num_cores):
        core = (i % gx, i // gx)
        g_start = i * base + min(i, extra_cores) * extra_per
        g_count = base + (extra_per if i < extra_cores else 0)
        if g_start >= total_q_chunks:
            g_start, g_count = total_q_chunks, 0
        elif g_start + g_count > total_q_chunks:
            g_count = total_q_chunks - g_start
        r = [q_addr, k_addr, v_addr, 0, 0, 0, 0, 1, 0, 0]
        if not is_causal:
            r += [0] * 12  # not a K/V chain participant
        r += [g_start, g_count, 0, 0, 0, 0]
        rt_r.append((core, r))
        rt_w.append((core, [o_addr, 1, 0, 0, 0, 0, 0, g_start, g_count, 0, 0, 0, 0]))
        rt_c.append((core, [1, 0, 0, 0, g_start, g_count]))

    ccfg = ttnn.ComputeConfigDescriptor()
    ccfg.math_fidelity = compute_kernel_config.math_fidelity
    ccfg.fp32_dest_acc_en = fp32_acc
    ccfg.dst_full_sync_en = bool(getattr(compute_kernel_config, "dst_full_sync_en", False))
    ccfg.math_approx_mode = bool(getattr(compute_kernel_config, "math_approx_mode", False))

    kern = lambda src, ct, rt, cfg, d: ttnn.KernelDescriptor(
        kernel_source=src,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=ct,
        runtime_args=rt,
        config=cfg,
        defines=d,
    )
    read_skip = is_causal and CAUSAL_SKIP and CAUSAL_READ_SKIP and k_num_chunks == 1
    reader_src = _CAUSAL_READER if read_skip else _READER
    reader_defines = defines + ([("SDPA_CAUSAL_READ_SKIP", "1")] if read_skip else [])
    kernels = [
        kern(reader_src, reader_ct, rt_r, ttnn.ReaderConfigDescriptor(), reader_defines),
        kern(_WRITER, writer_ct, rt_w, ttnn.WriterConfigDescriptor(), defines),
        kern(_COMPUTE, compute_ct, rt_c, ccfg, compute_defines),
    ]
    ttnn.generic_op([q, k, v, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))
    return out

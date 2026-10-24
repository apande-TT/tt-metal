// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The sampler's cross-device top-k (pipeline.py MergeTopK): the K = 32 best of the TP gathered shard lists.
//
// ttnn.topk (single core, insertion cascade) treats the gathered row as unsorted: it transposes every tile and
// runs a full 64-wide bitonic sort (local sort phases 0..5, 21 compare steps) per new tile -- three of them for
// four tiles. But each gathered tile is already a sorted run (VocabTopK's output, largest first), so only the
// merge half of the bitonic algorithm is needed:
//   * rebuild (the 32-run bitonic merge network, 5 steps) turns two descending runs into a descending and an
//     ascending one -- a monotone run is bitonic, so the network sorts it either way;
//   * merge (1 step) keeps the elementwise maxima of the two runs: the top 32 of the 64, as a bitonic run;
//   * rebuild sorts that run, descending for the first pair and ascending for the second, so the two pairs'
//     winners form the next bitonic pair; one more merge + rebuild leaves the top 32, sorted largest first.
// 28 compare steps instead of 63, with no packs between a pair's steps. The ids ride the swaps (index tracking)
// as uint32 tiles exactly as in ttnn.topk; equal values order arbitrarily, as ttnn.topk's unstable network does.
//
// PACKED_IN (pipeline.py MergeTopK over VocabTopK.PACKED_OUT): the gathered input is ONE tensor of VocabTopK's raw
// packed [bf16 value | u16 position'] key tiles, already in column layout (cb in_val, UInt32 pages, raw into DEST).
// Each pair is split in DEST by topk_defuse_tile's own sweep with an INT32 index store -- the value words
// [bf16 | 0] at DST 0 / 1 (the bits a bf16 tile widens to) and the positions [0 | u16] at DST 2 / 3 -- and the
// device's vocab offset (device << ct arg 10, the log2 shard width) is OR-ed into the positions: the same words the
// split tensors' transposes + typecast + offset add put there, so the merge below and its result are unchanged.
//
// SCALE (ct arg 9, the fp32 bits of the sampler's 1 / temperature; 0 = off): the values leave as fp32 scores
// value * scale -- the sampler's typecast + multiply, folded. The transpose widens bf16 exactly into 32-bit DEST
// and the SFPU multiply is the one ttnn.multiply's fp32 scalar path runs (val * param), so the bits are the same.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/bitwise.h"
#include "api/compute/topk.h"
#include "api/compute/transpose.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr std::uint32_t in_val = get_compile_time_arg_val(0);   // gathered values (bf16), one sorted run a tile
    constexpr std::uint32_t in_idx = get_compile_time_arg_val(1);   // their vocab ids (uint32)
    constexpr std::uint32_t mid_val = get_compile_time_arg_val(2);  // the pairs' winners, transposed
    constexpr std::uint32_t mid_idx = get_compile_time_arg_val(3);
    constexpr std::uint32_t fin_val = get_compile_time_arg_val(4);  // the final top K, transposed
    constexpr std::uint32_t fin_idx = get_compile_time_arg_val(5);
    constexpr std::uint32_t out_val = get_compile_time_arg_val(6);  // the final top K, row layout
    constexpr std::uint32_t out_idx = get_compile_time_arg_val(7);
    constexpr std::uint32_t pairs = get_compile_time_arg_val(8);  // gathered tiles / 2 (2: TP = 4)
    constexpr std::uint32_t scale = get_compile_time_arg_val(9);  // fp32 bits of the score scale, 0 = none
    constexpr int K = 32;
    constexpr int logk = 5;
#ifdef PACKED_IN
    constexpr std::uint32_t shard_log2 = get_compile_time_arg_val(10);  // log2 of each device's vocab shard width
#endif

    compute_kernel_hw_startup(in_val, in_idx, out_val);
    ckernel::topk_tile_init();

    DataflowBuffer in_val_dfb(in_val);
    DataflowBuffer in_idx_dfb(in_idx);
    DataflowBuffer mid_val_dfb(mid_val);
    DataflowBuffer mid_idx_dfb(mid_idx);
    DataflowBuffer fin_val_dfb(fin_val);
    DataflowBuffer fin_idx_dfb(fin_idx);
    DataflowBuffer out_val_dfb(out_val);
    DataflowBuffer out_idx_dfb(out_idx);

    // Level 1: each pair of sorted runs -> its top K, sorted in alternating directions across the pairs.
    for (std::uint32_t p = 0; p < pairs; ++p) {
#ifdef PACKED_IN
        in_val_dfb.wait_front(2);
        tile_regs_acquire();
        reconfig_data_format_srca(in_val);
        copy_init(in_val);
        copy_tile(in_val, 0, 0);
        copy_tile(in_val, 1, 1);
        // split: values in place at DST 0 / 1, positions [0 | u16] at DST 2 / 3 (INT32 store, not the pack-half move)
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_topk_defuse, (true, /*largest=*/true), 0, VectorMode::RC_custom, 2));
        bitwise_or_tile_init();
        bitwise_or_tile<DataFormat::Int32>(2, (2 * p) << shard_log2);      // + the vocab offset of the pair's first device
        bitwise_or_tile<DataFormat::Int32>(3, (2 * p + 1) << shard_log2);  // ... and of its second
        ckernel::topk_tile_init();                         // index tracking back on (the unary init reset it)
#else
        in_val_dfb.wait_front(2);
        in_idx_dfb.wait_front(2);
        tile_regs_acquire();
        reconfig_data_format_srca(in_val);
        transpose_init(in_val);
        transpose_tile(in_val, 0, 0);
        transpose_tile(in_val, 1, 1);
        reconfig_data_format_srca(in_idx);
        transpose_init(in_idx);
        transpose_tile(in_idx, 0, 2);
        transpose_tile(in_idx, 1, 3);
#endif
        // The 32-bit transpose records its MOP into replay slots 16..31, over the rebuild's cached windows: make
        // the rebuild record them again (as the LLK's own fuse / defuse sweeps do after clobbering them).
        MATH((ckernel::sfpu::topk_replay_init = 0));
        ckernel::topk_rebuild(0, /*idir=*/false, 0, K, logk, 0);  // run 0 descending, run 1 ascending
        ckernel::topk_merge</*idir=*/false>(0, 0, K);             // DST 0 <- the top K (bitonic)
        ckernel::topk_rebuild(0, (p & 1) != 0, 0, K, logk, 1);
        tile_regs_commit();
        in_val_dfb.pop_front(2);
#ifndef PACKED_IN
        in_idx_dfb.pop_front(2);
#endif

        mid_val_dfb.reserve_back(1);
        mid_idx_dfb.reserve_back(1);
        tile_regs_wait();
        pack_reconfig_data_format(mid_val);
        pack_tile(0, mid_val);
        pack_reconfig_data_format(mid_idx);
        pack_tile(2, mid_idx);
        tile_regs_release();
        mid_val_dfb.push_back(1);
        mid_idx_dfb.push_back(1);
    }

    // Level 2: the two winners (descending, ascending) -> the top K, descending.
    mid_val_dfb.wait_front(2);
    mid_idx_dfb.wait_front(2);
    tile_regs_acquire();
    reconfig_data_format_srca(mid_val);
    copy_init(mid_val);
    copy_tile(mid_val, 0, 0);
    copy_tile(mid_val, 1, 1);
    reconfig_data_format_srca(mid_val, mid_idx);
    copy_init(mid_idx);
    copy_tile(mid_idx, 0, 2);
    copy_tile(mid_idx, 1, 3);
    ckernel::topk_merge</*idir=*/false>(0, 0, K);
    MATH((ckernel::sfpu::topk_replay_init = 0));
    ckernel::topk_rebuild(0, /*idir=*/false, 0, K, logk, 1);
    tile_regs_commit();
    mid_val_dfb.pop_front(2);
    mid_idx_dfb.pop_front(2);

    fin_val_dfb.reserve_back(1);
    fin_idx_dfb.reserve_back(1);
    tile_regs_wait();
    pack_reconfig_data_format(fin_val);
    pack_tile(0, fin_val);
    pack_reconfig_data_format(fin_idx);
    pack_tile(2, fin_idx);
    tile_regs_release();
    fin_val_dfb.push_back(1);
    fin_idx_dfb.push_back(1);

    // Back to row layout.
    fin_val_dfb.wait_front(1);
    fin_idx_dfb.wait_front(1);
    tile_regs_acquire();
    reconfig_data_format_srca(mid_idx, fin_val);
    transpose_init(fin_val);
    transpose_tile(fin_val, 0, 0);
    reconfig_data_format_srca(fin_val, fin_idx);
    transpose_init(fin_idx);
    transpose_tile(fin_idx, 0, 1);
    if constexpr (scale != 0) {
        binop_with_scalar_tile_init();
        mul_unary_tile(0, scale);
    }
    tile_regs_commit();
    fin_val_dfb.pop_front(1);
    fin_idx_dfb.pop_front(1);

    out_val_dfb.reserve_back(1);
    out_idx_dfb.reserve_back(1);
    tile_regs_wait();
    pack_reconfig_data_format(out_val);
    pack_tile(0, out_val);
    pack_reconfig_data_format(out_idx);
    pack_tile(1, out_idx);
    tile_regs_release();
    out_val_dfb.push_back(1);
    out_idx_dfb.push_back(1);
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel of tt/add_stats.py, per tile row: h = x + d on the SFPU in fp32 (into a row-wide private buffer),
// a copy of h to the writer, h^2 on the SFPU, and the row sums of h^2 by the accurate (fp32 SFPU) row reduce --
// the same steps rms_norm_pre_all_gather runs on an fp32 input, without re-reading h from DRAM. The row-sum tile then
// has its columns 1..31 (the reduce's partials) zeroed by an fp32 multiply with a column-0 mask (column 0 * 1.0 is
// exact), so the gathered stats need no masking before they are summed.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    const uint32_t nr = get_arg_val<uint32_t>(0);  // tile rows this core owns
    constexpr uint32_t WT = get_compile_time_arg_val(0);
    constexpr uint32_t BLK = get_compile_time_arg_val(1);

    constexpr uint32_t cb_x = tt::CBIndex::c_0;       // fp32 residual
    constexpr uint32_t cb_d = tt::CBIndex::c_1;       // delta
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;  // unit reduce scaler
    constexpr uint32_t cb_h = tt::CBIndex::c_3;       // h = x + d, the whole row (private)
    constexpr uint32_t cb_x2 = tt::CBIndex::c_4;      // h^2, the whole row (private)
    constexpr uint32_t cb_out = tt::CBIndex::c_16;    // h to the writer
    constexpr uint32_t cb_s = tt::CBIndex::c_17;      // row sums of h^2, columns 1..31 zeroed
    constexpr uint32_t cb_mask = tt::CBIndex::c_5;    // fp32 column-0 mask (the reader's)
    constexpr uint32_t cb_sraw = tt::CBIndex::c_18;   // the reduce's row-sum tile

#ifdef ONE_PASS
    compute_kernel_hw_startup(cb_x, cb_d, cb_out);  // (no cb_h)
#else
    compute_kernel_hw_startup(cb_x, cb_d, cb_h);
#endif

    constexpr auto shape = ckl::IterationShape::tiles(WT).block_size(BLK);
    constexpr auto h_row =
        ckl::input(cb_h, ckl::WaitPolicy::Cumulative, ckl::PopPolicy::None, ckl::InputTileMapping::Block);

    cb_wait_front(cb_mask, 1);
    for (uint32_t r = 0; r < nr; ++r) {
#ifdef ONE_PASS
        // ONE_PASS (cpp try, run 8): h, its copy for the writer and h^2 from ONE DEST pass a P-tile group -- x and
        // d copied in, add_binary_tile, copy_dest_values into the next slot and square_tile there, h packed to
        // cb_out and h^2 to cb_x2 -- instead of h packed to the row buffer cb_h and unpacked twice (copy, square).
        // The same add / square on the same fp32 values: bit-identical.
        constexpr uint32_t P = 2;  // tiles a DEST acquire (fp32 half-sync DEST: 4 slots = h, h^2 for 2 tiles)
        pack_reconfig_data_format(cb_out);  // (cb_out and cb_x2 are both fp32)
        for (uint32_t t = 0; t < WT; t += BLK) {
            cb_wait_front(cb_x, BLK);
            cb_wait_front(cb_d, BLK);
            cb_reserve_back(cb_out, BLK);
            cb_reserve_back(cb_x2, BLK);
            for (uint32_t j = 0; j < BLK; j += P) {
                tile_regs_acquire();
                reconfig_data_format_srca(cb_x);
                copy_init(cb_x);
                for (uint32_t i = 0; i < P; ++i) {
                    copy_tile(cb_x, j + i, 2 * i);
                }
                reconfig_data_format_srca(cb_d);
                copy_init(cb_d);
                for (uint32_t i = 0; i < P; ++i) {
                    copy_tile(cb_d, j + i, 2 * i + 1);
                }
                add_binary_tile_init();
                for (uint32_t i = 0; i < P; ++i) {
                    add_binary_tile(2 * i, 2 * i + 1, 2 * i);
                }
                copy_dest_values_init();
                for (uint32_t i = 0; i < P; ++i) {
                    copy_dest_values<DataFormat::Float32>(2 * i, 2 * i + 1);
                }
                square_tile_init();
                for (uint32_t i = 0; i < P; ++i) {
                    square_tile(2 * i + 1);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < P; ++i) {
                    pack_tile(2 * i, cb_out);
                }
                for (uint32_t i = 0; i < P; ++i) {
                    pack_tile(2 * i + 1, cb_x2);
                }
                tile_regs_release();
            }
            cb_push_back(cb_out, BLK);
            cb_push_back(cb_x2, BLK);
            cb_pop_front(cb_x, BLK);
            cb_pop_front(cb_d, BLK);
        }
#else
        ckl::binary_sfpu<
            ckl::AddBinary<>,
            ckl::input(cb_x, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::InputTileMapping::Block),
            ckl::input(cb_d, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::InputTileMapping::Block),
            ckl::output(cb_h, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(shape);
        ckl::copy<h_row, ckl::output(cb_out, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(shape);
        ckl::unary<ckl::Square<>, h_row, ckl::output(cb_x2, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
            shape);
#endif
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            cb_x2,
            cb_scaler,
            cb_sraw,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
            ReduceFp32Mode::Accurate>(compute_kernel_lib::ReduceInputBlockShape::row(WT));
#ifndef ONE_PASS
        cb_pop_front(cb_h, WT);
#endif

        cb_wait_front(cb_sraw, 1);
        cb_reserve_back(cb_s, 1);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_sraw);
        copy_init(cb_sraw);
        copy_tile(cb_sraw, 0, 0);
        copy_init(cb_mask);
        copy_tile(cb_mask, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_s);
        pack_tile(0, cb_s);
        tile_regs_release();
        cb_push_back(cb_s, 1);
        cb_pop_front(cb_sraw, 1);
    }
    cb_pop_front(cb_mask, 1);
    cb_pop_front(cb_scaler, 1);
}

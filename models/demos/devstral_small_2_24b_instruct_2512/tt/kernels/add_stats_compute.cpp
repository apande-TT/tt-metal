// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel of tt/add_stats.py, per tile row: h = x + d on the SFPU in fp32 (into a row-wide private buffer),
// a copy of h to the writer, h^2 on the SFPU, and the row sums of h^2 by the accurate (fp32 SFPU) row reduce --
// the same steps rms_norm_pre_all_gather runs on an fp32 input, without re-reading h from DRAM.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
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
    constexpr uint32_t cb_s = tt::CBIndex::c_17;      // row sums of h^2

    compute_kernel_hw_startup(cb_x, cb_d, cb_h);

    constexpr auto shape = ckl::IterationShape::tiles(WT).block_size(BLK);
    constexpr auto h_row =
        ckl::input(cb_h, ckl::WaitPolicy::Cumulative, ckl::PopPolicy::None, ckl::InputTileMapping::Block);

    for (uint32_t r = 0; r < nr; ++r) {
        ckl::binary_sfpu<
            ckl::AddBinary<>,
            ckl::input(cb_x, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::InputTileMapping::Block),
            ckl::input(cb_d, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::InputTileMapping::Block),
            ckl::output(cb_h, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(shape);
        ckl::copy<h_row, ckl::output(cb_out, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(shape);
        ckl::unary<ckl::Square<>, h_row, ckl::output(cb_x2, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
            shape);
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            cb_x2,
            cb_scaler,
            cb_s,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
            ReduceFp32Mode::Accurate>(compute_kernel_lib::ReduceInputBlockShape::row(WT));
        cb_pop_front(cb_h, WT);
    }
    cb_pop_front(cb_scaler, 1);
}

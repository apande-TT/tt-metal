// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel of tt/add_stats.py's apply_bf16, per tile row: h = x + d on the SFPU in fp32, rounded to bf16 by the
// SFPU typecast ttnn.typecast runs (so the result matches the fp32 add followed by that cast bit for bit), packed into
// the bf16 output.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    const uint32_t nr = get_arg_val<uint32_t>(0);  // tile rows this core owns
    constexpr uint32_t WT = get_compile_time_arg_val(0);
    constexpr uint32_t BLK = get_compile_time_arg_val(1);

    constexpr uint32_t cb_x = tt::CBIndex::c_0;       // residual (fp32 or bf16)
    constexpr uint32_t cb_d = tt::CBIndex::c_1;       // delta
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;  // the shared reader's reduce scaler (unused here)
    constexpr uint32_t cb_out = tt::CBIndex::c_16;    // bf16(h) to the writer

    compute_kernel_hw_startup(cb_x, cb_d, cb_out);

    constexpr auto shape = ckl::IterationShape::tiles(WT).block_size(BLK);
    for (uint32_t r = 0; r < nr; ++r) {
        ckl::eltwise_chain(
            shape,
            ckl::CopyTile<ckl::input(
                cb_x, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::InputTileMapping::Block)>{},
            ckl::CopyTile<
                ckl::input(
                    cb_d, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::InputTileMapping::Block),
                ckl::Dst::D1>{},
            ckl::AddBinary<>{},
            ckl::Typecast<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>{},
            ckl::PackTile<ckl::output(cb_out, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>{});
    }
    cb_wait_front(cb_scaler, 1);
    cb_pop_front(cb_scaler, 1);
}

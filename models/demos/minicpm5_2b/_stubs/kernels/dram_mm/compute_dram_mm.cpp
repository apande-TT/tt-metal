// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode projection compute (M = one tile row). For each group of group_w output tiles, one
// uninterrupted fp32 DEST accumulation over kt = 0..Kt-1 in order with ttnn's matmul_block calls (rt = 1,
// ct = group_w), then pack to bf16. The 1-D multicast matmul this replaces reloads its fp32 partials straight into
// DEST (UnpackToDestFp32) between K blocks, so its result is the same uninterrupted K-order accumulation.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/compute_kernel_hw_startup.h"

using std::uint32_t;

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t group_w = get_compile_time_arg_val(1);
    constexpr uint32_t num_groups = get_compile_time_arg_val(2);
    constexpr uint32_t k_block = get_compile_time_arg_val(3);
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(4);
    constexpr uint32_t cb_in1 = get_compile_time_arg_val(5);
    constexpr uint32_t cb_out = get_compile_time_arg_val(6);
    constexpr uint32_t block_tiles = k_block * group_w;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);
    matmul_block_init(cb_in0, cb_in1, 0, group_w, 1, k_block);

    for (uint32_t g = 0; g < num_groups; ++g) {
        tile_regs_acquire();
        for (uint32_t kb = 0; kb < Kt; kb += k_block) {
            if (g == 0) {
                cb_wait_front(cb_in0, kb + k_block);  // the activations land in chunks during the first group
            }
            cb_wait_front(cb_in1, block_tiles);
            uint32_t in1_index = 0;
            for (uint32_t k = 0; k < k_block; ++k) {
                matmul_block(cb_in0, cb_in1, kb + k, in1_index, 0, false, group_w, 1, k_block);
                in1_index += group_w;
            }
            cb_pop_front(cb_in1, block_tiles);
        }
        tile_regs_commit();

        cb_reserve_back(cb_out, group_w);
        tile_regs_wait();
        for (uint32_t j = 0; j < group_w; ++j) {
            pack_tile(j, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, group_w);
    }
    cb_pop_front(cb_in0, Kt);
}

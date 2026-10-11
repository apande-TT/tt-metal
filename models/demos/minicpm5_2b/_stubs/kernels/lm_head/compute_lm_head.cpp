// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode LM head compute (M = one tile row).
// Adapted from programming_examples/matmul/matmul_multi_core/kernels/compute/mm.cpp and
// ttnn matmul compute/bmm.cpp. For each group of `group_w` output tiles, one uninterrupted DEST
// accumulation over kt = 0..Kt-1 in order (DST[j] += x[kt] @ w[kt, n+j]), then pack to bf16.
// Activations (cb_in0, Kt tiles) stay resident for the whole kernel.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/compute_kernel_hw_startup.h"

using std::uint32_t;

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t group_w = get_compile_time_arg_val(1);
    constexpr uint32_t k_block = get_compile_time_arg_val(2);
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_in1 = get_compile_time_arg_val(4);
    constexpr uint32_t cb_out = get_compile_time_arg_val(5);
    constexpr uint32_t block_tiles = k_block * group_w;

    const uint32_t num_groups = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);
    matmul_init(cb_in0, cb_in1);

    cb_wait_front(cb_in0, Kt);

    for (uint32_t g = 0; g < num_groups; ++g) {
        tile_regs_acquire();
        for (uint32_t kb = 0; kb < Kt; kb += k_block) {
            cb_wait_front(cb_in1, block_tiles);
            for (uint32_t k = 0; k < k_block; ++k) {
                for (uint32_t j = 0; j < group_w; ++j) {
                    matmul_tiles(cb_in0, cb_in1, kb + k, k * group_w + j, j);
                }
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

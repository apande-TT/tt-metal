// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// rotate_half RoPE (rope_swap.cpp) straight into the bf16 limbs (hi, lo) of its float32 output (split_bf16.cpp), one
// output tile at a time: the RoPE runs rope_swap's exact calls in DEST,
//   a = x * cos                     mul_binary_tile
//   r = -x2 (first half) or x1      negative_tile on the half-swapped tile
//   b = r * sin                     mul_binary_tile
//   y = a + b                       add_binary_tile<NearestEven>
// and y is then split as split_bf16 splits a float32 tensor: packed to hi (cb 16, bf16), to a bf16 scratch (cb 24) and
// to a float32 scratch (cb 25, the float32 tensor rope_swap wrote); lo = bf16(y - float32(hi)) by the SFPU float32
// subtract. The same limbs as rope_swap + split_bf16, without the float32 RoPE output's DRAM round trip and the split
// program. Tiles arrive on cb 0 in fours: x, the half-swapped x, cos, sin.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    constexpr uint32_t Dt = get_compile_time_arg_val(0);
    constexpr uint32_t negate = get_compile_time_arg_val(1);

    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_hi = 16;
    constexpr uint32_t cb_lo = 17;
    constexpr uint32_t cb_tmp = 24;
    constexpr uint32_t cb_y = 25;
    CircularBuffer in(cb_in);
    CircularBuffer hi(cb_hi);
    CircularBuffer lo(cb_lo);
    CircularBuffer tmp(cb_tmp);
    CircularBuffer y(cb_y);

    compute_kernel_hw_startup(cb_in, cb_hi);
    for (uint32_t t = start_tile; t < start_tile + num_tiles; ++t) {
        in.wait_front(4);
        hi.reserve_back(1);
        tmp.reserve_back(1);
        y.reserve_back(1);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_in);
        copy_init(cb_in);
        copy_tile(cb_in, 0, 0);
        copy_tile(cb_in, 2, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        copy_init(cb_in);
        copy_tile(cb_in, 1, 1);
        if (negate && t % Dt < Dt / 2) {
            negative_tile_init();
            negative_tile(1);
        }
        copy_init(cb_in);
        copy_tile(cb_in, 3, 2);
        mul_binary_tile_init();
        mul_binary_tile(1, 2, 1);
        add_binary_tile_init();
        add_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_hi);
        pack_tile(0, cb_hi);
        pack_tile(0, cb_tmp);
        pack_reconfig_data_format(cb_y);
        pack_tile(0, cb_y);
        tile_regs_release();
        hi.push_back(1);
        tmp.push_back(1);
        y.push_back(1);
        in.pop_front(4);

        // lo = bf16(y - float32(hi))
        tmp.wait_front(1);
        y.wait_front(1);
        lo.reserve_back(1);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_y);
        copy_init(cb_y);
        copy_tile(cb_y, 0, 0);
        reconfig_data_format_srca(cb_tmp);
        copy_init(cb_tmp);
        copy_tile(cb_tmp, 0, 1);
        sub_binary_tile_init();
        sub_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_lo);
        pack_tile(0, cb_lo);
        tile_regs_release();
        lo.push_back(1);
        tmp.pop_front(1);
        y.pop_front(1);
    }
}

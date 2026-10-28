// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// rotate_half RoPE x * cos + rotate_half(x) * sin, one output tile at a time, bit-identical to the stock float32
// chain it replaces (slice / slice / neg / concat / multiply / multiply / add; float32 binary_ng runs on the SFPU):
//   a = x * cos                     mul_binary_tile
//   r = -x2 (first half) or x1      negative_tile on the half-swapped tile (concat / slice only move data)
//   b = r * sin                     mul_binary_tile
//   y = a + b                       add_binary_tile<NearestEven>
// Tiles arrive on cb 0 in fours: x, the half-swapped x, cos, sin. a, r and b stay in DEST.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    constexpr uint32_t Dt = get_compile_time_arg_val(0);

    constexpr uint32_t cb_in_id = 0;
    constexpr uint32_t cb_out_id = 16;
    CircularBuffer cb_in(cb_in_id);
    CircularBuffer cb_out(cb_out_id);

    compute_kernel_hw_startup(cb_in_id, cb_out_id);
    for (uint32_t t = start_tile; t < start_tile + num_tiles; ++t) {
        cb_in.wait_front(4);
        tile_regs_acquire();
        copy_init(cb_in_id);
        copy_tile(cb_in_id, 0, 0);
        copy_tile(cb_in_id, 2, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        copy_init(cb_in_id);
        copy_tile(cb_in_id, 1, 1);
        if (t % Dt < Dt / 2) {
            negative_tile_init();
            negative_tile(1);
        }
        copy_init(cb_in_id);
        copy_tile(cb_in_id, 3, 2);
        mul_binary_tile_init();
        mul_binary_tile(1, 2, 1);
        add_binary_tile_init();
        add_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
        tile_regs_commit();
        cb_in.pop_front(4);
        cb_out.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, cb_out_id);
        tile_regs_release();
        cb_out.push_back(1);
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The tail (x * rs) * w of the precise float32 RMSNorm, one output tile at a time, bit-identical to the two
// stock binary_ng multiplies it replaces (both float32: the SFPU path, LLK broadcast):
//   t = x * bcast_col(rs)    unary_bcast<COL> + mul_binary_tile
//   y = t * bcast_row(w)     unary_bcast<ROW> + mul_binary_tile
// t stays in DEST instead of a float32 tensor. rs (column 0 valid) is the row scale of rms_scale.cpp, one tile
// per tile row; w (row 0 valid) the [1, W] float32 weight, tile t % Wt. Tiles arrive in order: x, w per output
// tile, and rs whenever the output tile starts a new tile row.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_x = 0;
constexpr uint32_t cb_rs = 1;
constexpr uint32_t cb_w = 2;
constexpr uint32_t cb_out = 16;

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);

    constexpr uint32_t Wt = get_compile_time_arg_val(0);

    CircularBuffer x(cb_x);
    CircularBuffer rs(cb_rs);
    CircularBuffer w(cb_w);
    CircularBuffer out(cb_out);

    compute_kernel_hw_startup(cb_x, cb_out);
    uint32_t row = 0xFFFFFFFF;
    for (uint32_t t = start_tile; t < start_tile + num_tiles; ++t) {
        if (t / Wt != row) {
            if (row != 0xFFFFFFFF) {
                rs.pop_front(1);
            }
            row = t / Wt;
            rs.wait_front(1);
        }
        x.wait_front(1);
        w.wait_front(1);
        tile_regs_acquire();
        unary_bcast_init<ckernel::BroadcastType::COL>(cb_rs);
        unary_bcast<ckernel::BroadcastType::COL>(cb_rs, 0, 1);
        copy_init(cb_x);
        copy_tile(cb_x, 0, 0);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        unary_bcast_init<ckernel::BroadcastType::ROW>(cb_w);
        unary_bcast<ckernel::BroadcastType::ROW>(cb_w, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        out.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        out.push_back(1);
        x.pop_front(1);
        w.pop_front(1);
    }
    if (row != 0xFFFFFFFF) {
        rs.pop_front(1);
    }
}

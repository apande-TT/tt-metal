// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The exact all_to_all reduce's local sum with the block's gated residual update, one output tile at a time:
//   s = ((p0 + p1) + ...) + p_{n-1} [+ bias]   as sum_blocks.cpp: SFPU float32 add<NearestEven> in source order
//   y = r + g * s                               unary_bcast<ROW>(g) + mul_binary_tile, then add_binary_tile<NE>
// bit-identical to fused_block_sum(_bias) followed by ttnn.multiply(g, s) and ttnn.add(r, ...) on the device's
// slice (float32 binary_ng is SFPU; IEEE multiply / add are commutative). s and g * s stay in DEST.
// Tiles: cb 0 the n_in source (+ bias) tiles in order, cb 1 the residual tile, cb 2 the gate tile (row 0 valid).

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

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t n_in = get_compile_time_arg_val(0);

    constexpr uint32_t cb_src = 0;
    constexpr uint32_t cb_res = 1;
    constexpr uint32_t cb_gate = 2;
    constexpr uint32_t cb_out = 16;
    CircularBuffer src(cb_src);
    CircularBuffer rb(cb_res);
    CircularBuffer gb(cb_gate);
    CircularBuffer out(cb_out);

    compute_kernel_hw_startup(cb_src, cb_out);
    for (uint32_t t = 0; t < num_tiles; ++t) {
        out.reserve_back(1);
        tile_regs_acquire();
        src.wait_front(1);
        copy_init(cb_src);
        copy_tile(cb_src, 0, 0);
        src.pop_front(1);
        for (uint32_t i = 1; i < n_in; ++i) {
            src.wait_front(1);
            copy_init(cb_src);
            copy_tile(cb_src, 0, 1);
            src.pop_front(1);
            add_binary_tile_init();
            add_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
        }
        gb.wait_front(1);
        unary_bcast_init<ckernel::BroadcastType::ROW>(cb_gate);
        unary_bcast<ckernel::BroadcastType::ROW>(cb_gate, 0, 1);
        gb.pop_front(1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        rb.wait_front(1);
        copy_init(cb_res);
        copy_tile(cb_res, 0, 1);
        rb.pop_front(1);
        add_binary_tile_init();
        add_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        out.push_back(1);
    }
}

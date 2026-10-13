// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode score scale + mask compute: out = scores * scale + mask, the two bf16 ops the multi-op
// path runs, each as ttnn runs it for bf16 operands: the product as binary_ng's SFPU multiply with the
// scalar as a bf16 tile (copy both to DEST, mul_binary_tile, pack bf16), the sum as binary_ng's default FPU
// add (binary_tiles_init<true, ELWADD> + add_tiles, bf16 DEST).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/compute_kernel_hw_startup.h"

using std::uint32_t;

void kernel_main() {
    constexpr uint32_t cb_s = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scale = get_compile_time_arg_val(1);
    constexpr uint32_t cb_mask = get_compile_time_arg_val(2);
    constexpr uint32_t cb_t = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);

    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_s, cb_out);
    cb_wait_front(cb_scale, 1);

    for (uint32_t t = 0; t < num_tiles; ++t) {
        // t = scores * scale
        cb_wait_front(cb_s, 1);
        cb_reserve_back(cb_t, 1);
        mul_binary_tile_init();
        tile_regs_acquire();
        copy_init(cb_s);
        copy_tile(cb_s, 0, 0);
        copy_init(cb_scale);
        copy_tile(cb_scale, 0, 1);
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_t);
        tile_regs_release();
        cb_push_back(cb_t, 1);
        cb_pop_front(cb_s, 1);

        // out = t + mask
        cb_wait_front(cb_t, 1);
        cb_wait_front(cb_mask, 1);
        cb_reserve_back(cb_out, 1);
        binary_tiles_init<true, EltwiseBinaryType::ELWADD>(cb_t, cb_mask);
        tile_regs_acquire();
        add_tiles(cb_t, cb_mask, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_t, 1);
        cb_pop_front(cb_mask, 1);
    }
    cb_pop_front(cb_scale, 1);
}

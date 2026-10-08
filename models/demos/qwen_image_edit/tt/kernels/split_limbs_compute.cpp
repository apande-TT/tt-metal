// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// float32 x -> its two bf16 limbs, one tile at a time:
//
//     hi = bf16(x);  lo = bf16(x - hi)
//
// as split_bf16 (models/demos/qwen_image_edit_text_encoder/_stubs/attention.py) spells it with ttnn
// typecast / subtract, with the same SFPU calls (typecast_tile<Float32, Float16_b>, sub_binary_tile) on x
// unpacked straight to the float32 DST; hi -> c_16, lo -> c_17 (bf16). The rounded hi is exactly a bf16, so
// subtracting it in DST is the float32 subtract of its widened copy (as conv3d's split_operand_block does).

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_hw_startup.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_x = 0, cb_hi = 16, cb_lo = 17;
    constexpr uint32_t X = 0, HI = 1;

    compute_kernel_hw_startup(cb_x, cb_hi);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(cb_x, 1);
        cb_reserve_back(cb_hi, 1);
        cb_reserve_back(cb_lo, 1);

        tile_regs_acquire();
        copy_init(cb_x);
        copy_tile(cb_x, 0, X);
        copy_tile(cb_x, 0, HI);
        typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
        typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(HI);  // hi = bf16(x)
        sub_binary_tile_init();
        sub_binary_tile(X, HI, X);  // x - hi
        typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
        typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(X);  // lo = bf16(x - hi)
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(HI, cb_hi);
        pack_tile(X, cb_lo);
        tile_regs_release();

        cb_push_back(cb_hi, 1);
        cb_push_back(cb_lo, 1);
        cb_pop_front(cb_x, 1);
    }
}

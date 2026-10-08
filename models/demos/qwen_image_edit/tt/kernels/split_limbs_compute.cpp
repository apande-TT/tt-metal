// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// float32 x -> its bf16 limbs, one tile at a time (compile-time LIMBS = 2 or 3):
//
//     hi = bf16(x);  r = x - hi;  [mid = bf16(r);  r = r - mid;]  lo = bf16(r)
//
// as split_bf16 (models/demos/qwen_image_edit_text_encoder/_stubs/attention.py) spells it with ttnn
// typecast / subtract, with the same SFPU calls (typecast_tile<Float32, Float16_b>, sub_binary_tile) on x
// unpacked straight to the float32 DST; hi -> c_16, (mid -> c_17,) lo -> the last output CB (bf16). A rounded
// limb is exactly a bf16, so subtracting it in DST is the float32 subtract of its widened copy (as conv3d's
// split_operand_block does). The 3-limb remainder is formed twice (slots X and MID) instead of copied in DST.

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

constexpr uint32_t F32 = (uint32_t)DataFormat::Float32, BF16 = (uint32_t)DataFormat::Float16_b;

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t limbs = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = 0, cb_hi = 16, cb_mid = 17, cb_lo = limbs == 3 ? 18 : 17;
    constexpr uint32_t X = 0, HI = 1, MID = 2;

    compute_kernel_hw_startup(cb_x, cb_hi);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(cb_x, 1);
        cb_reserve_back(cb_hi, 1);
        if constexpr (limbs == 3) {
            cb_reserve_back(cb_mid, 1);
        }
        cb_reserve_back(cb_lo, 1);

        tile_regs_acquire();
        copy_init(cb_x);
        copy_tile(cb_x, 0, X);
        copy_tile(cb_x, 0, HI);
        if constexpr (limbs == 3) {
            copy_tile(cb_x, 0, MID);
        }
        typecast_tile_init<F32, BF16>();
        typecast_tile<F32, BF16>(HI);  // hi = bf16(x)
        sub_binary_tile_init();
        sub_binary_tile(X, HI, X);  // r = x - hi
        if constexpr (limbs == 3) {
            sub_binary_tile(MID, HI, MID);  // r again
            typecast_tile_init<F32, BF16>();
            typecast_tile<F32, BF16>(MID);  // mid = bf16(r)
            sub_binary_tile_init();
            sub_binary_tile(X, MID, X);  // r = r - mid
        }
        typecast_tile_init<F32, BF16>();
        typecast_tile<F32, BF16>(X);  // lo = bf16(r)
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(HI, cb_hi);
        if constexpr (limbs == 3) {
            pack_tile(MID, cb_mid);
        }
        pack_tile(X, cb_lo);
        tile_regs_release();

        cb_push_back(cb_hi, 1);
        if constexpr (limbs == 3) {
            cb_push_back(cb_mid, 1);
        }
        cb_push_back(cb_lo, 1);
        cb_pop_front(cb_x, 1);
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The compute kernel of the C++ decode RoPE (see tt/cpp_rope_dec.py): per tile row of x, for each column tile c
//   y[c] = x[c] * cos[c] + x[(c + DT/2) % DT] * sin_signed[c]
// with binary_ng's float32 SFPU ops in its order (mul_binary_tile, mul_binary_tile, add_binary_tile NearestEven),
// every operand unpacked straight to DEST -- the stock `x * cos + cat(x2, x1) * sin_signed` bit for bit (the
// rotate-half is a whole-tile swap: half the head is DT/2 tiles).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"

constexpr uint32_t DT = get_compile_time_arg_val(0);

constexpr auto cb_x = tt::CBIndex::c_0;
constexpr auto cb_cos = tt::CBIndex::c_1;
constexpr auto cb_sin = tt::CBIndex::c_2;
constexpr auto cb_y = tt::CBIndex::c_16;

void kernel_main() {
    const uint32_t nrows = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_x, cb_cos, cb_y);
    cb_wait_front(cb_cos, DT);
    cb_wait_front(cb_sin, DT);
    for (uint32_t r = 0; r < nrows; ++r) {
        cb_wait_front(cb_x, DT);
        cb_reserve_back(cb_y, DT);
        for (uint32_t c = 0; c < DT; ++c) {
            tile_regs_acquire();
            copy_init(cb_x);
            copy_tile(cb_x, c, 0);
            copy_init(cb_cos);
            copy_tile(cb_cos, c, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            copy_init(cb_x);
            copy_tile(cb_x, (c + DT / 2) % DT, 1);
            copy_init(cb_sin);
            copy_tile(cb_sin, c, 2);
            mul_binary_tile_init();
            mul_binary_tile(1, 2, 1);
            add_binary_tile_init();
            add_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_y);
            tile_regs_release();
        }
        cb_push_back(cb_y, DT);
        cb_pop_front(cb_x, DT);
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode rope compute: out = x * cos + rotate_half(x) * sin per tile, the three bf16 ops the
// multi-op path runs, each as ttnn runs it for bf16 operands: the two products as binary_ng's SFPU
// multiply (both operands copied to DEST, mul_binary_tile, packed to bf16) and the sum as binary_ng's
// default FPU add (binary_tiles_init<true, ELWADD> + add_tiles, bf16 DEST). rotate_half's sign is carried
// by the signed sin (x * -s == -x * s exactly).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/compute_kernel_hw_startup.h"

using std::uint32_t;

namespace {
// lhs * rhs -> one bf16 tile in cb_dst (binary_ng eltwise_binary_sfpu_no_bcast, one tile per cycle).
inline void sfpu_mul(uint32_t cb_lhs, uint32_t cb_rhs, uint32_t cb_dst) {
    cb_wait_front(cb_lhs, 1);
    cb_wait_front(cb_rhs, 1);
    cb_reserve_back(cb_dst, 1);
    mul_binary_tile_init();
    tile_regs_acquire();
    copy_init(cb_lhs);
    copy_tile(cb_lhs, 0, 0);
    copy_init(cb_rhs);
    copy_tile(cb_rhs, 0, 1);
    mul_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_dst);
    tile_regs_release();
    cb_push_back(cb_dst, 1);
    cb_pop_front(cb_lhs, 1);
    cb_pop_front(cb_rhs, 1);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_cos = get_compile_time_arg_val(1);
    constexpr uint32_t cb_xr = get_compile_time_arg_val(2);
    constexpr uint32_t cb_sin = get_compile_time_arg_val(3);
    constexpr uint32_t cb_t1 = get_compile_time_arg_val(4);
    constexpr uint32_t cb_t2 = get_compile_time_arg_val(5);
    constexpr uint32_t cb_out = get_compile_time_arg_val(6);

    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_x, cb_out);

    for (uint32_t t = 0; t < num_tiles; ++t) {
        sfpu_mul(cb_x, cb_cos, cb_t1);
        sfpu_mul(cb_xr, cb_sin, cb_t2);

        cb_wait_front(cb_t1, 1);
        cb_wait_front(cb_t2, 1);
        cb_reserve_back(cb_out, 1);
        binary_tiles_init<true, EltwiseBinaryType::ELWADD>(cb_t1, cb_t2);
        tile_regs_acquire();
        add_tiles(cb_t1, cb_t2, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_t1, 1);
        cb_pop_front(cb_t2, 1);
    }
}

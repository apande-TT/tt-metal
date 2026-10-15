// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode attention bmm compute: each output tile is one uninterrupted fp32 DEST accumulation over
// k = 0..Kt-1 in order (DST += in0[k] @ in1[k], in1 transposed by the unpacker for scores), packed to bf16 --
// the arithmetic of the reuse-matmul it replaces, whose in0 block spans all of K (no partial reload).
// With fuse_scale_mask (scores) the packed bf16 score tile then gets the scale + mask of kernels/scale_mask on the
// same core: t = score * scale (SFPU multiply of the bf16 score and scale tiles, packed bf16), out = t + mask
// (FPU add), instead of a separate op reading the scores back.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/compute_kernel_hw_startup.h"

using std::uint32_t;

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t transpose_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_in1 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);
    constexpr uint32_t fuse_scale_mask = get_compile_time_arg_val(5);
    constexpr uint32_t cb_scale = get_compile_time_arg_val(6);
    constexpr uint32_t cb_mask = get_compile_time_arg_val(7);
    constexpr uint32_t cb_t = get_compile_time_arg_val(8);
    constexpr uint32_t cb_t2 = get_compile_time_arg_val(9);
    constexpr uint32_t cb_mm = fuse_scale_mask ? cb_t : cb_out;

    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_mm);
    matmul_init(cb_in0, cb_in1, transpose_b);
    if constexpr (fuse_scale_mask) {
        cb_wait_front(cb_scale, 1);
    }

    for (uint32_t t = 0; t < num_tiles; ++t) {
        if constexpr (fuse_scale_mask) {
            matmul_init(cb_in0, cb_in1, transpose_b);
        }
        cb_wait_front(cb_in0, Kt);
        cb_wait_front(cb_in1, Kt);
        tile_regs_acquire();
        for (uint32_t k = 0; k < Kt; ++k) {
            matmul_tiles(cb_in0, cb_in1, k, k, 0);
        }
        tile_regs_commit();
        cb_pop_front(cb_in0, Kt);
        cb_pop_front(cb_in1, Kt);
        cb_reserve_back(cb_mm, 1);
        tile_regs_wait();
        pack_tile(0, cb_mm);
        tile_regs_release();
        cb_push_back(cb_mm, 1);

        if constexpr (fuse_scale_mask) {
            // t2 = t * scale
            cb_wait_front(cb_t, 1);
            cb_reserve_back(cb_t2, 1);
            mul_binary_tile_init();
            tile_regs_acquire();
            copy_init(cb_t);
            copy_tile(cb_t, 0, 0);
            copy_init(cb_scale);
            copy_tile(cb_scale, 0, 1);
            mul_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_t2);
            tile_regs_release();
            cb_push_back(cb_t2, 1);
            cb_pop_front(cb_t, 1);

            // out = t2 + mask
            cb_wait_front(cb_t2, 1);
            cb_wait_front(cb_mask, 1);
            cb_reserve_back(cb_out, 1);
            binary_tiles_init<true, EltwiseBinaryType::ELWADD>(cb_t2, cb_mask);
            tile_regs_acquire();
            add_tiles(cb_t2, cb_mask, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 1);
            cb_pop_front(cb_t2, 1);
            cb_pop_front(cb_mask, 1);
        }
    }
    if constexpr (fuse_scale_mask) {
        cb_pop_front(cb_scale, 1);
    }
}

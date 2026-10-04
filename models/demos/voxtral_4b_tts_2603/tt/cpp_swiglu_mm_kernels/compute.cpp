// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The compute kernel of tt/cpp_swiglu_mm.py: minimal_matmul(fuse_swiglu=True)'s arithmetic, tile for tile.
//
// Every output tile goes through exactly minimal_matmul's sequence (compute_metal2.cpp): per K block of KB
// tiles, a fresh 16-bit DEST accumulates the block's products in K order (matmul_block, LoFi); the block is
// packed into a Float16_b accumulator CB -- plain for the first block, packer-L1-accumulated for the rest;
// then each (gate, up) tile pair is copied back to DEST, silu runs on the gate, the SFPU multiplies it by the
// up tile and the product is packed bf16. Only WHICH tiles a core owns differs: here all MT rows of W fused
// columns, so the weight is spread over every core instead of minimal_matmul's grid.x column readers.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t MT = get_compile_time_arg_val(0);  // output tile rows
    constexpr uint32_t KB = get_compile_time_arg_val(1);  // K tiles a block (minimal_matmul's K_block_size)
    constexpr uint32_t NB = get_compile_time_arg_val(2);  // K blocks
    constexpr uint32_t W = get_compile_time_arg_val(3);   // fused gate / up column tiles this core owns
    constexpr uint32_t SW = get_compile_time_arg_val(4);  // subblock width (divides W, <= 8)

    constexpr auto cb_x = tt::CBIndex::c_0;
    constexpr auto cb_w = tt::CBIndex::c_1;
    constexpr auto cb_y = tt::CBIndex::c_16;
    constexpr auto cb_p = tt::CBIndex::c_24;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_p);
    matmul_init(cb_x, cb_w);
    reconfig_data_format(cb_w, cb_x);
    pack_reconfig_data_format(cb_p);
    matmul_block_init(cb_x, cb_w, false, SW, 1, KB);

    // The accumulator: every partial of this core's MT x W tiles, packed in place block after block.
    cb_reserve_back(cb_p, MT * W);
    for (uint32_t b = 0; b < NB; ++b) {
        cb_wait_front(cb_x, MT * KB);
        cb_wait_front(cb_w, KB * W);
        for (uint32_t m = 0; m < MT; ++m) {
            for (uint32_t n0 = 0; n0 < W; n0 += SW) {
                tile_regs_acquire();
                uint32_t i0 = m * KB;
                uint32_t i1 = n0;
                for (uint32_t k = 0; k < KB; ++k) {
                    matmul_block(cb_x, cb_w, i0, i1, 0, false, SW, 1, KB);
                    ++i0;
                    i1 += W;
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t w = 0; w < SW; ++w) {
                    pack_tile<true>(w, cb_p, m * W + n0 + w);
                }
                tile_regs_release();
            }
        }
        cb_pop_front(cb_x, MT * KB);
        cb_pop_front(cb_w, KB * W);
        if (b == 0) {
            pack_reconfig_l1_acc(1);
        }
    }
    cb_push_back(cb_p, MT * W);
    pack_reconfig_l1_acc(0);

    // minimal_matmul's swiglu_block: silu(gate) * up per interleaved pair, from the accumulator.
    cb_wait_front(cb_p, MT * W);
    reconfig_data_format_srca(cb_p);
    pack_reconfig_data_format(cb_y);
    for (uint32_t m = 0; m < MT; ++m) {
        cb_reserve_back(cb_y, W / 2);
        for (uint32_t p = 0; p < W / 2; ++p) {
            const uint32_t g = m * W + 2 * p;
            tile_regs_acquire();
            copy_init(cb_p);
            copy_tile(cb_p, g, 0);
            copy_tile(cb_p, g + 1, 1);
            silu_tile_init();
            silu_tile(0);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_y);
            tile_regs_release();
        }
        cb_push_back(cb_y, W / 2);
    }
    cb_pop_front(cb_p, MT * W);
}

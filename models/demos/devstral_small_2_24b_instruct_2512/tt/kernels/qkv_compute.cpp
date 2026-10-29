// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute of tt/qkv_decode.py: one tile row of tokens times this core's one weight column, accumulated in fp32 DEST
// in K order (weight push p in cb_w1 for the first W1 odd p, else cb_w0: the RISC that streamed it,
// kernels/qkv_stream.cpp) and packed to bf16 straight into the core's output shard. KB < Kt: the fp32 partial is
// packed to cb_part every KB K tiles and reloaded through srcA (19 bits kept), as the stock DRAM-sharded matmul
// (bmm_large_block_zm_fused_bias_activation, no packer L1 acc) does between its in0_block_w blocks.
#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);  // K tiles
    constexpr uint32_t KB = get_compile_time_arg_val(1);  // K tiles a block (the stock op's in0_block_w)
    constexpr uint32_t RB = get_compile_time_arg_val(2);  // K rows a weight push
    constexpr uint32_t CH = get_compile_time_arg_val(3);  // x tiles a chunk
    constexpr uint32_t W1 = get_compile_time_arg_val(4);  // the odd pushes 1 .. 2 W1 - 1 come from RISCV_1
    constexpr uint32_t cb_x = 0, cb_w0 = 1, cb_w1 = 2, cb_out = 16, cb_part = 24;
    static_assert(Kt % KB == 0 && KB % RB == 0 && RB % CH == 0, "qkv blocking");
    constexpr uint32_t NBLK = Kt / KB;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w0, cb_part);
    matmul_block_init(cb_x, cb_w0, 0, 1, 1, KB);
    for (uint32_t b = 0; b < NBLK; ++b) {
        tile_regs_acquire();
        if (b > 0) {
            reconfig_data_format_srca(cb_w0, cb_part);
            copy_init(cb_part);
            cb_wait_front(cb_part, 1);
            copy_tile(cb_part, 0, 0);
            cb_pop_front(cb_part, 1);
            reconfig_data_format_srca(cb_part, cb_w0);
            matmul_block_init(cb_x, cb_w0, 0, 1, 1, KB);
        }
        for (uint32_t p = 0; p < KB / RB; ++p) {
            const uint32_t k0 = b * KB + p * RB;
            const uint32_t q = k0 / RB;
            const uint32_t cb_w = (q % 2 == 1 && q / 2 < W1) ? cb_w1 : cb_w0;
            cb_wait_front(cb_x, k0 + RB);  // x tiles 0 .. k0 + RB - 1 (cumulative: x is popped at the end)
            cb_wait_front(cb_w, RB);
            for (uint32_t i = 0; i < RB; ++i) {
                matmul_block(cb_x, cb_w, k0 + i, i, 0, 0, 1, 1, KB);
            }
            cb_pop_front(cb_w, RB);
        }
        tile_regs_commit();
        if (b + 1 < NBLK) {
            cb_reserve_back(cb_part, 1);
            tile_regs_wait();
            pack_tile(0, cb_part);
            tile_regs_release();
            cb_push_back(cb_part, 1);
        } else {
            cb_reserve_back(cb_out, 1);
            tile_regs_wait();
            pack_reconfig_data_format(cb_out);
            pack_tile(0, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 1);
        }
    }
    cb_pop_front(cb_x, Kt);
}

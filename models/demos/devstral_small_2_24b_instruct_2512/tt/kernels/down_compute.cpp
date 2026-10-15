// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute of tt/down_decode.py: one tile row of tokens times this core's two weight columns (column 0 streamed into
// cb_w0 by RISCV_0, column 1 into cb_w1 by RISCV_1), each accumulated over the whole K in fp32 DEST in K order and
// packed once. ttnn's 1D-multicast matmul runs the same sequence of tile products: its fp32 partials between K blocks
// are reloaded exactly (UnpackToDestFp32), so the result is bit-identical to it.
#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);  // K tiles
    constexpr uint32_t RB = get_compile_time_arg_val(1);  // K rows a push
    constexpr uint32_t cb_x = 0, cb_w0 = 1, cb_w1 = 2, cb_out = 16;
    static_assert(Kt % RB == 0, "down blocking");

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w0, cb_out);
    matmul_block_init(cb_x, cb_w0, 0, 1, 1, RB);
    cb_wait_front(cb_x, Kt);

    tile_regs_acquire();
    for (uint32_t k0 = 0; k0 < Kt; k0 += RB) {
        cb_wait_front(cb_w0, RB);
        for (uint32_t i = 0; i < RB; ++i) {
            matmul_block(cb_x, cb_w0, k0 + i, i, 0, 0, 1, 1, RB);
        }
        cb_pop_front(cb_w0, RB);
        cb_wait_front(cb_w1, RB);
        for (uint32_t i = 0; i < RB; ++i) {
            matmul_block(cb_x, cb_w1, k0 + i, i, 1, 0, 1, 1, RB);
        }
        cb_pop_front(cb_w1, RB);
    }
    tile_regs_commit();

    cb_reserve_back(cb_out, 2);
    tile_regs_wait();
    pack_tile(0, cb_out);
    pack_tile(1, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 2);
    cb_pop_front(cb_x, Kt);
}

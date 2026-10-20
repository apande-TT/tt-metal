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
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);  // K tiles
    constexpr uint32_t RB = get_compile_time_arg_val(1);  // K rows a push
    constexpr uint32_t cb_x = 0, cb_w0 = 1, cb_w1 = 2, cb_out = 16;
    static_assert(Kt % RB == 0, "down blocking");

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w0, cb_out);
    matmul_block_init(cb_x, cb_w0, 0, 1, 1, RB);
    cb_wait_front(cb_x, Kt);

#if defined(DOWN_ONE_COL) || defined(DOWN_SEQ)
    // one column at a time, its pushes alternating between the two RISCs' CBs (push 2q in cb_w0, 2q + 1 in cb_w1):
    // each column still one DEST tile accumulated in K order, packed as soon as it is done
#ifdef DOWN_SEQ
    constexpr uint32_t NCOL = DOWN_SEQ;
#else
    constexpr uint32_t NCOL = 1;
#endif
    static_assert(Kt % (2 * RB) == 0, "down blocking");
    for (uint32_t col = 0; col < NCOL; ++col) {
        tile_regs_acquire();
        for (uint32_t k0 = 0; k0 < Kt; k0 += 2 * RB) {
            cb_wait_front(cb_w0, RB);
            for (uint32_t i = 0; i < RB; ++i) {
                matmul_block(cb_x, cb_w0, k0 + i, i, 0, 0, 1, 1, RB);
            }
            cb_pop_front(cb_w0, RB);
            cb_wait_front(cb_w1, RB);
            for (uint32_t i = 0; i < RB; ++i) {
                matmul_block(cb_x, cb_w1, k0 + RB + i, i, 0, 0, 1, 1, RB);
            }
            cb_pop_front(cb_w1, RB);
        }
        tile_regs_commit();

        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
#else
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
#endif
    cb_pop_front(cb_x, Kt);

#ifdef RES_TAIL
    // tt/down_decode.py ar_call, on the core of residual shard i: the all_reduce's sum of the TP partial slots as
    // ttnn.experimental.all_reduce_async forms it (pairs added into an fp32 DEST, (s0 + s1) + (s2 + s3), rounded once to
    // bf16), then the residual add h = x + sum as ttnn.add runs it for an fp32 x and a bf16 sum (FPU ELWADD, x in
    // the default unpack mode, fp32 DEST) -- norm_shard_ar_compute.cpp's first two steps.
    constexpr uint32_t TP = get_compile_time_arg_val(2);  // partial slots (devices)
    constexpr uint32_t WT = get_compile_time_arg_val(3);  // tiles of the shard
    constexpr uint32_t cb_x0 = 11, cb_ar = 12, cb_a = 13, cb_hout = 14;
    reconfig_data_format(cb_ar, cb_ar);
    pack_reconfig_data_format(cb_a);
    cb_wait_front(cb_ar, TP * WT);
    cb_reserve_back(cb_a, WT);
    tile_regs_acquire();
    for (uint32_t s = 0; s < TP; s += 2) {
        add_init(cb_ar, cb_ar, s != 0);
        for (uint32_t w = 0; w < WT; ++w) {
            add_tiles(cb_ar, cb_ar, s * WT + w, (s + 1) * WT + w, w);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t w = 0; w < WT; ++w) {
        pack_tile(w, cb_a);
    }
    tile_regs_release();
    cb_push_back(cb_a, WT);
    cb_pop_front(cb_ar, TP * WT);

    cb_wait_front(cb_x0, WT);
    cb_wait_front(cb_a, WT);
    reconfig_data_format(cb_x0, cb_a);
    pack_reconfig_data_format(cb_hout);
    add_init(cb_x0, cb_a);
    cb_reserve_back(cb_hout, WT);
    tile_regs_acquire();
    for (uint32_t w = 0; w < WT; ++w) {
        add_tiles(cb_x0, cb_a, w, w, w);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t w = 0; w < WT; ++w) {
        pack_tile(w, cb_hout);
    }
    tile_regs_release();
    cb_push_back(cb_hout, WT);
    cb_pop_front(cb_x0, WT);
    cb_pop_front(cb_a, WT);
#endif
}

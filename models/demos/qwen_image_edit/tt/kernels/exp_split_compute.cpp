// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The softmax's exp(s - max) and its bf16 limb split in one pass, per float32 tile:
//
//     e = exp(s - m)      (the subtract with the fused EXP post-activation, as ttnn's binary_ng spells it:
//                          sub_binary_tile<NearestEven>, then exp_tile<approx> per SOFTMAX_EXP_APPROX)
//     hi = bf16(e), lo = bf16(e - hi)   (split_limbs_compute.cpp's typecast / subtract)
//
// e (float32, for the row sum) to c_16, hi and lo (bf16) to c_17 / c_18. Compile-time: approx (0 / 1).
// Runtime: tile count, first tile index, Nt (a tile row starts at every multiple of Nt, and at the first tile).

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reconfig_data_format.h"

constexpr uint32_t F32 = (uint32_t)DataFormat::Float32, BF16 = (uint32_t)DataFormat::Float16_b;

void kernel_main() {
    const uint32_t count = get_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t nt = get_arg_val<uint32_t>(2);
    constexpr uint32_t approx = get_compile_time_arg_val(0);
    // rowsum (compile-time 1, with the column-chunk width cw): e is not written; each (tile row, cw-column chunk)
    // item's e tiles are summed elementwise by the packer's float32 L1 accumulation into ONE c_16 tile (the item's
    // partial row sums, column-spread), pushed when the item ends -- the row sum's input without e's round trip
    constexpr uint32_t rowsum = get_compile_time_arg_val(1);
    constexpr uint32_t cw = get_compile_time_arg_val(2);
    constexpr uint32_t cb_s = 0, cb_m = 1, cb_e = 16, cb_hi = 17, cb_lo = 18;
    constexpr uint32_t X = 0, M = 1, HI = 2, LO = 3;

    compute_kernel_hw_startup(cb_s, cb_e);
    for (uint32_t i = start; i < start + count; ++i) {
        const uint32_t c = i % nt;
        if (i == start || c == 0) {
            cb_wait_front(cb_m, 1);
        }
        cb_wait_front(cb_s, 1);
        const bool item_first = rowsum && c % cw == 0;
        const bool item_last = rowsum && ((c + 1) % cw == 0 || c + 1 == nt);
        if (!rowsum || item_first) {
            cb_reserve_back(cb_e, 1);
        }
        cb_reserve_back(cb_hi, 1);
        cb_reserve_back(cb_lo, 1);

        tile_regs_acquire();
        copy_init(cb_s);
        copy_tile(cb_s, 0, X);
        copy_init(cb_m);
        copy_tile(cb_m, 0, M);
        sub_binary_tile_init();
        sub_binary_tile<ckernel::DstRoundingMode::NearestEven>(X, M, X);  // s - m
        exp_tile_init<approx != 0>();
        exp_tile<approx != 0>(X);  // e
        copy_dest_values_init();
        copy_dest_values<DataFormat::Float32>(X, HI);
        copy_dest_values<DataFormat::Float32>(X, LO);
        typecast_tile_init<F32, BF16>();
        typecast_tile<F32, BF16>(HI);  // hi = bf16(e)
        sub_binary_tile_init();
        sub_binary_tile(LO, HI, LO);  // r = e - hi
        typecast_tile_init<F32, BF16>();
        typecast_tile<F32, BF16>(LO);  // lo = bf16(r)
        tile_regs_commit();

        tile_regs_wait();
        pack_reconfig_data_format(cb_e);  // float32 out
        if constexpr (rowsum) {
            pack_reconfig_l1_acc(item_first ? 0 : 1);
            pack_tile<true>(X, cb_e, 0);
            pack_reconfig_l1_acc(0);
        } else {
            pack_tile(X, cb_e);
        }
        pack_reconfig_data_format(cb_hi);  // bf16 outs
        pack_tile(HI, cb_hi);
        pack_tile(LO, cb_lo);
        tile_regs_release();

        if (!rowsum || item_last) {
            cb_push_back(cb_e, 1);
        }
        cb_push_back(cb_hi, 1);
        cb_push_back(cb_lo, 1);
        cb_pop_front(cb_s, 1);
        if (i + 1 == start + count || c + 1 == nt) {
            cb_pop_front(cb_m, 1);
        }
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode down projection, helper-core compute under binary_ng's compute config (16-bit DEST, HiFi4,
// no approximation), each half reproducing the stand-alone bf16 op:
//   prologue: h = silu(gate) * up, the calls of binary_ng's SFPU kernel (eltwise_binary_sfpu_no_bcast.cpp, one
//             tile per cycle) for ttnn.multiply with input_tensor_a_activations=[SILU]: the LHS preprocess pass
//             copy -> silu -> pack bf16, then both operands into DEST and mul_binary_tile;
//   epilogue: out = residual + projection (ttnn.add's bf16 sums: the FPU add).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/compute_kernel_hw_startup.h"

namespace {
// process_sfpu_tiles for one tile: lhs and rhs copied into DEST 0 / 1, op, pack DEST 0.
inline void sfpu_mul_tile(uint32_t cb_lhs, uint32_t cb_rhs, uint32_t cb_out) {
    cb_wait_front(cb_lhs, 1);
    cb_wait_front(cb_rhs, 1);
    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    copy_init(cb_lhs);
    copy_tile(cb_lhs, 0, 0);
    reconfig_data_format_srca(cb_lhs, cb_rhs);
    copy_init(cb_rhs);
    copy_tile(cb_rhs, 0, 1);
    mul_binary_tile(0, 1, 0);
    reconfig_data_format_srca(cb_rhs, cb_lhs);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
    cb_pop_front(cb_lhs, 1);
    cb_pop_front(cb_rhs, 1);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_gate = get_compile_time_arg_val(0);
    constexpr uint32_t cb_up = get_compile_time_arg_val(1);
    constexpr uint32_t cb_post = get_compile_time_arg_val(2);  // silu(gate) (binary_ng's c_3)
    constexpr uint32_t cb_h = get_compile_time_arg_val(3);
    constexpr uint32_t cb_res = get_compile_time_arg_val(4);
    constexpr uint32_t cb_mm = get_compile_time_arg_val(5);
    constexpr uint32_t cb_sum = get_compile_time_arg_val(6);
    const uint32_t p_tiles = get_arg_val<uint32_t>(1);
    const uint32_t e_tiles = get_arg_val<uint32_t>(3);

    // ---- prologue: the silu * up kernel ----
    compute_kernel_hw_startup(cb_post, cb_h);
    copy_init(cb_post);
    for (uint32_t t = 0; t < p_tiles; ++t) {
        reconfig_data_format_srca(cb_post, cb_gate);
        pack_reconfig_data_format(cb_h, cb_post);
        cb_wait_front(cb_gate, 1);
        cb_reserve_back(cb_post, 1);
        tile_regs_acquire();
        copy_init(cb_gate);
        copy_tile(cb_gate, 0, 0);
        silu_tile_init();
        silu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_post);
        tile_regs_release();
        cb_pop_front(cb_gate, 1);
        cb_push_back(cb_post, 1);
        reconfig_data_format_srca(cb_gate, cb_post);
        pack_reconfig_data_format(cb_post, cb_h);
        mul_binary_tile_init();
        sfpu_mul_tile(cb_post, cb_up, cb_h);
    }

    // ---- epilogue: residual + projection, the FPU add of eltwise_binary_no_bcast.cpp (the bf16 sums of the
    // stand-alone ttnn.add; the SFPU add_binary_tile differs on ~10% of the elements) ----
    reconfig_data_format(cb_res, cb_mm);
    pack_reconfig_data_format(cb_sum);
    binary_tiles_init<true, EltwiseBinaryType::ELWADD>(cb_res, cb_mm);
    for (uint32_t t = 0; t < e_tiles; ++t) {
        cb_wait_front(cb_res, 1);
        cb_wait_front(cb_mm, 1);
        cb_reserve_back(cb_sum, 1);
        tile_regs_acquire();
        add_tiles(cb_res, cb_mm, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_sum);
        tile_regs_release();
        cb_push_back(cb_sum, 1);
        cb_pop_front(cb_res, 1);
        cb_pop_front(cb_mm, 1);
    }
}

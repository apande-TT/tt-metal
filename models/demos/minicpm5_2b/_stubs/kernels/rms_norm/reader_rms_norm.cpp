// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode RMSNorm reader (one tile row of x, Wt tiles). Every core reads the whole row (each core
// recomputes the row statistic) plus the gamma tiles of its own output slice, and builds the two constant tiles
// as ttnn's layernorm reader does: the reduce scaler (1.0 bf16 in row 0 of each face, zeros elsewhere) and eps
// (fp32 eps truncated to bf16, in column 0 of every row of faces 0 and 2).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(2);
    constexpr uint32_t cb_eps = get_compile_time_arg_val(3);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(4);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(5);
    constexpr auto x_args = TensorAccessorArgs<6>();
    constexpr auto g_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = 32 * 32 * 2;

    const uint32_t t_start = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0));
    const auto gamma = TensorAccessor(g_args, get_common_arg_val<uint32_t>(1));

    // Constant tiles.
    cb_reserve_back(cb_scaler, 1);
    cb_reserve_back(cb_eps, 1);
    auto* scaler = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_scaler));
    auto* eps = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_eps));
    for (uint32_t i = 0; i < tile_bytes / 2; ++i) {
        scaler[i] = 0;
        eps[i] = 0;
    }
    for (uint32_t face = 0; face < 4; ++face) {
        for (uint32_t c = 0; c < 16; ++c) {
            scaler[face * 256 + c] = 0x3F80;  // 1.0
        }
    }
    const uint16_t eps_bf16 = static_cast<uint16_t>(eps_bits >> 16);
    for (uint32_t face = 0; face < 4; face += 2) {
        for (uint32_t r = 0; r < 16; ++r) {
            eps[face * 256 + r * 16] = eps_bf16;
        }
    }
    cb_push_back(cb_scaler, 1);
    cb_push_back(cb_eps, 1);

    // The whole input row, and gamma for this core's slice.
    cb_reserve_back(cb_in, Wt);
    const uint32_t in_l1 = get_write_ptr(cb_in);
    for (uint32_t t = 0; t < Wt; ++t) {
        noc_async_read_page(t, x, in_l1 + t * tile_bytes);
    }
    cb_reserve_back(cb_gamma, num_tiles);
    const uint32_t g_l1 = get_write_ptr(cb_gamma);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        noc_async_read_page(t_start + i, gamma, g_l1 + i * tile_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_in, Wt);
    cb_push_back(cb_gamma, num_tiles);
}

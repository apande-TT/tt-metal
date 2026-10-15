// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode RMSNorm reader (one tile row of x, Wt tiles). Every core reads the whole row (each core
// recomputes the row statistic) plus the gamma tiles of its own output slice, and the two constant tiles of ttnn's
// layernorm reader, built once on the host: the reduce scaler (1.0 bf16 in row 0 of each face, zeros elsewhere)
// and eps (fp32 eps truncated to bf16, in column 0 of every row of faces 0 and 2).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(2);
    constexpr uint32_t cb_eps = get_compile_time_arg_val(3);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(4);
    constexpr auto x_args = TensorAccessorArgs<5>();
    constexpr auto g_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = 32 * 32 * 2;

    const uint32_t t_start = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0));
    const auto gamma = TensorAccessor(g_args, get_common_arg_val<uint32_t>(1));
    const auto consts = TensorAccessor(c_args, get_common_arg_val<uint32_t>(2));

    // Constant tiles, host-built pages 0..1 of the [32, 64] constants tensor: the reduce scaler and eps.
    cb_reserve_back(cb_scaler, 1);
    cb_reserve_back(cb_eps, 1);
    noc_async_read_page(0, consts, get_write_ptr(cb_scaler));
    noc_async_read_page(1, consts, get_write_ptr(cb_eps));
    noc_async_read_barrier();
    cb_push_back(cb_scaler, 1);
    cb_push_back(cb_eps, 1);

    // Gamma for this core's slice, then the whole input row pushed in chunks so compute squares a chunk while the
    // next is in flight.
    constexpr uint32_t chunk = 8;
    cb_reserve_back(cb_gamma, num_tiles);
    const uint32_t g_l1 = get_write_ptr(cb_gamma);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        noc_async_read_page(t_start + i, gamma, g_l1 + i * tile_bytes);
    }
    cb_reserve_back(cb_in, Wt);
    const uint32_t in_l1 = get_write_ptr(cb_in);
    for (uint32_t t0 = 0; t0 < Wt; t0 += chunk) {
        const uint32_t n = (Wt - t0) < chunk ? (Wt - t0) : chunk;
        for (uint32_t t = t0; t < t0 + n; ++t) {
            noc_async_read_page(t, x, in_l1 + t * tile_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_in, n);
    }
    cb_push_back(cb_gamma, num_tiles);
}

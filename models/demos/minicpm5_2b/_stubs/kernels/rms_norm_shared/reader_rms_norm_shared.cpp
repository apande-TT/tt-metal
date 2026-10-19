// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode RMSNorm reader, shared-statistic variant. Each core reads only its own slice of the input row
// (num_tiles tiles from t_start) and the matching gamma tiles. The reducer core also reads the two host-built
// constant tiles (reduce scaler, eps) and, once every core's writer has dropped its x^2 tiles into the reducer's
// cb_x2all and bumped sem_gather, publishes the whole Wt-tile x^2 row to its compute. Every core then waits for the
// reducer to drop the rstd tile into its cb_ex2pe (sem_bcast) and publishes it.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(2);
    constexpr uint32_t cb_eps = get_compile_time_arg_val(3);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(4);
    constexpr uint32_t cb_x2all = get_compile_time_arg_val(5);
    constexpr uint32_t cb_ex2pe = get_compile_time_arg_val(6);
    constexpr uint32_t sem_gather = get_compile_time_arg_val(7);
    constexpr uint32_t sem_bcast = get_compile_time_arg_val(8);
    constexpr uint32_t n_cores = get_compile_time_arg_val(9);
    // As the prologue of a projection program: once this core's DRAM reads have landed, bump start_sem on each
    // projection core (common args from 3: count, then x / y pairs) -- their weight streams wait for every norm
    // core, so these few small reads are not queued behind the stream at the DRAM banks.
    constexpr uint32_t start_sem = get_compile_time_arg_val(10);  // ~0u: none
    constexpr auto x_args = TensorAccessorArgs<11>();
    constexpr auto g_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = 32 * 32 * 2;

    const uint32_t t_start = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t is_reducer = get_arg_val<uint32_t>(2);
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0));
    const auto gamma = TensorAccessor(g_args, get_common_arg_val<uint32_t>(1));
    const auto consts = TensorAccessor(c_args, get_common_arg_val<uint32_t>(2));

    cb_reserve_back(cb_in, num_tiles);
    const uint32_t in_l1 = get_write_ptr(cb_in);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        noc_async_read_page(t_start + i, x, in_l1 + i * tile_bytes);
    }
    cb_reserve_back(cb_gamma, num_tiles);
    const uint32_t g_l1 = get_write_ptr(cb_gamma);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        noc_async_read_page(t_start + i, gamma, g_l1 + i * tile_bytes);
    }
    if (is_reducer) {
        cb_reserve_back(cb_scaler, 1);
        cb_reserve_back(cb_eps, 1);
        noc_async_read_page(0, consts, get_write_ptr(cb_scaler));
        noc_async_read_page(1, consts, get_write_ptr(cb_eps));
    }
    noc_async_read_barrier();
    if constexpr (start_sem != 0xFFFFFFFF) {
        const uint32_t n_mm = get_common_arg_val<uint32_t>(3);
        const uint32_t sem_l1 = get_semaphore(start_sem);
        for (uint32_t i = 0; i < n_mm; ++i) {
            const uint32_t cx = get_common_arg_val<uint32_t>(4 + 2 * i);
            const uint32_t cy = get_common_arg_val<uint32_t>(5 + 2 * i);
            noc_semaphore_inc(get_noc_addr(cx, cy, sem_l1), 1);
        }
    }
    cb_push_back(cb_in, num_tiles);
    if (is_reducer) {
        cb_push_back(cb_scaler, 1);
        cb_push_back(cb_eps, 1);
    }

    if (is_reducer) {
        cb_reserve_back(cb_x2all, Wt);
        auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_gather));
        noc_semaphore_wait(sem, n_cores);
        noc_semaphore_set(sem, 0);
        cb_push_back(cb_x2all, Wt);
    }

    cb_reserve_back(cb_ex2pe, 1);
    {
        auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_bcast));
        noc_semaphore_wait(sem, 1);
        noc_semaphore_set(sem, 0);
    }
    cb_push_back(cb_ex2pe, 1);
    cb_push_back(cb_gamma, num_tiles);
    if constexpr (start_sem != 0xFFFFFFFF) {
        noc_async_atomic_barrier();
    }
}

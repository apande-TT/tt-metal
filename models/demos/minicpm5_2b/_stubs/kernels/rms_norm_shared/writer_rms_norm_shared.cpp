// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode RMSNorm writer, shared-statistic variant.
//   1. Sends this core's num_tiles fp32 x^2 tiles into slot t_start of the reducer's cb_x2all, then bumps the
//      reducer's sem_gather.
//   2. Reducer only: copies the rstd tile its compute packed (cb_ex2pe_send) into every core's cb_ex2pe and bumps
//      each core's sem_bcast.
//   3. Writes this core's output tiles [t_start, t_start + num_tiles).
// CBs are allocated at the same L1 address on every core of the program, and cb_x2all / cb_ex2pe are untouched
// before these writes, so their base write pointers are the remote destinations.
// Common args: [out_addr, reducer_x, reducer_y, x_0, y_0, ..., x_{n-1}, y_{n-1}] (NoC coordinates).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_x2 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x2all = get_compile_time_arg_val(1);
    constexpr uint32_t cb_ex2pe = get_compile_time_arg_val(2);
    constexpr uint32_t cb_ex2pe_send = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);
    constexpr uint32_t sem_gather = get_compile_time_arg_val(5);
    constexpr uint32_t sem_bcast = get_compile_time_arg_val(6);
    constexpr uint32_t n_cores = get_compile_time_arg_val(7);
    // As the prologue of a projection program: once this core's output slice is stored, bump pro_sem on each of
    // the projection's cores (common args after the norm cores' coordinates: [n_mm, x_0, y_0, ...]).
    constexpr uint32_t pro_sem = get_compile_time_arg_val(8);  // ~0u: stand-alone
    constexpr auto out_args = TensorAccessorArgs<9>();
    constexpr uint32_t f32_tile_bytes = 32 * 32 * 4;

    const uint32_t t_start = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t is_reducer = get_arg_val<uint32_t>(2);
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(0));
    const uint32_t red_x = get_common_arg_val<uint32_t>(1);
    const uint32_t red_y = get_common_arg_val<uint32_t>(2);

    // 1. x^2 slice -> reducer.
    cb_wait_front(cb_x2, num_tiles);
    noc_async_write(
        get_read_ptr(cb_x2),
        get_noc_addr(red_x, red_y, get_write_ptr(cb_x2all) + t_start * f32_tile_bytes),
        num_tiles * f32_tile_bytes);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(red_x, red_y, get_semaphore(sem_gather)), 1);
    noc_async_atomic_barrier();
    cb_pop_front(cb_x2, num_tiles);

    // 2. rstd -> every core.
    if (is_reducer) {
        cb_wait_front(cb_ex2pe_send, 1);
        const uint32_t src = get_read_ptr(cb_ex2pe_send);
        const uint32_t dst = get_write_ptr(cb_ex2pe);
        for (uint32_t c = 0; c < n_cores; ++c) {
            const uint32_t cx = get_common_arg_val<uint32_t>(3 + 2 * c);
            const uint32_t cy = get_common_arg_val<uint32_t>(4 + 2 * c);
            noc_async_write(src, get_noc_addr(cx, cy, dst), f32_tile_bytes);
        }
        noc_async_write_barrier();
        const uint32_t sem_l1 = get_semaphore(sem_bcast);
        for (uint32_t c = 0; c < n_cores; ++c) {
            const uint32_t cx = get_common_arg_val<uint32_t>(3 + 2 * c);
            const uint32_t cy = get_common_arg_val<uint32_t>(4 + 2 * c);
            noc_semaphore_inc(get_noc_addr(cx, cy, sem_l1), 1);
        }
        noc_async_atomic_barrier();
        cb_pop_front(cb_ex2pe_send, 1);
    }

    // 3. Output slice.
    for (uint32_t t = t_start; t < t_start + num_tiles; ++t) {
        cb_wait_front(cb_out, 1);
        noc_async_write_page(t, out, get_read_ptr(cb_out));
        noc_async_writes_flushed();
        cb_pop_front(cb_out, 1);
    }
    noc_async_write_barrier();
    if constexpr (pro_sem != 0xFFFFFFFF) {
        const uint32_t base = 3 + 2 * n_cores;
        const uint32_t n_mm = get_common_arg_val<uint32_t>(base);
        const uint32_t sem_l1 = get_semaphore(pro_sem);
        for (uint32_t c = 0; c < n_mm; ++c) {
            const uint32_t cx = get_common_arg_val<uint32_t>(base + 1 + 2 * c);
            const uint32_t cy = get_common_arg_val<uint32_t>(base + 2 + 2 * c);
            noc_semaphore_inc(get_noc_addr(cx, cy, sem_l1), 1);
        }
        noc_async_atomic_barrier();
    }
}

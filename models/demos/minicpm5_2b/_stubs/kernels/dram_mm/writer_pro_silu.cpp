// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode down projection, helper-core writer: the prologue's h tiles [p_start, p_start + p_tiles),
// then one bump of every projection core's prologue semaphore (they read h once all helpers are in); then the
// epilogue's summed tiles [e_start, e_start + e_tiles).
// Common args: [h, out, n_mm, x_0, y_0, ..., x_{n_mm-1}, y_{n_mm-1}] (projection cores' NoC coordinates).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_h = get_compile_time_arg_val(0);
    constexpr uint32_t cb_sum = get_compile_time_arg_val(1);
    constexpr uint32_t pro_sem = get_compile_time_arg_val(2);
    constexpr auto h_args = TensorAccessorArgs<3>();
    constexpr auto o_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();

    const uint32_t p_start = get_arg_val<uint32_t>(0);
    const uint32_t p_tiles = get_arg_val<uint32_t>(1);
    const uint32_t e_start = get_arg_val<uint32_t>(2);
    const uint32_t e_tiles = get_arg_val<uint32_t>(3);
    const auto h = TensorAccessor(h_args, get_common_arg_val<uint32_t>(0));
    const auto out = TensorAccessor(o_args, get_common_arg_val<uint32_t>(1));
    const uint32_t n_mm = get_common_arg_val<uint32_t>(2);

    for (uint32_t t = p_start; t < p_start + p_tiles; ++t) {
        cb_wait_front(cb_h, 1);
        noc_async_write_page(t, h, get_read_ptr(cb_h));
        noc_async_writes_flushed();
        cb_pop_front(cb_h, 1);
    }
    noc_async_write_barrier();
    const uint32_t sem_l1 = get_semaphore(pro_sem);
    for (uint32_t c = 0; c < n_mm; ++c) {
        const uint32_t cx = get_common_arg_val<uint32_t>(3 + 2 * c);
        const uint32_t cy = get_common_arg_val<uint32_t>(4 + 2 * c);
        noc_semaphore_inc(get_noc_addr(cx, cy, sem_l1), 1);
    }
    noc_async_atomic_barrier();

    for (uint32_t t = e_start; t < e_start + e_tiles; ++t) {
        cb_wait_front(cb_sum, 1);
        noc_async_write_page(t, out, get_read_ptr(cb_sum));
        noc_async_writes_flushed();
        cb_pop_front(cb_sum, 1);
    }
    noc_async_write_barrier();
}

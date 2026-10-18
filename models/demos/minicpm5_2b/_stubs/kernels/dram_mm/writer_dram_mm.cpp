// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode projection, activation reader + output writer (BRISC). Reads the whole activation row
// x[0, 0..Kt) into cb_in0 (kept resident by compute), pushed in chunks so compute starts on the first K rows, then
// writes this core's output tiles [n_start, n_start + num_groups * group_w) of the one-tile-row output (columns
// from split_t on into the second output tensor, when the program computes two projections side by side).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t group_w = get_compile_time_arg_val(1);
    constexpr uint32_t in0_chunk = get_compile_time_arg_val(2);
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);
    constexpr uint32_t split_t = get_compile_time_arg_val(5);  // columns >= split_t go to the second output
    // Epilogue (residual add on helper cores): signal each stored output tile's helper (common args from ep_off).
    constexpr uint32_t ep_on = get_compile_time_arg_val(6);
    constexpr uint32_t ep_sem = get_compile_time_arg_val(7);
    constexpr uint32_t ep_tpc = get_compile_time_arg_val(8);  // output tiles per helper core
    constexpr uint32_t ep_off = get_compile_time_arg_val(9);
    constexpr auto x_args = TensorAccessorArgs<10>();
    constexpr auto out_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = 32 * 32 * 2;

    const uint32_t n_start = get_arg_val<uint32_t>(2);
    const uint32_t num_groups = get_arg_val<uint32_t>(4);
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(1));
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(2));
    const auto out_hi = TensorAccessor(out_args, get_common_arg_val<uint32_t>(3));  // same spec, other buffer

    cb_reserve_back(cb_in0, Kt);
    uint32_t l1 = get_write_ptr(cb_in0);
    for (uint32_t k0 = 0; k0 < Kt; k0 += in0_chunk) {
        for (uint32_t k = k0; k < k0 + in0_chunk; ++k) {
            noc_async_read_page(k, x, l1);
            l1 += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_in0, in0_chunk);
    }

    uint32_t n = n_start;
    for (uint32_t g = 0; g < num_groups; ++g) {
        cb_wait_front(cb_out, group_w);
        uint32_t src = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < group_w; ++j) {
            if (n + j < split_t) {
                noc_async_write_page(n + j, out, src);
            } else {
                noc_async_write_page(n + j - split_t, out_hi, src);
            }
            src += tile_bytes;
        }
        noc_async_writes_flushed();
        if constexpr (ep_on) {
            // The tiles are stored: hand each to its epilogue core.
            noc_async_write_barrier();
            const uint32_t sem_l1 = get_semaphore(ep_sem);
            for (uint32_t j = 0; j < group_w; ++j) {
                const uint32_t h = (n + j) / ep_tpc;
                const uint32_t hx = get_common_arg_val<uint32_t>(ep_off + 2 * h);
                const uint32_t hy = get_common_arg_val<uint32_t>(ep_off + 2 * h + 1);
                noc_semaphore_inc(get_noc_addr(hx, hy, sem_l1), 1);
            }
        }
        cb_pop_front(cb_out, group_w);
        n += group_w;
    }
    noc_async_write_barrier();
    if constexpr (ep_on) {
        noc_async_atomic_barrier();
    }
}

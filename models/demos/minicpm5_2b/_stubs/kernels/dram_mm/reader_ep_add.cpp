// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode projection epilogue (helper cores of a dram_mm program): residual + projection output for
// output tiles [t_start, t_start + n_tiles). The residual tiles are read at once; the projection's tiles once its
// writers have stored them and bumped this core's semaphore once per tile.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_lhs = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rhs = get_compile_time_arg_val(1);
    constexpr uint32_t ep_sem = get_compile_time_arg_val(2);
    constexpr auto lhs_args = TensorAccessorArgs<3>();
    constexpr auto rhs_args = TensorAccessorArgs<lhs_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = 32 * 32 * 2;

    const uint32_t t_start = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);
    const auto lhs = TensorAccessor(lhs_args, get_common_arg_val<uint32_t>(0));
    const auto rhs = TensorAccessor(rhs_args, get_common_arg_val<uint32_t>(1));

    cb_reserve_back(cb_lhs, n_tiles);
    uint32_t l1 = get_write_ptr(cb_lhs);
    for (uint32_t t = t_start; t < t_start + n_tiles; ++t) {
        noc_async_read_page(t, lhs, l1);
        l1 += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_lhs, n_tiles);

    auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ep_sem));
    noc_semaphore_wait(sem, n_tiles);
    noc_semaphore_set(sem, 0);
    cb_reserve_back(cb_rhs, n_tiles);
    l1 = get_write_ptr(cb_rhs);
    for (uint32_t t = t_start; t < t_start + n_tiles; ++t) {
        noc_async_read_page(t, rhs, l1);
        l1 += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_rhs, n_tiles);
}

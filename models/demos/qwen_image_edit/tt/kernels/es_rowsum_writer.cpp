// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// exp_split's writer in its rowsum mode: hi (c_17) and lo (c_18) tile by tile to the same pages as the input, and
// each (tile row, cw-column chunk) item's partial row-sum tile (c_16, pushed when the item ends) to page
// row * S + chunk of p. Runtime: hi, lo, p addresses, tile count, first tile, Nt. Compile-time: cw, S, then the
// hi, lo, p accessor args.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t hi_addr = get_arg_val<uint32_t>(0);
    const uint32_t lo_addr = get_arg_val<uint32_t>(1);
    const uint32_t p_addr = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    const uint32_t start = get_arg_val<uint32_t>(4);
    const uint32_t nt = get_arg_val<uint32_t>(5);
    constexpr uint32_t cw = get_compile_time_arg_val(0);
    constexpr uint32_t n_chunks = get_compile_time_arg_val(1);
    constexpr auto hi_args = TensorAccessorArgs<2>();
    constexpr auto lo_args = TensorAccessorArgs<hi_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<lo_args.next_compile_time_args_offset()>();
    const auto s_hi = TensorAccessor(hi_args, hi_addr);
    const auto s_lo = TensorAccessor(lo_args, lo_addr);
    const auto s_p = TensorAccessor(p_args, p_addr);

    constexpr uint32_t cb_p = 16, cb_hi = 17, cb_lo = 18;
    const uint32_t b_half = get_local_cb_interface(cb_hi).fifo_page_size;
    const uint32_t b_p = get_local_cb_interface(cb_p).fifo_page_size;

    for (uint32_t i = start; i < start + count; ++i) {
        const uint32_t row = i / nt, c = i - row * nt;
        cb_wait_front(cb_hi, 1);
        cb_wait_front(cb_lo, 1);
        noc_async_write(get_read_ptr(cb_hi), s_hi.get_noc_addr(i), b_half);
        noc_async_write(get_read_ptr(cb_lo), s_lo.get_noc_addr(i), b_half);
        if ((c + 1) % cw == 0 || c + 1 == nt) {
            cb_wait_front(cb_p, 1);
            noc_async_write(get_read_ptr(cb_p), s_p.get_noc_addr(row * n_chunks + c / cw), b_p);
            noc_async_writes_flushed();
            cb_pop_front(cb_p, 1);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_hi, 1);
        cb_pop_front(cb_lo, 1);
    }
    noc_async_write_barrier();
}

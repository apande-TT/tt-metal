// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The activation multicaster of tt/cpp_swiglu_mm.py, on one core outside the compute grid. Every compute core
// needs all of x; read 96 times over the NoC it cost ~35 us of a ~150 us call. Here x is read ONCE, a K block at
// a time, into this core's copy of the x CB, and each block is multicast to the same CB offset on every compute
// core (one or two rectangles), followed by a running block count on their VALID semaphore. Blocks go out in
// the order 0, NB-1, 1, NB-2, ... so the cores walking K forward and those walking it backward both get their
// first blocks first. Writing starts only once every compute core has checked in on READY (its L1 is free).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    // two destination rectangles in NoC coordinates: x0, y0, x1, y1, count (count 0 = unused)
    const uint32_t a_x0 = get_arg_val<uint32_t>(1);
    const uint32_t a_y0 = get_arg_val<uint32_t>(2);
    const uint32_t a_x1 = get_arg_val<uint32_t>(3);
    const uint32_t a_y1 = get_arg_val<uint32_t>(4);
    const uint32_t a_n = get_arg_val<uint32_t>(5);
    const uint32_t b_x0 = get_arg_val<uint32_t>(6);
    const uint32_t b_y0 = get_arg_val<uint32_t>(7);
    const uint32_t b_x1 = get_arg_val<uint32_t>(8);
    const uint32_t b_y1 = get_arg_val<uint32_t>(9);
    const uint32_t b_n = get_arg_val<uint32_t>(10);

    constexpr uint32_t MT = get_compile_time_arg_val(0);
    constexpr uint32_t KB = get_compile_time_arg_val(1);
    constexpr uint32_t NB = get_compile_time_arg_val(2);
    constexpr uint32_t KT = get_compile_time_arg_val(3);
    constexpr auto ax = TensorAccessorArgs<4>();
    const auto sx = TensorAccessor(ax, x_addr);

    constexpr uint32_t cb_x = 0;
    const uint32_t x_bytes = get_tile_size(cb_x);
    const uint32_t blk_bytes = MT * KB * x_bytes;
    const uint32_t base = get_write_ptr(cb_x);

    const uint32_t ready_addr = get_semaphore(0);
    const uint32_t valid_addr = get_semaphore(1);
    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
    volatile tt_l1_ptr uint32_t* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_addr);

    // NoC 1 runs mirrored: a rectangle's corners swap.
    const bool flip = noc_index == 1;
    auto rect_a = [&](uint32_t addr) {
        return flip ? get_noc_multicast_addr(a_x1, a_y1, a_x0, a_y0, addr)
                    : get_noc_multicast_addr(a_x0, a_y0, a_x1, a_y1, addr);
    };
    auto rect_b = [&](uint32_t addr) {
        return flip ? get_noc_multicast_addr(b_x1, b_y1, b_x0, b_y0, addr)
                    : get_noc_multicast_addr(b_x0, b_y0, b_x1, b_y1, addr);
    };

    noc_semaphore_wait(ready, a_n + b_n);
    noc_semaphore_set(ready, 0);
    for (uint32_t s = 0; s < NB; ++s) {
        const uint32_t b = (s % 2 == 0) ? s / 2 : NB - 1 - s / 2;
        const uint32_t dst = base + b * blk_bytes;
        for (uint32_t r = 0; r < MT; ++r) {
            const uint32_t t = r * KT + b * KB;
            for (uint32_t k = 0; k < KB; ++k) {
                noc_async_read_page(t + k, sx, dst + (r * KB + k) * x_bytes);
            }
        }
        noc_async_read_barrier();
        // The VALID count this core multicasts must not change before the previous one left.
        noc_async_writes_flushed();
        *valid = s + 1;
        if (a_n) {
            noc_async_write_multicast(dst, rect_a(dst), blk_bytes, a_n, true);
            noc_semaphore_set_multicast(valid_addr, rect_a(valid_addr), a_n);
        }
        if (b_n) {
            noc_async_write_multicast(dst, rect_b(dst), blk_bytes, b_n, true);
            noc_semaphore_set_multicast(valid_addr, rect_b(valid_addr), b_n);
        }
    }
    noc_async_write_barrier();
}

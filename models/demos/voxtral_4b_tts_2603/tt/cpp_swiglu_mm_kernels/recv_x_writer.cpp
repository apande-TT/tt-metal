// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The activation receiver and output writer of tt/cpp_swiglu_mm.py (multicast form, see send_x.cpp). x arrives
// whole, block b at CB offset b * MT * KB tiles, written by the multicaster; this core checks in on its READY
// semaphore, then hands the compute its blocks in its own K order -- one push per block as soon as the VALID
// count shows that block has landed (the compute indexes block b at its absolute offset and pops x only at the
// end). Then it writes its MT x W/2 gated output tiles, a row at a time.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t n0 = get_arg_val<uint32_t>(1);   // first output column tile
    const uint32_t rev = get_arg_val<uint32_t>(2);  // K blocks last to first
    const uint32_t sx = get_arg_val<uint32_t>(3);   // the multicaster's NoC coordinates
    const uint32_t sy = get_arg_val<uint32_t>(4);

    constexpr uint32_t MT = get_compile_time_arg_val(0);
    constexpr uint32_t KB = get_compile_time_arg_val(1);
    constexpr uint32_t NB = get_compile_time_arg_val(2);
    constexpr uint32_t PW = get_compile_time_arg_val(3);  // output tiles a row (W / 2)
    constexpr uint32_t NT = get_compile_time_arg_val(4);  // output column tiles
    constexpr auto ay = TensorAccessorArgs<5>();
    const auto syy = TensorAccessor(ay, y_addr);

    constexpr uint32_t cb_x = 0;
    constexpr uint32_t cb_y = 16;
    const uint32_t y_bytes = get_tile_size(cb_y);

    volatile tt_l1_ptr uint32_t* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    cb_reserve_back(cb_x, NB * MT * KB);
    noc_semaphore_inc(get_noc_addr(sx, sy, get_semaphore(0)), 1);
    for (uint32_t i = 0; i < NB; ++i) {
        const uint32_t b = rev ? NB - 1 - i : i;
        // send_x's order is 0, NB-1, 1, NB-2, ...
        const uint32_t pos = (b < (NB + 1) / 2) ? 2 * b : 2 * (NB - 1 - b) + 1;
        noc_semaphore_wait_min(valid, pos + 1);
        cb_push_back(cb_x, MT * KB);
    }

    for (uint32_t m = 0; m < MT; ++m) {
        cb_wait_front(cb_y, PW);
        uint32_t l1 = get_read_ptr(cb_y);
        for (uint32_t j = 0; j < PW; ++j) {
            noc_async_write_page(m * NT + n0 + j, syy, l1);
            l1 += y_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_y, PW);
    }
    noc_async_atomic_barrier();
}

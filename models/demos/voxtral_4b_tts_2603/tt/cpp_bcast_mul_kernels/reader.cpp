// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The reader of the C++ column-broadcast multiply (see tt/cpp_bcast_mul.py): y = a * b, b one value a row.
// A core owns a contiguous run of a's tiles and reads it CH tiles a chunk with ONE barrier (the stock
// binary_ng reader waits on every tile), plus the b tiles of the rows that chunk touches. Each b tile is
// column-filled -- every element of a row set to the row's first value, the data binary_ng's reader
// leaves for a float32 column-broadcast operand (fill_tile_with_first_column) -- and pushed as soon as it
// is filled, the first one before the a tiles, so the math starts on row 0 while later rows fill.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// The same data as binary_ng's fill_tile_with_first_column (kernels/dataflow/fill_tile_utils.hpp): for both
// face pairs, each of the 16 face rows takes the value of its first element across the left and the right
// face. The stock loop (volatile, word by word, index arithmetic per store) is what makes every float32
// column-broadcast binary_ng op slow here -- its reader runs ~5 us a filled tile. Straight-line stores, 32 a
// face row; the compiler barrier orders them before the push.
#define FILL4(p, o, v) \
    (p)[(o) + 0] = (v); \
    (p)[(o) + 1] = (v); \
    (p)[(o) + 2] = (v); \
    (p)[(o) + 3] = (v)
#define FILL16(p, v) \
    FILL4(p, 0, v);  \
    FILL4(p, 4, v);  \
    FILL4(p, 8, v);  \
    FILL4(p, 12, v)

FORCE_INLINE void fill_tile_with_first_column(uint32_t l1_write_ptr) {
    uint32_t* ptr = reinterpret_cast<uint32_t*>(l1_write_ptr);
    for (uint32_t r = 0; r < 32; ++r) {
        // face row r: faces 0/1 for r < 16, faces 2/3 after (512 words a face pair, 16 a face row)
        uint32_t* left = ptr + ((r >> 4) << 9) + ((r & 15) << 4);
        uint32_t* right = left + 256;
        const uint32_t v = left[0];
        FILL16(left, v);
        FILL16(right, v);
    }
    asm volatile("" ::: "memory");
}

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t t0 = get_arg_val<uint32_t>(2);
    const uint32_t nt = get_arg_val<uint32_t>(3);

    constexpr uint32_t WT = get_compile_time_arg_val(0);   // a's tiles a row
    constexpr uint32_t CH = get_compile_time_arg_val(1);   // a tiles a chunk
    constexpr uint32_t NBR = get_compile_time_arg_val(2);  // b slots a chunk (rows a chunk can touch)
    constexpr auto aa = TensorAccessorArgs<3>();
    constexpr auto ab = TensorAccessorArgs<aa.next_compile_time_args_offset()>();
    const auto sa = TensorAccessor(aa, a_addr);
    const auto sb = TensorAccessor(ab, b_addr);

    constexpr uint32_t cb_a = 0;
    constexpr uint32_t cb_b = 1;
    const uint32_t a_bytes = get_tile_size(cb_a);
    const uint32_t b_bytes = get_tile_size(cb_b);

    // Every chunk pushes CH a tiles and NBR b tiles (the tail of a short chunk unused), so both rings wrap on
    // chunk boundaries.
    for (uint32_t c0 = 0; c0 < nt; c0 += CH) {
        const uint32_t n = (nt - c0 < CH) ? (nt - c0) : CH;
        const uint32_t g0 = t0 + c0;
        const uint32_t r0 = g0 / WT;
        const uint32_t nr = (g0 + n - 1) / WT - r0 + 1;
        cb_reserve_back(cb_b, NBR);
        cb_reserve_back(cb_a, CH);
        const uint32_t lb0 = get_write_ptr(cb_b);
        uint32_t lb = lb0;
        for (uint32_t r = 0; r < nr; ++r) {
            noc_async_read_page(r0 + r, sb, lb);
            lb += b_bytes;
        }
        uint32_t la = get_write_ptr(cb_a);
        for (uint32_t j = 0; j < n; ++j) {
            noc_async_read_page(g0 + j, sa, la);
            la += a_bytes;
        }
        noc_async_read_barrier();
        fill_tile_with_first_column(lb0);
        cb_push_back(cb_b, 1);
        cb_push_back(cb_a, CH);
        lb = lb0;
        for (uint32_t r = 1; r < nr; ++r) {
            lb += b_bytes;
            fill_tile_with_first_column(lb);
            cb_push_back(cb_b, 1);
        }
        if (NBR > nr) {
            cb_push_back(cb_b, NBR - nr);
        }
    }
}

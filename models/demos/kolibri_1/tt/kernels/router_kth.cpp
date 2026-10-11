// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// MoE router selection: per token row, the K-th largest of its E fp32 scores (counting repeats, as ttnn.topk's
// K-th sorted value does), then the row of the [N, E] output = 1.0 where score >= that threshold, else 0.0 (what
// ttnn.ge(scores, threshold) gave). Rows are dealt round-robin over the cores. Reads and writes a row as two 64 B
// face rows per tile; fp32 values are compared through an order-preserving integer key (-0 taken as +0, so key
// order is float order).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

inline uint32_t order_key(uint32_t bits) { return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u); }
inline uint32_t float_key(uint32_t bits) { return order_key(bits == 0x80000000u ? 0u : bits); }

void kernel_main() {
    constexpr uint32_t ET = get_compile_time_arg_val(0);  // tiles across the E scores
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t N = get_compile_time_arg_val(2);   // token rows
    constexpr uint32_t NC = get_compile_time_arg_val(3);  // cores
    constexpr auto s_args = TensorAccessorArgs<4>();
    constexpr auto o_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    const auto s = TensorAccessor(s_args, get_arg_val<uint32_t>(0));
    const auto out = TensorAccessor(o_args, get_arg_val<uint32_t>(1));
    const uint32_t core_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_row = 0;
    constexpr uint32_t face = 1024, frow = 64;  // fp32 tile geometry

    cb_reserve_back(cb_row, 1);
    const uint32_t l1 = (get_write_ptr(cb_row) + 63) & ~63u;  // DRAM reads need a 64 B-aligned landing address
    const uint32_t w_l1 = l1 + ET * 2 * frow;  // the row's output, same column order
    // The scores land before the read barrier (which also drops stale cache lines): plain, cacheable reads.
    const uint32_t* v = reinterpret_cast<const uint32_t*>(l1);
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w_l1);
    constexpr uint32_t ONE = 0x3f800000u;  // 1.0f

    // w stays all zeros between rows except the ones the last row set (cleared once its writes have left).
    for (uint32_t c = 0; c < ET * 32; ++c) {
        w[c] = 0;
    }
    uint32_t set_pos[K];
    bool set_all = false;
    for (uint32_t i = 0; i < K; ++i) {
        set_pos[i] = 0;
    }

    for (uint32_t b = core_id; b < N; b += NC) {
        const uint32_t tile_row = b / 32, r = b % 32;
        const uint32_t off = (r / 16) * 2 * face + (r % 16) * frow;
        for (uint32_t t = 0; t < ET; ++t) {
            noc_async_read(s.get_noc_addr(tile_row * ET + t, off), l1 + t * 2 * frow, frow);
            noc_async_read(s.get_noc_addr(tile_row * ET + t, off + face), l1 + t * 2 * frow + frow, frow);
        }
        noc_async_read_barrier();
        asm volatile("" ::: "memory");  // v changed under the compiler: reload it

        // top[0 .. K-1]: the K largest keys so far, descending, and their columns. The row's mask is every column with
        // key >= top[K - 1] at the end: the listed ones, plus any other column EQUAL to it -- which can only be one
        // skipped while equal to the list's minimum, or dropped off the list's end while equal to the new minimum (the
        // minimum never falls). `tie` records either event; without one the list is exactly the mask.
        uint32_t top[K], pos[K];
        for (uint32_t i = 0; i < K; ++i) {
            top[i] = 0;
            pos[i] = 0;
        }
        bool tie = false;
        for (uint32_t c = 0; c < ET * 32; ++c) {
            const uint32_t key = float_key(v[c]);
            if (key <= top[K - 1]) {
                tie |= key == top[K - 1];
                continue;
            }
            const uint32_t dropped = top[K - 1];
            uint32_t i = K - 1;
            while (i > 0 && top[i - 1] < key) {
                top[i] = top[i - 1];
                pos[i] = pos[i - 1];
                --i;
            }
            top[i] = key;
            pos[i] = c;
            tie |= dropped != 0 && dropped == top[K - 1];
        }
        const uint32_t kth = top[K - 1];

        noc_async_writes_flushed();  // the previous row's writes have left w
        if (set_all) {
            for (uint32_t c = 0; c < ET * 32; ++c) {
                w[c] = 0;
            }
        } else {
            for (uint32_t i = 0; i < K; ++i) {
                w[set_pos[i]] = 0;
            }
        }
        set_all = tie;
        if (tie) {
            for (uint32_t c = 0; c < ET * 32; ++c) {
                w[c] = static_cast<uint32_t>(float_key(v[c]) >= kth) * ONE;
            }
        } else {
            for (uint32_t i = 0; i < K; ++i) {
                w[pos[i]] = ONE;
                set_pos[i] = pos[i];
            }
        }
        for (uint32_t t = 0; t < ET; ++t) {
            noc_async_write(w_l1 + t * 2 * frow, out.get_noc_addr(tile_row * ET + t, off), frow);
            noc_async_write(w_l1 + t * 2 * frow + frow, out.get_noc_addr(tile_row * ET + t, off + face), frow);
        }
    }
    noc_async_write_barrier();
}

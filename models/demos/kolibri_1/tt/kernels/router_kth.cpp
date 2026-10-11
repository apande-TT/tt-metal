// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// MoE router selection threshold: per token row, the K-th largest of its E fp32 scores (counting repeats, as
// ttnn.topk's K-th sorted value does), written to column 0 of the row of the [N, 1] output. Rows are dealt
// round-robin over the cores. Reads a row as two 64 B face rows per tile; fp32 values are compared through an
// order-preserving integer key.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

inline uint32_t order_key(uint32_t bits) { return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u); }
inline uint32_t key_bits(uint32_t key) { return (key & 0x80000000u) ? (key & 0x7fffffffu) : ~key; }

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
    const uint32_t w_l1 = l1 + ET * 2 * frow;
    volatile tt_l1_ptr uint32_t* v = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w_l1);

    for (uint32_t b = core_id; b < N; b += NC) {
        const uint32_t tile_row = b / 32, r = b % 32;
        const uint32_t off = (r / 16) * 2 * face + (r % 16) * frow;
        for (uint32_t t = 0; t < ET; ++t) {
            noc_async_read(s.get_noc_addr(tile_row * ET + t, off), l1 + t * 2 * frow, frow);
            noc_async_read(s.get_noc_addr(tile_row * ET + t, off + face), l1 + t * 2 * frow + frow, frow);
        }
        noc_async_read_barrier();

        // top[0 .. K-1]: the K largest keys so far, descending.
        uint32_t top[K];
        for (uint32_t i = 0; i < K; ++i) {
            top[i] = 0;
        }
        for (uint32_t c = 0; c < ET * 32; ++c) {
            uint32_t key = order_key(v[c]);
            if (key <= top[K - 1]) {
                continue;
            }
            uint32_t i = K - 1;
            while (i > 0 && top[i - 1] < key) {
                top[i] = top[i - 1];
                --i;
            }
            top[i] = key;
        }

        w[0] = key_bits(top[K - 1]);
        w[1] = 0;
        w[2] = 0;
        w[3] = 0;
        noc_async_write(w_l1, out.get_noc_addr(tile_row, off), 16);
        noc_async_write_barrier();
    }
}

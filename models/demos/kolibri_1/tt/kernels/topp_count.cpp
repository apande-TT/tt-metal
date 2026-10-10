// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Sampler top_p count, one core per user row b: n_keep[b] = #{k : sum_{j<k} p[b, j] < top_p} over the descending
// top-k probabilities, the exclusive sums accumulated one by one in fp32 (as exact as ttnn.cumsum, which ran this as
// a 128-step one-core tile walk; the 0/1-triangle matmul scan was off by up to 8.6e-4 and flipped the cut).
// Reads the user's row as two 64 B face rows per tile, writes n_keep to column 0 of its row of the [B, 1] output.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);          // tiles across the k candidates
    constexpr uint32_t TOP_P_BITS = get_compile_time_arg_val(1);  // top_p as fp32 bits
    constexpr auto p_args = TensorAccessorArgs<2>();
    constexpr auto o_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    const auto p = TensorAccessor(p_args, get_arg_val<uint32_t>(0));
    const auto out = TensorAccessor(o_args, get_arg_val<uint32_t>(1));
    const uint32_t b = get_arg_val<uint32_t>(2);  // the user row this core owns

    constexpr uint32_t cb_row = 0;
    constexpr uint32_t face = 1024, frow = 64;  // fp32 tile geometry

    cb_reserve_back(cb_row, 1);
    const uint32_t l1 = (get_write_ptr(cb_row) + 63) & ~63u;  // DRAM reads need a 64 B-aligned landing address
    const uint32_t tile_row = b / 32, r = b % 32;
    const uint32_t off = (r / 16) * 2 * face + (r % 16) * frow;
    for (uint32_t t = 0; t < KT; ++t) {
        noc_async_read(p.get_noc_addr(tile_row * KT + t, off), l1 + t * 2 * frow, frow);
        noc_async_read(p.get_noc_addr(tile_row * KT + t, off + face), l1 + t * 2 * frow + frow, frow);
    }
    noc_async_read_barrier();

    volatile tt_l1_ptr float* v = reinterpret_cast<volatile tt_l1_ptr float*>(l1);
    union {
        uint32_t u;
        float f;
    } top_p{TOP_P_BITS};
    float excl = 0.0f;
    uint32_t n = 0;
    for (uint32_t k = 0; k < KT * 32 && excl < top_p.f; ++k) {
        ++n;
        excl += v[k];
    }

    const uint32_t w_l1 = l1 + KT * 2 * frow;
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w_l1);
    union {
        float f;
        uint32_t u;
    } nf{static_cast<float>(n)};
    w[0] = nf.u;
    w[1] = 0;
    w[2] = 0;
    w[3] = 0;
    noc_async_write(w_l1, out.get_noc_addr(tile_row, off), 16);
    noc_async_write_barrier();
}

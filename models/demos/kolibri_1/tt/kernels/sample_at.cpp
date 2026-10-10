// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Sampler inverse CDF, one core per user row b: q[b, :] are the kept candidates' weights in token-id order (0 where
// dropped); z = their sum and at = #{j : q[b, 0] + ... + q[b, j] <= u[b] * z}, the cumulative sums taken one by one
// in fp32 (host_sample does the same cumsum in fp64; the 0/1-triangle matmul scan this replaces was off by up to
// 2.4e-4, enough to land u on the wrong side of a CDF edge). Writes `at` (fp32) to column 0 of row b of [B, 1].
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);  // tiles across the k candidates
    constexpr auto q_args = TensorAccessorArgs<1>();
    constexpr auto u_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<u_args.next_compile_time_args_offset()>();
    const auto q = TensorAccessor(q_args, get_arg_val<uint32_t>(0));
    const auto uu = TensorAccessor(u_args, get_arg_val<uint32_t>(1));
    const auto out = TensorAccessor(o_args, get_arg_val<uint32_t>(2));
    const uint32_t b = get_arg_val<uint32_t>(3);  // the user row this core owns

    constexpr uint32_t cb_row = 0;
    constexpr uint32_t face = 1024, frow = 64;  // fp32 tile geometry

    cb_reserve_back(cb_row, 1);
    const uint32_t l1 = (get_write_ptr(cb_row) + 63) & ~63u;  // DRAM reads need a 64 B-aligned landing address
    const uint32_t u_l1 = l1 + KT * 2 * frow;
    const uint32_t w_l1 = u_l1 + frow;
    const uint32_t tile_row = b / 32, r = b % 32;
    const uint32_t off = (r / 16) * 2 * face + (r % 16) * frow;
    for (uint32_t t = 0; t < KT; ++t) {
        noc_async_read(q.get_noc_addr(tile_row * KT + t, off), l1 + t * 2 * frow, frow);
        noc_async_read(q.get_noc_addr(tile_row * KT + t, off + face), l1 + t * 2 * frow + frow, frow);
    }
    noc_async_read(uu.get_noc_addr(tile_row, off), u_l1, frow);  // column 0 of row b
    noc_async_read_barrier();

    volatile tt_l1_ptr float* v = reinterpret_cast<volatile tt_l1_ptr float*>(l1);
    const float u = reinterpret_cast<volatile tt_l1_ptr float*>(u_l1)[0];
    float z = 0.0f;
    for (uint32_t k = 0; k < KT * 32; ++k) {
        z += v[k];
    }
    const float thr = u * z;
    float c = 0.0f;
    uint32_t at = 0;
    for (uint32_t k = 0; k < KT * 32; ++k) {
        c += v[k];
        at += c <= thr ? 1u : 0u;
    }

    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w_l1);
    union {
        float f;
        uint32_t u;
    } af{static_cast<float>(at)};
    w[0] = af.u;
    w[1] = 0;
    w[2] = 0;
    w[3] = 0;
    noc_async_write(w_l1, out.get_noc_addr(tile_row, off), 16);
    noc_async_write_barrier();
}

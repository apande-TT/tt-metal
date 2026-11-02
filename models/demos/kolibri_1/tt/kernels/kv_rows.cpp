// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Served prefill: the K and V of N prompts run as one batch, [N, HKV, RT, W] tiles each (RT = padded prompt
// length / 32, W = head_dim / 32), into their own users' cache rows [B, HKV, CT, W] (CT = KV capacity / 32):
// prompt b's tiles go to user slot[b] (runtime arg 6 + b), positions [0, RT * 32). Whole-tile copies (same dtype
// and layout), the 2 N HKV RT W tiles (all of K, then all of V) dealt in contiguous ranges over the cores; every
// other user's rows are untouched.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t HKV = get_compile_time_arg_val(1);
    constexpr uint32_t RT = get_compile_time_arg_val(2);
    constexpr uint32_t W = get_compile_time_arg_val(3);
    constexpr uint32_t CT = get_compile_time_arg_val(4);
    constexpr uint32_t TILE = get_compile_time_arg_val(5);  // bytes
    constexpr uint32_t SLOTS = get_compile_time_arg_val(6);
    constexpr auto a0 = TensorAccessorArgs<7>();
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    constexpr auto a3 = TensorAccessorArgs<a2.next_compile_time_args_offset()>();
    const auto k = TensorAccessor(a0, get_arg_val<uint32_t>(0));
    const auto v = TensorAccessor(a1, get_arg_val<uint32_t>(1));
    const auto kc = TensorAccessor(a2, get_arg_val<uint32_t>(2));
    const auto vc = TensorAccessor(a3, get_arg_val<uint32_t>(3));
    const uint32_t start = get_arg_val<uint32_t>(4), end = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb = 0;
    constexpr uint32_t PER = N * HKV * RT * W;  // tiles in K (and in V)
    cb_reserve_back(cb, SLOTS);
    const uint32_t l1 = get_write_ptr(cb);

    for (uint32_t g0 = start; g0 < end; g0 += SLOTS) {
        const uint32_t n = end - g0 < SLOTS ? end - g0 : SLOTS;
        uint32_t dst_page[SLOTS];
        bool is_v[SLOTS];
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t g = g0 + i;
            const bool sel = g >= PER;
            const uint32_t t = sel ? g - PER : g;
            const uint32_t c = t % W, r = (t / W) % RT, h = (t / (W * RT)) % HKV, b = t / (W * RT * HKV);
            const uint32_t slot = get_arg_val<uint32_t>(6 + b);
            dst_page[i] = ((slot * HKV + h) * CT + r) * W + c;
            is_v[i] = sel;
            noc_async_read(sel ? v.get_noc_addr(t) : k.get_noc_addr(t), l1 + i * TILE, TILE);
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t a = l1 + i * TILE;
            noc_async_write(a, is_v[i] ? vc.get_noc_addr(dst_page[i]) : kc.get_noc_addr(dst_page[i]), TILE);
        }
        noc_async_write_barrier();
    }
}

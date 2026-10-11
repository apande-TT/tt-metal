// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Prefill head split / merge as whole-tile copies (DRAM -> DRAM), the output tiles dealt in contiguous ranges over
// the cores. In the fused projection [B, 1, T, (HQ + 2 HKV) * D] the heads are column blocks of W = D / 32 tiles
// ([q heads | k heads | v heads]), so with Rt = T / 32 tile rows per user:
//   split (MODE 0): q[b, h, r, c] = x[b, r, h W + c]; k and v likewise from column blocks HQ + h and HQ + HKV + h
//                   (what ttnn.experimental.nlp_create_qkv_heads gives, which ran one core per tile row);
//   merge (MODE 1): y[b, r, h W + c] = a[b, h, r, c] over the HQ heads (nlp_concat_heads).
// A tile copy moves the bytes unchanged.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t MODE = get_compile_time_arg_val(0);
    constexpr uint32_t B = get_compile_time_arg_val(1);
    constexpr uint32_t RT = get_compile_time_arg_val(2);   // tile rows per user
    constexpr uint32_t W = get_compile_time_arg_val(3);    // tiles across a head
    constexpr uint32_t HQ = get_compile_time_arg_val(4);
    constexpr uint32_t HKV = get_compile_time_arg_val(5);  // 0 for MODE 1
    constexpr uint32_t TILE = get_compile_time_arg_val(6);  // bytes
    constexpr uint32_t SLOTS = get_compile_time_arg_val(7);
    constexpr auto a0 = TensorAccessorArgs<8>();
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    constexpr auto a3 = TensorAccessorArgs<a2.next_compile_time_args_offset()>();
    const auto src = TensorAccessor(a0, get_arg_val<uint32_t>(0));
    const auto d0 = TensorAccessor(a1, get_arg_val<uint32_t>(1));  // q (split) / y (merge)
    const auto d1 = TensorAccessor(a2, get_arg_val<uint32_t>(2));  // k (split)
    const auto d2 = TensorAccessor(a3, get_arg_val<uint32_t>(3));  // v (split)
    const uint32_t start = get_arg_val<uint32_t>(4), end = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb = 0;
    constexpr uint32_t HT = HQ + 2 * HKV;  // head column blocks in the fused row (split)
    constexpr uint32_t PER_Q = HQ * RT * W, PER_KV = HKV * RT * W;  // output tiles per user
    cb_reserve_back(cb, SLOTS);
    const uint32_t l1 = get_write_ptr(cb);

    for (uint32_t g0 = start; g0 < end; g0 += SLOTS) {
        const uint32_t n = end - g0 < SLOTS ? end - g0 : SLOTS;
        uint32_t dst_page[SLOTS], which[SLOTS];
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t g = g0 + i;
            uint32_t src_page;
            if constexpr (MODE == 0) {
                // Output order: all of q, then all of k, then all of v; each [B, heads, RT, W] tiles.
                uint32_t t = g, base = 0, heads = HQ, sel = 0;
                if (t >= B * PER_Q) {
                    t -= B * PER_Q;
                    base = HQ;
                    heads = HKV;
                    sel = 1;
                    if (t >= B * PER_KV) {
                        t -= B * PER_KV;
                        base = HQ + HKV;
                        sel = 2;
                    }
                }
                const uint32_t c = t % W, r = (t / W) % RT, h = (t / (W * RT)) % heads, b = t / (W * RT * heads);
                src_page = (b * RT + r) * (HT * W) + (base + h) * W + c;
                dst_page[i] = t;
                which[i] = sel;
            } else {
                const uint32_t c = g % W, h = (g / W) % HQ, r = (g / (W * HQ)) % RT, b = g / (W * HQ * RT);
                src_page = ((b * HQ + h) * RT + r) * W + c;
                dst_page[i] = g;
                which[i] = 0;
            }
            noc_async_read(src.get_noc_addr(src_page), l1 + i * TILE, TILE);
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t a = l1 + i * TILE;
            if (which[i] == 0) {
                noc_async_write(a, d0.get_noc_addr(dst_page[i]), TILE);
            } else if (which[i] == 1) {
                noc_async_write(a, d1.get_noc_addr(dst_page[i]), TILE);
            } else {
                noc_async_write(a, d2.get_noc_addr(dst_page[i]), TILE);
            }
        }
        noc_async_write_barrier();
    }
}

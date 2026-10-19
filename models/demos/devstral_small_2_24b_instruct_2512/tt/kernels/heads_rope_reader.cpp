// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of tt/heads_rope.py: for each of this core's (tile row, head) units the head's q and k tiles of the fused
// QKV row (to the compute kernel) and its v tiles (straight to the writer), plus the cos / signed-sin tiles of the
// unit's sequence tile whenever that changes.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t qkv_addr = get_arg_val<uint32_t>(0);
    const uint32_t cos_addr = get_arg_val<uint32_t>(1);
    const uint32_t sin_addr = get_arg_val<uint32_t>(2);
    const uint32_t u0 = get_arg_val<uint32_t>(3);  // first unit (tile row * H + head) this core owns
    const uint32_t nu = get_arg_val<uint32_t>(4);  // units this core owns

    constexpr uint32_t H = get_compile_time_arg_val(0);   // heads
    constexpr uint32_t HT = get_compile_time_arg_val(1);  // tiles a head
    constexpr uint32_t ST = get_compile_time_arg_val(2);  // sequence tiles an image
    constexpr auto aqkv = TensorAccessorArgs<3>();
    constexpr auto acos = TensorAccessorArgs<aqkv.next_compile_time_args_offset()>();
    constexpr auto asin = TensorAccessorArgs<acos.next_compile_time_args_offset()>();
    const auto sqkv = TensorAccessor(aqkv, qkv_addr);
    const auto scos = TensorAccessor(acos, cos_addr);
    const auto ssin = TensorAccessor(asin, sin_addr);

    constexpr uint32_t C = 3 * H * HT;  // tiles a QKV row: [q heads | k heads | v heads]
    constexpr uint32_t cb_x = 0;        // q tiles then k tiles of the unit
    constexpr uint32_t cb_cos = 1;
    constexpr uint32_t cb_sin = 2;
    constexpr uint32_t cb_v = 3;
    const uint32_t tb = get_tile_size(cb_x);

    uint32_t cur_st = 0xFFFFFFFF;
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t r = u / H;
        const uint32_t h = u % H;
        const uint32_t st = r % ST;
        if (st != cur_st) {
            cb_reserve_back(cb_cos, HT);
            cb_reserve_back(cb_sin, HT);
            const uint32_t lc = get_write_ptr(cb_cos);
            const uint32_t ls = get_write_ptr(cb_sin);
            for (uint32_t j = 0; j < HT; ++j) {
                noc_async_read_page(st * HT + j, scos, lc + j * tb);
                noc_async_read_page(st * HT + j, ssin, ls + j * tb);
            }
            noc_async_read_barrier();
            cb_push_back(cb_cos, HT);
            cb_push_back(cb_sin, HT);
            cur_st = st;
        }
        cb_reserve_back(cb_x, 2 * HT);
        cb_reserve_back(cb_v, HT);
        const uint32_t lx = get_write_ptr(cb_x);
        const uint32_t lv = get_write_ptr(cb_v);
        const uint32_t row = r * C;
        for (uint32_t j = 0; j < HT; ++j) {
            noc_async_read_page(row + h * HT + j, sqkv, lx + j * tb);
            noc_async_read_page(row + (H + h) * HT + j, sqkv, lx + (HT + j) * tb);
            noc_async_read_page(row + (2 * H + h) * HT + j, sqkv, lv + j * tb);
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, 2 * HT);
        cb_push_back(cb_v, HT);
    }
}

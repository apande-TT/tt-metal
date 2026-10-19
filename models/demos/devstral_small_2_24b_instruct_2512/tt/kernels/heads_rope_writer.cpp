// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer of tt/heads_rope.py: each unit's rotated q and k tiles (from the compute kernel) and its v tiles (from the
// reader) into the (B, H, S, hd) head tensors.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t u0 = get_arg_val<uint32_t>(3);
    const uint32_t nu = get_arg_val<uint32_t>(4);

    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t HT = get_compile_time_arg_val(1);
    constexpr uint32_t ST = get_compile_time_arg_val(2);
    constexpr auto aq = TensorAccessorArgs<3>();
    constexpr auto ak = TensorAccessorArgs<aq.next_compile_time_args_offset()>();
    constexpr auto av = TensorAccessorArgs<ak.next_compile_time_args_offset()>();
    const auto sq = TensorAccessor(aq, q_addr);
    const auto sk = TensorAccessor(ak, k_addr);
    const auto sv = TensorAccessor(av, v_addr);

    constexpr uint32_t cb_v = 3;
    constexpr uint32_t cb_out = 16;
    const uint32_t tb = get_tile_size(cb_out);

    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t r = u / H;
        const uint32_t h = u % H;
        const uint32_t base = (((r / ST) * H + h) * ST + r % ST) * HT;
        cb_wait_front(cb_out, 2 * HT);
        cb_wait_front(cb_v, HT);
        const uint32_t lo = get_read_ptr(cb_out);
        const uint32_t lv = get_read_ptr(cb_v);
        for (uint32_t j = 0; j < HT; ++j) {
            noc_async_write_page(base + j, sq, lo + j * tb);
            noc_async_write_page(base + j, sk, lo + (HT + j) * tb);
            noc_async_write_page(base + j, sv, lv + j * tb);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, 2 * HT);
        cb_pop_front(cb_v, HT);
    }
}

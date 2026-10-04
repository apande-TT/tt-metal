// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The output stream of the C++ prefill k/v join (see tt/cpp_kv_join.py): unit u's WT tiles to joined tile row
// (h, r) of the k (s = 0) or v (s = 1) output [1, H, P + T, D].

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t jk_addr = get_arg_val<uint32_t>(0);
    const uint32_t jv_addr = get_arg_val<uint32_t>(1);
    const uint32_t u0 = get_arg_val<uint32_t>(2);
    const uint32_t nu = get_arg_val<uint32_t>(3);

    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t PT = get_compile_time_arg_val(1);
    constexpr uint32_t TT = get_compile_time_arg_val(2);
    constexpr uint32_t WT = get_compile_time_arg_val(3);
    constexpr uint32_t page = get_compile_time_arg_val(4);
    constexpr auto ajk = TensorAccessorArgs<5>();
    constexpr auto ajv = TensorAccessorArgs<ajk.next_compile_time_args_offset()>();
    const auto sjk = TensorAccessor(ajk, jk_addr);
    const auto sjv = TensorAccessor(ajv, jv_addr);
    constexpr uint32_t OT = PT + TT;

    constexpr uint32_t cb = 0;

    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t s = u / (H * OT);
        const uint32_t rem = u % (H * OT);
        cb_wait_front(cb, WT);
        uint32_t src = get_read_ptr(cb);
        for (uint32_t w = 0; w < WT; ++w) {
            if (s) {
                noc_async_write_page(rem * WT + w, sjv, src);
            } else {
                noc_async_write_page(rem * WT + w, sjk, src);
            }
            src += page;
        }
        noc_async_write_barrier();
        cb_pop_front(cb, WT);
    }
}

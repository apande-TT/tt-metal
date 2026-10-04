// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The input stream of the C++ prefill k/v join (see tt/cpp_kv_join.py). Unit u = (operand s: 0 = k, 1 = v,
// head h, joined tile row r): rows r < PT come from the shared prefix [1, H, P, D], the rest from the compact
// tail [1, H, T, D]. Each unit's WT tiles are read into the CB in one batch.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t pk_addr = get_arg_val<uint32_t>(0);
    const uint32_t tk_addr = get_arg_val<uint32_t>(1);
    const uint32_t pv_addr = get_arg_val<uint32_t>(2);
    const uint32_t tv_addr = get_arg_val<uint32_t>(3);
    const uint32_t u0 = get_arg_val<uint32_t>(4);
    const uint32_t nu = get_arg_val<uint32_t>(5);

    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t PT = get_compile_time_arg_val(1);  // prefix tile rows a head
    constexpr uint32_t TT = get_compile_time_arg_val(2);  // tail tile rows a head
    constexpr uint32_t WT = get_compile_time_arg_val(3);  // tiles a row
    constexpr uint32_t page = get_compile_time_arg_val(4);  // tile bytes
    constexpr auto apk = TensorAccessorArgs<5>();
    constexpr auto atk = TensorAccessorArgs<apk.next_compile_time_args_offset()>();
    constexpr auto apv = TensorAccessorArgs<atk.next_compile_time_args_offset()>();
    constexpr auto atv = TensorAccessorArgs<apv.next_compile_time_args_offset()>();
    const auto spk = TensorAccessor(apk, pk_addr);
    const auto stk = TensorAccessor(atk, tk_addr);
    const auto spv = TensorAccessor(apv, pv_addr);
    const auto stv = TensorAccessor(atv, tv_addr);
    constexpr uint32_t OT = PT + TT;

    constexpr uint32_t cb = 0;

    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t s = u / (H * OT);
        const uint32_t rem = u % (H * OT);
        const uint32_t h = rem / OT;
        const uint32_t r = rem % OT;
        cb_reserve_back(cb, WT);
        uint32_t dst = get_write_ptr(cb);
        for (uint32_t w = 0; w < WT; ++w) {
            if (r < PT) {
                const uint32_t p = (h * PT + r) * WT + w;
                if (s) {
                    noc_async_read_page(p, spv, dst);
                } else {
                    noc_async_read_page(p, spk, dst);
                }
            } else {
                const uint32_t p = (h * TT + r - PT) * WT + w;
                if (s) {
                    noc_async_read_page(p, stv, dst);
                } else {
                    noc_async_read_page(p, stk, dst);
                }
            }
            dst += page;
        }
        noc_async_read_barrier();
        cb_push_back(cb, WT);
    }
}

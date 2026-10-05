// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The output stream of tt/cpp_qkv_rope.py: the RoPE'd q / k tile rows (from the stock compute kernel) to
// q `[1, NQ, S, D]` / k `[1, NKV, S, D]`, and the passed-through v tile rows to v `[1, NKV, S, D]`.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t u0 = get_arg_val<uint32_t>(3);
    const uint32_t nqk = get_arg_val<uint32_t>(4);
    const uint32_t w0 = get_arg_val<uint32_t>(5);
    const uint32_t nv = get_arg_val<uint32_t>(6);

    constexpr uint32_t WT = get_compile_time_arg_val(0);
    constexpr uint32_t NQ = get_compile_time_arg_val(1);
    constexpr uint32_t NKV = get_compile_time_arg_val(2);
    constexpr uint32_t RT = get_compile_time_arg_val(3);  // tile rows of the sequence
    constexpr auto aq = TensorAccessorArgs<4>();
    constexpr auto ak = TensorAccessorArgs<aq.next_compile_time_args_offset()>();
    constexpr auto av = TensorAccessorArgs<ak.next_compile_time_args_offset()>();
    const auto sq = TensorAccessor(aq, q_addr);
    const auto sk = TensorAccessor(ak, k_addr);
    const auto sv = TensorAccessor(av, v_addr);

    constexpr uint32_t cb_out = 16;
    constexpr uint32_t cb_v = 17;
    const uint32_t bytes = get_tile_size(cb_out);

    for (uint32_t u = u0; u < u0 + nqk; ++u) {
        const uint32_t r = u / (NQ + NKV);
        const uint32_t hh = u % (NQ + NKV);
        cb_wait_front(cb_out, WT);
        uint32_t l1 = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < WT; ++j) {
            if (hh < NQ) {
                noc_async_write_page((hh * RT + r) * WT + j, sq, l1);
            } else {
                noc_async_write_page(((hh - NQ) * RT + r) * WT + j, sk, l1);
            }
            l1 += bytes;
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_out, WT);
    }
    for (uint32_t w = w0; w < w0 + nv; ++w) {
        const uint32_t r = w / NKV;
        const uint32_t hv = w % NKV;
        cb_wait_front(cb_v, WT);
        uint32_t l1 = get_read_ptr(cb_v);
        for (uint32_t j = 0; j < WT; ++j) {
            noc_async_write_page((hv * RT + r) * WT + j, sv, l1);
            l1 += bytes;
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_v, WT);
    }
    noc_async_write_barrier();
}

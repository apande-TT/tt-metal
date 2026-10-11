// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode down projection over the ROUTED experts only, phase 2 (reader). Output column tiles are dealt round-robin
// over the cores; per tile: the shared expert's bf16 partial tile, then the R routed experts' fp32 partial tiles
// (planes 0 .. R-1, in batches of B).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t E = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t B = get_compile_time_arg_val(3);
    constexpr auto p_args = TensorAccessorArgs<4>();
    constexpr auto s_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto m_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();

    const auto planes = TensorAccessor(p_args, get_arg_val<uint32_t>(0));
    const auto shared = TensorAccessor(s_args, get_arg_val<uint32_t>(1));
    const auto mask_acc = TensorAccessor(m_args, get_arg_val<uint32_t>(2));
    const uint32_t core_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_p = 0, cb_s = 1, cb_mask = 3, cb_count = 4;
    constexpr uint32_t p_bytes = 4096, s_bytes = 2048;

    cb_reserve_back(cb_mask, 1);
    const uint32_t mask_l1 = get_write_ptr(cb_mask);
    noc_async_read_page(0, mask_acc, mask_l1);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* mask = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mask_l1);
    uint32_t R = 0;
    for (uint32_t e = 0; e < E; ++e) {
        R += (mask[e] & 0x7fffffffu) != 0;
    }
    cb_reserve_back(cb_count, 1);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_count)) = R;
    cb_push_back(cb_count, 1);

    for (uint32_t n = core_id; n < Nt; n += NC) {
        cb_reserve_back(cb_s, 1);
        noc_async_read_page(n, shared, get_write_ptr(cb_s));
        noc_async_read_barrier();
        cb_push_back(cb_s, 1);
        for (uint32_t r0 = 0; r0 < R; r0 += B) {
            const uint32_t b = R - r0 < B ? R - r0 : B;
            cb_reserve_back(cb_p, b);
            const uint32_t l1 = get_write_ptr(cb_p);
            for (uint32_t i = 0; i < b; ++i) {
                noc_async_read_page((r0 + i) * Nt + n, planes, l1 + i * p_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb_p, b);
        }
    }
}

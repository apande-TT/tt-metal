// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The input stream of the C++ acoustic attention softmax (see tt/cpp_softmax.py): per unit (a head's
// 32-row tile row) the ST raw score tiles and the ST additive-mask tiles of that row.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t raw_addr = get_arg_val<uint32_t>(0);
    const uint32_t mask_addr = get_arg_val<uint32_t>(1);
    const uint32_t u0 = get_arg_val<uint32_t>(2);
    const uint32_t nu = get_arg_val<uint32_t>(3);

    constexpr uint32_t ST = get_compile_time_arg_val(0);
    constexpr uint32_t MT = get_compile_time_arg_val(1);  // row tiles a head
    constexpr uint32_t MH = get_compile_time_arg_val(2);  // mask heads: 1 (broadcast) or H
    constexpr auto ar = TensorAccessorArgs<3>();
    constexpr auto am = TensorAccessorArgs<ar.next_compile_time_args_offset()>();
    const auto sr = TensorAccessor(ar, raw_addr);
    const auto sm = TensorAccessor(am, mask_addr);

    constexpr uint32_t cb_raw = 0;
    constexpr uint32_t cb_mask = 1;

    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t mu = (MH > 1 ? u : u % MT);
        for (uint32_t w = 0; w < ST; ++w) {
            cb_reserve_back(cb_raw, 1);
            cb_reserve_back(cb_mask, 1);
            noc_async_read_page(u * ST + w, sr, get_write_ptr(cb_raw));
            noc_async_read_page(mu * ST + w, sm, get_write_ptr(cb_mask));
            noc_async_read_barrier();
            cb_push_back(cb_raw, 1);
            cb_push_back(cb_mask, 1);
        }
    }
}

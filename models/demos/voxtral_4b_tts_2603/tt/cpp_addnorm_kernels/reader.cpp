// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The input stream of the C++ codec residual add + RMS norm (see tt/cpp_addnorm.py): per unit (one 32-row
// tile row of the [rows, dim] float32 residual stream) its WT tiles of h and of the branch r, in step.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t h_addr = get_arg_val<uint32_t>(0);
    const uint32_t r_addr = get_arg_val<uint32_t>(1);
    const uint32_t u0 = get_arg_val<uint32_t>(2);
    const uint32_t nu = get_arg_val<uint32_t>(3);

    constexpr uint32_t WT = get_compile_time_arg_val(0);
    constexpr auto ah = TensorAccessorArgs<1>();
    constexpr auto ar = TensorAccessorArgs<ah.next_compile_time_args_offset()>();
    const auto sh = TensorAccessor(ah, h_addr);
    const auto sr = TensorAccessor(ar, r_addr);

    constexpr uint32_t cb_h = 0;
    constexpr uint32_t cb_r = 1;

    for (uint32_t u = u0; u < u0 + nu; ++u) {
        for (uint32_t w = 0; w < WT; ++w) {
            cb_reserve_back(cb_h, 1);
            cb_reserve_back(cb_r, 1);
            noc_async_read_page(u * WT + w, sh, get_write_ptr(cb_h));
            noc_async_read_page(u * WT + w, sr, get_write_ptr(cb_r));
            noc_async_read_barrier();
            cb_push_back(cb_h, 1);
            cb_push_back(cb_r, 1);
        }
    }
}

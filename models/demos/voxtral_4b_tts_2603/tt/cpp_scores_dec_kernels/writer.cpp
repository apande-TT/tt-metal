// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The output writer of the C++ decode attention scores (see tt/cpp_scores_dec.py): each unit's ST score tiles,
// in order. PACKED: the unit (b, h) holds only G real rows (its kv head's G query heads); they are written
// into rows h * G .. h * G + G - 1 of the batch's packed tile row (b, j) -- every kv head's rows sharing one
// [32-row] tile, so the softmax ops after it touch NKV-times fewer tiles. Each face's G rows are one contiguous
// G * 64 B run, written as it is.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t u0 = get_arg_val<uint32_t>(1);
    const uint32_t nu = get_arg_val<uint32_t>(2);

    constexpr uint32_t ST = get_compile_time_arg_val(0);
    constexpr uint32_t PACKED = get_compile_time_arg_val(1);
    constexpr uint32_t G = get_compile_time_arg_val(2);
    constexpr uint32_t NKV = get_compile_time_arg_val(3);
    constexpr auto ay = TensorAccessorArgs<4>();
    const auto sy = TensorAccessor(ay, y_addr);

    constexpr uint32_t cb_y = 16;
    constexpr uint32_t face = 1024;
    constexpr uint32_t seg = 64;
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t b = u / NKV;
        const uint32_t r0 = (u % NKV) * G;
        const uint32_t dof = ((r0 >> 4) * 2) * face + (r0 & 15) * seg;
        for (uint32_t j = 0; j < ST; ++j) {
            cb_wait_front(cb_y, 1);
            if constexpr (PACKED) {
                const uint32_t src = get_read_ptr(cb_y);
                for (uint32_t f = 0; f < 2; ++f) {
                    noc_async_write(src + f * face, sy.get_noc_addr(b * ST + j, dof + f * face), G * seg);
                }
            } else {
                noc_async_write_page(u * ST + j, sy, get_read_ptr(cb_y));
            }
            noc_async_write_barrier();
            cb_pop_front(cb_y, 1);
        }
    }
}

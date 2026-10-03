// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The writer of the C++ waveform flatten (see tt/cpp_wave.py): each unit's row-major [rows, C] block
// is ONE contiguous run of output row `row` at sample offset tt * 32 * C -- out[row, t * C + c] =
// x[row, t, c], the "b (c h) t -> b c (t h)" de-patch with c = 1.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t u0 = get_arg_val<uint32_t>(1);
    const uint32_t nu = get_arg_val<uint32_t>(2);

    constexpr uint32_t LT = get_compile_time_arg_val(0);
    constexpr uint32_t L = get_compile_time_arg_val(1);
    constexpr uint32_t C = get_compile_time_arg_val(2);
    constexpr auto ay = TensorAccessorArgs<3>();
    const auto sy = TensorAccessor(ay, y_addr);

    constexpr uint32_t cb_rm = 1;
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t row = u / LT;
        const uint32_t tt = u % LT;
        const uint32_t rows = (L - tt * 32) < 32 ? (L - tt * 32) : 32;
        cb_wait_front(cb_rm, 1);
        noc_async_write(get_read_ptr(cb_rm), sy.get_noc_addr(row, tt * 32 * C * 4), rows * C * 4);
        noc_async_write_barrier();
        cb_pop_front(cb_rm, 1);
    }
}

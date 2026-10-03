// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The writer of the C++ convolution shift-add (see tt/cpp_shift_add.py): unit u's summed tile to
// output tile (b, r, c) of [B, 1, T, C_out].

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t o_addr = get_arg_val<uint32_t>(0);
    const uint32_t u0 = get_arg_val<uint32_t>(1);
    const uint32_t nu = get_arg_val<uint32_t>(2);

    constexpr auto ao = TensorAccessorArgs<0>();
    const auto so = TensorAccessor(ao, o_addr);

    constexpr uint32_t cb_out = 16;
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        cb_wait_front(cb_out, 1);
        noc_async_write_page(u, so, get_read_ptr(cb_out));
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}

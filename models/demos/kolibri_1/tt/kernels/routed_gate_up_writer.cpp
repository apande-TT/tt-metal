// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode gate/up over the ROUTED experts only (writer): each computed gate/up tile to its column of the
// persistent [32, n_local*inter] outputs. Unrouted columns are left as they are -- their routing weight is
// exactly 0 downstream, so whatever finite values they hold contribute nothing.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto g_args = TensorAccessorArgs<0>();
    constexpr auto u_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    const auto g = TensorAccessor(g_args, get_arg_val<uint32_t>(0));
    const auto u = TensorAccessor(u_args, get_arg_val<uint32_t>(1));

    constexpr uint32_t cb_list = 5, cb_out_g = 16, cb_out_u = 17;

    cb_wait_front(cb_list, 1);
    volatile tt_l1_ptr uint32_t* list = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_list));
    const uint32_t n = list[0];
    for (uint32_t s = 0; s < n; ++s) {
        const uint32_t col = list[1 + s];
        cb_wait_front(cb_out_g, 1);
        cb_wait_front(cb_out_u, 1);
        noc_async_write_page(col, g, get_read_ptr(cb_out_g));
        noc_async_write_page(col, u, get_read_ptr(cb_out_u));
        noc_async_writes_flushed();
        cb_pop_front(cb_out_g, 1);
        cb_pop_front(cb_out_u, 1);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_list, 1);
}

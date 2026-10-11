// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode gate/up -> act over the ROUTED experts only (writer): each computed act tile to its column of the
// persistent [32, n_local*inter] output, and a zero tile to every column of the experts no token routed to (the
// down projection reads all columns; the buffer persists across calls, so an expert routed last step would
// otherwise leave its act behind).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(0);
    constexpr auto a_args = TensorAccessorArgs<1>();
    const auto act = TensorAccessor(a_args, get_arg_val<uint32_t>(0));

    constexpr uint32_t cb_list = 5, cb_act = 16, cb_zero = 17;

    cb_reserve_back(cb_zero, 1);
    const uint32_t zero_l1 = get_write_ptr(cb_zero);
    volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero_l1);
    for (uint32_t i = 0; i < tile_bytes / 4; ++i) {
        z[i] = 0;
    }

    cb_wait_front(cb_list, 1);
    volatile tt_l1_ptr uint32_t* list = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_list));
    const uint32_t n = list[0], nz = list[1];
    for (uint32_t s = 0; s < nz; ++s) {
        noc_async_write_page(list[2 + n + s], act, zero_l1);
    }
    for (uint32_t s = 0; s < n; ++s) {
        cb_wait_front(cb_act, 1);
        noc_async_write_page(list[2 + s], act, get_read_ptr(cb_act));
        noc_async_writes_flushed();
        cb_pop_front(cb_act, 1);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_list, 1);
}

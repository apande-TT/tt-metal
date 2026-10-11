// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode down projection over the ROUTED experts only, phase 2 (writer): each summed bf16 tile to its column of
// the [32, hidden] output.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Nt = get_compile_time_arg_val(0);
    constexpr uint32_t NC = get_compile_time_arg_val(1);
    constexpr auto o_args = TensorAccessorArgs<2>();
    const auto out = TensorAccessor(o_args, get_arg_val<uint32_t>(0));
    const uint32_t core_id = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_out = 16;

    for (uint32_t n = core_id; n < Nt; n += NC) {
        cb_wait_front(cb_out, 1);
        noc_async_write_page(n, out, get_read_ptr(cb_out));
        noc_async_writes_flushed();
        cb_pop_front(cb_out, 1);
    }
    noc_async_write_barrier();
}

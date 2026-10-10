// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Prefill routed experts, combine (writer): each summed tile to its place in the [M, hidden] bf16 output.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t MT = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t CW = get_compile_time_arg_val(3);  // output tiles per unit
    constexpr auto o_args = TensorAccessorArgs<4>();
    const auto out = TensorAccessor(o_args, get_arg_val<uint32_t>(0));
    const uint32_t core = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_out = 16;

    for (uint32_t unit = core; unit < MT * (Kt / CW); unit += NC) {
        const uint32_t r = unit / (Kt / CW), n0 = (unit % (Kt / CW)) * CW;
        cb_wait_front(cb_out, CW);
        const uint32_t l1 = get_read_ptr(cb_out);
        for (uint32_t c = 0; c < CW; ++c) {
            noc_async_write_page(r * Kt + n0 + c, out, l1 + c * 2048);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_out, CW);
    }
    noc_async_write_barrier();
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode q/k prep (writer): the unit's W output tiles back to the same tile columns of the fused [B, NT * 32] output.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0);
    constexpr uint32_t NT = get_compile_time_arg_val(1);
    constexpr auto o_args = TensorAccessorArgs<2>();
    const auto out = TensorAccessor(o_args, get_arg_val<uint32_t>(0));
    const uint32_t tr = get_arg_val<uint32_t>(1), col0 = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = 16;
    constexpr uint32_t TILE = 2048;  // bf16
    cb_wait_front(cb_out, W);
    const uint32_t l1 = get_read_ptr(cb_out);
    for (uint32_t t = 0; t < W; ++t) {
        noc_async_write(l1 + t * TILE, out.get_noc_addr(tr * NT + col0 + t), TILE);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, W);
}

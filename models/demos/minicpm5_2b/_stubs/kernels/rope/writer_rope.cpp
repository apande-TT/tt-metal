// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode rope writer: output tiles [t_start, t_start + num_tiles) of the roped [B, n_kv, rows, D].

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr auto out_args = TensorAccessorArgs<1>();

    const uint32_t t_start = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(0));

    for (uint32_t t = t_start; t < t_start + num_tiles; ++t) {
        cb_wait_front(cb_out, 1);
        noc_async_write_page(t, out, get_read_ptr(cb_out));
        noc_async_writes_flushed();
        cb_pop_front(cb_out, 1);
    }
    noc_async_write_barrier();
}

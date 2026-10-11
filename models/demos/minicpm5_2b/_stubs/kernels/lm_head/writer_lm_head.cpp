// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode LM head writer: output tiles [n_start, n_start + num_groups * group_w) of the
// interleaved [.., 32, V] bf16 output (M = one tile row, so page id == global column tile).
// Adapted from programming_examples/matmul/matmul_multi_core writer_unary_interleaved_start_id.cpp.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t group_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(2);
    constexpr auto out_args = TensorAccessorArgs<3>();

    const uint32_t n_start = get_arg_val<uint32_t>(0);
    const uint32_t num_groups = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);

    const auto out = TensorAccessor(out_args, out_addr);

    uint32_t n = n_start;
    for (uint32_t g = 0; g < num_groups; ++g) {
        cb_wait_front(cb_out, group_w);
        uint32_t l1_addr = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < group_w; ++j) {
            noc_async_write_page(n + j, out, l1_addr);
            l1_addr += out_tile_bytes;
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_out, group_w);
        n += group_w;
    }
    noc_async_write_barrier();
}

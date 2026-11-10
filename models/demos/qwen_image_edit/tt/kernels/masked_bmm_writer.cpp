// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// masked_lane_bmm's writer: each tile row's nt float32 output tiles from c_16, row by row.
// Compile-time: y accessor args, nt. Runtime: y addr, first row, row count.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);
    const uint32_t rows = get_arg_val<uint32_t>(2);

    constexpr auto y_args = TensorAccessorArgs<0>();
    constexpr uint32_t nt = get_compile_time_arg_val(y_args.next_compile_time_args_offset());
    const auto y = TensorAccessor(y_args, y_addr);
    constexpr uint32_t cb_out = 16;
    const uint32_t bytes = get_local_cb_interface(cb_out).fifo_page_size;

    Noc noc;
    CircularBuffer cb(cb_out);
    for (uint32_t m = 0; m < rows; ++m) {
        cb.wait_front(nt);
        for (uint32_t j = 0; j < nt; ++j) {
            noc.async_write(cb, y, bytes, {.offset_bytes = j * bytes}, {.page_id = (row_start + m) * nt + j});
        }
        noc.async_write_barrier();
        cb.pop_front(nt);
    }
}

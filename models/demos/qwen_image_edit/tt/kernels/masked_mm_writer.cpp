// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The masked-lane linear's writer: each N chunk's finished float32 block (rows x nc tiles, row-major) from c_16 to
// y pages (row_start + m) * nt + n0 + j.
// Compile-time: y's accessor args, then nt, nc. Runtime: y addr, first row, row count.

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
    constexpr uint32_t nc = get_compile_time_arg_val(y_args.next_compile_time_args_offset() + 1);
    const auto y = TensorAccessor(y_args, y_addr);
    constexpr uint32_t cb_out = 16;
    const uint32_t bytes = get_local_cb_interface(cb_out).fifo_page_size;

    Noc noc;
    CircularBuffer cb(cb_out);
    for (uint32_t n0 = 0; n0 < nt; n0 += nc) {
        cb.wait_front(rows * nc);
        for (uint32_t m = 0; m < rows; ++m) {
            for (uint32_t j = 0; j < nc; ++j) {
                noc.async_write(
                    cb, y, bytes, {.offset_bytes = (m * nc + j) * bytes}, {.page_id = (row_start + m) * nt + n0 + j});
            }
        }
        noc.async_write_barrier();
        cb.pop_front(rows * nc);
    }
}

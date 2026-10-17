// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A page-range copy's reader: pages [start, start + count) of the source into c_0, `chunk` pages per barrier
// (row-major pages are small: one read per page, many in flight).
// Compile-time: the source's accessor args, then chunk. Runtime: source addr, first page, page count.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);

    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr uint32_t chunk = get_compile_time_arg_val(src_args.next_compile_time_args_offset());
    const auto src = TensorAccessor(src_args, src_addr);
    const uint32_t bytes = get_local_cb_interface(0).fifo_page_size;

    Noc noc;
    CircularBuffer cb(0);
    for (uint32_t p = 0; p < count; p += chunk) {
        const uint32_t n = count - p < chunk ? count - p : chunk;
        cb.reserve_back(chunk);
        for (uint32_t j = 0; j < n; ++j) {
            noc.async_read(src, cb, bytes, {.page_id = start + p + j}, {.offset_bytes = j * bytes});
        }
        noc.async_read_barrier();
        cb.push_back(chunk);
    }
}

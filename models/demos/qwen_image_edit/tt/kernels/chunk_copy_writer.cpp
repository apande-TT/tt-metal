// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A page-range copy's writer: c_0's pages to the destination pages [start, start + count), `chunk` per flush.
// Compile-time: the destination's accessor args, then chunk. Runtime: destination addr, first page, page count.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);

    constexpr auto dst_args = TensorAccessorArgs<0>();
    constexpr uint32_t chunk = get_compile_time_arg_val(dst_args.next_compile_time_args_offset());
    const auto dst = TensorAccessor(dst_args, dst_addr);
    const uint32_t bytes = get_local_cb_interface(0).fifo_page_size;

    Noc noc;
    CircularBuffer cb(0);
    for (uint32_t p = 0; p < count; p += chunk) {
        const uint32_t n = count - p < chunk ? count - p : chunk;
        cb.wait_front(chunk);
        for (uint32_t j = 0; j < n; ++j) {
            noc.async_write(cb, dst, bytes, {.offset_bytes = j * bytes}, {.page_id = start + p + j});
        }
        noc.async_writes_flushed();
        cb.pop_front(chunk);
    }
    noc.async_write_barrier();
}

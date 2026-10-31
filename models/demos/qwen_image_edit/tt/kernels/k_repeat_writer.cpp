// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A K-repeat's writer: x [rows, Kt tiles] -> y [rows, reps * Kt tiles] with y's tile row holding x's tile row reps
// times side by side. c_0 carries x's tile pages [start, start + count) (chunk_copy_reader.cpp, `chunk` pages per
// push); x page p (tile row p / Kt, tile column p % Kt) goes to y pages (p / Kt) * reps * Kt + r * Kt + p % Kt.
// Compile-time: y's accessor args, then chunk, Kt, reps. Runtime: y addr, first x page, x page count.

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
    constexpr uint32_t kt = get_compile_time_arg_val(dst_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t reps = get_compile_time_arg_val(dst_args.next_compile_time_args_offset() + 2);
    const auto dst = TensorAccessor(dst_args, dst_addr);
    const uint32_t bytes = get_local_cb_interface(0).fifo_page_size;

    Noc noc;
    CircularBuffer cb(0);
    for (uint32_t p = 0; p < count; p += chunk) {
        const uint32_t n = count - p < chunk ? count - p : chunk;
        cb.wait_front(chunk);
        // copy-major: the chunk's consecutive pages for one copy, then the next copy -- consecutive pages spread over
        // the DRAM banks, where one tile's 8 copies (page stride Kt, a multiple of the bank count at Kt = 96) would
        // all land in the same bank back to back
        for (uint32_t r = 0; r < reps; ++r) {
            for (uint32_t j = 0; j < n; ++j) {
                const uint32_t src_page = start + p + j;
                const uint32_t base = (src_page / kt) * reps * kt + src_page % kt;
                noc.async_write(cb, dst, bytes, {.offset_bytes = j * bytes}, {.page_id = base + r * kt});
            }
        }
        noc.async_writes_flushed();
        cb.pop_front(chunk);
    }
    noc.async_write_barrier();
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The row gather's writer: each unit's mt float32 tiles (c_0) to the output tiles (entry e, tile rows 0 .. mt,
// column c). Compile-time: the output's accessor args, then mt, nt. Runtime: output addr, first unit, unit count.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t unit_start = get_arg_val<uint32_t>(1);
    const uint32_t unit_count = get_arg_val<uint32_t>(2);

    constexpr auto dst_args = TensorAccessorArgs<0>();
    constexpr uint32_t base = dst_args.next_compile_time_args_offset();
    constexpr uint32_t mt = get_compile_time_arg_val(base);
    constexpr uint32_t nt = get_compile_time_arg_val(base + 1);
    constexpr uint32_t tile_bytes = 4096;
    const auto dst = TensorAccessor(dst_args, dst_addr);

    Noc noc;
    CircularBuffer cb(0);
    for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
        const uint32_t e = u / nt, c = u % nt;
        cb.wait_front(mt);
        for (uint32_t t = 0; t < mt; ++t) {
            noc.async_write(cb, dst, tile_bytes, {.offset_bytes = t * tile_bytes}, {.page_id = (e * mt + t) * nt + c});
        }
        noc.async_writes_flushed();
        cb.pop_front(mt);
    }
    noc.async_write_barrier();
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Row sums of squares' reader: the reduce's ones scaler into c_1 (once), then the input's tile rows
// [row0, row0 + rows) into c_0, Kt tiles per row in order. Compile-time: the input's accessor args, then Kt.
// Runtime: input addr, first tile row, tile row count.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t row0 = get_arg_val<uint32_t>(1);
    const uint32_t rows = get_arg_val<uint32_t>(2);

    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr uint32_t kt = get_compile_time_arg_val(src_args.next_compile_time_args_offset());
    const auto src = TensorAccessor(src_args, src_addr);
    const uint32_t bytes = get_local_cb_interface(0).fifo_page_size;

    wh_generate_reduce_scaler<true>(1, 0x3F803F80);  // bf16 1.0 in both halves

    Noc noc;
    CircularBuffer cb(0);
    for (uint32_t r = row0; r < row0 + rows; ++r) {
        for (uint32_t k = 0; k < kt; ++k) {
            cb.reserve_back(1);
            noc.async_read(src, cb, bytes, {.page_id = r * kt + k}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb.push_back(1);
        }
    }
}

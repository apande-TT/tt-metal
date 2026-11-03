// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The N-stacked exact-lane product's reader: the product x @ [w_0 | ... | w_{L-1}] holds lane r's output in
// column block r (nt tiles wide) of a float32 tile tensor L * nt tiles wide. Output tile i (row i / nt, column
// i % nt of one lane block) reads its L lane tiles, lane order, into c_0.
// Compile-time: the stacked tensor's accessor args, then L and nt. Runtime: addr, num_tiles, start_id.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr auto args = TensorAccessorArgs<0>();
    constexpr uint32_t lanes = get_compile_time_arg_val(args.next_compile_time_args_offset());
    constexpr uint32_t nt = get_compile_time_arg_val(args.next_compile_time_args_offset() + 1);
    const auto src = TensorAccessor(args, addr);
    const uint32_t bytes = get_local_cb_interface(0).fifo_page_size;

    Noc noc;
    CircularBuffer cb(0);
    for (uint32_t i = start_id; i < start_id + num_tiles; ++i) {
        const uint32_t base = (i / nt) * (lanes * nt) + i % nt;
        cb.reserve_back(lanes);
        for (uint32_t r = 0; r < lanes; ++r) {
            noc.async_read(src, cb, bytes, {.page_id = base + r * nt}, {.offset_bytes = r * bytes});
        }
        noc.async_read_barrier();
        cb.push_back(lanes);
    }
}

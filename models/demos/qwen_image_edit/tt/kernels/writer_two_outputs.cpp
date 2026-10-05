// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writes tile t of two outputs (cb 16 -> out0, cb 17 -> out1) for this core's tile range.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t dst0_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst1_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);
    const uint32_t start_tile = get_arg_val<uint32_t>(3);

    constexpr auto dst0_args = TensorAccessorArgs<0>();
    constexpr auto dst1_args = TensorAccessorArgs<dst0_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb0 = 16;
    constexpr uint32_t cb1 = 17;
    const uint32_t page0 = get_local_cb_interface(cb0).fifo_page_size;
    const uint32_t page1 = get_local_cb_interface(cb1).fifo_page_size;
    const auto s0 = TensorAccessor(dst0_args, dst0_addr);
    const auto s1 = TensorAccessor(dst1_args, dst1_addr);

    Noc noc;
    DataflowBuffer d0(cb0);
    DataflowBuffer d1(cb1);
    for (uint32_t t = start_tile; t < start_tile + num_tiles; ++t) {
        d0.wait_front(1);
        noc.async_write(d0, s0, page0, {}, {.page_id = t});
        d1.wait_front(1);
        noc.async_write(d1, s1, page1, {}, {.page_id = t});
        noc.async_writes_flushed();
        d0.pop_front(1);
        d1.pop_front(1);
    }
    noc.async_write_barrier();
}

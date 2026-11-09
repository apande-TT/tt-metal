// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Copies pages [src_off + start, src_off + start + n) of src to pages [start, start + n) of dst through one L1 page
// (pure data movement; a device's own block of a replicated tensor).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t src_off = get_arg_val<uint32_t>(2);
    const uint32_t num_pages = get_arg_val<uint32_t>(3);
    const uint32_t start = get_arg_val<uint32_t>(4);

    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_id = 0;
    const uint32_t page = get_local_cb_interface(cb_id).fifo_page_size;
    const auto src = TensorAccessor(src_args, src_addr);
    const auto dst = TensorAccessor(dst_args, dst_addr);

    Noc noc;
    DataflowBuffer dfb(cb_id);
    for (uint32_t t = start; t < start + num_pages; ++t) {
        dfb.reserve_back(1);
        noc.async_read(src, dfb, page, {.page_id = src_off + t}, {.offset_bytes = 0});
        noc.async_read_barrier();
        noc.async_write(dfb, dst, page, {}, {.page_id = t});
        noc.async_write_barrier();
        dfb.push_back(1);
        dfb.wait_front(1);
        dfb.pop_front(1);
    }
}

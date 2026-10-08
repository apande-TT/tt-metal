// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Drains N (compile-time, 2 or 3) output CBs (c_16, c_17[, c_18]) tile by tile into N interleaved tensors, the
// same page range of each. Runtime: the N output addresses, num_pages, start_id. Compile-time: N, then the
// outputs' accessor args (the third repeats the first when N == 2).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t n_out = get_compile_time_arg_val(0);
    const uint32_t dst0_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst1_addr = get_arg_val<uint32_t>(1);
    const uint32_t dst2_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_pages = get_arg_val<uint32_t>(3);
    const uint32_t start_id = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb0 = 16, cb1 = 17, cb2 = 18;
    constexpr auto dst0_args = TensorAccessorArgs<1>();
    constexpr auto dst1_args = TensorAccessorArgs<dst0_args.next_compile_time_args_offset()>();
    constexpr auto dst2_args = TensorAccessorArgs<dst1_args.next_compile_time_args_offset()>();
    const auto s0 = TensorAccessor(dst0_args, dst0_addr);
    const auto s1 = TensorAccessor(dst1_args, dst1_addr);
    const auto s2 = TensorAccessor(dst2_args, dst2_addr);
    const uint32_t bytes0 = get_local_cb_interface(cb0).fifo_page_size;
    const uint32_t bytes1 = get_local_cb_interface(cb1).fifo_page_size;

    Noc noc;
    DataflowBuffer dfb0(cb0), dfb1(cb1), dfb2(cb2);
    for (uint32_t i = start_id; i < start_id + num_pages; ++i) {
        dfb0.wait_front(1);
        dfb1.wait_front(1);
        noc.async_write(dfb0, s0, bytes0, {}, {.page_id = i});
        noc.async_write(dfb1, s1, bytes1, {}, {.page_id = i});
        if constexpr (n_out > 2) {
            dfb2.wait_front(1);
            noc.async_write(dfb2, s2, get_local_cb_interface(cb2).fifo_page_size, {}, {.page_id = i});
        }
        noc.async_writes_flushed();
        dfb0.pop_front(1);
        dfb1.pop_front(1);
        if constexpr (n_out > 2) {
            dfb2.pop_front(1);
        }
    }
    noc.async_write_barrier();
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Drains the 8 output CBs c_16..c_23 tile by tile into 8 interleaved tensors, the same page range of each.
// Compile-time: the 8 outputs' accessor args. Runtime: the 8 output addresses, num_pages, start_id.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t num_pages = get_arg_val<uint32_t>(8);
    const uint32_t start_id = get_arg_val<uint32_t>(9);

    constexpr auto a0 = TensorAccessorArgs<0>();
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    constexpr auto a3 = TensorAccessorArgs<a2.next_compile_time_args_offset()>();
    constexpr auto a4 = TensorAccessorArgs<a3.next_compile_time_args_offset()>();
    constexpr auto a5 = TensorAccessorArgs<a4.next_compile_time_args_offset()>();
    constexpr auto a6 = TensorAccessorArgs<a5.next_compile_time_args_offset()>();
    constexpr auto a7 = TensorAccessorArgs<a6.next_compile_time_args_offset()>();
    const auto s0 = TensorAccessor(a0, get_arg_val<uint32_t>(0));
    const auto s1 = TensorAccessor(a1, get_arg_val<uint32_t>(1));
    const auto s2 = TensorAccessor(a2, get_arg_val<uint32_t>(2));
    const auto s3 = TensorAccessor(a3, get_arg_val<uint32_t>(3));
    const auto s4 = TensorAccessor(a4, get_arg_val<uint32_t>(4));
    const auto s5 = TensorAccessor(a5, get_arg_val<uint32_t>(5));
    const auto s6 = TensorAccessor(a6, get_arg_val<uint32_t>(6));
    const auto s7 = TensorAccessor(a7, get_arg_val<uint32_t>(7));
    const uint32_t bytes = get_local_cb_interface(16).fifo_page_size;

    Noc noc;
    DataflowBuffer d0(16), d1(17), d2(18), d3(19), d4(20), d5(21), d6(22), d7(23);
    for (uint32_t i = start_id; i < start_id + num_pages; ++i) {
        d0.wait_front(1);
        noc.async_write(d0, s0, bytes, {}, {.page_id = i});
        d1.wait_front(1);
        noc.async_write(d1, s1, bytes, {}, {.page_id = i});
        d2.wait_front(1);
        noc.async_write(d2, s2, bytes, {}, {.page_id = i});
        d3.wait_front(1);
        noc.async_write(d3, s3, bytes, {}, {.page_id = i});
        d4.wait_front(1);
        noc.async_write(d4, s4, bytes, {}, {.page_id = i});
        d5.wait_front(1);
        noc.async_write(d5, s5, bytes, {}, {.page_id = i});
        d6.wait_front(1);
        noc.async_write(d6, s6, bytes, {}, {.page_id = i});
        d7.wait_front(1);
        noc.async_write(d7, s7, bytes, {}, {.page_id = i});
        noc.async_writes_flushed();
        d0.pop_front(1);
        d1.pop_front(1);
        d2.pop_front(1);
        d3.pop_front(1);
        d4.pop_front(1);
        d5.pop_front(1);
        d6.pop_front(1);
        d7.pop_front(1);
    }
    noc.async_write_barrier();
}

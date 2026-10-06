// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Streams the same tile range of the five guarded-tail operands (ex, dn, nn, tol, lo) into CBs c_0..c_4.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t addr0 = get_arg_val<uint32_t>(0);
    const uint32_t addr1 = get_arg_val<uint32_t>(1);
    const uint32_t addr2 = get_arg_val<uint32_t>(2);
    const uint32_t addr3 = get_arg_val<uint32_t>(3);
    const uint32_t addr4 = get_arg_val<uint32_t>(4);
    const uint32_t num_tiles = get_arg_val<uint32_t>(5);
    const uint32_t start_id = get_arg_val<uint32_t>(6);

    constexpr auto args0 = TensorAccessorArgs<0>();
    constexpr auto args1 = TensorAccessorArgs<args0.next_compile_time_args_offset()>();
    constexpr auto args2 = TensorAccessorArgs<args1.next_compile_time_args_offset()>();
    constexpr auto args3 = TensorAccessorArgs<args2.next_compile_time_args_offset()>();
    constexpr auto args4 = TensorAccessorArgs<args3.next_compile_time_args_offset()>();
    const auto src0 = TensorAccessor(args0, addr0);
    const auto src1 = TensorAccessor(args1, addr1);
    const auto src2 = TensorAccessor(args2, addr2);
    const auto src3 = TensorAccessor(args3, addr3);
    const auto src4 = TensorAccessor(args4, addr4);
    const uint32_t bytes = get_local_cb_interface(0).fifo_page_size;

    Noc noc;
    CircularBuffer cb0(0), cb1(1), cb2(2), cb3(3), cb4(4);
    for (uint32_t i = start_id; i < start_id + num_tiles; ++i) {
        cb0.reserve_back(1);
        cb1.reserve_back(1);
        cb2.reserve_back(1);
        cb3.reserve_back(1);
        cb4.reserve_back(1);
        noc.async_read(src0, cb0, bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read(src1, cb1, bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read(src2, cb2, bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read(src3, cb3, bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read(src4, cb4, bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb0.push_back(1);
        cb1.push_back(1);
        cb2.push_back(1);
        cb3.push_back(1);
        cb4.push_back(1);
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of a precise linear's epilogue (a + b) + bias (sum_blocks.cpp with n = 3): for each output tile t of
// this core it streams a[t], b[t] and the bias tile of t's column, bias[t % Nt] (bias rows replicated over the
// tile), one barrier per output tile.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t bias_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_tiles = get_arg_val<uint32_t>(3);
    const uint32_t start_tile = get_arg_val<uint32_t>(4);

    constexpr uint32_t Nt = get_compile_time_arg_val(0);
    constexpr auto a_args = TensorAccessorArgs<1>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto bias_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_id_in = 0;
    const uint32_t page = get_local_cb_interface(cb_id_in).fifo_page_size;
    const auto a = TensorAccessor(a_args, a_addr);
    const auto b = TensorAccessor(b_args, b_addr);
    const auto bias = TensorAccessor(bias_args, bias_addr);

    Noc noc;
    DataflowBuffer dfb(cb_id_in);
    for (uint32_t t = start_tile; t < start_tile + num_tiles; ++t) {
        dfb.reserve_back(3);
        noc.async_read(a, dfb, page, {.page_id = t}, {.offset_bytes = 0});
        noc.async_read(b, dfb, page, {.page_id = t}, {.offset_bytes = page});
        noc.async_read(bias, dfb, page, {.page_id = t % Nt}, {.offset_bytes = 2 * page});
        noc.async_read_barrier();
        dfb.push_back(3);
    }
}

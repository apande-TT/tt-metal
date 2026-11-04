// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of a precise linear's epilogue (a + b) + bias (sum_blocks.cpp with n = 3) for a product y whose tile rows
// hold both limbs' products: a = y's rows 0..15 (the hi limb's), b = its rows 16..31 (the lo limb's). For each output
// tile t of this core it streams y[t] as a, then y[t]'s bottom two faces (rows 16..31) into the TOP two faces of b's
// slot (b's rows 0..15; its bottom faces are left as they are: output rows 16..31 are padding), and the bias tile of
// t's column, bias[t % Nt]. One barrier per output tile.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t bias_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);
    const uint32_t start_tile = get_arg_val<uint32_t>(3);

    constexpr uint32_t Nt = get_compile_time_arg_val(0);
    constexpr auto y_args = TensorAccessorArgs<1>();
    constexpr auto bias_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_id_in = 0;
    const uint32_t page = get_local_cb_interface(cb_id_in).fifo_page_size;
    const uint32_t half = page / 2;  // two 16 x 16 faces: one tile's rows 0..15 (or 16..31)
    const auto y = TensorAccessor(y_args, y_addr);
    const auto bias = TensorAccessor(bias_args, bias_addr);

    Noc noc;
    DataflowBuffer dfb(cb_id_in);
    for (uint32_t t = start_tile; t < start_tile + num_tiles; ++t) {
        dfb.reserve_back(3);
        noc.async_read(y, dfb, page, {.page_id = t}, {.offset_bytes = 0});
        noc.async_read(y, dfb, half, {.page_id = t, .offset_bytes = half}, {.offset_bytes = page});
        noc.async_read(bias, dfb, page, {.page_id = t % Nt}, {.offset_bytes = 2 * page});
        noc.async_read_barrier();
        dfb.push_back(3);
    }
}

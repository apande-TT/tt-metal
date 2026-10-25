// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of the exact all_to_all reduce's local sum + the row-parallel layer's bias (sum_blocks.cpp with
// n = n_blocks + 1): the input holds n_blocks equal blocks of block_tiles tiles (block i = source device i's
// partial); for each output tile t of this core it streams the n_blocks source tiles i * block_tiles + t, in
// source order, then the bias tile of t's column, bias[t % Nt] (bias rows replicated over the tile). One barrier
// per output tile.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t bias_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);
    const uint32_t start_tile = get_arg_val<uint32_t>(3);

    constexpr uint32_t block_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t n_blocks = get_compile_time_arg_val(1);
    constexpr uint32_t Nt = get_compile_time_arg_val(2);
    constexpr auto src_args = TensorAccessorArgs<3>();
    constexpr auto bias_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_id_in = 0;
    const uint32_t page_bytes = get_local_cb_interface(cb_id_in).fifo_page_size;
    const auto s = TensorAccessor(src_args, src_addr);
    const auto bias = TensorAccessor(bias_args, bias_addr);

    Noc noc;
    DataflowBuffer dfb(cb_id_in);
    for (uint32_t t = start_tile; t < start_tile + num_tiles; ++t) {
        dfb.reserve_back(n_blocks + 1);
        for (uint32_t i = 0; i < n_blocks; ++i) {
            noc.async_read(s, dfb, page_bytes, {.page_id = i * block_tiles + t}, {.offset_bytes = i * page_bytes});
        }
        noc.async_read(bias, dfb, page_bytes, {.page_id = t % Nt}, {.offset_bytes = n_blocks * page_bytes});
        noc.async_read_barrier();
        dfb.push_back(n_blocks + 1);
    }
}

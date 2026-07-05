// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// C++ Metalium reader for the fused-FFN kernel. Streams:
//   - fc1 activation (x) tiles into cb_in0
//   - fc1 weight (W1) tiles into cb_fc1_w
//   - fc2 weight (W2) tiles into cb_fc2_w
// The fc1 intermediate stays in L1 (managed by the compute kernel via
// cb_hidden), so this reader never fetches the hidden activation from DRAM.
//
// Adapted from tt_metal/programming_examples/matmul/matmul_multi_core/
// kernels/dataflow/reader_mm_output_tiles_partitioned.cpp.

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime args
    uint32_t src_x_addr = get_arg_val<uint32_t>(0);   // x [M, in]
    uint32_t src_w1_addr = get_arg_val<uint32_t>(1);  // W1 [in, hidden]
    uint32_t src_w2_addr = get_arg_val<uint32_t>(2);  // W2 [hidden, out]
    uint32_t Mt = get_arg_val<uint32_t>(3);
    uint32_t Kt_in = get_arg_val<uint32_t>(4);
    uint32_t Ht = get_arg_val<uint32_t>(5);
    uint32_t Nt_out = get_arg_val<uint32_t>(6);
    uint32_t output_tile_start_id = get_arg_val<uint32_t>(7);
    uint32_t num_output_tiles = get_arg_val<uint32_t>(8);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_fc1_w = tt::CBIndex::c_1;
    constexpr uint32_t cb_fc2_w = tt::CBIndex::c_2;

    constexpr auto x_args = TensorAccessorArgs<0>();
    const auto x = TensorAccessor(x_args, src_x_addr);
    constexpr auto w1_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto w1 = TensorAccessor(w1_args, src_w1_addr);
    constexpr auto w2_args = TensorAccessorArgs<w1_args.next_compile_time_args_offset()>();
    const auto w2 = TensorAccessor(w2_args, src_w2_addr);

    for (uint32_t output_tile = 0; output_tile < num_output_tiles; ++output_tile) {
        uint32_t current_tile_id = output_tile_start_id + output_tile;
        uint32_t out_row = current_tile_id / Nt_out;
        uint32_t out_col = current_tile_id % Nt_out;

        // Stage 1 stream: for each hidden tile column, stream Kt_in tiles
        // of x[out_row, :] and W1[:, ht] into the fc1 input CBs.
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            for (uint32_t k = 0; k < Kt_in; ++k) {
                cb_reserve_back(cb_in0, 1);
                uint32_t l1_a = get_write_ptr(cb_in0);
                noc_async_read_page(out_row * Kt_in + k, x, l1_a);
                noc_async_read_barrier();
                cb_push_back(cb_in0, 1);

                cb_reserve_back(cb_fc1_w, 1);
                uint32_t l1_w1 = get_write_ptr(cb_fc1_w);
                noc_async_read_page(k * Ht + ht, w1, l1_w1);
                noc_async_read_barrier();
                cb_push_back(cb_fc1_w, 1);
            }
        }

        // Stage 2 stream: for each fc2 output column tile, stream Ht tiles
        // of W2[:, out_col]. The hidden activation is already in L1 (cb_hidden)
        // produced by the compute kernel — no DRAM fetch here.
        for (uint32_t kt = 0; kt < Ht; ++kt) {
            cb_reserve_back(cb_fc2_w, 1);
            uint32_t l1_w2 = get_write_ptr(cb_fc2_w);
            noc_async_read_page(kt * Nt_out + out_col, w2, l1_w2);
            noc_async_read_barrier();
            cb_push_back(cb_fc2_w, 1);
        }
    }
}

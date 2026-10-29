// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of sum_gated.cpp, the exact all_to_all reduce's local sum with the block's gated residual update:
// for each output tile t of this core (the device's slice [B_local, Rt_local tile rows, Wt]) it streams
//   cb 0: the n_blocks source tiles i * block_tiles + t in source order, then (has_bias) the bias tile t % Wt
//   cb 1: the residual tile of t in the whole residual [B, Rt tile rows, Wt]: batch bl + b_off, tile row rt + r_off
//         (this device's slice: its batch items or its tile rows, per device runtime args)
//   cb 2: row 0 (faces 0 and 1's first rows) of the gate tile of t's batch item and column -- all the ROW
//         broadcast reads
// One barrier per output tile.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t bias_addr = get_arg_val<uint32_t>(1);
    const uint32_t res_addr = get_arg_val<uint32_t>(2);
    const uint32_t gate_addr = get_arg_val<uint32_t>(3);
    const uint32_t b_off = get_arg_val<uint32_t>(4);
    const uint32_t r_off = get_arg_val<uint32_t>(5);
    const uint32_t num_tiles = get_arg_val<uint32_t>(6);
    const uint32_t start_tile = get_arg_val<uint32_t>(7);

    constexpr uint32_t block_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t n_blocks = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);
    constexpr uint32_t Rt_local = get_compile_time_arg_val(3);
    constexpr uint32_t Rt = get_compile_time_arg_val(4);
    constexpr uint32_t has_bias = get_compile_time_arg_val(5);
    constexpr uint32_t gate_batched = get_compile_time_arg_val(6);
    constexpr auto src_args = TensorAccessorArgs<7>();
    constexpr auto bias_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto res_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();
    constexpr auto gate_args = TensorAccessorArgs<res_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_src = 0;
    constexpr uint32_t cb_res = 1;
    constexpr uint32_t cb_gate = 2;
    constexpr uint32_t face_bytes = 16 * 16 * 4;  // float32 face
    constexpr uint32_t row_bytes = 16 * 4;        // one face row
    const uint32_t page = get_local_cb_interface(cb_src).fifo_page_size;
    const auto s = TensorAccessor(src_args, src_addr);
    const auto bias = TensorAccessor(bias_args, bias_addr);
    const auto res = TensorAccessor(res_args, res_addr);
    const auto gate = TensorAccessor(gate_args, gate_addr);

    Noc noc;
    DataflowBuffer src(cb_src);
    DataflowBuffer rb(cb_res);
    DataflowBuffer gb(cb_gate);
    constexpr uint32_t n_in = n_blocks + has_bias;
    for (uint32_t t = start_tile; t < start_tile + num_tiles; ++t) {
        const uint32_t c = t % Wt;
        const uint32_t rt = (t / Wt) % Rt_local;
        const uint32_t b = t / (Wt * Rt_local) + b_off;
        src.reserve_back(n_in);
        rb.reserve_back(1);
        gb.reserve_back(1);
        for (uint32_t i = 0; i < n_blocks; ++i) {
            noc.async_read(s, src, page, {.page_id = i * block_tiles + t}, {.offset_bytes = i * page});
        }
        if constexpr (has_bias) {
            noc.async_read(bias, src, page, {.page_id = c}, {.offset_bytes = n_blocks * page});
        }
        noc.async_read(res, rb, page, {.page_id = (b * Rt + rt + r_off) * Wt + c}, {.offset_bytes = 0});
        const uint32_t gp = gate_batched ? b * Wt + c : c;
        for (uint32_t f = 0; f < 2; ++f) {
            noc.async_read(gate, gb, row_bytes, {.page_id = gp, .offset_bytes = f * face_bytes}, {.offset_bytes = f * face_bytes});
        }
        noc.async_read_barrier();
        src.push_back(n_in);
        rb.push_back(1);
        gb.push_back(1);
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The masked-lane linear's reader: the 8 column-lane mask tiles into c_2 once; then for each of this core's N chunks
// (nc tiles wide) and each K tile k: this core's x tiles (rows row_start.. row_start + rows - 1, column k) into c_0
// and the weight row k's chunk tiles w[k, n0 .. n0 + nc - 1] into c_1.
// Compile-time: x's accessor args, w's, the masks', then kt, nt, nc. Runtime: x addr, w addr, mask addr, first row,
// row count.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t m_addr = get_arg_val<uint32_t>(2);
    const uint32_t row_start = get_arg_val<uint32_t>(3);
    const uint32_t rows = get_arg_val<uint32_t>(4);

    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto m_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr uint32_t base = m_args.next_compile_time_args_offset();
    constexpr uint32_t kt = get_compile_time_arg_val(base);
    constexpr uint32_t nt = get_compile_time_arg_val(base + 1);
    constexpr uint32_t nc = get_compile_time_arg_val(base + 2);
    constexpr uint32_t lanes = get_compile_time_arg_val(base + 3);  // column lanes of x (8, or 4: 8 K terms per 32)
    // 1: a second input (x's trailing bf16 limb) read after each x tile, its product folded in as one more lane
    constexpr uint32_t extra = get_compile_time_arg_val(base + 4);
    constexpr auto lo_args = TensorAccessorArgs<base + 5>();
    constexpr uint32_t S = 1 + extra;  // x tiles per row slot in c_0
    const auto x = TensorAccessor(x_args, x_addr);
    const auto lo = TensorAccessor(lo_args, extra ? get_arg_val<uint32_t>(5) : x_addr);
    // 1: the float32 output block of each N chunk is pre-filled with the bias rows ([32, N] float32, every row the
    // bias) before compute packs into it with L1 accumulation from k = 0; c_5 tells compute the block is ready
    constexpr uint32_t has_bias = get_compile_time_arg_val(lo_args.next_compile_time_args_offset());
    constexpr auto b_args = TensorAccessorArgs<lo_args.next_compile_time_args_offset() + 1>();
    const auto bias = TensorAccessor(b_args, has_bias ? get_arg_val<uint32_t>(6) : x_addr);
    const auto w = TensorAccessor(w_args, w_addr);
    const auto masks = TensorAccessor(m_args, m_addr);
    const uint32_t bytes = get_local_cb_interface(0).fifo_page_size;

    Noc noc;
    CircularBuffer cb_x(0), cb_w(1), cb_m(2);
    cb_m.reserve_back(lanes + extra);
    for (uint32_t r = 0; r < lanes + extra; ++r) {
        noc.async_read(masks, cb_m, bytes, {.page_id = r}, {.offset_bytes = r * bytes});
    }
    noc.async_read_barrier();
    cb_m.push_back(lanes + extra);

    for (uint32_t n0 = 0; n0 < nt; n0 += nc) {
        if constexpr (has_bias) {
            const uint32_t obytes = get_local_cb_interface(16).fifo_page_size;
            cb_reserve_back(16, rows * nc);  // (the writer has drained the previous block)
            const uint32_t base_ptr = get_write_ptr(16);
            for (uint32_t m = 0; m < rows; ++m) {
                for (uint32_t j = 0; j < nc; ++j) {
                    noc_async_read(bias.get_noc_addr(n0 + j), base_ptr + (m * nc + j) * obytes, obytes);
                }
            }
            noc_async_read_barrier();
            cb_reserve_back(5, 1);
            cb_push_back(5, 1);
        }
        for (uint32_t k = 0; k < kt; ++k) {
            cb_x.reserve_back(rows * S);
            for (uint32_t m = 0; m < rows; ++m) {
                const uint32_t page = (row_start + m) * kt + k;
                noc.async_read(x, cb_x, bytes, {.page_id = page}, {.offset_bytes = m * S * bytes});
                if constexpr (extra) {
                    noc.async_read(lo, cb_x, bytes, {.page_id = page}, {.offset_bytes = (m * S + 1) * bytes});
                }
            }
            cb_w.reserve_back(nc);
            for (uint32_t j = 0; j < nc; ++j) {
                noc.async_read(w, cb_w, bytes, {.page_id = k * nt + n0 + j}, {.offset_bytes = j * bytes});
            }
            noc.async_read_barrier();
            cb_x.push_back(rows * S);
            cb_w.push_back(nc);
        }
    }
}

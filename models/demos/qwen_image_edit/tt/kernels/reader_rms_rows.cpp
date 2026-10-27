// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of the precise RMSNorm kernels.
//   mode 0 (rms_scale.cpp): the Wt x tiles of each tile row r of this core, r * Wt + wt, one barrier per
//     `batch` tiles (batch divides Wt).
//   mode 1 (rms_apply.cpp): for each output tile t of this core, rs tile t / Wt when t starts a new tile row
//     (or is the first), then x tile t and w tile t % Wt, one barrier per output tile.
// Runtime args: x, rs, w addresses, the tile row / tile count and the first one.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t rs_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t num = get_arg_val<uint32_t>(3);
    const uint32_t start = get_arg_val<uint32_t>(4);

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t mode = get_compile_time_arg_val(1);
    constexpr uint32_t batch = get_compile_time_arg_val(2);
    constexpr auto x_args = TensorAccessorArgs<3>();
    constexpr auto rs_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<rs_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_x = 0;
    constexpr uint32_t cb_rs = 1;
    constexpr uint32_t cb_w = 2;
    const uint32_t page = get_local_cb_interface(cb_x).fifo_page_size;
    const auto xs = TensorAccessor(x_args, x_addr);

    Noc noc;
    DataflowBuffer x(cb_x);
    if constexpr (mode == 0) {
        for (uint32_t r = start; r < start + num; ++r) {
            for (uint32_t wt = 0; wt < Wt; wt += batch) {
                x.reserve_back(batch);
                for (uint32_t k = 0; k < batch; ++k) {
                    noc.async_read(xs, x, page, {.page_id = r * Wt + wt + k}, {.offset_bytes = k * page});
                }
                noc.async_read_barrier();
                x.push_back(batch);
            }
        }
    } else {
        const auto rss = TensorAccessor(rs_args, rs_addr);
        const auto ws = TensorAccessor(w_args, w_addr);
        DataflowBuffer rs(cb_rs);
        DataflowBuffer w(cb_w);
        uint32_t row = 0xFFFFFFFF;
        for (uint32_t t = start; t < start + num; ++t) {
            if (t / Wt != row) {
                row = t / Wt;
                rs.reserve_back(1);
                noc.async_read(rss, rs, page, {.page_id = row}, {.offset_bytes = 0});
                noc.async_read_barrier();
                rs.push_back(1);
            }
            x.reserve_back(1);
            w.reserve_back(1);
            noc.async_read(xs, x, page, {.page_id = t}, {.offset_bytes = 0});
            noc.async_read(ws, w, page, {.page_id = t % Wt}, {.offset_bytes = 0});
            noc.async_read_barrier();
            x.push_back(1);
            w.push_back(1);
        }
    }
}

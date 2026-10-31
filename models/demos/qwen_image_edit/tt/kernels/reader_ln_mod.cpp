// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of ln_modulate.cpp: for each tile row i of this core it reads the row's Wt x tiles once (`batch` tiles per
// barrier; the compute keeps the row in cb_x for its three passes), then for the apply pass
// the (1 + scale) and shift tiles of each column for i's batch item (b = i / St; one table row when the tables are
// shared) -- only their row 0 (faces 0 and 1's first rows), the one the ROW broadcast reads. Runtime args: x, (1 + scale), shift addresses, the tile row count and the first one, and the
// apply pass's column range [c0, c1).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t sc_addr = get_arg_val<uint32_t>(1);
    const uint32_t sh_addr = get_arg_val<uint32_t>(2);
    const uint32_t num = get_arg_val<uint32_t>(3);
    const uint32_t start = get_arg_val<uint32_t>(4);
    const uint32_t c0 = get_arg_val<uint32_t>(5);  // the apply pass's columns [c0, c1) (the whole row: 0, Wt)
    const uint32_t c1 = get_arg_val<uint32_t>(6);

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t St = get_compile_time_arg_val(1);
    constexpr uint32_t batched = get_compile_time_arg_val(2);
    constexpr uint32_t batch = get_compile_time_arg_val(3);
    constexpr auto x_args = TensorAccessorArgs<4>();
    constexpr auto sc_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto sh_args = TensorAccessorArgs<sc_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_x = 0;
    constexpr uint32_t cb_ss = 3;
    const uint32_t page = get_local_cb_interface(cb_x).fifo_page_size;
    constexpr uint32_t face_bytes = 16 * 16 * 4;  // float32 face
    constexpr uint32_t row_bytes = 16 * 4;        // one face row
    const auto xs = TensorAccessor(x_args, x_addr);
    const auto scs = TensorAccessor(sc_args, sc_addr);
    const auto shs = TensorAccessor(sh_args, sh_addr);

    Noc noc;
    DataflowBuffer x(cb_x);
    DataflowBuffer ss(cb_ss);
    for (uint32_t i = start; i < start + num; ++i) {
        // the row once: it stays in cb_x (two rows deep) for all three of the compute's passes
        for (uint32_t wt = 0; wt < Wt; wt += batch) {
            x.reserve_back(batch);
            for (uint32_t k = 0; k < batch; ++k) {
                noc.async_read(xs, x, page, {.page_id = i * Wt + wt + k}, {.offset_bytes = k * page});
            }
            noc.async_read_barrier();
            x.push_back(batch);
        }
        const uint32_t tab = batched ? (i / St) * Wt : 0;
        for (uint32_t wt = c0; wt < c1; ++wt) {
            ss.reserve_back(2);
            // only row 0 of a table tile is read by the ROW broadcast: faces 0 and 1's first rows
            for (uint32_t f = 0; f < 2; ++f) {
                const uint32_t o = f * face_bytes;
                noc.async_read(scs, ss, row_bytes, {.page_id = tab + wt, .offset_bytes = o}, {.offset_bytes = o});
                noc.async_read(shs, ss, row_bytes, {.page_id = tab + wt, .offset_bytes = o}, {.offset_bytes = page + o});
            }
            noc.async_read_barrier();
            ss.push_back(2);
        }
    }
}

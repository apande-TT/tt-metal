// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of rope_swap.cpp: for each output tile t of this core (x [B, H, S, D] tiled, Dt = D / 32 tiles per row,
// St tile rows per head) it streams x[t], the half-swapped tile x[t - c + (c + Dt / 2) % Dt] (c = t % Dt), and the
// cos / sin tiles of t's (row, column) -- of t's batch item when the tables carry one ([B, 1, S, D]), else the one
// [1, 1, S, D] table -- four tiles per barrier.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t cos_addr = get_arg_val<uint32_t>(1);
    const uint32_t sin_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_tiles = get_arg_val<uint32_t>(3);
    const uint32_t start_tile = get_arg_val<uint32_t>(4);

    constexpr uint32_t Dt = get_compile_time_arg_val(0);
    constexpr uint32_t St = get_compile_time_arg_val(1);
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr uint32_t batched = get_compile_time_arg_val(3);
    constexpr auto x_args = TensorAccessorArgs<4>();
    constexpr auto cos_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_id_in = 0;
    const uint32_t page = get_local_cb_interface(cb_id_in).fifo_page_size;
    const auto x = TensorAccessor(x_args, x_addr);
    const auto cs = TensorAccessor(cos_args, cos_addr);
    const auto sn = TensorAccessor(sin_args, sin_addr);

    Noc noc;
    DataflowBuffer dfb(cb_id_in);
    for (uint32_t t = start_tile; t < start_tile + num_tiles; ++t) {
        const uint32_t c = t % Dt;
        const uint32_t rc = t % (St * Dt);  // (row, column) within the head
        const uint32_t tab = batched ? (t / (St * Dt * H)) * (St * Dt) + rc : rc;
        dfb.reserve_back(4);
        noc.async_read(x, dfb, page, {.page_id = t}, {.offset_bytes = 0});
        noc.async_read(x, dfb, page, {.page_id = t - c + (c + Dt / 2) % Dt}, {.offset_bytes = page});
        noc.async_read(cs, dfb, page, {.page_id = tab}, {.offset_bytes = 2 * page});
        noc.async_read(sn, dfb, page, {.page_id = tab}, {.offset_bytes = 3 * page});
        noc.async_read_barrier();
        dfb.push_back(4);
    }
}

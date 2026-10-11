// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// LM head, one tile row of tokens (reader). A compute core streams its NC weight columns, RB K rows a push. The
// [32, K] activation every core needs whole is read once, by a sender core that computes nothing (SENDER = 1), and
// multicast into every compute core's x CB a rectangle at a time; the compute cores' writers take it (see
// head_stream_writer.cpp), so no weight stream waits on it. Each core pulling x from one place instead cost ~60 us.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);  // K tiles
    constexpr uint32_t Nt = get_compile_time_arg_val(1);  // weight column tiles
    constexpr uint32_t NC = get_compile_time_arg_val(2);  // columns a core
    constexpr uint32_t RB = get_compile_time_arg_val(3);  // K rows a push
    constexpr uint32_t x_tile = get_compile_time_arg_val(4);
    constexpr uint32_t w_tile = get_compile_time_arg_val(5);
    constexpr uint32_t n_cores = get_compile_time_arg_val(6);  // compute cores
    constexpr uint32_t sem_ready = get_compile_time_arg_val(7);
    constexpr uint32_t sem_valid = get_compile_time_arg_val(8);
    constexpr bool SENDER = get_compile_time_arg_val(9);
    constexpr auto x_args = TensorAccessorArgs<10>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    static_assert(Kt % RB == 0, "head_stream blocking");

    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(0));
    const auto w = TensorAccessor(w_args, get_arg_val<uint32_t>(1));
    const uint32_t core = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_x = 0, cb_w = 1;

    if constexpr (SENDER) {
        cb_reserve_back(cb_x, Kt);
        const uint32_t x_l1 = get_write_ptr(cb_x);
        for (uint32_t k = 0; k < Kt; ++k) {
            noc_async_read_page(k, x, x_l1 + k * x_tile);
        }
        noc_async_read_barrier();
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_ready)), n_cores);
        const uint32_t valid_l1 = get_semaphore(sem_valid);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_l1) = 1;
        // The compute cores as rectangles (NOC 0 order: start <= end, NOC coordinates): count, then x0 y0 x1 y1 cores.
        const uint32_t rects = get_arg_val<uint32_t>(3);
        for (uint32_t r = 0; r < rects; ++r) {
            const uint32_t x0 = get_arg_val<uint32_t>(4 + 5 * r), y0 = get_arg_val<uint32_t>(5 + 5 * r);
            const uint32_t x1 = get_arg_val<uint32_t>(6 + 5 * r), y1 = get_arg_val<uint32_t>(7 + 5 * r);
            const uint32_t cores = get_arg_val<uint32_t>(8 + 5 * r);
            if (cores == 1) {
                noc_async_write(x_l1, get_noc_addr(x0, y0, x_l1), Kt * x_tile);
                noc_semaphore_set_remote(valid_l1, get_noc_addr(x0, y0, valid_l1));
            } else {
                noc_async_write_multicast(x_l1, get_noc_multicast_addr(x0, y0, x1, y1, x_l1), Kt * x_tile, cores);
                noc_semaphore_set_multicast(valid_l1, get_noc_multicast_addr(x0, y0, x1, y1, valid_l1), cores);
            }
        }
        noc_async_write_barrier();
        return;
    }

    // CB layout of a push: RB rows of NC tiles, row-major (the matmul's in1 block).
    const uint32_t n0 = core * NC;
    for (uint32_t k0 = 0; k0 < Kt; k0 += RB) {
        cb_reserve_back(cb_w, RB * NC);
        const uint32_t w_l1 = get_write_ptr(cb_w);
        for (uint32_t r = 0; r < RB; ++r) {
            for (uint32_t c = 0; c < NC; ++c) {
                noc_async_read_page((k0 + r) * Nt + n0 + c, w, w_l1 + (r * NC + c) * w_tile);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_w, RB * NC);
    }
}

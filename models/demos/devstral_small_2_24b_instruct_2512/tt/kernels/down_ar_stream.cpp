// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Weight stream of tt/down_decode.py's all_reduce fold (StreamDownDecode.ar_call; one instance per data-movement
// RISC). The core's NCOL columns stream one after the other, each by BOTH RISCs (this RISC: pushes PHASE, PHASE + 2,
// ... of the column, its first page in the runtime args), so every core's first column is done about two thirds of
// the way through the program and only the two-column cores' second columns finish at its end.
// The OWNER (RISCV_1) takes the multicast activation as gate_up_stream.cpp does, and writes each output tile into the
// partials' width shards as soon as compute packs it (polled between pushes), then counts it on the fabric-link
// worker that sends that tile's shard (on that worker's wave semaphore: columns below W1_COLS are wave 1).
// RES: this core holds residual shard i. After its columns the OWNER waits until all TP devices' partials of the
// shard are in its scratch slots (TP counts on the global semaphore), hands them and the residual shard x to compute
// (kernels/down_compute.cpp RES_TAIL: their sum, rounded once, plus x) and writes the new residual h.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// GU_NSEND: senders the activation comes from (gate_up_sender.cpp X_SLICE: one valid count each)
#ifndef GU_NSEND
#define GU_NSEND 1
#endif

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);  // K tiles
    constexpr uint32_t NB = get_compile_time_arg_val(1);  // DRAM banks the weight is interleaved over
    constexpr uint32_t RB = get_compile_time_arg_val(2);  // K rows a push
    constexpr uint32_t w_tile = get_compile_time_arg_val(3);
    constexpr uint32_t cb_w = get_compile_time_arg_val(4);
    constexpr uint32_t DEPTH = get_compile_time_arg_val(5);  // pushes cb_w holds
    constexpr bool OWNER = get_compile_time_arg_val(6);
    constexpr uint32_t sem_ready = get_compile_time_arg_val(7);
    constexpr uint32_t sem_valid = get_compile_time_arg_val(8);
    constexpr uint32_t out_tile = get_compile_time_arg_val(9);
    constexpr uint32_t NCOL = get_compile_time_arg_val(10);  // columns of this core
    constexpr uint32_t sem_w1 = get_compile_time_arg_val(11);
    constexpr uint32_t sem_w2 = get_compile_time_arg_val(12);
    constexpr uint32_t W1_COLS = get_compile_time_arg_val(13);  // columns below this are wave 1
    constexpr bool RES = get_compile_time_arg_val(14);
    constexpr uint32_t TP = get_compile_time_arg_val(15);
    constexpr uint32_t WT = get_compile_time_arg_val(16);  // tiles of a shard
    constexpr uint32_t ar_tile = get_compile_time_arg_val(17);
    constexpr uint32_t f32_tile = get_compile_time_arg_val(18);
    constexpr auto w_args = TensorAccessorArgs<19>();
    constexpr auto o_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    static_assert(Kt % (2 * RB) == 0, "down blocking");
    constexpr uint32_t cb_x = 0, cb_out = 16, cb_x0 = 11, cb_ar = 12, cb_hout = 14;

    const auto w = TensorAccessor(w_args, get_arg_val<uint32_t>(0));
    volatile tt_l1_ptr uint32_t* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_valid));

    bool have_x = !OWNER;
    if constexpr (OWNER) {
        const uint32_t sx = get_arg_val<uint32_t>(2), sy = get_arg_val<uint32_t>(3);  // the sender, NOC coordinates
        cb_reserve_back(cb_x, Kt);
        noc_semaphore_inc(get_noc_addr(sx, sy, get_semaphore(sem_ready)), 1);
    }

    // OWNER: write finished output tiles (all remaining ones if `all`, else those already packed) and count them
    uint32_t written = 0;
    auto flush_out = [&](bool all) {
        if constexpr (OWNER) {
            const auto out = TensorAccessor(o_args, get_arg_val<uint32_t>(1));
            while (written < NCOL && (all || cb_pages_available_at_front(cb_out, 1))) {
                cb_wait_front(cb_out, 1);
                const uint32_t col = get_arg_val<uint32_t>(6 + written);
                noc_async_write_page(col, out, get_read_ptr(cb_out));
                noc_async_write_barrier();
                const uint32_t link = (col / WT) % 2;
                const uint32_t lx = get_arg_val<uint32_t>(8 + 2 * link), ly = get_arg_val<uint32_t>(9 + 2 * link);
                noc_semaphore_inc(get_noc_addr(lx, ly, get_semaphore(col < W1_COLS ? sem_w1 : sem_w2)), 1);
                cb_pop_front(cb_out, 1);
                ++written;
            }
        }
    };

    uint32_t push = 0;
    for (uint32_t col = 0; col < NCOL; ++col) {
        uint32_t page = get_arg_val<uint32_t>(4 + col);
        for (uint32_t k0 = 0; k0 < Kt / 2; k0 += RB, ++push) {
            if (!have_x) {
                // compute consumes no weight before x: once cb_w is full, x must be handed over before reserving more
                invalidate_l1_cache();
                if (push >= DEPTH || *valid >= GU_NSEND) {
                    noc_semaphore_wait(valid, GU_NSEND);
                    cb_push_back(cb_x, Kt);
                    have_x = true;
                }
            }
            if (col > 0) {
                flush_out(false);
            }
            cb_reserve_back(cb_w, RB);
            noc_async_read(w.get_noc_addr(page), get_write_ptr(cb_w), RB * w_tile);
            page += RB * NB * 2;
            noc_async_read_barrier();
            cb_push_back(cb_w, RB);
        }
    }

    if constexpr (OWNER) {
        if (!have_x) {
            noc_semaphore_wait(valid, GU_NSEND);
            cb_push_back(cb_x, Kt);
        }
        flush_out(true);
        if constexpr (RES) {
            const uint32_t x_addr = get_arg_val<uint32_t>(12);
            const uint32_t scratch_addr = get_arg_val<uint32_t>(13);
            const uint32_t h_addr = get_arg_val<uint32_t>(14);
            volatile tt_l1_ptr uint32_t* ar_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(15));
            cb_reserve_back(cb_x0, WT);
            noc_async_read(get_noc_addr(x_addr), get_write_ptr(cb_x0), WT * f32_tile);
            noc_semaphore_wait(ar_sem, TP);
            noc_semaphore_set(ar_sem, 0);
            cb_reserve_back(cb_ar, TP * WT);
            noc_async_read(get_noc_addr(scratch_addr), get_write_ptr(cb_ar), TP * WT * ar_tile);
            noc_async_read_barrier();
            cb_push_back(cb_x0, WT);
            cb_push_back(cb_ar, TP * WT);
            cb_wait_front(cb_hout, WT);
            noc_async_write(get_read_ptr(cb_hout), get_noc_addr(h_addr), WT * f32_tile);
            noc_async_write_barrier();
            cb_pop_front(cb_hout, WT);
        }
    }
}

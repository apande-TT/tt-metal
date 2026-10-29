// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Weight stream of tt/gate_up_decode.py (compute cores; one instance per data-movement RISC): streams this core's
// NC columns of one weight (gate on RISCV_0, up on RISCV_1), RB K rows a push, into its CB; a push is RB rows of NC
// tiles, row-major (the compute kernel's in1 block). The weight is bank-major (FusedGateUpDecode.bank_major): a push
// is RB * NC consecutive slots of one DRAM bank, so it is ONE read of RB * NC tiles from the address of its first
// page, and the next push starts RB * NC * NB pages later. The OWNER instance (RISCV_1) also takes the activation the
// sender multicasts -- it tells the sender this core's x CB is reserved, then hands x to compute as soon as the
// valid semaphore is up, checked between pushes so its weight stream never waits on x until its CB is full -- and
// finally writes the core's OUT_NC finished output tiles (silu(gate) * up for tt/gate_up_decode.py, the down projection
// for tt/down_decode.py, whose RISCs stream one column each).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// GU_NSEND: senders the activation comes from (gate_up_sender.cpp X_SLICE: one valid count each)
#ifndef GU_NSEND
#define GU_NSEND 1
#endif

// GU_CTA / GU_RTA: where this stream's compile-time / runtime args start (norm_gate_stream_dm.cpp appends a stream
// to the norm cores' dataflow, after the norm's own args)
#ifndef GU_CTA
#define GU_CTA 0
#endif
#ifndef GU_RTA
#define GU_RTA 0
#endif
// GU_PSTEP: this RISC streams every GU_PSTEP-th push of its column(s) -- the first at runtime arg 5's page, each next
// GU_PSTEP pushes on (tt/down_decode.py's one-column cores split a column's pushes over both RISCs)
#ifndef GU_PSTEP
#define GU_PSTEP 1
#endif
// GU_PIPE2 (see the push loop) unless GU_PIPE1
#ifndef GU_PIPE1
#define GU_PIPE2 1
#endif

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(GU_CTA + 0);  // K tiles
    constexpr uint32_t NB = get_compile_time_arg_val(GU_CTA + 1);  // DRAM banks the weight is interleaved over
    constexpr uint32_t NC = get_compile_time_arg_val(GU_CTA + 2);  // columns a core
    constexpr uint32_t RB = get_compile_time_arg_val(GU_CTA + 3);  // K rows a push
    constexpr uint32_t w_tile = get_compile_time_arg_val(GU_CTA + 4);
    constexpr uint32_t cb_w = get_compile_time_arg_val(GU_CTA + 5);
    constexpr uint32_t DEPTH = get_compile_time_arg_val(GU_CTA + 6);  // pushes cb_w holds
    constexpr bool OWNER = get_compile_time_arg_val(GU_CTA + 7);
    constexpr uint32_t sem_ready = get_compile_time_arg_val(GU_CTA + 8);
    constexpr uint32_t sem_valid = get_compile_time_arg_val(GU_CTA + 9);
    constexpr uint32_t out_tile = get_compile_time_arg_val(GU_CTA + 10);
    constexpr uint32_t OUT_NC = get_compile_time_arg_val(GU_CTA + 11);  // output tiles the OWNER writes
    constexpr auto w_args = TensorAccessorArgs<GU_CTA + 12>();
    constexpr auto o_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    static_assert(Kt % (RB * GU_PSTEP) == 0, "gate_up blocking");

    const auto w = TensorAccessor(w_args, get_arg_val<uint32_t>(GU_RTA + 0));
    const uint32_t n0 = get_arg_val<uint32_t>(GU_RTA + 1);  // this core's first output column
    uint32_t page = get_arg_val<uint32_t>(GU_RTA + 5);      // first page of this core's first push
    constexpr uint32_t T = RB * NC;                // tiles a push
    constexpr uint32_t cb_x = 0, cb_out = 16;
    volatile tt_l1_ptr uint32_t* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_valid));

    bool have_x = !OWNER;
    if constexpr (OWNER) {
        // the sender, NOC coordinates
        const uint32_t sx = get_arg_val<uint32_t>(GU_RTA + 3), sy = get_arg_val<uint32_t>(GU_RTA + 4);
        cb_reserve_back(cb_x, Kt);
        noc_semaphore_inc(get_noc_addr(sx, sy, get_semaphore(sem_ready)), 1);
    }

#ifdef GU_HEADS
    // GU_HEADS (tt/down_decode.py StreamDownDecode heads=True, HEADS_ON_STREAM): this RISC also assembles half of one
    // x tile -- 16 rows of the decode head concat, row b = row h of SDPA-decode's input tile b * DT + d for x tile
    // q = h * DT + d -- while its first weight push is in flight, writes it into the sender's x CB (the same L1 address
    // on every core of the program) and counts on the sender's gather semaphore (gate_up_sender.cpp X_GATHER).
    // Runtime args GU_RTA + 6, 7: SDPA's output, job = 2 q + half (0xFFFFFFFF: none).
    constexpr uint32_t HB = o_args.next_compile_time_args_offset();
    constexpr uint32_t H_DT = get_compile_time_arg_val(HB + 0);    // tiles across a head
    constexpr uint32_t H_B = get_compile_time_arg_val(HB + 1);     // batch rows
    constexpr uint32_t H_CB = get_compile_time_arg_val(HB + 2);    // scratch (one face pair)
    constexpr uint32_t H_SEM = get_compile_time_arg_val(HB + 3);   // the sender's gather count
    constexpr auto h_args = TensorAccessorArgs<HB + 4>();
    const auto h_in = TensorAccessor(h_args, get_arg_val<uint32_t>(GU_RTA + 6));
    const uint32_t h_job = get_arg_val<uint32_t>(GU_RTA + 7);
    constexpr uint32_t H_ROW = 32, H_FACE = 512;
#endif

#ifdef GU_PIPE2
    // GU_PIPE2: two pushes in flight -- push p is read tagged with NOC transaction id 1 + p % 2 (the read command
    // buffer keeps the tag for every packet of the read) into the CB slot after the still-pending push p - 1, and only
    // then is p - 1 waited for (its own id) and pushed: the RISC no longer idles a whole DRAM round trip per push.
    // Same tiles in the same CB slots, pushed in order.
    const uint32_t cb_limit = get_local_cb_interface(cb_w).fifo_limit;
    const uint32_t cb_size = get_local_cb_interface(cb_w).fifo_size;
    uint32_t prev_trid = 0;
#endif
    uint32_t push = 0;
    for (uint32_t k0 = 0; k0 < Kt / GU_PSTEP; k0 += RB, ++push) {
        if (!have_x) {
            // compute consumes no weight before x: once cb_w is full, x must be handed over before reserving more
            invalidate_l1_cache();
            if (push >= DEPTH || *valid >= GU_NSEND) {
                noc_semaphore_wait(valid, GU_NSEND);
                cb_push_back(cb_x, Kt);
                have_x = true;
            }
        }
#ifdef GU_PIPE2
        cb_reserve_back(cb_w, prev_trid ? 2 * T : T);
        uint32_t wr = get_write_ptr(cb_w);
        if (prev_trid) {
            wr += T * w_tile;
            if (wr >= cb_limit) {
                wr -= cb_size;
            }
        }
        const uint32_t trid = 1 + (push & 1);
        noc_async_read_set_trid(trid);
        noc_async_read(w.get_noc_addr(page), wr, T * w_tile);
#else
        cb_reserve_back(cb_w, T);
        noc_async_read(w.get_noc_addr(page), get_write_ptr(cb_w), T * w_tile);
#endif
        page += T * NB * GU_PSTEP;
#ifdef GU_HEADS
        const bool h_now = push == 0 && h_job != 0xFFFFFFFF;
        const uint32_t h_q = h_job / 2, h_half = h_job % 2;
        const uint32_t h_scratch = get_write_ptr(H_CB);
        if (h_now) {
            const uint32_t h = h_q / H_DT, d = h_q % H_DT;
            const uint32_t src_off = (h / 16) * 2 * H_FACE + (h % 16) * H_ROW;  // head h's face rows in its tile
            const uint32_t b0 = 16 * h_half, b1 = (b0 + 16 < H_B) ? b0 + 16 : H_B;
            for (uint32_t b = b0; b < b1; ++b) {
                const uint64_t src = h_in.get_noc_addr(b * H_DT + d, src_off);
                const uint32_t dst = h_scratch + (b - b0) * H_ROW;
                noc_async_read(src, dst, H_ROW);
                noc_async_read(src + H_FACE, dst + H_FACE, H_ROW);
            }
        }
#endif
#ifdef GU_PIPE2
#ifdef GU_HEADS
        if (h_now) {
            noc_async_read_barrier();  // the head rows (and this push) have landed
        }
#endif
#else
        noc_async_read_barrier();
#endif
#ifdef GU_HEADS
        if (h_now) {
            const uint32_t sx = get_arg_val<uint32_t>(GU_RTA + 3), sy = get_arg_val<uint32_t>(GU_RTA + 4);
            const uint32_t off = h_q * 4 * H_FACE + h_half * 2 * H_FACE;
            noc_async_write(h_scratch, get_noc_addr(sx, sy, get_write_ptr(cb_x) + off), 2 * H_FACE);
            noc_async_write_barrier();  // the half tile has landed before it is counted
            noc_semaphore_inc(get_noc_addr(sx, sy, get_semaphore(H_SEM)), 1);
            noc_async_atomic_barrier();
        }
#endif
#ifdef GU_PIPE2
        if (prev_trid) {
            noc_async_read_barrier_with_trid(prev_trid);
            cb_push_back(cb_w, T);
        }
        prev_trid = trid;
#else
        cb_push_back(cb_w, T);
#endif
    }
#ifdef GU_PIPE2
    if (prev_trid) {
        noc_async_read_barrier_with_trid(prev_trid);
        cb_push_back(cb_w, T);
    }
    noc_async_read_set_trid(0);
#endif

    if constexpr (OWNER) {
        if (!have_x) {
            noc_semaphore_wait(valid, GU_NSEND);
            cb_push_back(cb_x, Kt);
        }
        const auto out = TensorAccessor(o_args, get_arg_val<uint32_t>(GU_RTA + 2));
        cb_wait_front(cb_out, OUT_NC);
        const uint32_t l1 = get_read_ptr(cb_out);
        for (uint32_t c = 0; c < OUT_NC; ++c) {
            noc_async_write_page(n0 + c, out, l1 + c * out_tile);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, OUT_NC);
    }
}

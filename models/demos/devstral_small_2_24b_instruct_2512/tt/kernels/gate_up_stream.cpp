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

// GU_CTA / GU_RTA: where this stream's compile-time / runtime args start (norm_gate_stream_dm.cpp appends a stream
// to the norm cores' dataflow, after the norm's own args)
#ifndef GU_CTA
#define GU_CTA 0
#endif
#ifndef GU_RTA
#define GU_RTA 0
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
    static_assert(Kt % RB == 0, "gate_up blocking");

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

    uint32_t push = 0;
    for (uint32_t k0 = 0; k0 < Kt; k0 += RB, ++push) {
        if (!have_x) {
            // compute consumes no weight before x: once cb_w is full, x must be handed over before reserving more
            invalidate_l1_cache();
            if (push >= DEPTH || *valid != 0) {
                noc_semaphore_wait(valid, 1);
                cb_push_back(cb_x, Kt);
                have_x = true;
            }
        }
        cb_reserve_back(cb_w, T);
        noc_async_read(w.get_noc_addr(page), get_write_ptr(cb_w), T * w_tile);
        page += T * NB;
        noc_async_read_barrier();
        cb_push_back(cb_w, T);
    }

    if constexpr (OWNER) {
        if (!have_x) {
            noc_semaphore_wait(valid, 1);
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

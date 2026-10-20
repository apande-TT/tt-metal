// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Weight stream of tt/lm_head_decode.py (compute cores; one instance per data-movement RISC): kernels/gate_up_stream.cpp
// over PASSES column groups. In pass p this RISC streams the NC columns of stream p * n_cores + c of its bank-major
// weight copy (A on RISCV_0, B on RISCV_1), RB K rows a push, each push ONE read of RB * NC consecutive slots of one
// DRAM bank. The OWNER instance (RISCV_1) takes the multicast activation once (as gate_up_stream.cpp does) and, after
// every pass, writes the core's 2 * NC finished output tiles of that pass.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// GU_NSEND: senders the activation comes from (gate_up_sender.cpp X_SLICE: one valid count each)
#ifndef GU_NSEND
#define GU_NSEND 1
#endif

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);  // K tiles
    constexpr uint32_t NB = get_compile_time_arg_val(1);  // DRAM banks the weight is interleaved over
    constexpr uint32_t NC = get_compile_time_arg_val(2);  // columns a stream a pass
    constexpr uint32_t RB = get_compile_time_arg_val(3);  // K rows a push
    constexpr uint32_t w_tile = get_compile_time_arg_val(4);
    constexpr uint32_t cb_w = get_compile_time_arg_val(5);
    constexpr uint32_t DEPTH = get_compile_time_arg_val(6);  // pushes cb_w holds
    constexpr bool OWNER = get_compile_time_arg_val(7);
    constexpr uint32_t sem_ready = get_compile_time_arg_val(8);
    constexpr uint32_t sem_valid = get_compile_time_arg_val(9);
    constexpr uint32_t out_tile = get_compile_time_arg_val(10);
    constexpr uint32_t PASSES = get_compile_time_arg_val(11);
    constexpr uint32_t PASS_PAGES = get_compile_time_arg_val(12);  // pages from a stream's first push to the next pass's
    constexpr auto w_args = TensorAccessorArgs<13>();
    constexpr auto o_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    static_assert(Kt % RB == 0, "lm_head blocking");

    const auto w = TensorAccessor(w_args, get_arg_val<uint32_t>(0));
    const uint32_t n0 = get_arg_val<uint32_t>(1);     // this core's first output column (pass 0)
    const uint32_t page0 = get_arg_val<uint32_t>(5);  // first page of this core's pass-0 stream
    constexpr uint32_t T = RB * NC;                   // tiles a push
    constexpr uint32_t OUT_NC = 2 * NC;               // output tiles a pass
    constexpr uint32_t cb_x = 0, cb_out = 16;
    volatile tt_l1_ptr uint32_t* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_valid));

    bool have_x = !OWNER;
    if constexpr (OWNER) {
        const uint32_t sx = get_arg_val<uint32_t>(3), sy = get_arg_val<uint32_t>(4);  // the sender, NOC coordinates
        cb_reserve_back(cb_x, Kt);
        noc_semaphore_inc(get_noc_addr(sx, sy, get_semaphore(sem_ready)), 1);
    }

    uint32_t push = 0;
    for (uint32_t p = 0; p < PASSES; ++p) {
        uint32_t page = page0 + p * PASS_PAGES;
        for (uint32_t k0 = 0; k0 < Kt; k0 += RB, ++push) {
            if (!have_x) {
                // compute consumes no weight before x: once cb_w is full, x must be handed over before reserving more
                invalidate_l1_cache();
                if (push >= DEPTH || *valid >= GU_NSEND) {
                    noc_semaphore_wait(valid, GU_NSEND);
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
                noc_semaphore_wait(valid, GU_NSEND);
                cb_push_back(cb_x, Kt);
                have_x = true;
            }
            const auto out = TensorAccessor(o_args, get_arg_val<uint32_t>(2));
            cb_wait_front(cb_out, OUT_NC);
            const uint32_t l1 = get_read_ptr(cb_out);
            for (uint32_t c = 0; c < OUT_NC; ++c) {
                noc_async_write_page(n0 + p * OUT_NC + c, out, l1 + c * out_tile);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, OUT_NC);
        }
    }
}

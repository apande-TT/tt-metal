// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Weight stream of tt/qkv_decode.py (every compute core, both data-movement RISCs): the core's one column, RB K rows
// a push = one contiguous read of the bank-major copy; RISCV_1 streams the first W1 odd pushes (1, 3, ..,
// 2 W1 - 1), RISCV_0 all the others (its NOC 0 reads land faster), each into its own CB with two in flight (a push read
// tagged with NOC transaction id 1 + n % 2, n = the RISC's own push count, then the previous one waited for on its own
// id and pushed: gate_up_stream.cpp's GU_PIPE2).
// X_RECV (RISCV_1): also takes the activation the sender multicasts (kernels/qkv_xsend.cpp) -- reserves the whole x
// CB, tells the sender it may write, and while it waits for its own reads hands compute each CH-tile x chunk in K
// order as soon as the sender RISC multicasting it (chunk c: RISC c % 2) has counted it into valid semaphore c % 2.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);  // K tiles
    constexpr uint32_t NB = get_compile_time_arg_val(1);  // DRAM banks the weight is interleaved over
    constexpr uint32_t RB = get_compile_time_arg_val(2);  // K rows a push
    constexpr uint32_t w_tile = get_compile_time_arg_val(3);
    constexpr uint32_t cb_w = get_compile_time_arg_val(4);
    constexpr uint32_t CH = get_compile_time_arg_val(5);  // x tiles a chunk
    constexpr uint32_t sem_ready = get_compile_time_arg_val(6);
    constexpr uint32_t sem_v0 = get_compile_time_arg_val(7);
    constexpr uint32_t sem_v1 = get_compile_time_arg_val(8);
    constexpr uint32_t W1 = get_compile_time_arg_val(9);  // RISCV_1's pushes
    constexpr uint32_t ROLE = get_compile_time_arg_val(10);  // 0: RISCV_0, 1: RISCV_1
    constexpr auto w_args = TensorAccessorArgs<11>();
    constexpr uint32_t NPT = Kt / RB;  // the column's pushes
    static_assert(Kt % RB == 0 && Kt % CH == 0 && 2 * W1 <= NPT, "qkv blocking");

    const auto w = TensorAccessor(w_args, get_arg_val<uint32_t>(0));
    const uint32_t page0 = get_arg_val<uint32_t>(1);  // first page of the column's first push

#ifdef X_RECV
    constexpr uint32_t cb_x = 0, NCH = Kt / CH;
    const uint32_t sx = get_arg_val<uint32_t>(2), sy = get_arg_val<uint32_t>(3);  // the sender, NOC coordinates
    cb_reserve_back(cb_x, Kt);
    noc_semaphore_inc(get_noc_addr(sx, sy, get_semaphore(sem_ready)), 1);
    volatile tt_l1_ptr uint32_t* v0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_v0));
    volatile tt_l1_ptr uint32_t* v1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_v1));
    uint32_t xc = 0;  // x chunks handed to compute
    auto hand_x = [&]() {
        invalidate_l1_cache();
        while (xc < NCH && *((xc & 1) ? v1 : v0) >= xc / 2 + 1) {
            cb_push_back(cb_x, CH);
            ++xc;
        }
    };
#endif
    auto wait_trid = [&](uint32_t trid) {
        while (!ncrisc_noc_read_with_transaction_id_flushed(noc_index, trid)) {
#ifdef X_RECV
            hand_x();
#endif
        }
        invalidate_l1_cache();
    };

    constexpr uint32_t T = RB;  // tiles a push
    const uint32_t cb_limit = get_local_cb_interface(cb_w).fifo_limit;
    const uint32_t cb_size = get_local_cb_interface(cb_w).fifo_size;
    uint32_t prev_trid = 0, n = 0;
    for (uint32_t p = 0; p < NPT; ++p) {
        if ((p % 2 == 1 && p / 2 < W1) != (ROLE == 1)) {
            continue;
        }
        cb_reserve_back(cb_w, prev_trid ? 2 * T : T);
        uint32_t wr = get_write_ptr(cb_w);
        if (prev_trid) {
            wr += T * w_tile;
            if (wr >= cb_limit) {
                wr -= cb_size;
            }
        }
        const uint32_t trid = 1 + (n++ & 1);
        noc_async_read_set_trid(trid);
        noc_async_read(w.get_noc_addr(page0 + p * T * NB), wr, T * w_tile);
        if (prev_trid) {
            wait_trid(prev_trid);
            cb_push_back(cb_w, T);
        }
        prev_trid = trid;
    }
    wait_trid(prev_trid);
    cb_push_back(cb_w, T);
    noc_async_read_set_trid(0);

#ifdef X_RECV
    while (xc < NCH) {
        noc_semaphore_wait_min((xc & 1) ? v1 : v0, xc / 2 + 1);
        cb_push_back(cb_x, CH);
        ++xc;
    }
    noc_async_atomic_barrier();
#endif

#ifdef HEADS_OUT
    // the head split (nlp_create_qkv_heads_decode) folded in: row u of this core's output tile (user u's 32 values of
    // column j) is row h of tile t of user u's q / k / v head shard (column j = head h's d-tile t of q, k or v): two
    // 32-byte face-row writes a user, into the shard of the core that holds user u's heads. Runtime args 4..: the
    // destination L1 address (buffer + t * tile + row h's face offset), users, then each user's core (NOC x, y).
    constexpr uint32_t cb_out = 16, ROW = 32, FACE = 512;
    const uint32_t dst_l1 = get_arg_val<uint32_t>(4), users = get_arg_val<uint32_t>(5);
    cb_wait_front(cb_out, 1);
    const uint32_t src = get_read_ptr(cb_out);
    for (uint32_t u = 0; u < users; ++u) {
        const uint64_t dst = get_noc_addr(get_arg_val<uint32_t>(6 + 2 * u), get_arg_val<uint32_t>(7 + 2 * u), dst_l1);
        const uint32_t so = u < 16 ? u * ROW : 2 * FACE + (u - 16) * ROW;
        noc_async_write(src + so, dst, ROW);
        noc_async_write(src + so + FACE, dst + FACE, ROW);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
#endif
}

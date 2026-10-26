// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Decode draw of tt/pipeline.py DrawTail, one RISC per 16-row half (HALF) of the (32, 32) candidate tiles:
//   drawn[b] = ids[b, argmax_j scores[b, j]]                (the top-1 of the noisy scores, labelled with its id)
//   nxt[b] = unfinished[b] ? drawn[b] : PAD;  unfinished[b] &= nxt[b] != EOS;  tok[b] = nxt[b]   (as SampleTail)
// Rows 16 * HALF .. 16 * HALF + 15 of a 32-bit tile are faces 2 * HALF (columns 0-15) and 2 * HALF + 1 (16-31). The
// fp32 scores are compared through an order-preserving integer key (sign-magnitude -> unsigned), so the argmax is
// the comparison's; a tie keeps the lowest column. drawn and unfinished keep row b in column 0 (face 2 * HALF,
// element 16 * (b % 16)); this RISC writes its face of each and its 16 ids of the row-major next-token row.
// SPLIT8 (DrawTail.SPLIT, 2 cores, HALF = the core): each RISC takes 8 rows of its core's half (SUB = 0 / 1, rows
// 16 * HALF + 8 * SUB ..): it reads and writes only those rows of each face (8 x 64 B, 64-B aligned) and puts its 8
// ids into the core's shared next-token segment; SUB 0 then bumps the core's semaphore and SUB 1, once it sees it,
// writes the 16-id (64 B) segment.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

static inline uint32_t order_key(uint32_t f) { return (f & 0x80000000u) ? ~f : (f | 0x80000000u); }

void kernel_main() {
    const uint32_t scores_addr = get_arg_val<uint32_t>(0);
    const uint32_t ids_addr = get_arg_val<uint32_t>(1);
    const uint32_t unf_addr = get_arg_val<uint32_t>(2);
    const uint32_t tok_addr = get_arg_val<uint32_t>(3);
    const uint32_t drawn_addr = get_arg_val<uint32_t>(4);

    constexpr uint32_t HALF = get_compile_time_arg_val(0);  // rows 16 * HALF .. 16 * HALF + 15
    constexpr uint32_t PAD = get_compile_time_arg_val(1);
    constexpr uint32_t EOS = get_compile_time_arg_val(2);
    constexpr uint32_t CB = get_compile_time_arg_val(3);  // scratch: 2 KB scores + 2 KB ids + 1 KB unf + 1 KB drawn
    constexpr auto a_scores = TensorAccessorArgs<4>();
    constexpr auto a_ids = TensorAccessorArgs<a_scores.next_compile_time_args_offset()>();
    constexpr auto a_unf = TensorAccessorArgs<a_ids.next_compile_time_args_offset()>();
    constexpr auto a_tok = TensorAccessorArgs<a_unf.next_compile_time_args_offset()>();
    constexpr auto a_drawn = TensorAccessorArgs<a_tok.next_compile_time_args_offset()>();
    const auto s_scores = TensorAccessor(a_scores, scores_addr);
    const auto s_ids = TensorAccessor(a_ids, ids_addr);
    const auto s_unf = TensorAccessor(a_unf, unf_addr);
    const auto s_tok = TensorAccessor(a_tok, tok_addr);
    const auto s_drawn = TensorAccessor(a_drawn, drawn_addr);

    constexpr uint32_t FACE = 256 * 4;  // bytes of a 16 x 16 face of 32-bit elements
    constexpr uint32_t ONE = 0x3F800000u;  // fp32 1.0
    constexpr uint32_t F0 = 2 * HALF * FACE;  // this half's first face in a tile
#ifdef SPLIT8
    constexpr uint32_t SUB = SPLIT8_SUB;  // 8-row block within the half (0: BRISC, 1: NCRISC)
    constexpr uint32_t ROWS = 8, R0 = 8 * SUB, SEG = ROWS * 64;  // this RISC's rows of a face: SEG bytes at R0 * 64
    // per RISC: scores / ids face pairs (rows R0.. only, at their face offsets), the unfinished and drawn faces
    const uint32_t base = get_write_ptr(CB) + SUB * 6 * FACE;
    const uint32_t l_scores = base, l_ids = base + 2 * FACE, l_unf = base + 4 * FACE, l_drawn = base + 5 * FACE;
    const uint32_t l_tok = get_write_ptr(CB) + 12 * FACE;  // the core's 16 ids (shared by its two RISCs)
    for (uint32_t f = 0; f < 2; ++f) {
        noc_async_read(s_scores.get_noc_addr(0, F0 + f * FACE + R0 * 64), l_scores + f * FACE + R0 * 64, SEG);
        noc_async_read(s_ids.get_noc_addr(0, F0 + f * FACE + R0 * 64), l_ids + f * FACE + R0 * 64, SEG);
    }
    noc_async_read(s_unf.get_noc_addr(0, F0 + R0 * 64), l_unf + R0 * 64, SEG);
    noc_async_read_barrier();
#else
    constexpr uint32_t ROWS = 16, R0 = 0;
    const uint32_t base = get_write_ptr(CB) + HALF * 6 * FACE;
    const uint32_t l_scores = base, l_ids = base + 2 * FACE, l_unf = base + 4 * FACE, l_drawn = base + 5 * FACE;
    const uint32_t l_tok = get_write_ptr(CB) + 12 * FACE + HALF * 64;

    noc_async_read(s_scores.get_noc_addr(0, F0), l_scores, 2 * FACE);
    noc_async_read(s_ids.get_noc_addr(0, F0), l_ids, 2 * FACE);
    noc_async_read(s_unf.get_noc_addr(0, F0), l_unf, FACE);
    noc_async_read_barrier();
#endif

    const volatile tt_l1_ptr uint32_t* scores = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_scores);
    const volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_ids);
    volatile tt_l1_ptr uint32_t* unf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_unf);
    volatile tt_l1_ptr uint32_t* drawn = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_drawn);
    volatile tt_l1_ptr uint32_t* tok = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_tok);
    for (uint32_t r = R0; r < R0 + ROWS; ++r) {
        // columns 0-15 in the first face, 16-31 in the second, row r at 16 * r of each
        uint32_t best = order_key(scores[r * 16]), arg = r * 16;
        for (uint32_t j = 1; j < 32; ++j) {
            const uint32_t e = (j >> 4) * 256 + r * 16 + (j & 15);
            const uint32_t key = order_key(scores[e]);
            if (key > best) {
                best = key;
                arg = e;
            }
        }
        const uint32_t d = ids[arg];
        const bool live = unf[r * 16] == ONE;
        const uint32_t nxt = live ? d : PAD;
        drawn[r * 16] = d;
        unf[r * 16] = (live && nxt != EOS) ? ONE : 0u;
        tok[r] = nxt;
    }

#ifdef SPLIT8
    noc_async_write(l_unf + R0 * 64, s_unf.get_noc_addr(0, F0 + R0 * 64), SEG);
    noc_async_write(l_drawn + R0 * 64, s_drawn.get_noc_addr(0, F0 + R0 * 64), SEG);
    const uint32_t sem = get_semaphore(SPLIT8_SEM);
    if constexpr (SUB == 0) {
        // the 8 ids are in L1; tell the other RISC of this core (an atomic through the NoC to this core)
        noc_semaphore_inc(get_noc_addr(my_x[noc_index], my_y[noc_index], sem), 1);
        noc_async_atomic_barrier();
    } else {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem), 1);
        noc_async_write(l_tok, s_tok.get_noc_addr(0, HALF * 64), 64);
    }
    noc_async_write_barrier();
#else
    noc_async_write(l_unf, s_unf.get_noc_addr(0, F0), FACE);
    noc_async_write(l_drawn, s_drawn.get_noc_addr(0, F0), FACE);
    noc_async_write(l_tok, s_tok.get_noc_addr(0, HALF * 64), 64);
    noc_async_write_barrier();
#endif
}

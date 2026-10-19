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
    const uint32_t base = get_write_ptr(CB) + HALF * 6 * FACE;
    const uint32_t l_scores = base, l_ids = base + 2 * FACE, l_unf = base + 4 * FACE, l_drawn = base + 5 * FACE;
    const uint32_t l_tok = get_write_ptr(CB) + 12 * FACE + HALF * 64;

    noc_async_read(s_scores.get_noc_addr(0, F0), l_scores, 2 * FACE);
    noc_async_read(s_ids.get_noc_addr(0, F0), l_ids, 2 * FACE);
    noc_async_read(s_unf.get_noc_addr(0, F0), l_unf, FACE);
    noc_async_read_barrier();

    const volatile tt_l1_ptr uint32_t* scores = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_scores);
    const volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_ids);
    volatile tt_l1_ptr uint32_t* unf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_unf);
    volatile tt_l1_ptr uint32_t* drawn = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_drawn);
    volatile tt_l1_ptr uint32_t* tok = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_tok);
    for (uint32_t r = 0; r < 16; ++r) {
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

    noc_async_write(l_unf, s_unf.get_noc_addr(0, F0), FACE);
    noc_async_write(l_drawn, s_drawn.get_noc_addr(0, F0), FACE);
    noc_async_write(l_tok, s_tok.get_noc_addr(0, HALF * 64), 64);
    noc_async_write_barrier();
}

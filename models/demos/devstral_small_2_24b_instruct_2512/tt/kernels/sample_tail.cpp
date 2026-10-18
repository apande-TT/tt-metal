// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// End of the decode sampler of tt/pipeline.py SampleTail: for each row b of the (B, 1) draw,
//   nxt[b] = unfinished[b] ? drawn[b] : PAD          (generate(): a finished row emits pad)
//   unfinished[b] = unfinished[b] && nxt[b] != EOS   (a row finishes on the step it emits eos)
//   tok[b] = nxt[b]                                  (the next step's input ids, row-major uint32)
// unfinished is the fp32 0 / 1 mask tile, drawn the uint32 id tile; both hold row b in column 0, i.e. faces 0 / 2
// (rows 0-15 / 16-31), element 16 * (b % 16) of each face. The ids are exact integers below 2^24, so this integer
// select is what the fp32 (drawn - PAD) * u + PAD, ne and multiply chain computes, value for value.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t drawn_addr = get_arg_val<uint32_t>(0);
    const uint32_t unf_addr = get_arg_val<uint32_t>(1);
    const uint32_t tok_addr = get_arg_val<uint32_t>(2);

    constexpr uint32_t B = get_compile_time_arg_val(0);  // rows (<= 32)
    constexpr uint32_t PAD = get_compile_time_arg_val(1);
    constexpr uint32_t EOS = get_compile_time_arg_val(2);
    constexpr uint32_t CB = get_compile_time_arg_val(3);  // scratch: 2 faces of each tile + the id row
    constexpr auto a_drawn = TensorAccessorArgs<4>();
    constexpr auto a_unf = TensorAccessorArgs<a_drawn.next_compile_time_args_offset()>();
    constexpr auto a_tok = TensorAccessorArgs<a_unf.next_compile_time_args_offset()>();
    const auto s_drawn = TensorAccessor(a_drawn, drawn_addr);
    const auto s_unf = TensorAccessor(a_unf, unf_addr);
    const auto s_tok = TensorAccessor(a_tok, tok_addr);

    constexpr uint32_t FACE = 256 * 4;  // bytes of a 16 x 16 face of 32-bit elements
    constexpr uint32_t ONE = 0x3F800000u;  // fp32 1.0
    const uint32_t l_drawn = get_write_ptr(CB);
    const uint32_t l_unf = l_drawn + 2 * FACE;
    const uint32_t l_tok = l_unf + 2 * FACE;

    for (uint32_t f = 0; f < 2; ++f) {  // faces 0 and 2: column 0..15 of rows 0-15 and 16-31
        noc_async_read(s_drawn.get_noc_addr(0, 2 * f * FACE), l_drawn + f * FACE, FACE);
        noc_async_read(s_unf.get_noc_addr(0, 2 * f * FACE), l_unf + f * FACE, FACE);
    }
    noc_async_read_barrier();

    const volatile tt_l1_ptr uint32_t* drawn = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_drawn);
    volatile tt_l1_ptr uint32_t* unf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_unf);
    volatile tt_l1_ptr uint32_t* tok = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_tok);
    for (uint32_t b = 0; b < B; ++b) {
        const uint32_t e = (b >> 4) * 256 + (b & 15) * 16;  // row b, column 0 of the two read faces
        const bool live = unf[e] == ONE;
        const uint32_t nxt = live ? drawn[e] : PAD;
        unf[e] = (live && nxt != EOS) ? ONE : 0u;
        tok[b] = nxt;
    }

    for (uint32_t f = 0; f < 2; ++f) {
        noc_async_write(l_unf + f * FACE, s_unf.get_noc_addr(0, 2 * f * FACE), FACE);
    }
    noc_async_write(l_tok, s_tok.get_noc_addr(0), B * 4);
    noc_async_write_barrier();
}

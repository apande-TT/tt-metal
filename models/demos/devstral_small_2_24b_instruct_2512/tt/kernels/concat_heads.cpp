// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Prefill head concat of tt/concat_heads.py: SDPA's head-major output (B, H, S, hd) -> (B, 1, S, H * hd), tiles.
// The input is L1-interleaved, so every tile already sits in some core's L1: each data-movement RISC pushes the tiles
// of its OWN core's bank(s) straight to their concatenated slots -- one local-source NOC write a tile, no read round
// trip, no circular buffer. Input tile p = ((b * H + h) * ST + s) * DT + d lands on output tile
// (b * ST + s) * H * DT + h * DT + d. The RISCs of a core take alternate pages of its banks (FIRST, STRIDE).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t DT = get_compile_time_arg_val(0);      // tiles across a head (hd / 32)
    constexpr uint32_t ST = get_compile_time_arg_val(1);      // sequence tiles
    constexpr uint32_t H = get_compile_time_arg_val(2);       // heads
    constexpr uint32_t NPAGES = get_compile_time_arg_val(3);  // B * H * ST * DT
    constexpr uint32_t PAGE = get_compile_time_arg_val(4);    // tile bytes (= the aligned page of both tensors)
    constexpr uint32_t FIRST = get_compile_time_arg_val(5);   // this RISC's first page of each local bank
    constexpr uint32_t STRIDE = get_compile_time_arg_val(6);  // RISCs sharing a core's banks
    constexpr auto a_in = TensorAccessorArgs<7>();
    constexpr auto a_out = TensorAccessorArgs<a_in.next_compile_time_args_offset()>();
    const auto s_in = TensorAccessor(a_in, in_addr);
    const auto s_out = TensorAccessor(a_out, out_addr);

    for (uint32_t bank = 0; bank < NUM_L1_BANKS; ++bank) {
        const uint64_t a0 = s_in.get_noc_addr(bank);  // page `bank` is the bank's first
        if (!noc_address_backend::is_local(a0, noc_index)) {
            continue;
        }
        uint32_t src = noc_address_backend::extract_local_address(a0) + FIRST * PAGE;
        for (uint32_t p = bank + FIRST * NUM_L1_BANKS; p < NPAGES; p += STRIDE * NUM_L1_BANKS, src += STRIDE * PAGE) {
            const uint32_t d = p % DT, r = p / DT;
            const uint32_t s = r % ST, bh = r / ST;
            const uint32_t h = bh % H, b = bh / H;
            noc_async_write(src, s_out.get_noc_addr((b * ST + s) * (H * DT) + h * DT + d), PAGE);
        }
    }
    noc_async_write_barrier();
}

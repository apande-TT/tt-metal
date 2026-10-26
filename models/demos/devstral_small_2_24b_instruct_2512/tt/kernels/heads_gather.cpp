// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Head-concat producer of tt/down_decode.py (StreamDownDecode, heads=True): assembles x tiles Q0 .. Q0 + T - 1 of the
// o_proj input from SDPA-decode's head-major (1, B, H, hd) output -- tile q holds, in row b, row h of input tile
// b * DT + d (q = h * DT + d), the copy tt/concat_heads.py's decode concat makes -- in a local scratch laid out as the
// tiles, writes them straight into the sender core's x CB (the same L1 address on every core of the program) at once
// and counts itself on the sender's gather semaphore (gate_up_sender.cpp X_GATHER). A pure copy of bf16 face rows.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t sx = get_arg_val<uint32_t>(1), sy = get_arg_val<uint32_t>(2);  // the sender (NOC coordinates)
    const uint32_t q0 = get_arg_val<uint32_t>(3);                                  // first x tile

    constexpr uint32_t DT = get_compile_time_arg_val(0);    // tiles across a head (hd / 32)
    constexpr uint32_t B = get_compile_time_arg_val(1);     // batch rows (<= 32)
    constexpr uint32_t SLOT = get_compile_time_arg_val(2);  // this RISC's scratch (T tiles)
    constexpr uint32_t CB_X = get_compile_time_arg_val(3);  // the stream's x CB (the sender's is the target)
    constexpr uint32_t CB_S = get_compile_time_arg_val(4);  // scratch (T tiles a RISC)
    constexpr uint32_t SEM = get_compile_time_arg_val(5);   // the sender's gather count
    constexpr uint32_t T = get_compile_time_arg_val(6);     // x tiles a RISC
    constexpr auto a_in = TensorAccessorArgs<7>();
    const auto s_in = TensorAccessor(a_in, in_addr);

    constexpr uint32_t ROW = 32;    // bytes of a 16-element bf16 face row
    constexpr uint32_t FACE = 512;  // bytes of a 16 x 16 bf16 face
    constexpr uint32_t TILE = 4 * FACE;
    const uint32_t scratch = get_write_ptr(CB_S) + SLOT * T * TILE;

    for (uint32_t t = 0; t < T; ++t) {
        const uint32_t q = q0 + t, h = q / DT, d = q % DT;
        const uint32_t src_off = (h / 16) * 2 * FACE + (h % 16) * ROW;  // head h's face rows in its input tile
        for (uint32_t b = 0; b < B; ++b) {
            const uint64_t src = s_in.get_noc_addr(b * DT + d, src_off);
            const uint32_t dst = scratch + t * TILE + (b / 16) * 2 * FACE + (b % 16) * ROW;
            noc_async_read(src, dst, ROW);
            noc_async_read(src + FACE, dst + FACE, ROW);
        }
    }
    noc_async_read_barrier();
    noc_async_write(scratch, get_noc_addr(sx, sy, get_write_ptr(CB_X) + q0 * TILE), T * TILE);
    noc_async_write_barrier();  // the tiles have landed before they are counted
    noc_semaphore_inc(get_noc_addr(sx, sy, get_semaphore(SEM)), 1);
    noc_async_atomic_barrier();
}

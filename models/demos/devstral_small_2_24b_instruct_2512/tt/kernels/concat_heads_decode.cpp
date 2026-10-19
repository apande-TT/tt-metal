// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Decode head concat of tt/concat_heads.py (ConcatHeadsDecode): SDPA-decode's (1, B, H, hd) output -- one tile row a
// batch row b, its H heads in tile rows 0..H-1 -- -> the (1, 1, B, H * hd) row-per-user o_proj input. Output tile
// q = h * DT + d holds, in row b, row h of input tile b * DT + d. This RISC assembles HALF (= 16 rows, faces 2 * HALF
// and 2 * HALF + 1) of output tile Q: per row, the two 16-element face rows of head h are read into a local scratch laid
// out as those two output faces, which then go out as ONE write. A pure copy of bf16 face rows.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t q = get_arg_val<uint32_t>(2);  // output tile

    constexpr uint32_t DT = get_compile_time_arg_val(0);    // tiles across a head (hd / 32)
    constexpr uint32_t B = get_compile_time_arg_val(1);     // batch rows (<= 32)
    constexpr uint32_t HALF = get_compile_time_arg_val(2);  // rows 16 * HALF .. 16 * HALF + 15 of the output tile
    constexpr uint32_t CB = get_compile_time_arg_val(3);    // scratch (1 KB a RISC)
    constexpr auto a_in = TensorAccessorArgs<4>();
    constexpr auto a_out = TensorAccessorArgs<a_in.next_compile_time_args_offset()>();
    const auto s_in = TensorAccessor(a_in, in_addr);
    const auto s_out = TensorAccessor(a_out, out_addr);

    constexpr uint32_t ROW = 32;    // bytes of a 16-element bf16 face row
    constexpr uint32_t FACE = 512;  // bytes of a 16 x 16 bf16 face
    const uint32_t h = q / DT, d = q % DT;
    const uint32_t src_off = (h / 16) * 2 * FACE + (h % 16) * ROW;  // head h's face rows in its input tile
    const uint32_t scratch = get_write_ptr(CB) + HALF * 2 * FACE;

    constexpr uint32_t b0 = 16 * HALF;
    constexpr uint32_t b1 = (b0 + 16 < B) ? b0 + 16 : B;
    if constexpr (b0 < B) {
        for (uint32_t b = b0; b < b1; ++b) {
            const uint64_t src = s_in.get_noc_addr(b * DT + d, src_off);
            const uint32_t dst = scratch + (b - b0) * ROW;
            noc_async_read(src, dst, ROW);
            noc_async_read(src + FACE, dst + FACE, ROW);
        }
        noc_async_read_barrier();
        noc_async_write(scratch, s_out.get_noc_addr(q) + HALF * 2 * FACE, 2 * FACE);
        noc_async_write_barrier();
    }
}

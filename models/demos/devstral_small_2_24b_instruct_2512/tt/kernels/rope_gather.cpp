// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Decode RoPE tables of tt/rope_gather.py: core i of the fused q / k RoPE grid holds batch row b = i % B of the
// (1, 2B, 1, HD) cos / sin tables -- q's rows, then k's -- as one height shard of HD / 32 tiles. It reads cur_pos[b],
// that position's rows of the resident row-major bf16 cos / sin tables, and writes them into row 0 of its shard's
// tiles (face 0 and face 1 row 0: 16 + 16 elements a tile), the only row the RoPE's row broadcast reads.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t pos_addr = get_arg_val<uint32_t>(0);
    const uint32_t cos_tab_addr = get_arg_val<uint32_t>(1);
    const uint32_t sin_tab_addr = get_arg_val<uint32_t>(2);
    const uint32_t cos_out = get_arg_val<uint32_t>(3);  // this core's shard (the sharded buffer's L1 address)
    const uint32_t sin_out = get_arg_val<uint32_t>(4);
    const uint32_t b = get_arg_val<uint32_t>(5);

    constexpr uint32_t HD = get_compile_time_arg_val(0);  // head dim (a multiple of 32)
    constexpr uint32_t CB = get_compile_time_arg_val(1);  // scratch: 64 B of positions + two table rows
    constexpr auto a_pos = TensorAccessorArgs<2>();
    constexpr auto a_cos = TensorAccessorArgs<a_pos.next_compile_time_args_offset()>();
    constexpr auto a_sin = TensorAccessorArgs<a_cos.next_compile_time_args_offset()>();
    const auto s_pos = TensorAccessor(a_pos, pos_addr);
    const auto s_cos = TensorAccessor(a_cos, cos_tab_addr);
    const auto s_sin = TensorAccessor(a_sin, sin_tab_addr);

    constexpr uint32_t ROW = HD * 2;  // bytes of a bf16 table row
    constexpr uint32_t POS_BYTES = 128;  // cur_pos row: up to 32 int32
    const uint32_t l_pos = get_write_ptr(CB);
    const uint32_t l_cos = l_pos + POS_BYTES;
    const uint32_t l_sin = l_cos + ROW;

    noc_async_read(s_pos.get_noc_addr(0), l_pos, POS_BYTES);
    noc_async_read_barrier();
    const uint32_t p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_pos)[b];
    noc_async_read(s_cos.get_noc_addr(p), l_cos, ROW);
    noc_async_read(s_sin.get_noc_addr(p), l_sin, ROW);
    noc_async_read_barrier();

    // table elements 32 d .. 32 d + 15 -> tile d face 0 row 0, 32 d + 16 .. 32 d + 31 -> tile d face 1 row 0
    for (uint32_t t = 0; t < 2; ++t) {
        const volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(t ? l_sin : l_cos);
        volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(t ? sin_out : cos_out);
        for (uint32_t d = 0; d < HD / 32; ++d) {
            for (uint32_t w = 0; w < 8; ++w) {  // 8 words = 16 bf16
                dst[d * 512 + w] = src[d * 16 + w];             // tile d (2048 B = 512 words), face 0 row 0
                dst[d * 512 + 128 + w] = src[d * 16 + 8 + w];   // face 1 row 0 (+512 B)
            }
        }
    }
}

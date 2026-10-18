// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of tt/embed_decode.py: this core owns columns [C * WT * 32, (C + 1) * WT * 32) of the decode residual. It reads
// the B next-token ids, then each id's slice of those columns from the row-major bf16 embedding table into row b of
// a WT-tile-wide row-major block (WT bf16 tile pages), the compute kernel's tilize input.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t tok_addr = get_arg_val<uint32_t>(0);
    const uint32_t table_addr = get_arg_val<uint32_t>(1);
    const uint32_t c = get_arg_val<uint32_t>(2);  // this core's width shard

    constexpr uint32_t B = get_compile_time_arg_val(0);   // token rows (<= 32)
    constexpr uint32_t WT = get_compile_time_arg_val(1);  // tiles across a shard
    constexpr uint32_t CB_TOK = get_compile_time_arg_val(2);
    constexpr uint32_t CB_RM = get_compile_time_arg_val(3);
    constexpr auto a_tok = TensorAccessorArgs<4>();
    constexpr auto a_table = TensorAccessorArgs<a_tok.next_compile_time_args_offset()>();
    const auto s_tok = TensorAccessor(a_tok, tok_addr);
    const auto s_table = TensorAccessor(a_table, table_addr);

    constexpr uint32_t ROW = WT * 32 * 2;  // bytes of one token's shard slice
    const uint32_t l_tok = get_write_ptr(CB_TOK);
    noc_async_read(s_tok.get_noc_addr(0), l_tok, B * 4);
    noc_async_read_barrier();
    const volatile tt_l1_ptr uint32_t* tok = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_tok);

    cb_reserve_back(CB_RM, WT);
    const uint32_t l_rm = get_write_ptr(CB_RM);
    for (uint32_t b = 0; b < B; ++b) {
        noc_async_read(s_table.get_noc_addr(tok[b], c * ROW), l_rm + b * ROW, ROW);
    }
    noc_async_read_barrier();
    cb_push_back(CB_RM, WT);
}

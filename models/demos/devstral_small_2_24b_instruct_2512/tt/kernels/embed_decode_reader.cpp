// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of tt/embed_decode.py: this core owns columns [C * WT * 32, (C + 1) * WT * 32) of the decode residual.
// Phase 1: core b < B reads the WHOLE table row of token b (one contiguous DRAM read) into its row scratch (the same
// L1 address on every core) and counts itself in on every core's arrival semaphore. Phase 2: once all B rows have
// arrived, this core gathers its slice of each from the holders' L1 into row b of a WT-tile-wide row-major block
// (WT bf16 tile pages), the compute kernel's tilize input. B large reads instead of B x cores small DRAM reads.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t tok_addr = get_arg_val<uint32_t>(0);
    const uint32_t table_addr = get_arg_val<uint32_t>(1);
    const uint32_t c = get_arg_val<uint32_t>(2);  // this core's width shard (and, below B, the token row it holds)
    // runtime args 3 .. 3 + 2 * N - 1: the NOC (x, y) of every core of the program, in shard order

    constexpr uint32_t B = get_compile_time_arg_val(0);   // token rows (<= 32, <= cores)
    constexpr uint32_t WT = get_compile_time_arg_val(1);  // tiles across a shard
    constexpr uint32_t CB_TOK = get_compile_time_arg_val(2);
    constexpr uint32_t CB_RM = get_compile_time_arg_val(3);
    constexpr uint32_t CB_ROW = get_compile_time_arg_val(4);
    constexpr uint32_t SEM = get_compile_time_arg_val(5);
    constexpr uint32_t N = get_compile_time_arg_val(6);         // cores
    constexpr uint32_t ROW_BYTES = get_compile_time_arg_val(7);  // bytes of a whole table row
    constexpr auto a_tok = TensorAccessorArgs<8>();
    constexpr auto a_table = TensorAccessorArgs<a_tok.next_compile_time_args_offset()>();
    const auto s_tok = TensorAccessor(a_tok, tok_addr);
    const auto s_table = TensorAccessor(a_table, table_addr);

    constexpr uint32_t SLICE = WT * 32 * 2;  // bytes of one token's shard slice
    const uint32_t sem = get_semaphore(SEM);
    const uint32_t l_row = get_write_ptr(CB_ROW);

    if (c < B) {
        const uint32_t l_tok = get_write_ptr(CB_TOK);
        noc_async_read(s_tok.get_noc_addr(0), l_tok, B * 4);
        noc_async_read_barrier();
        const uint32_t t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_tok)[c];
        noc_async_read(s_table.get_noc_addr(t), l_row, ROW_BYTES);
        noc_async_read_barrier();
        for (uint32_t i = 0; i < N; ++i) {
            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(3 + 2 * i), get_arg_val<uint32_t>(4 + 2 * i), sem), 1);
        }
        noc_async_atomic_barrier();
    }
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem), B);

    cb_reserve_back(CB_RM, WT);
    const uint32_t l_rm = get_write_ptr(CB_RM);
    for (uint32_t b = 0; b < B; ++b) {
        const uint64_t src = get_noc_addr(get_arg_val<uint32_t>(3 + 2 * b), get_arg_val<uint32_t>(4 + 2 * b), l_row);
        noc_async_read(src + c * SLICE, l_rm + b * SLICE, SLICE);
    }
    noc_async_read_barrier();
    cb_push_back(CB_RM, WT);
}

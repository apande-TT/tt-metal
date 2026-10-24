// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Data movement of tt/kv_append.py, one per RISC: this RISC appends kv heads HEAD0, HEAD0 + HSTEP, ... of user b
// (the core's index in its input's shard grid) at row u = idx[b] of the bfloat8_b cache (B, NKV, ST * 32, WT * 32).
// In a bfp8 tile, row r lives in faces F = 2 * (r / 16) and F + 1 at face row i = r % 16: one exponent byte each
// (bytes F * 16 + i, F * 16 + 16 + i of the 64-byte exponent section) and 16 mantissa bytes each
// (64 + F * 256 + i * 16, 64 + (F + 1) * 256 + i * 16). The compute packs the input row h to bfp8 with the same
// packer the stock tilize uses, so this RISC reads each of the head's WT cache tiles' exponent section (64 B),
// splices in row h's two exponent bytes, and writes back the two 16-byte exponent groups plus the row's two
// mantissa face rows -- every other byte of the tile stays as it is (the stock untilize -> tilize round trip
// rewrites them unchanged). u == 0xFFFFFFFF skips the update, as the stock op does.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t cache_addr = get_arg_val<uint32_t>(0);
    const uint32_t idx_addr = get_arg_val<uint32_t>(1);
    const uint32_t b = get_arg_val<uint32_t>(2);

    constexpr uint32_t WT = get_compile_time_arg_val(0);     // tiles across a head row
    constexpr uint32_t HEAD0 = get_compile_time_arg_val(1);  // first kv head of this RISC
    constexpr uint32_t HSTEP = get_compile_time_arg_val(2);
    constexpr uint32_t NKV = get_compile_time_arg_val(3);
    constexpr uint32_t ST = get_compile_time_arg_val(4);  // tile rows of one head's cache
    constexpr uint32_t CB_IN = get_compile_time_arg_val(5);
    constexpr uint32_t CB_BF8 = get_compile_time_arg_val(6);
    constexpr uint32_t CB_SCR = get_compile_time_arg_val(7);
    constexpr uint32_t IDX_BYTES = get_compile_time_arg_val(8);
    constexpr uint32_t SCR_OFF = get_compile_time_arg_val(9);  // this RISC's part of the scratch CB
    constexpr auto a_cache = TensorAccessorArgs<10>();
    constexpr auto a_idx = TensorAccessorArgs<a_cache.next_compile_time_args_offset()>();
    const auto s_cache = TensorAccessor(a_cache, cache_addr);
    const auto s_idx = TensorAccessor(a_idx, idx_addr);

    constexpr uint32_t BF8_TILE = 1088, EXP = 64, FACE = 256;
    if constexpr (HEAD0 == 0) {
        cb_reserve_back(CB_IN, WT);  // the input shard is the CB
        cb_push_back(CB_IN, WT);
    }

    // DRAM reads land at the same 64-byte offset they start at: the scratch is 64-aligned
    const uint32_t scr = ((get_write_ptr(CB_SCR) + SCR_OFF + 63) & ~63u);
    const uint32_t l_idx = scr;
    const uint32_t l_exp = scr + ((IDX_BYTES + 63) & ~63u);
    noc_async_read(s_idx.get_noc_addr(0), l_idx, IDX_BYTES);
    noc_async_read_barrier();
    const uint32_t u = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_idx)[b];
    if (u == 0xFFFFFFFFu) {
        return;
    }
    const uint32_t r = u % 32, i = r % 16, f = 2 * (r / 16);
    const uint32_t row_tile0 = (b * NKV * ST + u / 32) * WT;

    for (uint32_t h = HEAD0; h < NKV; h += HSTEP) {
        const uint32_t tile0 = row_tile0 + h * ST * WT;
        const uint32_t l_head = l_exp + (h - HEAD0) / HSTEP * WT * EXP;
        for (uint32_t w = 0; w < WT; ++w) {
            noc_async_read(s_cache.get_noc_addr(tile0 + w), l_head + w * EXP, EXP);
        }
    }
    cb_wait_front(CB_BF8, WT);
    const uint32_t l_bf8 = get_read_ptr(CB_BF8);
    noc_async_read_barrier();
    for (uint32_t h = HEAD0; h < NKV; h += HSTEP) {
        const uint32_t tile0 = row_tile0 + h * ST * WT;
        for (uint32_t w = 0; w < WT; ++w) {
            const uint32_t src = l_bf8 + w * BF8_TILE;  // input row h: faces 0 and 1, face row h
            const uint32_t ex = l_exp + ((h - HEAD0) / HSTEP * WT + w) * EXP;
            volatile tt_l1_ptr uint8_t* e = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(ex);
            volatile tt_l1_ptr uint8_t* s = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(src);
            e[f * 16 + i] = s[h];
            e[f * 16 + 16 + i] = s[16 + h];
            const uint32_t t = tile0 + w;
            noc_async_write(ex + f * 16, s_cache.get_noc_addr(t, f * 16), 32);
            noc_async_write(src + EXP + h * 16, s_cache.get_noc_addr(t, EXP + f * FACE + i * 16), 16);
            noc_async_write(src + EXP + FACE + h * 16, s_cache.get_noc_addr(t, EXP + (f + 1) * FACE + i * 16), 16);
        }
    }
    noc_async_write_barrier();
}

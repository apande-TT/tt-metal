// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The input stream of tt/cpp_qkv_rope_dec.py, one user (b) a core. The fused qkv projection `[1, 1, 32, (NQ +
// 2 NKV) * D]` holds every user as a ROW of each (head, column tile) tile; the decode layout wants each user as
// ONE tile row whose rows are the heads. So user b's q tile j gathers row b of fused tiles h * DT + j (h < NQ)
// into its rows h -- two 64-byte face segments a row, float32 -- and likewise k (NKV heads, rows past NKV zero).
// v needs no RoPE: its rows go straight into this core's own shard of the height-sharded v output. The cos /
// signed-sin full tile rows (cpp_rope_dec.full_rows) are read once.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

inline uint32_t seg(uint32_t r, uint32_t s) { return (((r >> 4) * 2 + s) * 1024) + (r & 15) * 64; }

void kernel_main() {
    const uint32_t f_addr = get_arg_val<uint32_t>(0);
    const uint32_t cos_addr = get_arg_val<uint32_t>(1);
    const uint32_t sin_addr = get_arg_val<uint32_t>(2);
    const uint32_t b = get_arg_val<uint32_t>(3);       // this core's user (row of the fused tiles)
    const uint32_t v_addr = get_arg_val<uint32_t>(4);  // this core's v shard (local L1)

    constexpr uint32_t DT = get_compile_time_arg_val(0);
    constexpr uint32_t NQ = get_compile_time_arg_val(1);
    constexpr uint32_t NKV = get_compile_time_arg_val(2);
    constexpr auto af = TensorAccessorArgs<3>();
    constexpr auto ac = TensorAccessorArgs<af.next_compile_time_args_offset()>();
    constexpr auto as = TensorAccessorArgs<ac.next_compile_time_args_offset()>();
    const auto sf = TensorAccessor(af, f_addr);
    const auto sc = TensorAccessor(ac, cos_addr);
    const auto ss = TensorAccessor(as, sin_addr);

    constexpr uint32_t cb_x = 0;
    constexpr uint32_t cb_cos = 1;
    constexpr uint32_t cb_sin = 2;
    constexpr uint32_t bytes = 4096;  // float32 tile

    cb_reserve_back(cb_cos, DT);
    cb_reserve_back(cb_sin, DT);
    const uint32_t lc = get_write_ptr(cb_cos);
    const uint32_t ls = get_write_ptr(cb_sin);
    for (uint32_t j = 0; j < DT; ++j) {
        noc_async_read_page(j, sc, lc + j * bytes);
        noc_async_read_page(j, ss, ls + j * bytes);
    }

    // q: a full tile row (NQ = 32 head rows)
    cb_reserve_back(cb_x, DT);
    uint32_t lq = get_write_ptr(cb_x);
    for (uint32_t j = 0; j < DT; ++j) {
        for (uint32_t h = 0; h < NQ; ++h) {
            for (uint32_t s = 0; s < 2; ++s) {
                noc_async_read(sf.get_noc_addr(h * DT + j, seg(b, s)), lq + j * bytes + seg(h, s), 64);
            }
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_cos, DT);
    cb_push_back(cb_sin, DT);
    cb_push_back(cb_x, DT);

    // k: NKV head rows, the rest zero; v: the same gather into the local shard
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    cb_reserve_back(cb_x, DT);
    const uint32_t lk = get_write_ptr(cb_x);
    for (uint32_t off = 0; off < DT * bytes; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, lk + off, MEM_ZEROS_SIZE);
        noc_async_read(zeros, v_addr + off, MEM_ZEROS_SIZE);
    }
    noc_async_read_barrier();
    for (uint32_t j = 0; j < DT; ++j) {
        for (uint32_t h = 0; h < NKV; ++h) {
            for (uint32_t s = 0; s < 2; ++s) {
                noc_async_read(sf.get_noc_addr((NQ + h) * DT + j, seg(b, s)), lk + j * bytes + seg(h, s), 64);
                noc_async_read(
                    sf.get_noc_addr((NQ + NKV + h) * DT + j, seg(b, s)), v_addr + j * bytes + seg(h, s), 64);
            }
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, DT);
}

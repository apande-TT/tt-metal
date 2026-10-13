// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode KV-cache write, reader.
// A work unit is (cache c in {K, V}, batch row b, kv head h): the D-wide row at slot pos[b] of cache[b, h].
// For each of the Dt column tiles of that row:
//   cb_old <- the cache tile holding the slot (row r = pos[b] % 32 is the slot's current value)
//   cb_new <- a tile whose row r is the new K/V row (other rows are don't-care; only row r is written back)
// New K comes from the roped k [B, kv, rows, D] (row k_row of tile (b, h)); new V straight from the fused-QKV
// output [1, B, (n + 2 kv) * D] (row b % 32 of its column tiles), so no reshape/reshard runs before it.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t kFaceBytes = 16 * 16 * 2;  // bf16 16x16 face
constexpr uint32_t kFaceRowBytes = 16 * 2;

// Byte offset of the half-row (cols 0-15 or 16-31) of tile row `row` in a bf16 32x32 tile.
inline uint32_t face_row_offset(uint32_t row, uint32_t half) {
    return (((row >> 4) << 1) + half) * kFaceBytes + (row & 15) * kFaceRowBytes;
}

inline void copy_row(uint32_t src_tile, uint32_t src_row, uint32_t dst_tile, uint32_t dst_row) {
    for (uint32_t half = 0; half < 2; ++half) {
        auto* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_tile + face_row_offset(src_row, half));
        auto* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_tile + face_row_offset(dst_row, half));
        for (uint32_t i = 0; i < kFaceRowBytes / 4; ++i) {
            dst[i] = src[i];
        }
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t n_kv = get_compile_time_arg_val(1);
    constexpr uint32_t Ct = get_compile_time_arg_val(2);
    constexpr uint32_t Dt = get_compile_time_arg_val(3);
    constexpr uint32_t qkv_Nt = get_compile_time_arg_val(4);
    constexpr uint32_t v_col0 = get_compile_time_arg_val(5);
    constexpr uint32_t cb_old = get_compile_time_arg_val(6);
    constexpr uint32_t cb_new = get_compile_time_arg_val(7);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(8);
    constexpr uint32_t cb_pos = get_compile_time_arg_val(9);
    constexpr uint32_t k_row = get_compile_time_arg_val(10);
    constexpr auto cache_args = TensorAccessorArgs<11>();
    constexpr auto k_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    constexpr auto qkv_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<qkv_args.next_compile_time_args_offset()>();

    const uint32_t unit_start = get_arg_val<uint32_t>(0);
    const uint32_t num_units = get_arg_val<uint32_t>(1);
    // common args: [cache_k, cache_v, k_new, qkv, cur_pos]
    const auto cache_k = TensorAccessor(cache_args, get_common_arg_val<uint32_t>(0));
    const auto cache_v = TensorAccessor(cache_args, get_common_arg_val<uint32_t>(1));
    const auto k_new = TensorAccessor(k_args, get_common_arg_val<uint32_t>(2));
    const auto qkv = TensorAccessor(qkv_args, get_common_arg_val<uint32_t>(3));
    const auto cur_pos = TensorAccessor(pos_args, get_common_arg_val<uint32_t>(4));

    const uint32_t pos_l1 = get_write_ptr(cb_pos);
    noc_async_read_page(0, cur_pos, pos_l1);
    noc_async_read_barrier();
    auto* pos = reinterpret_cast<volatile tt_l1_ptr int32_t*>(pos_l1);

    const uint32_t scr = get_write_ptr(cb_scr);
    for (uint32_t u = unit_start; u < unit_start + num_units; ++u) {
        const uint32_t is_v = u / (B * n_kv);
        const uint32_t bh = u - is_v * (B * n_kv);
        const uint32_t b = bh / n_kv;
        const uint32_t h = bh - b * n_kv;
        const uint32_t p = static_cast<uint32_t>(pos[b]);
        const uint32_t r = p & 31;
        const uint32_t cache_page0 = (bh * Ct + (p >> 5)) * Dt;
        for (uint32_t wt = 0; wt < Dt; ++wt) {
            cb_reserve_back(cb_old, 1);
            cb_reserve_back(cb_new, 1);
            const uint32_t old_l1 = get_write_ptr(cb_old);
            uint32_t src_row;
            if (is_v) {
                noc_async_read_page(cache_page0 + wt, cache_v, old_l1);
                noc_async_read_page((b >> 5) * qkv_Nt + v_col0 + h * Dt + wt, qkv, scr);
                src_row = b & 31;
            } else {
                noc_async_read_page(cache_page0 + wt, cache_k, old_l1);
                noc_async_read_page(bh * Dt + wt, k_new, scr);
                src_row = k_row;
            }
            noc_async_read_barrier();
            copy_row(scr, src_row, get_write_ptr(cb_new), r);
            cb_push_back(cb_old, 1);
            cb_push_back(cb_new, 1);
        }
    }
}

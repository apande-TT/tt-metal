// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode KV-cache write, writer: for each (cache, b, h) unit and column tile, re-read the cache
// tile holding slot pos[b], replace only row pos[b] % 32 with the computed row, and write the tile back in
// place. Each cache tile belongs to exactly one unit, so no other core touches it.
// With `signal` set (the write fused into the decode attention program), once a unit's tiles are written the unit's
// attention core (row block b * n_kv + h, NoC coordinates in the common args from index 5) gets its K or V
// semaphore bumped, so it streams the slot's tile only after the write landed.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t kFaceBytes = 16 * 16 * 2;
constexpr uint32_t kFaceRowBytes = 16 * 2;

inline uint32_t face_row_offset(uint32_t row, uint32_t half) {
    return (((row >> 4) << 1) + half) * kFaceBytes + (row & 15) * kFaceRowBytes;
}
}  // namespace

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t n_kv = get_compile_time_arg_val(1);
    constexpr uint32_t Ct = get_compile_time_arg_val(2);
    constexpr uint32_t Dt = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);
    constexpr uint32_t cb_wscr = get_compile_time_arg_val(5);
    constexpr uint32_t cb_wpos = get_compile_time_arg_val(6);
    constexpr auto cache_args = TensorAccessorArgs<7>();
    constexpr auto pos_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    constexpr uint32_t signal = get_compile_time_arg_val(pos_args.next_compile_time_args_offset());
    constexpr uint32_t sem_k_id = get_compile_time_arg_val(pos_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t sem_v_id = get_compile_time_arg_val(pos_args.next_compile_time_args_offset() + 2);

    const uint32_t unit_start = get_arg_val<uint32_t>(0);
    const uint32_t num_units = get_arg_val<uint32_t>(1);
    const auto cache_k = TensorAccessor(cache_args, get_common_arg_val<uint32_t>(0));
    const auto cache_v = TensorAccessor(cache_args, get_common_arg_val<uint32_t>(1));
    const auto cur_pos = TensorAccessor(pos_args, get_common_arg_val<uint32_t>(4));

    const uint32_t pos_l1 = get_write_ptr(cb_wpos);
    noc_async_read_page(0, cur_pos, pos_l1);
    noc_async_read_barrier();
    auto* pos = reinterpret_cast<volatile tt_l1_ptr int32_t*>(pos_l1);

    // A unit's Dt cache tiles are read under one barrier and written back under one barrier.
    constexpr uint32_t tile_bytes = 32 * 32 * 2;
    const uint32_t tiles = get_write_ptr(cb_wscr);
    for (uint32_t u = unit_start; u < unit_start + num_units; ++u) {
        const uint32_t is_v = u / (B * n_kv);
        const uint32_t bh = u - is_v * (B * n_kv);
        const uint32_t b = bh / n_kv;
        const uint32_t p = static_cast<uint32_t>(pos[b]);
        const uint32_t r = p & 31;
        const uint32_t cache_page0 = (bh * Ct + (p >> 5)) * Dt;
        for (uint32_t wt = 0; wt < Dt; ++wt) {
            noc_async_read_page(cache_page0 + wt, is_v ? cache_v : cache_k, tiles + wt * tile_bytes);
        }
        noc_async_read_barrier();
        for (uint32_t wt = 0; wt < Dt; ++wt) {
            const uint32_t page = cache_page0 + wt;
            const uint32_t tile = tiles + wt * tile_bytes;
            cb_wait_front(cb_out, 1);
            const uint32_t out = get_read_ptr(cb_out);
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t off = face_row_offset(r, half);
                auto* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out + off);
                auto* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile + off);
                for (uint32_t i = 0; i < kFaceRowBytes / 4; ++i) {
                    dst[i] = src[i];
                }
            }
            cb_pop_front(cb_out, 1);
            noc_async_write_page(page, is_v ? cache_v : cache_k, tile);
        }
        noc_async_write_barrier();  // the next unit reuses the scratch tiles
        if constexpr (signal) {
            const uint32_t x = get_common_arg_val<uint32_t>(5 + 2 * bh);
            const uint32_t y = get_common_arg_val<uint32_t>(6 + 2 * bh);
            noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(is_v ? sem_v_id : sem_k_id)), 1);
        }
    }
    if constexpr (signal) {
        noc_async_atomic_barrier();
    }
}

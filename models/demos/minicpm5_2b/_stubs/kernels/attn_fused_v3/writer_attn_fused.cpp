// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B fused decode attention writer with the inline KV-cache write: once the reader has spliced this
// step's key and value rows into its staged position tiles (cb_kv_old: K's Kt tiles, then V's) and set kv_flag,
// stores them to the caches; then the probs.V head merge of kernels/attn_bmm/writer_attn_merge.cpp.

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
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t n_kv = get_compile_time_arg_val(1);
    constexpr uint32_t groups = get_compile_time_arg_val(2);
    constexpr uint32_t Dt = get_compile_time_arg_val(3);
    constexpr uint32_t Ct = get_compile_time_arg_val(4);
    constexpr uint32_t cb_kv_old = get_compile_time_arg_val(5);
    constexpr uint32_t cb_wpos = get_compile_time_arg_val(6);
    constexpr uint32_t kv_flag = get_compile_time_arg_val(7);
    constexpr auto out_args = TensorAccessorArgs<8>();
    constexpr auto cache_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    // The row block's G = groups + 1 head rows of the roped QKV layout q [1, B, H * G * D] (row b = batch b),
    // gathered into rows 0..G-1 of its Dt q tiles for compute (the reader's q_rows gather, moved to this RISC so it
    // runs on the other NoC while the reader fetches the KV-write operands).
    constexpr uint32_t cb_q = get_compile_time_arg_val(pos_args.next_compile_time_args_offset());
    constexpr auto q_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset() + 1>();
    constexpr uint32_t tile_bytes = 32 * 32 * 2;

    const uint32_t t_start = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(0));
    const auto kc = TensorAccessor(cache_args, get_common_arg_val<uint32_t>(1));
    const auto vc = TensorAccessor(cache_args, get_common_arg_val<uint32_t>(2));
    const auto cur_pos = TensorAccessor(pos_args, get_common_arg_val<uint32_t>(3));
    const uint32_t bg = t_start / Dt;
    const uint32_t b = bg / n_kv;

    {
        constexpr uint32_t G = groups + 1;
        const auto q = TensorAccessor(q_args, get_common_arg_val<uint32_t>(4));
        const uint32_t h = bg - b * n_kv;
        cb_reserve_back(cb_q, Dt);
        const uint32_t l1 = get_write_ptr(cb_q);
        for (uint32_t j = 0; j < G; ++j) {
            for (uint32_t k = 0; k < Dt; ++k) {
                for (uint32_t half = 0; half < 2; ++half) {
                    noc_async_read(
                        q.get_noc_addr((h * G + j) * Dt + k, face_row_offset(b & 31, half)),
                        l1 + k * tile_bytes + face_row_offset(j, half),
                        kFaceRowBytes);
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_q, Dt);
    }

    const uint32_t pos_l1 = get_write_ptr(cb_wpos);
    noc_async_read_page(0, cur_pos, pos_l1);
    noc_async_read_barrier();
    const uint32_t last_key_tile = static_cast<uint32_t>(reinterpret_cast<volatile tt_l1_ptr int32_t*>(pos_l1)[b]) >> 5;

    auto* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(kv_flag));
    noc_semaphore_wait(flag, 1);
    noc_semaphore_set(flag, 0);
    const uint32_t staged = get_write_ptr(cb_kv_old);
    for (uint32_t dt = 0; dt < Dt; ++dt) {
        const uint32_t page = (bg * Ct + last_key_tile) * Dt + dt;
        noc_async_write_page(page, kc, staged + dt * tile_bytes);
        noc_async_write_page(page, vc, staged + (Dt + dt) * tile_bytes);
    }

    for (uint32_t t = t_start; t < t_start + num_tiles; ++t) {
        const uint32_t dt = t % Dt;
        const uint32_t g = bg % n_kv;
        cb_wait_front(cb_out, 1);
        const uint32_t src = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < groups; ++j) {
            const uint32_t page = (g * groups + j) * Dt + dt;
            for (uint32_t half = 0; half < 2; ++half) {
                noc_async_write(
                    src + face_row_offset(j, half), out.get_noc_addr(page, face_row_offset(b, half)), kFaceRowBytes);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
    noc_async_write_barrier();
}

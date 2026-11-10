// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B fused decode attention reader for the serving layout (q_rows, inline_kv, the mask table): one (batch,
// KV head) row block bg per core. The tiles of kernels/attn_fused's reader, in an order that keeps fewer reads
// waiting on one another:
//   1. the position; then in one batch the score scale, the softmax scalers, the position tile's mask tile and this
//      step's KV-write operands (per unit, K then V, the Kt old position tiles and the new row);
//   2. the key tiles before the position tile, Kt cache tiles each and no mask tile (theirs is the table's -0 tile,
//      whose addition compute skips), which compute scores before it does the KV-write arithmetic;
//   3. the position tile with this step's key row spliced in;
//   4. the value tiles per key tile (its Kt column tiles), V_BLK key tiles per push into a ring of whole blocks.
// Only the nt = position tile + 1 key tiles are streamed (nt handed to compute through cb_cnt): the tiles past it
// are fully masked, so their scores add exact zeros to the softmax sum and to probs.V.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);  // head_dim tiles
    constexpr uint32_t Ct = get_compile_time_arg_val(1);  // cache capacity tiles
    constexpr uint32_t H = get_compile_time_arg_val(2);   // KV heads
    constexpr uint32_t cb_k = get_compile_time_arg_val(4);
    constexpr uint32_t cb_scale = get_compile_time_arg_val(5);
    constexpr uint32_t cb_mask = get_compile_time_arg_val(6);
    constexpr uint32_t cb_pos = get_compile_time_arg_val(7);
    constexpr uint32_t cb_max_scaler = get_compile_time_arg_val(9);
    constexpr uint32_t cb_sum_scaler = get_compile_time_arg_val(10);
    constexpr uint32_t cb_v = get_compile_time_arg_val(11);
    constexpr uint32_t cb_cnt = get_compile_time_arg_val(12);
    constexpr uint32_t tile_bytes = 32 * 32 * 2;
    constexpr auto q_args = TensorAccessorArgs<13>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto scale_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto mask_args = TensorAccessorArgs<scale_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<mask_args.next_compile_time_args_offset()>();
    constexpr auto zero_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    constexpr auto mx_args = TensorAccessorArgs<zero_args.next_compile_time_args_offset()>();
    constexpr auto sm_args = TensorAccessorArgs<mx_args.next_compile_time_args_offset()>();
    constexpr uint32_t base = sm_args.next_compile_time_args_offset();
    constexpr uint32_t wait_kv = get_compile_time_arg_val(base);
    constexpr uint32_t q_rows = get_compile_time_arg_val(base + 3);
    // q is the roped QKV layout [1, B, H * G * D] (row b = batch b, G = groups + 1 heads per KV head; the writer
    // gathers the row block's q rows).
    constexpr uint32_t G = get_compile_time_arg_val(base + 4);
    // inline_kv: this step's KV-cache write is done by this row block's cores: the old position tiles and the new
    // rows (K from q, V from qkv) go to compute, which forms old + bf16(new - old) with kernels/kv_write's arithmetic;
    // its row r is spliced into the staged position tiles (which the writer stores) before they are streamed.
    constexpr uint32_t inline_kv = get_compile_time_arg_val(base + 5);
    constexpr uint32_t cb_kv_old = get_compile_time_arg_val(base + 6);
    constexpr uint32_t cb_kv_new = get_compile_time_arg_val(base + 7);
    constexpr uint32_t cb_kv_out = get_compile_time_arg_val(base + 9);
    constexpr uint32_t v_col0 = get_compile_time_arg_val(base + 10);
    constexpr uint32_t qkv_Nt = get_compile_time_arg_val(base + 11);
    // kv_flag: semaphore set once both staged position tiles hold this step's row (the writer stores them).
    constexpr uint32_t kv_flag = get_compile_time_arg_val(base + 12);
    constexpr auto qkv_args = TensorAccessorArgs<base + 13>();
    // `mask` is the table [tiles r = the position at column r, r < 32 | R unmasked (-0) tiles | R score-scale tiles |
    // R max-scaler tiles] (one copy per row block, read by row block bg only) and the sum scaler holds R tiles.
    constexpr uint32_t R = get_compile_time_arg_val(qkv_args.next_compile_time_args_offset());
    constexpr uint32_t V_BLK = get_compile_time_arg_val(qkv_args.next_compile_time_args_offset() + 1);
    static_assert(inline_kv && q_rows && !wait_kv && R, "the serving layout: q_rows, inline_kv and the mask table");
    constexpr uint32_t face_bytes = 16 * 16 * 2, face_row_bytes = 16 * 2;
    auto face_row_off = [](uint32_t row, uint32_t half) {
        return (((row >> 4) << 1) + half) * face_bytes + (row & 15) * face_row_bytes;
    };
    auto copy_row = [&](uint32_t src_tile, uint32_t src_row, uint32_t dst_tile, uint32_t dst_row) {
        for (uint32_t half = 0; half < 2; ++half) {
            auto* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_tile + face_row_off(src_row, half));
            auto* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_tile + face_row_off(dst_row, half));
            for (uint32_t i = 0; i < face_row_bytes / 4; ++i) {
                dst[i] = src[i];
            }
        }
    };

    const uint32_t bg = get_arg_val<uint32_t>(0);
    const auto q = TensorAccessor(q_args, get_common_arg_val<uint32_t>(0));
    const auto kc = TensorAccessor(k_args, get_common_arg_val<uint32_t>(1));
    const auto vc = TensorAccessor(v_args, get_common_arg_val<uint32_t>(2));
    const auto scale = TensorAccessor(scale_args, get_common_arg_val<uint32_t>(3));
    const auto mask = TensorAccessor(mask_args, get_common_arg_val<uint32_t>(4));
    const auto cur_pos = TensorAccessor(pos_args, get_common_arg_val<uint32_t>(5));
    const auto max_scaler = TensorAccessor(mx_args, get_common_arg_val<uint32_t>(7));
    const auto sum_scaler = TensorAccessor(sm_args, get_common_arg_val<uint32_t>(8));
    const auto qkv = TensorAccessor(qkv_args, get_common_arg_val<uint32_t>(9));
    const uint32_t b = bg / H;
    const uint32_t hh = bg - b * H;

    // 1. The position, then the constants and the KV-write operands in one batch.
    const uint32_t pos_l1 = get_write_ptr(cb_pos);
    noc_async_read_page(0, cur_pos, pos_l1);
    noc_async_read_barrier();
    const uint32_t pos = static_cast<uint32_t>(reinterpret_cast<volatile tt_l1_ptr int32_t*>(pos_l1)[b]);
    const uint32_t last_key_tile = pos >> 5;
    const uint32_t r = pos & 31;
    const uint32_t nt = last_key_tile + 1 < Ct ? last_key_tile + 1 : Ct;
    const uint32_t n_full = nt - 1;  // key tiles streamed before the position tile
    cb_reserve_back(cb_cnt, 1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_cnt))[0] = nt;
    cb_push_back(cb_cnt, 1);

    cb_reserve_back(cb_scale, 1);
    cb_reserve_back(cb_max_scaler, 1);
    cb_reserve_back(cb_sum_scaler, 1);
    cb_reserve_back(cb_mask, 1);
    noc_async_read_page(32 + R + bg, scale, get_write_ptr(cb_scale));
    noc_async_read_page(32 + 2 * R + bg, max_scaler, get_write_ptr(cb_max_scaler));
    noc_async_read_page(bg, sum_scaler, get_write_ptr(cb_sum_scaler));
    noc_async_read_page(n_full == last_key_tile ? r : 32 + bg, mask, get_write_ptr(cb_mask));
    // The KV write's operands: per unit (K, then V) the Kt old position tiles and Kt tiles whose row r is the new row
    // (the rest of those tiles is not used: only row r of the result is kept). The new row: only row b of its source
    // tile (two 32 B face rows) lands straight in row r -- every core of a KV head reads the same source tiles, so
    // whole-tile reads made those pages hotspots.
    cb_reserve_back(cb_kv_old, 2 * Kt);
    cb_reserve_back(cb_kv_new, 2 * Kt);
    const uint32_t kv_old_base = get_write_ptr(cb_kv_old);  // K's Kt staged tiles, then V's (the CB is filled once)
    const uint32_t kv_new_base = get_write_ptr(cb_kv_new);
    for (uint32_t unit = 0; unit < 2; ++unit) {
        for (uint32_t dt = 0; dt < Kt; ++dt) {
            const uint32_t page = (bg * Ct + last_key_tile) * Kt + dt;
            const uint32_t old_l1 = kv_old_base + (unit * Kt + dt) * tile_bytes;
            const uint32_t new_l1 = kv_new_base + (unit * Kt + dt) * tile_bytes;
            const uint32_t src_page =
                unit == 0 ? (hh * G + G - 1) * Kt + dt : (b >> 5) * qkv_Nt + v_col0 + hh * Kt + dt;
            if (unit == 0) {
                noc_async_read_page(page, kc, old_l1);
            } else {
                noc_async_read_page(page, vc, old_l1);
            }
            for (uint32_t half = 0; half < 2; ++half) {
                const uint64_t src = unit == 0 ? q.get_noc_addr(src_page, face_row_off(b & 31, half))
                                               : qkv.get_noc_addr(src_page, face_row_off(b & 31, half));
                noc_async_read(src, new_l1 + face_row_off(r, half), face_row_bytes);
            }
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_scale, 1);
    cb_push_back(cb_max_scaler, 1);
    cb_push_back(cb_sum_scaler, 1);
    cb_push_back(cb_mask, 1);
    cb_push_back(cb_kv_old, 2 * Kt);
    cb_push_back(cb_kv_new, 2 * Kt);

    // 2. The key tiles before the position tile.
    for (uint32_t n = 0; n < n_full; ++n) {
        cb_reserve_back(cb_k, Kt);
        const uint32_t l1 = get_write_ptr(cb_k);
        for (uint32_t k = 0; k < Kt; ++k) {
            noc_async_read_page((bg * Ct + n) * Kt + k, kc, l1 + k * tile_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_k, Kt);
    }

    // 3. The position tile: this step's key row spliced into the staged old tiles, copied in locally.
    cb_reserve_back(cb_k, Kt);
    {
        const uint32_t l1 = get_write_ptr(cb_k);
        if (n_full == last_key_tile) {
            cb_wait_front(cb_kv_out, Kt);
            const uint32_t upd = get_read_ptr(cb_kv_out);
            for (uint32_t k = 0; k < Kt; ++k) {
                copy_row(upd + k * tile_bytes, r, kv_old_base + k * tile_bytes, r);
                noc_async_read(get_noc_addr(kv_old_base + k * tile_bytes), l1 + k * tile_bytes, tile_bytes);
            }
            noc_async_read_barrier();
            cb_pop_front(cb_kv_out, Kt);
        } else {
            for (uint32_t k = 0; k < Kt; ++k) {
                noc_async_read_page((bg * Ct + n_full) * Kt + k, kc, l1 + k * tile_bytes);
            }
            noc_async_read_barrier();
        }
    }
    cb_push_back(cb_k, Kt);

    // 4. probs @ V: this step's value row into the staged old V tiles (then both staged units are complete for the
    // writer); then per key tile k its cache tiles (k, dt), dt = 0..Kt-1, V_BLK key tiles per push, each block in a
    // whole V_BLK * Kt slot of the ring (the rest of the last block unwritten and unread).
    cb_wait_front(cb_kv_out, Kt);
    {
        const uint32_t upd = get_read_ptr(cb_kv_out);
        for (uint32_t dt = 0; dt < Kt; ++dt) {
            copy_row(upd + dt * tile_bytes, r, kv_old_base + (Kt + dt) * tile_bytes, r);
        }
    }
    cb_pop_front(cb_kv_out, Kt);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(kv_flag)) = 1;
    for (uint32_t k0 = 0; k0 < nt; k0 += V_BLK) {
        const uint32_t m = nt - k0 < V_BLK ? nt - k0 : V_BLK;
        cb_reserve_back(cb_v, V_BLK * Kt);
        uint32_t l1 = get_write_ptr(cb_v);
        for (uint32_t k = k0; k < k0 + m; ++k) {
            for (uint32_t dt = 0; dt < Kt; ++dt) {
                if (k == last_key_tile) {
                    noc_async_read(get_noc_addr(kv_old_base + (Kt + dt) * tile_bytes), l1, tile_bytes);
                } else {
                    noc_async_read_page((bg * Ct + k) * Kt + dt, vc, l1);
                }
                l1 += tile_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_v, V_BLK * Kt);
    }
}

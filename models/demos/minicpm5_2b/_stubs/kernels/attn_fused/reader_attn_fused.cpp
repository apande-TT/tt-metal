// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B fused decode attention reader: one (batch, KV head) row block bg per core. Pushes, in the order the
// compute consumes them: the bf16 score-scale tile, the softmax reduce scalers (host-built tiles, the values
// ttnn's softmax reader generates), the Kt query/key rows of bg, then per key tile n its Kt cache tiles (cache^T for
// the scores) and the batch's key-mask tile, then per output column tile dt the value-cache tiles.
// Only the nt = position tile + 1 key tiles are streamed (nt handed to compute through cb_cnt): the tiles past it
// are fully masked, so their scores add exact zeros to the softmax sum and to probs.V.
// With `wait_kv` set, this step's KV-cache write runs in the same program on other cores (kernels/kv_write); the
// position tile of K (resp. the V tiles) is streamed only after that write signalled the K (resp. V) semaphore.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);  // head_dim tiles
    constexpr uint32_t Ct = get_compile_time_arg_val(1);  // cache capacity tiles
    constexpr uint32_t H = get_compile_time_arg_val(2);   // KV heads
    constexpr uint32_t cb_q = get_compile_time_arg_val(3);
    constexpr uint32_t cb_k = get_compile_time_arg_val(4);
    constexpr uint32_t cb_scale = get_compile_time_arg_val(5);
    constexpr uint32_t cb_mask = get_compile_time_arg_val(6);
    constexpr uint32_t cb_pos = get_compile_time_arg_val(7);
    [[maybe_unused]] constexpr uint32_t cb_zero = get_compile_time_arg_val(8);
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
    constexpr uint32_t wait_kv = get_compile_time_arg_val(sm_args.next_compile_time_args_offset());
    constexpr uint32_t sem_k_id = get_compile_time_arg_val(sm_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t sem_v_id = get_compile_time_arg_val(sm_args.next_compile_time_args_offset() + 2);
    // q_rows: q is the roped QKV-layout [1, B, H * G * D] (row b = batch b, G = groups + 1 heads per KV head):
    // the row block's G head rows are gathered into rows 0..G-1 of its Kt q tiles (the head-split layout).
    constexpr uint32_t q_rows = get_compile_time_arg_val(sm_args.next_compile_time_args_offset() + 3);
    constexpr uint32_t G = get_compile_time_arg_val(sm_args.next_compile_time_args_offset() + 4);
    // inline_kv: this step's KV-cache write is done here instead of by separate cores: the old position tiles and
    // the new rows (K from q = the roped QKV layout, V from qkv) go to compute, which forms
    // old + bf16(new - old) with kernels/kv_write's arithmetic; its row r is spliced into the position tile read
    // for the scores (resp. the values), which is written back to the cache before compute gets it.
    constexpr uint32_t inline_kv = get_compile_time_arg_val(sm_args.next_compile_time_args_offset() + 5);
    constexpr uint32_t cb_kv_old = get_compile_time_arg_val(sm_args.next_compile_time_args_offset() + 6);
    constexpr uint32_t cb_kv_new = get_compile_time_arg_val(sm_args.next_compile_time_args_offset() + 7);
    constexpr uint32_t cb_kv_scr = get_compile_time_arg_val(sm_args.next_compile_time_args_offset() + 8);
    constexpr uint32_t cb_kv_out = get_compile_time_arg_val(sm_args.next_compile_time_args_offset() + 9);
    constexpr uint32_t v_col0 = get_compile_time_arg_val(sm_args.next_compile_time_args_offset() + 10);
    constexpr uint32_t qkv_Nt = get_compile_time_arg_val(sm_args.next_compile_time_args_offset() + 11);
    // kv_flag: semaphore set once both staged position tiles hold this step's row (the writer stores them).
    constexpr uint32_t kv_flag = get_compile_time_arg_val(sm_args.next_compile_time_args_offset() + 12);
    constexpr auto qkv_args = TensorAccessorArgs<sm_args.next_compile_time_args_offset() + 13>();
    // mask_table = R (0: off): `mask` is the table [tiles r = the position at column r, r < 32 | R unmasked tiles |
    // R score-scale tiles | R max-scaler tiles] (one copy per row block, read by row block bg only) and the sum
    // scaler holds R tiles, instead of the [B, 1, 32, C] mask and single-tile scalers.
    constexpr uint32_t mask_table = get_compile_time_arg_val(qkv_args.next_compile_time_args_offset());
    constexpr uint32_t R = mask_table;
    constexpr uint32_t face_bytes_ = 16 * 16 * 2, face_row_bytes_ = 16 * 2;
    auto face_row_off = [](uint32_t row, uint32_t half) {
        return (((row >> 4) << 1) + half) * face_bytes_ + (row & 15) * face_row_bytes_;
    };
    auto copy_row = [&](uint32_t src_tile, uint32_t src_row, uint32_t dst_tile, uint32_t dst_row) {
        for (uint32_t half = 0; half < 2; ++half) {
            auto* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_tile + face_row_off(src_row, half));
            auto* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_tile + face_row_off(dst_row, half));
            for (uint32_t i = 0; i < face_row_bytes_ / 4; ++i) {
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
    const uint32_t b = bg / H;
    const auto qkv = TensorAccessor(qkv_args, get_common_arg_val<uint32_t>(9));

    // Position (the zero-tile accessor is unused: no tile past the position tile is streamed).
    const uint32_t pos_l1 = get_write_ptr(cb_pos);
    noc_async_read_page(0, cur_pos, pos_l1);
    noc_async_read_barrier();
    const uint32_t last_key_tile = static_cast<uint32_t>(reinterpret_cast<volatile tt_l1_ptr int32_t*>(pos_l1)[b]) >> 5;
    const uint32_t nt = last_key_tile + 1 < Ct ? last_key_tile + 1 : Ct;
    cb_reserve_back(cb_cnt, 1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_cnt))[0] = nt;
    cb_push_back(cb_cnt, 1);

    // Scale and softmax scalers.
    cb_reserve_back(cb_scale, 1);
    cb_reserve_back(cb_max_scaler, 1);
    cb_reserve_back(cb_sum_scaler, 1);
    noc_async_read_page(R ? 32 + R + bg : 0, scale, get_write_ptr(cb_scale));
    noc_async_read_page(R ? 32 + 2 * R + bg : 0, max_scaler, get_write_ptr(cb_max_scaler));
    noc_async_read_page(R ? bg : 0, sum_scaler, get_write_ptr(cb_sum_scaler));
    noc_async_read_barrier();
    cb_push_back(cb_scale, 1);
    cb_push_back(cb_max_scaler, 1);
    cb_push_back(cb_sum_scaler, 1);

    const uint32_t r = static_cast<uint32_t>(reinterpret_cast<volatile tt_l1_ptr int32_t*>(pos_l1)[b]) & 31;
    const uint32_t hh = bg - b * H;
    uint32_t kv_old_base = 0;
    if constexpr (inline_kv) {
        // The KV write's operands for compute: per unit (K, then V) the Kt old position tiles and Kt tiles whose
        // row r is the new row (the rest of those tiles is not used: only row r of the result is kept).
        const uint32_t scr = get_write_ptr(cb_kv_scr);
        kv_old_base = get_write_ptr(cb_kv_old);  // K's Kt staged tiles, then V's (the CB is filled once)
        for (uint32_t unit = 0; unit < 2; ++unit) {
            cb_reserve_back(cb_kv_old, Kt);
            cb_reserve_back(cb_kv_new, Kt);
            const uint32_t old_l1 = get_write_ptr(cb_kv_old);
            const uint32_t new_l1 = get_write_ptr(cb_kv_new);
            for (uint32_t dt = 0; dt < Kt; ++dt) {
                const uint32_t page = (bg * Ct + last_key_tile) * Kt + dt;
                if (unit == 0) {
                    noc_async_read_page(page, kc, old_l1 + dt * tile_bytes);
                    noc_async_read_page((hh * G + G - 1) * Kt + dt, q, scr + dt * tile_bytes);
                } else {
                    noc_async_read_page(page, vc, old_l1 + dt * tile_bytes);
                    noc_async_read_page((b >> 5) * qkv_Nt + v_col0 + hh * Kt + dt, qkv, scr + dt * tile_bytes);
                }
            }
            noc_async_read_barrier();
            for (uint32_t dt = 0; dt < Kt; ++dt) {
                copy_row(scr + dt * tile_bytes, b & 31, new_l1 + dt * tile_bytes, r);
            }
            cb_push_back(cb_kv_old, Kt);
            cb_push_back(cb_kv_new, Kt);
        }
    }

    // q (with the discarded key row) of bg. With q_rows and inline_kv the writer gathers it (on the other NoC, in
    // parallel with the KV-write operand reads above).
    constexpr bool q_by_writer = q_rows && inline_kv;
    if constexpr (!q_by_writer) {
        cb_reserve_back(cb_q, Kt);
    }
    if constexpr (q_by_writer) {
        // (the writer pushes cb_q)
    } else if constexpr (q_rows) {
        constexpr uint32_t face_bytes = 16 * 16 * 2, face_row_bytes = 16 * 2;
        auto face_row_offset = [](uint32_t row, uint32_t half) {
            return (((row >> 4) << 1) + half) * face_bytes + (row & 15) * face_row_bytes;
        };
        const uint32_t h = bg - b * H;
        const uint32_t l1 = get_write_ptr(cb_q);
        for (uint32_t j = 0; j < G; ++j) {
            for (uint32_t k = 0; k < Kt; ++k) {
                for (uint32_t half = 0; half < 2; ++half) {
                    noc_async_read(
                        q.get_noc_addr((h * G + j) * Kt + k, face_row_offset(b & 31, half)),
                        l1 + k * tile_bytes + face_row_offset(j, half),
                        face_row_bytes);
                }
            }
        }
    } else {
        uint32_t l1 = get_write_ptr(cb_q);
        for (uint32_t k = 0; k < Kt; ++k) {
            noc_async_read_page(bg * Kt + k, q, l1);
            l1 += tile_bytes;
        }
    }
    if constexpr (!q_by_writer) {
        noc_async_read_barrier();
        cb_push_back(cb_q, Kt);
    }

    // Scores: per key tile n, cache tiles (n, k) and the mask tile.
    for (uint32_t n = 0; n < nt; ++n) {
        if constexpr (wait_kv) {
            if (n == last_key_tile) {
                auto* sem_k = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_k_id));
                noc_semaphore_wait(sem_k, 1);
                noc_semaphore_set(sem_k, 0);
            }
        }
        cb_reserve_back(cb_k, Kt);
        uint32_t l1 = get_write_ptr(cb_k);
        if (inline_kv && n == last_key_tile) {
            // The position tiles: this step's key row spliced into the staged old tiles (which the writer stores),
            // copied in locally.
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
                noc_async_read_page((bg * Ct + n) * Kt + k, kc, l1);
                l1 += tile_bytes;
            }
            noc_async_read_barrier();
        }
        cb_push_back(cb_k, Kt);
        cb_reserve_back(cb_mask, 1);
        const uint32_t mask_page = R ? (n == last_key_tile ? r : 32 + bg) : b * Ct + n;
        noc_async_read_page(mask_page, mask, get_write_ptr(cb_mask));
        noc_async_read_barrier();
        cb_push_back(cb_mask, 1);
    }

    // probs @ V: per output column tile dt, cache tiles (k, dt) for k = 0..nt-1, in a full Ct-tile slot (the rest
    // unwritten and unread) so every push stays aligned to the 2 * Ct ring and a block never runs past its end.
    if constexpr (inline_kv) {
        // This step's value row into the staged old V tiles; then both staged units are complete for the writer.
        cb_wait_front(cb_kv_out, Kt);
        const uint32_t upd = get_read_ptr(cb_kv_out);
        for (uint32_t dt = 0; dt < Kt; ++dt) {
            copy_row(upd + dt * tile_bytes, r, kv_old_base + (Kt + dt) * tile_bytes, r);
        }
        cb_pop_front(cb_kv_out, Kt);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(kv_flag)) = 1;
    }
    if constexpr (wait_kv) {
        auto* sem_v = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_v_id));
        noc_semaphore_wait(sem_v, 1);
        noc_semaphore_set(sem_v, 0);
    }
    for (uint32_t dt = 0; dt < Kt; ++dt) {
        cb_reserve_back(cb_v, Ct);
        uint32_t l1 = get_write_ptr(cb_v);
        for (uint32_t k = 0; k < nt; ++k) {
            if (inline_kv && k == last_key_tile) {
                noc_async_read(get_noc_addr(kv_old_base + (Kt + dt) * tile_bytes), l1, tile_bytes);
            } else {
                noc_async_read_page((bg * Ct + k) * Kt + dt, vc, l1);
            }
            l1 += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_v, Ct);
    }
}

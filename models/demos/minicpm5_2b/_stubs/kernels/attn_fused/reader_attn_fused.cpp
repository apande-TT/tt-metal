// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B fused decode attention reader: one (batch, KV head) row block bg per core. Pushes, in the order the
// compute consumes them: the bf16 score-scale tile, the softmax reduce scalers (host-built tiles, the values
// ttnn's softmax reader generates), the Kt query/key rows of bg, then per key tile n its Kt cache tiles (cache^T for
// the scores) and the batch's key-mask tile, then per output column tile dt the value-cache tiles.
// Only the nt = position tile + 1 key tiles are streamed (nt handed to compute through cb_cnt): the tiles past it
// are fully masked, so their scores add exact zeros to the softmax sum and to probs.V.

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
    noc_async_read_page(0, scale, get_write_ptr(cb_scale));
    noc_async_read_page(0, max_scaler, get_write_ptr(cb_max_scaler));
    noc_async_read_page(0, sum_scaler, get_write_ptr(cb_sum_scaler));
    noc_async_read_barrier();
    cb_push_back(cb_scale, 1);
    cb_push_back(cb_max_scaler, 1);
    cb_push_back(cb_sum_scaler, 1);

    // q (with the discarded key row) of bg.
    cb_reserve_back(cb_q, Kt);
    {
        uint32_t l1 = get_write_ptr(cb_q);
        for (uint32_t k = 0; k < Kt; ++k) {
            noc_async_read_page(bg * Kt + k, q, l1);
            l1 += tile_bytes;
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_q, Kt);

    // Scores: per key tile n, cache tiles (n, k) and the mask tile.
    for (uint32_t n = 0; n < nt; ++n) {
        cb_reserve_back(cb_k, Kt);
        uint32_t l1 = get_write_ptr(cb_k);
        for (uint32_t k = 0; k < Kt; ++k) {
            noc_async_read_page((bg * Ct + n) * Kt + k, kc, l1);
            l1 += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_k, Kt);
        cb_reserve_back(cb_mask, 1);
        noc_async_read_page(b * Ct + n, mask, get_write_ptr(cb_mask));
        noc_async_read_barrier();
        cb_push_back(cb_mask, 1);
    }

    // probs @ V: per output column tile dt, cache tiles (k, dt) for k = 0..nt-1, in a full Ct-tile slot (the rest
    // unwritten and unread) so every push stays aligned to the 2 * Ct ring and a block never runs past its end.
    for (uint32_t dt = 0; dt < Kt; ++dt) {
        cb_reserve_back(cb_v, Ct);
        uint32_t l1 = get_write_ptr(cb_v);
        for (uint32_t k = 0; k < nt; ++k) {
            noc_async_read_page((bg * Ct + k) * Kt + dt, vc, l1);
            l1 += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_v, Ct);
    }
}

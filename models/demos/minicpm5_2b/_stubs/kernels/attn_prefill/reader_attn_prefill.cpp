// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B fused prefill attention reader. A unit is one query tile row t of one (batch, KV head) row block bg
// (q viewed as [B, n_kv, G * S, D]: Rt = G * S / 32 tile rows per block); each core takes a contiguous unit range.
// Pushes, in the order compute consumes them: the bf16 score-scale tile and the softmax reduce scalers (once),
// then per unit: on a new bg, its whole K block (Ct key tiles x Kt, row-major) and V block (per output column
// tile dt the Ct value tiles), resident until the next bg; the unit's Kt query tiles; per key tile n the causal
// mask tile (s tile = t % Ct, n).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);  // head_dim tiles
    constexpr uint32_t Ct = get_compile_time_arg_val(1);  // sequence tiles (keys)
    constexpr uint32_t Rt = get_compile_time_arg_val(2);  // query tile rows per (batch, KV head) block
    constexpr uint32_t cb_q = get_compile_time_arg_val(3);
    constexpr uint32_t cb_k = get_compile_time_arg_val(4);
    constexpr uint32_t cb_scale = get_compile_time_arg_val(5);
    constexpr uint32_t cb_mask = get_compile_time_arg_val(6);
    constexpr uint32_t cb_max_scaler = get_compile_time_arg_val(7);
    constexpr uint32_t cb_sum_scaler = get_compile_time_arg_val(8);
    constexpr uint32_t cb_v = get_compile_time_arg_val(9);
    constexpr uint32_t tile_bytes = 32 * 32 * 2;
    constexpr auto q_args = TensorAccessorArgs<10>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto scale_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto mask_args = TensorAccessorArgs<scale_args.next_compile_time_args_offset()>();
    constexpr auto mx_args = TensorAccessorArgs<mask_args.next_compile_time_args_offset()>();
    constexpr auto sm_args = TensorAccessorArgs<mx_args.next_compile_time_args_offset()>();

    const uint32_t u_start = get_arg_val<uint32_t>(0);
    const uint32_t u_count = get_arg_val<uint32_t>(1);
    const auto q = TensorAccessor(q_args, get_common_arg_val<uint32_t>(0));
    const auto kt = TensorAccessor(k_args, get_common_arg_val<uint32_t>(1));
    const auto vt = TensorAccessor(v_args, get_common_arg_val<uint32_t>(2));
    const auto scale = TensorAccessor(scale_args, get_common_arg_val<uint32_t>(3));
    const auto mask = TensorAccessor(mask_args, get_common_arg_val<uint32_t>(4));
    const auto max_scaler = TensorAccessor(mx_args, get_common_arg_val<uint32_t>(5));
    const auto sum_scaler = TensorAccessor(sm_args, get_common_arg_val<uint32_t>(6));

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

    uint32_t cur_bg = 0xFFFFFFFF;
    for (uint32_t u = u_start; u < u_start + u_count; ++u) {
        const uint32_t bg = u / Rt;
        const uint32_t t = u - bg * Rt;
        if (bg != cur_bg) {
            cur_bg = bg;
            cb_reserve_back(cb_k, Ct * Kt);
            uint32_t l1 = get_write_ptr(cb_k);
            for (uint32_t i = 0; i < Ct * Kt; ++i) {  // (n, k) row-major = K's own tile order
                noc_async_read_page(bg * Ct * Kt + i, kt, l1);
                l1 += tile_bytes;
            }
            cb_reserve_back(cb_v, Ct * Kt);
            l1 = get_write_ptr(cb_v);
            for (uint32_t dt = 0; dt < Kt; ++dt) {
                for (uint32_t n = 0; n < Ct; ++n) {
                    noc_async_read_page((bg * Ct + n) * Kt + dt, vt, l1);
                    l1 += tile_bytes;
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_k, Ct * Kt);
            cb_push_back(cb_v, Ct * Kt);
        }
        cb_reserve_back(cb_q, Kt);
        uint32_t l1 = get_write_ptr(cb_q);
        for (uint32_t k = 0; k < Kt; ++k) {
            noc_async_read_page(u * Kt + k, q, l1);
            l1 += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_q, Kt);
        const uint32_t s_t = t % Ct;
        for (uint32_t n = 0; n < Ct; ++n) {
            cb_reserve_back(cb_mask, 1);
            noc_async_read_page(s_t * Ct + n, mask, get_write_ptr(cb_mask));
            noc_async_read_barrier();
            cb_push_back(cb_mask, 1);
        }
    }
}

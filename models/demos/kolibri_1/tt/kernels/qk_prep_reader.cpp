// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode q/k prep (reader). The fused projection [B, (HQ + 2 HKV) * D] holds one user per row, so a head is a block
// of W = D / 32 tile columns of a 32-user tile row: a unit is (tile row, head). A q / k unit (kind 0) reads the
// head's W tiles, its norm weight ([32, D]: every row the weight), and on RoPE layers the users' cos / sin rows
// ([B, D]); a v unit (kind 1) is passed through untouched (straight to the output buffer). Also builds the reduce
// scaler tile (1 / D in row 0 of each face, bf16) once.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0);   // tiles across a head
    constexpr uint32_t NT = get_compile_time_arg_val(1);  // tiles across a fused row
    constexpr uint32_t ROPE = get_compile_time_arg_val(2);
    constexpr uint32_t SCALER = get_compile_time_arg_val(3);  // bf16 bits of 1 / D
    constexpr auto x_args = TensorAccessorArgs<4>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(0));
    const auto wt = TensorAccessor(w_args, get_arg_val<uint32_t>(1));
    const auto cs = TensorAccessor(c_args, get_arg_val<uint32_t>(2));
    const auto sn = TensorAccessor(s_args, get_arg_val<uint32_t>(3));
    const uint32_t kind = get_arg_val<uint32_t>(4), tr = get_arg_val<uint32_t>(5), col0 = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_x = 0, cb_w = 1, cb_cos = 2, cb_sin = 3, cb_scaler = 4, cb_out = 16;
    constexpr uint32_t TILE = 2048;  // bf16

    if (kind == 1) {
        cb_reserve_back(cb_out, W);
        const uint32_t l1 = get_write_ptr(cb_out);
        for (uint32_t t = 0; t < W; ++t) {
            noc_async_read(x.get_noc_addr(tr * NT + col0 + t), l1 + t * TILE, TILE);
        }
        noc_async_read_barrier();
        cb_push_back(cb_out, W);
        return;
    }

    cb_reserve_back(cb_scaler, 1);
    volatile tt_l1_ptr uint32_t* sc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_scaler));
    for (uint32_t i = 0; i < TILE / 4; ++i) {
        sc[i] = 0;
    }
    for (uint32_t f = 0; f < 4; ++f) {  // row 0 of each 16 x 16 face: 16 bf16 = 8 words
        for (uint32_t i = 0; i < 8; ++i) {
            sc[f * 128 + i] = (SCALER << 16) | SCALER;
        }
    }
    cb_push_back(cb_scaler, 1);

    cb_reserve_back(cb_x, W);
    cb_reserve_back(cb_w, W);
    const uint32_t xl = get_write_ptr(cb_x), wl = get_write_ptr(cb_w);
    for (uint32_t t = 0; t < W; ++t) {
        noc_async_read(x.get_noc_addr(tr * NT + col0 + t), xl + t * TILE, TILE);
        noc_async_read(wt.get_noc_addr(t), wl + t * TILE, TILE);
    }
    if constexpr (ROPE) {
        cb_reserve_back(cb_cos, W);
        cb_reserve_back(cb_sin, W);
        const uint32_t cl = get_write_ptr(cb_cos), sl = get_write_ptr(cb_sin);
        for (uint32_t t = 0; t < W; ++t) {
            noc_async_read(cs.get_noc_addr(tr * W + t), cl + t * TILE, TILE);
            noc_async_read(sn.get_noc_addr(tr * W + t), sl + t * TILE, TILE);
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, W);
    cb_push_back(cb_w, W);
    if constexpr (ROPE) {
        cb_push_back(cb_cos, W);
        cb_push_back(cb_sin, W);
    }
}

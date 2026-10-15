// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode attention bmm reader: out[bg] (one tile row) = in0[bg] (one tile row, Kt tiles) @ B[bg],
// where B is the [B, n_kv, C, D] cache (transpose_b: B = cache^T, scores; else B = cache, probs @ V).
// For output tile (bg, n) it pushes the Kt in0 tiles of row bg and the Kt in1 tiles of column n, in K order.
// With fuse_scale_mask (scores only) it also pushes the bf16 scale tile once and, per output tile, the key-mask
// tile of its batch (mask [B, 1, 32, C], row-broadcast; b = bg / H), for the scale + mask in compute.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);
    constexpr uint32_t transpose_b = get_compile_time_arg_val(2);
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_in1 = get_compile_time_arg_val(4);
    constexpr uint32_t fuse_scale_mask = get_compile_time_arg_val(5);
    constexpr uint32_t H = get_compile_time_arg_val(6);
    constexpr uint32_t cb_scale = get_compile_time_arg_val(7);
    constexpr uint32_t cb_mask = get_compile_time_arg_val(8);
    constexpr uint32_t tile_bytes = 32 * 32 * 2;
    constexpr auto in0_args = TensorAccessorArgs<9>();
    constexpr auto in1_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    constexpr auto scale_args = TensorAccessorArgs<in1_args.next_compile_time_args_offset()>();
    constexpr auto mask_args = TensorAccessorArgs<scale_args.next_compile_time_args_offset()>();

    const uint32_t t_start = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const auto in0 = TensorAccessor(in0_args, get_common_arg_val<uint32_t>(0));
    const auto in1 = TensorAccessor(in1_args, get_common_arg_val<uint32_t>(1));
    const auto scale = TensorAccessor(scale_args, get_common_arg_val<uint32_t>(2));
    const auto mask = TensorAccessor(mask_args, get_common_arg_val<uint32_t>(3));

    if constexpr (fuse_scale_mask) {
        cb_reserve_back(cb_scale, 1);
        noc_async_read_page(0, scale, get_write_ptr(cb_scale));
        noc_async_read_barrier();
        cb_push_back(cb_scale, 1);
    }

    for (uint32_t t = t_start; t < t_start + num_tiles; ++t) {
        const uint32_t bg = t / Nt;
        const uint32_t n = t % Nt;
        cb_reserve_back(cb_in0, Kt);
        cb_reserve_back(cb_in1, Kt);
        const uint32_t in0_l1 = get_write_ptr(cb_in0);
        const uint32_t in1_l1 = get_write_ptr(cb_in1);
        for (uint32_t k = 0; k < Kt; ++k) {
            noc_async_read_page(bg * Kt + k, in0, in0_l1 + k * tile_bytes);
            // cache [bg, C, D]: scores read cache tile (n, k) (transposed by the unpacker), probs @ V tile (k, n).
            const uint32_t page = transpose_b ? (bg * Nt + n) * Kt + k : (bg * Kt + k) * Nt + n;
            noc_async_read_page(page, in1, in1_l1 + k * tile_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_in0, Kt);
        cb_push_back(cb_in1, Kt);
        if constexpr (fuse_scale_mask) {
            cb_reserve_back(cb_mask, 1);
            noc_async_read_page((bg / H) * Nt + n, mask, get_write_ptr(cb_mask));
            noc_async_read_barrier();
            cb_push_back(cb_mask, 1);
        }
    }
}

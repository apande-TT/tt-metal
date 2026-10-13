// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode score scale + mask reader. Scores [B, n_kv, 32, C]; the bf16 scale tile once; the per-batch
// additive key mask [B, 1, 32, C] already broadcast over the tile rows. For score tile (b, g, ct) it pushes the
// score tile and mask tile (b, ct).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t n_kv = get_compile_time_arg_val(0);
    constexpr uint32_t Ct = get_compile_time_arg_val(1);
    constexpr uint32_t cb_s = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scale = get_compile_time_arg_val(3);
    constexpr uint32_t cb_mask = get_compile_time_arg_val(4);
    constexpr auto s_args = TensorAccessorArgs<5>();
    constexpr auto scale_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto mask_args = TensorAccessorArgs<scale_args.next_compile_time_args_offset()>();

    const uint32_t t_start = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const auto scores = TensorAccessor(s_args, get_common_arg_val<uint32_t>(0));
    const auto scale = TensorAccessor(scale_args, get_common_arg_val<uint32_t>(1));
    const auto mask = TensorAccessor(mask_args, get_common_arg_val<uint32_t>(2));

    cb_reserve_back(cb_scale, 1);
    noc_async_read_page(0, scale, get_write_ptr(cb_scale));
    noc_async_read_barrier();
    cb_push_back(cb_scale, 1);

    for (uint32_t t = t_start; t < t_start + num_tiles; ++t) {
        const uint32_t ct = t % Ct;
        const uint32_t b = t / (Ct * n_kv);
        cb_reserve_back(cb_s, 1);
        cb_reserve_back(cb_mask, 1);
        noc_async_read_page(t, scores, get_write_ptr(cb_s));
        noc_async_read_page(b * Ct + ct, mask, get_write_ptr(cb_mask));
        noc_async_read_barrier();
        cb_push_back(cb_s, 1);
        cb_push_back(cb_mask, 1);
    }
}

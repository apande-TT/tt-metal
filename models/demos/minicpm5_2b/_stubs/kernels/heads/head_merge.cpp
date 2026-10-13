// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode head merge: [B, n_kv, rows_in, D] (rows 0..groups-1 valid) -> one-tile-row
// [1, B, n_kv * groups * D] in one data-movement pass (replaces slice + tiled reshape). Output column tile
// c = h * Dt + dt with h = g * groups + j takes in row b the row j of input tile (b, g, dt).
// Pure copies of bf16 row halves (32 B each), so bit-identical.

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
    constexpr uint32_t n_kv = get_compile_time_arg_val(0);
    constexpr uint32_t groups = get_compile_time_arg_val(1);
    constexpr uint32_t Dt = get_compile_time_arg_val(2);
    constexpr uint32_t B = get_compile_time_arg_val(3);
    constexpr uint32_t cb_tile = get_compile_time_arg_val(4);
    constexpr auto src_args = TensorAccessorArgs<5>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    const uint32_t t_start = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const auto src = TensorAccessor(src_args, get_common_arg_val<uint32_t>(0));
    const auto dst = TensorAccessor(dst_args, get_common_arg_val<uint32_t>(1));

    const uint32_t buf = get_write_ptr(cb_tile);
    for (uint32_t c = t_start; c < t_start + num_tiles; ++c) {
        const uint32_t dt = c % Dt;
        const uint32_t h = c / Dt;
        const uint32_t g = h / groups;
        const uint32_t j = h - g * groups;
        for (uint32_t b = 0; b < B; ++b) {
            const uint32_t page = (b * n_kv + g) * Dt + dt;
            for (uint32_t half = 0; half < 2; ++half) {
                noc_async_read(
                    src.get_noc_addr(page, face_row_offset(j, half)), buf + face_row_offset(b, half), kFaceRowBytes);
            }
        }
        noc_async_read_barrier();
        noc_async_write_page(c, dst, buf);
        noc_async_write_barrier();
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode head split: one-tile-row fused projection [1, B, cols] -> [B, n_kv, rows, D] in one
// data-movement pass (replaces slice + tiled reshape). Output tile (b, g, dt) holds in row j the 32-wide
// column tile dt of head (g * rows + j) for batch row b, i.e. input tile column (g * rows + j) * Dt + dt,
// row b. Rows >= `rows` are zero. Pure copies of bf16 row halves (32 B each), so bit-identical.

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
    constexpr uint32_t rows = get_compile_time_arg_val(1);
    constexpr uint32_t Dt = get_compile_time_arg_val(2);
    constexpr uint32_t cb_tile = get_compile_time_arg_val(3);
    constexpr uint32_t tile_bytes = 32 * 32 * 2;
    constexpr auto src_args = TensorAccessorArgs<4>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    const uint32_t t_start = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const auto src = TensorAccessor(src_args, get_common_arg_val<uint32_t>(0));
    const auto dst = TensorAccessor(dst_args, get_common_arg_val<uint32_t>(1));

    const uint32_t buf = get_write_ptr(cb_tile);
    // Zero once: rows >= `rows` stay zero, rows < `rows` are overwritten for every tile.
    auto* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);
    for (uint32_t i = 0; i < tile_bytes / 4; ++i) {
        z[i] = 0;
    }
    for (uint32_t t = t_start; t < t_start + num_tiles; ++t) {
        const uint32_t dt = t % Dt;
        const uint32_t bg = t / Dt;
        const uint32_t g = bg % n_kv;
        const uint32_t b = bg / n_kv;
        for (uint32_t j = 0; j < rows; ++j) {
            const uint32_t page = (g * rows + j) * Dt + dt;
            for (uint32_t half = 0; half < 2; ++half) {
                noc_async_read(
                    src.get_noc_addr(page, face_row_offset(b, half)), buf + face_row_offset(j, half), kFaceRowBytes);
            }
        }
        noc_async_read_barrier();
        noc_async_write_page(t, dst, buf);
        noc_async_write_barrier();
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode probs.V writer fused with the head merge: output tile (b, g, dt) of [B, n_kv, rows, D] is
// not stored; its rows j < groups go straight to row b of column tile (g * groups + j) * Dt + dt of the
// one-tile-row [1, B, n_kv * groups * D] output (kernels/heads/head_merge.cpp's gather, done as a scatter).
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
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t n_kv = get_compile_time_arg_val(1);
    constexpr uint32_t groups = get_compile_time_arg_val(2);
    constexpr uint32_t Dt = get_compile_time_arg_val(3);
    constexpr auto out_args = TensorAccessorArgs<4>();

    const uint32_t t_start = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(0));

    for (uint32_t t = t_start; t < t_start + num_tiles; ++t) {
        const uint32_t dt = t % Dt;
        const uint32_t bg = t / Dt;
        const uint32_t g = bg % n_kv;
        const uint32_t b = bg / n_kv;
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
}

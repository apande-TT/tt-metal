// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B sampler: Gumbel noise of each top-k candidate. For batch row b and candidate rank r < k, token
// t = idx[b, r] (uint32 [1, 1, B, kpad] tiles) sits in noise segment t / W at column t % W of the segmented
// noise buffer [1, S, B, W] (bf16 tiles); its value goes, widened to fp32 (exact), to out[b, r] ([1, 1, B, kpad]
// fp32 tiles). Ranks >= k are written 0. One core per batch row; pure data movement.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace {
// Byte offset of the 16-wide half-row `half` of tile row `row` in a 32x32 tile of `elem` bytes per element.
inline uint32_t face_row_offset(uint32_t row, uint32_t half, uint32_t elem) {
    return (((row >> 4) << 1) + half) * (16 * 16 * elem) + (row & 15) * (16 * elem);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t k = get_compile_time_arg_val(0);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t W = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(3);
    constexpr auto idx_args = TensorAccessorArgs<4>();
    constexpr auto noise_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<noise_args.next_compile_time_args_offset()>();
    constexpr uint32_t Wt = W / 32;
    constexpr uint32_t kSlot = 64;  // one aligned 64-byte read per candidate

    const uint32_t b_start = get_arg_val<uint32_t>(0);
    const uint32_t b_count = get_arg_val<uint32_t>(1);
    const auto idx = TensorAccessor(idx_args, get_common_arg_val<uint32_t>(0));
    const auto noise = TensorAccessor(noise_args, get_common_arg_val<uint32_t>(1));
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(2));

    // scratch: [k_tiles * 2 index half-rows of 64 B] [k_tiles * 32 noise slots of 64 B] [k_tiles * 2 out half-rows]
    const uint32_t idx_l1 = get_write_ptr(cb_scratch);
    const uint32_t noise_l1 = idx_l1 + k_tiles * 2 * 64;
    const uint32_t out_l1 = noise_l1 + k_tiles * 32 * kSlot;
    auto* idx_vals = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);
    auto* out_vals = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1);

    for (uint32_t b = b_start; b < b_start + b_count; ++b) {
        // Row b of the index tiles: rank r = tile * 32 + half * 16 + i at idx_vals[(tile * 2 + half) * 16 + i].
        for (uint32_t tile = 0; tile < k_tiles; ++tile) {
            for (uint32_t half = 0; half < 2; ++half) {
                noc_async_read(
                    idx.get_noc_addr(tile, face_row_offset(b, half, 4)), idx_l1 + (tile * 2 + half) * 64, 64);
            }
        }
        noc_async_read_barrier();

        for (uint32_t r = 0; r < k; ++r) {
            const uint32_t t = idx_vals[r];
            const uint32_t seg = t / W;
            const uint32_t col = t - seg * W;
            const uint32_t off = face_row_offset(b, (col & 31) >> 4, 2);
            noc_async_read(noise.get_noc_addr(seg * Wt + (col >> 5), off & ~63u), noise_l1 + r * kSlot, kSlot);
        }
        noc_async_read_barrier();

        for (uint32_t r = 0; r < k_tiles * 32; ++r) {
            uint32_t bits = 0;
            if (r < k) {
                const uint32_t col = idx_vals[r] % W;
                const uint32_t off = face_row_offset(b, (col & 31) >> 4, 2);
                const uint32_t byte = (off & 63u) + (col & 15) * 2;
                auto* slot = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(noise_l1 + r * kSlot + byte);
                bits = static_cast<uint32_t>(*slot) << 16;  // bf16 -> fp32, exact
            }
            out_vals[r] = bits;  // rank r at (tile r / 32, half (r % 32) / 16, i r % 16): same order as idx_vals
        }
        for (uint32_t tile = 0; tile < k_tiles; ++tile) {
            for (uint32_t half = 0; half < 2; ++half) {
                noc_async_write(
                    out_l1 + (tile * 2 + half) * 64, out.get_noc_addr(tile, face_row_offset(b, half, 4)), 64);
            }
        }
        noc_async_write_barrier();
    }
}

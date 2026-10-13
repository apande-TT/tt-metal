// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B sampler tail: per batch row b, the rank of the largest perturbed score over the k candidates
// (fp32 [1, 1, B, kpad] tiles; ttnn.argmax's comparison: NaN-aware float32_greater, +0 == -0, first max wins)
// and that rank's token id (uint32 [1, 1, B, kpad] tiles), written as row b of the [1, B] uint32 row-major token
// buffer. One core, pure scalar work.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/numeric/float32.h"

namespace {
constexpr uint32_t kExpMask = 0x7F800000u;
constexpr uint32_t kMantMask = 0x007FFFFFu;

// ttnn argmax's argmax_float32_greater.
inline bool greater(uint32_t a, uint32_t b) {
    const bool a_nan = (a & kExpMask) == kExpMask && (a & kMantMask) != 0;
    const bool b_nan = (b & kExpMask) == kExpMask && (b & kMantMask) != 0;
    if (a_nan || b_nan) {
        return a_nan && !b_nan;
    }
    return float32_greater(a, b);
}

// Element (row, col) of a 32x32 tile of 4-byte elements.
inline uint32_t elem_offset(uint32_t row, uint32_t col) {
    return (((row >> 4) << 1) + ((col & 31) >> 4)) * 1024 + (row & 15) * 64 + (col & 15) * 4;
}
}  // namespace

void kernel_main() {
    constexpr uint32_t k = get_compile_time_arg_val(0);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t B = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(3);
    constexpr auto val_args = TensorAccessorArgs<4>();
    constexpr auto idx_args = TensorAccessorArgs<val_args.next_compile_time_args_offset()>();
    constexpr auto tok_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = 32 * 32 * 4;

    const auto vals = TensorAccessor(val_args, get_common_arg_val<uint32_t>(0));
    const auto idx = TensorAccessor(idx_args, get_common_arg_val<uint32_t>(1));
    const auto tok = TensorAccessor(tok_args, get_common_arg_val<uint32_t>(2));

    const uint32_t val_l1 = get_write_ptr(cb_scratch);
    const uint32_t idx_l1 = val_l1 + k_tiles * tile_bytes;
    const uint32_t tok_l1 = idx_l1 + k_tiles * tile_bytes;
    for (uint32_t t = 0; t < k_tiles; ++t) {
        noc_async_read_page(t, vals, val_l1 + t * tile_bytes);
        noc_async_read_page(t, idx, idx_l1 + t * tile_bytes);
    }
    noc_async_read_barrier();

    auto* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tok_l1);
    for (uint32_t b = 0; b < B; ++b) {
        uint32_t best = 0;
        uint32_t best_val = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(val_l1 + elem_offset(b, 0));
        for (uint32_t r = 1; r < k; ++r) {
            const uint32_t v =
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(val_l1 + (r >> 5) * tile_bytes + elem_offset(b, r));
            if (greater(v, best_val)) {
                best = r;
                best_val = v;
            }
        }
        const uint32_t idx_addr = idx_l1 + (best >> 5) * tile_bytes + elem_offset(b, best);
        out[b] = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_addr);
    }
    noc_async_write(tok_l1, tok.get_noc_addr(0), B * 4);
    noc_async_write_barrier();
}

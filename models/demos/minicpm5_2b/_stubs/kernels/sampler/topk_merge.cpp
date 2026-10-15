// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B sampler: second stage of the segmented top-k. Stage 1 (ttnn.topk on the segmented logits
// [1, S, B, W]) leaves, per segment s and batch row b, the segment's top-Ks logits in descending order (bf16
// [1, S, B, Ks] tiles) and their in-segment ids (uint16 tiles, same layout). Every global top-k token is among its
// segment's top-Ks (k <= Ks), so the global top-k is the k-way merge of the S sorted lists: repeatedly take the
// largest head (ties -> the lower segment). Writes, for row b, the k values widened to fp32 (exact) and their
// global ids s * W + id as uint32 into [1, 1, B, kpad] tiles; ranks >= k get -inf / 0.
// One core per batch row; pure scalar work on the row's face chunks.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace {
// Byte offset of the 16-wide half-row `half` of tile row `row` in a 32x32 tile of `elem` bytes per element.
inline uint32_t face_row_offset(uint32_t row, uint32_t half, uint32_t elem) {
    return (((row >> 4) << 1) + half) * (16 * 16 * elem) + (row & 15) * (16 * elem);
}

// bf16 bits -> unsigned key with the same order as the float values (no NaNs here: stage 1 clamps -inf).
inline uint32_t order_key(uint32_t v) { return (v & 0x8000u) ? (~v & 0xFFFFu) : (v | 0x8000u); }

constexpr uint32_t kSlot = 64;  // one aligned 64-byte read per 16-element face chunk
}  // namespace

void kernel_main() {
    constexpr uint32_t k = get_compile_time_arg_val(0);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t S = get_compile_time_arg_val(2);
    constexpr uint32_t W = get_compile_time_arg_val(3);
    constexpr uint32_t Ks_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(5);
    constexpr auto val_args = TensorAccessorArgs<6>();
    constexpr auto id_args = TensorAccessorArgs<val_args.next_compile_time_args_offset()>();
    constexpr auto out_val_args = TensorAccessorArgs<id_args.next_compile_time_args_offset()>();
    constexpr auto out_id_args = TensorAccessorArgs<out_val_args.next_compile_time_args_offset()>();
    constexpr uint32_t Ks = Ks_tiles * 32;
    constexpr uint32_t slots_per_seg = Ks_tiles * 2;

    const uint32_t b = get_arg_val<uint32_t>(0);
    const auto vals = TensorAccessor(val_args, get_common_arg_val<uint32_t>(0));
    const auto ids = TensorAccessor(id_args, get_common_arg_val<uint32_t>(1));
    const auto out_vals = TensorAccessor(out_val_args, get_common_arg_val<uint32_t>(2));
    const auto out_ids = TensorAccessor(out_id_args, get_common_arg_val<uint32_t>(3));

    // scratch: [S * slots_per_seg value slots][S * slots_per_seg id slots][k_tiles * 2 * 64 out values][same, ids]
    const uint32_t val_l1 = get_write_ptr(cb_scratch);
    const uint32_t id_l1 = val_l1 + S * slots_per_seg * kSlot;
    const uint32_t out_val_l1 = id_l1 + S * slots_per_seg * kSlot;
    const uint32_t out_id_l1 = out_val_l1 + k_tiles * 2 * 64;

    // Row b's chunks: element r of segment s is in slot s * slots_per_seg + (r / 32) * 2 + (r % 32) / 16.
    const uint32_t off0 = face_row_offset(b, 0, 2);
    const uint32_t off1 = face_row_offset(b, 1, 2);
    for (uint32_t s = 0; s < S; ++s) {
        for (uint32_t t = 0; t < Ks_tiles; ++t) {
            const uint32_t page = s * Ks_tiles + t;
            const uint32_t slot = s * slots_per_seg + t * 2;
            noc_async_read(vals.get_noc_addr(page, off0 & ~63u), val_l1 + slot * kSlot, kSlot);
            noc_async_read(vals.get_noc_addr(page, off1 & ~63u), val_l1 + (slot + 1) * kSlot, kSlot);
            noc_async_read(ids.get_noc_addr(page, off0 & ~63u), id_l1 + slot * kSlot, kSlot);
            noc_async_read(ids.get_noc_addr(page, off1 & ~63u), id_l1 + (slot + 1) * kSlot, kSlot);
        }
    }
    noc_async_read_barrier();
    const uint32_t in_slot = off0 & 63u;  // both halves share the in-slot offset

    auto elem = [&](uint32_t base, uint32_t s, uint32_t r) -> uint32_t {
        const uint32_t slot = s * slots_per_seg + (r >> 5) * 2 + ((r & 31) >> 4);
        return *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + slot * kSlot + in_slot + (r & 15) * 2);
    };

    uint32_t head[S];
    uint32_t head_key[S];
    for (uint32_t s = 0; s < S; ++s) {
        head[s] = 0;
        head_key[s] = order_key(elem(val_l1, s, 0));
    }

    auto* ov = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_val_l1);
    auto* oi = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_id_l1);
    for (uint32_t n = 0; n < k_tiles * 32; ++n) {
        if (n >= k) {
            ov[n] = 0xFF800000u;  // -inf
            oi[n] = 0;
            continue;
        }
        uint32_t best = 0;
        for (uint32_t s = 1; s < S; ++s) {
            if (head_key[s] > head_key[best]) {
                best = s;
            }
        }
        const uint32_t r = head[best];
        ov[n] = elem(val_l1, best, r) << 16;  // bf16 -> fp32, exact
        oi[n] = best * W + elem(id_l1, best, r);
        head[best] = r + 1;
        head_key[best] = (r + 1 < Ks) ? order_key(elem(val_l1, best, r + 1)) : 0;
    }

    // Rank n sits at (tile n / 32, half (n % 32) / 16, i n % 16): ov / oi are in that order.
    for (uint32_t t = 0; t < k_tiles; ++t) {
        for (uint32_t half = 0; half < 2; ++half) {
            const uint32_t off = face_row_offset(b, half, 4);
            noc_async_write(out_val_l1 + (t * 2 + half) * 64, out_vals.get_noc_addr(t, off), 64);
            noc_async_write(out_id_l1 + (t * 2 + half) * 64, out_ids.get_noc_addr(t, off), 64);
        }
    }
    noc_async_write_barrier();
}

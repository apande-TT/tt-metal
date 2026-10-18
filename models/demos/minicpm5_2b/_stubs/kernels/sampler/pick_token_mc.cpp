// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B sampler tail, one core per batch row, over the k top-k candidates (fp32 / uint32 [1, 1, B, kpad]
// tiles; each core reads only its row's face rows):
//   keep      = tail_mass > (1 - top_p)                   (ttnn.gt)
//   threshold = min over kept candidates of top_vals       (ttnn.min of top_vals + (keep - 1) * NEG)
//   allowed   = top_vals >= threshold                      (ttnn.ge)
//   token     = idx[argmax over allowed of scored]         (scored = top_vals + noise, disallowed pushed down by
//                                                           1e9; ttnn.argmax's comparison, first max wins)
// Every comparison is on an order-preserving integer key of the fp32 bits, so no float arithmetic runs here.
// Every non-leader core writes its token id into a 16 B slot of the leader's scratch and bumps the leader's
// semaphore; the leader (row 0) waits for all B - 1, packs the B ids and writes the [1, B] uint32 token buffer.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace {
// Order-preserving key for float32: a NaN beats every non-NaN and ties another NaN (argmax's
// argmax_float32_greater), +0 and -0 tie, otherwise ordinary float order.
inline uint32_t order_key(uint32_t u) {
    if ((u & 0x7F800000u) == 0x7F800000u && (u & 0x007FFFFFu) != 0) {
        return 0xFFFFFFFFu;
    }
    if ((u & 0x7FFFFFFFu) == 0) {
        return 0x80000000u;
    }
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t k = get_compile_time_arg_val(0);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t B = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(3);
    constexpr uint32_t sem_id = get_compile_time_arg_val(4);
    constexpr uint32_t keep_above_bits = get_compile_time_arg_val(5);  // fp32 bits of 1 - top_p
    constexpr auto tail_args = TensorAccessorArgs<6>();
    constexpr auto val_args = TensorAccessorArgs<tail_args.next_compile_time_args_offset()>();
    constexpr auto score_args = TensorAccessorArgs<val_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<score_args.next_compile_time_args_offset()>();
    constexpr auto tok_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
    constexpr uint32_t face_row_bytes = 16 * 4;
    constexpr uint32_t row_bytes = k_tiles * 32 * 4;  // the row's kpad values, contiguous

    const auto tail = TensorAccessor(tail_args, get_common_arg_val<uint32_t>(0));
    const auto vals = TensorAccessor(val_args, get_common_arg_val<uint32_t>(1));
    const auto score = TensorAccessor(score_args, get_common_arg_val<uint32_t>(2));
    const auto idx = TensorAccessor(idx_args, get_common_arg_val<uint32_t>(3));
    const auto tok = TensorAccessor(tok_args, get_common_arg_val<uint32_t>(4));
    const uint32_t leader_x = get_common_arg_val<uint32_t>(5);
    const uint32_t leader_y = get_common_arg_val<uint32_t>(6);
    const uint32_t b = get_arg_val<uint32_t>(0);

    // Scratch (same address on every core): tail | vals | score | ids rows, B 16 B gather slots, packed [B] ids.
    const uint32_t tail_l1 = get_write_ptr(cb_scratch);
    const uint32_t val_l1 = tail_l1 + row_bytes;
    const uint32_t score_l1 = val_l1 + row_bytes;
    const uint32_t idx_l1 = score_l1 + row_bytes;
    const uint32_t slot_l1 = idx_l1 + row_bytes;
    const uint32_t tok_l1 = slot_l1 + B * 16;

    // Row b of a 32x32 tile of 4-byte elements: columns 0-15 in face (b / 16) * 2, columns 16-31 in the next face.
    const uint32_t face_off = ((b >> 4) << 1) * 1024 + (b & 15) * face_row_bytes;
    for (uint32_t t = 0; t < k_tiles; ++t) {
        for (uint32_t h = 0; h < 2; ++h) {
            const uint32_t src = face_off + h * 1024;
            const uint32_t dst = t * 128 + h * face_row_bytes;
            noc_async_read(tail.get_noc_addr(t, src), tail_l1 + dst, face_row_bytes);
            noc_async_read(vals.get_noc_addr(t, src), val_l1 + dst, face_row_bytes);
            noc_async_read(score.get_noc_addr(t, src), score_l1 + dst, face_row_bytes);
            noc_async_read(idx.get_noc_addr(t, src), idx_l1 + dst, face_row_bytes);
        }
    }
    noc_async_read_barrier();

    const auto* tm = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tail_l1);
    const auto* tv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(val_l1);
    const auto* sc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(score_l1);
    constexpr uint32_t keep_above = keep_above_bits;
    const uint32_t keep_key = order_key(keep_above);

    uint32_t threshold_key = 0xFFFFFFFFu;
    bool any_kept = false;
    for (uint32_t r = 0; r < k; ++r) {
        if (order_key(tm[r]) > keep_key) {
            const uint32_t key = order_key(tv[r]);
            if (!any_kept || key < threshold_key) {
                threshold_key = key;
            }
            any_kept = true;
        }
    }

    uint32_t best = 0;
    uint32_t best_key = 0;
    bool found = false;
    for (uint32_t r = 0; r < k; ++r) {
        if (any_kept && order_key(tv[r]) < threshold_key) {
            continue;
        }
        const uint32_t key = order_key(sc[r]);
        if (!found || key > best_key) {
            best = r;
            best_key = key;
            found = true;
        }
    }
    const uint32_t id = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1)[best];

    const uint32_t sem_l1 = get_semaphore(sem_id);
    if (b != 0) {
        auto* mine = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_l1 + b * 16);
        *mine = id;
        noc_async_write(slot_l1 + b * 16, get_noc_addr(leader_x, leader_y, slot_l1 + b * 16), 4);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(leader_x, leader_y, sem_l1), 1);
        noc_async_atomic_barrier();
        return;
    }

    auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_l1);
    noc_semaphore_wait(sem, B - 1);
    noc_semaphore_set(sem, 0);
    auto* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tok_l1);
    out[0] = id;
    for (uint32_t i = 1; i < B; ++i) {
        out[i] = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_l1 + i * 16);
    }
    noc_async_write(tok_l1, tok.get_noc_addr(0), B * 4);
    noc_async_write_barrier();
}

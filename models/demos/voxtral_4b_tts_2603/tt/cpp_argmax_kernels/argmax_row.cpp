// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The C++ semantic argmax (see tt/cpp_argmax.py). This core owns row b of the float32 TILE logits;
// RISC ROLE scans tile columns [T0, T1): it reads row b's two 64-byte face segments of each tile
// (all under one barrier) and keeps the first column holding the largest value, comparing floats as
// order-preserving unsigned keys (no FPU on the RISC). RISC 1 hands its (key, index) to RISC 0 through
// a CB; RISC 0 keeps the earlier index on a tie and writes the uint32 index to the output row.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace {
FORCE_INLINE uint32_t key_of(uint32_t bits) { return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u); }
}  // namespace

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t y_addr = get_arg_val<uint32_t>(1);
    const uint32_t b = get_arg_val<uint32_t>(2);
    const uint32_t t0 = get_arg_val<uint32_t>(3);
    const uint32_t t1 = get_arg_val<uint32_t>(4);

    constexpr uint32_t ROLE = get_compile_time_arg_val(0);
    constexpr uint32_t W = get_compile_time_arg_val(1);  // logical columns (pad columns are skipped)
    constexpr uint32_t CB_SCRATCH = get_compile_time_arg_val(2);
    constexpr uint32_t CB_MAIL = get_compile_time_arg_val(3);
    constexpr auto ax = TensorAccessorArgs<4>();
    constexpr auto ay = TensorAccessorArgs<ax.next_compile_time_args_offset()>();
    const auto sx = TensorAccessor(ax, x_addr);
    const auto sy = TensorAccessor(ay, y_addr);

    const uint32_t row_face = (b >> 4) * 2;
    const uint32_t row_off = (b & 15) * 64;
    const uint32_t buf = get_write_ptr(CB_SCRATCH);
    for (uint32_t t = t0; t < t1; ++t) {
        const uint32_t dst = buf + (t - t0) * 128;
        noc_async_read(sx.get_noc_addr(t, row_face * 1024 + row_off), dst, 64);
        noc_async_read(sx.get_noc_addr(t, (row_face + 1) * 1024 + row_off), dst + 64, 64);
    }
    noc_async_read_barrier();

    uint32_t best_key = 0;
    uint32_t best = t0 * 32;
    const uint32_t* v = reinterpret_cast<const uint32_t*>(buf);
    for (uint32_t t = t0; t < t1; ++t) {
        for (uint32_t j = 0; j < 32; ++j) {
            const uint32_t col = t * 32 + j;
            if (col >= W) {
                break;
            }
            const uint32_t k = key_of(v[(t - t0) * 32 + j]);
            if (k > best_key) {
                best_key = k;
                best = col;
            }
        }
    }

    if constexpr (ROLE == 1) {
        cb_reserve_back(CB_MAIL, 1);
        volatile tt_l1_ptr uint32_t* mail = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_MAIL));
        mail[0] = best_key;
        mail[1] = best;
        cb_push_back(CB_MAIL, 1);
    } else {
        cb_wait_front(CB_MAIL, 1);
        volatile tt_l1_ptr uint32_t* mail = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_MAIL));
        if (mail[0] > best_key) {
            best = mail[1];
        }
        cb_pop_front(CB_MAIL, 1);
        volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);
        out[0] = best;
        noc_async_write(buf, sy.get_noc_addr(b), 4);
        noc_async_write_barrier();
    }
}

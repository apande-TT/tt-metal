// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode rope rows from precomputed tables: for each batch row b of this core, row pos[b] of the cos
// table and of the signed-sin table ([P, D] bf16 tiles, made at init by the per-step rotary ops themselves) is copied
// into row b of the [1, 1, 32, D] cos / sin outputs. A tile row is two 32 B face rows per D tile, so a gathered row
// is 2 * Dt face-row copies per table. pos: the int32 [1, B] position tile (column b = batch b).

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
    constexpr uint32_t cb_pos = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rows = get_compile_time_arg_val(1);
    constexpr uint32_t Dt = get_compile_time_arg_val(2);
    constexpr uint32_t users = get_compile_time_arg_val(3);
    constexpr uint32_t pos_tile_bytes = 32 * 32 * 4;
    constexpr uint32_t tile_bytes = 32 * 32 * 2;
    constexpr auto pos_args = TensorAccessorArgs<4>();
    constexpr auto tab_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<tab_args.next_compile_time_args_offset()>();

    const uint32_t b0 = get_arg_val<uint32_t>(0);
    const auto pos = TensorAccessor(pos_args, get_common_arg_val<uint32_t>(0), pos_tile_bytes);
    const auto cos_tab = TensorAccessor(tab_args, get_common_arg_val<uint32_t>(1), tile_bytes);
    const auto sin_tab = TensorAccessor(tab_args, get_common_arg_val<uint32_t>(2), tile_bytes);
    const auto cos_out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(3), tile_bytes);
    const auto sin_out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(4), tile_bytes);

    const uint32_t l1_pos = get_write_ptr(cb_pos);
    noc_async_read_page(0, pos, l1_pos);
    noc_async_read_barrier();
    volatile tt_l1_ptr int32_t* p = reinterpret_cast<volatile tt_l1_ptr int32_t*>(l1_pos);

    // Gather: the DRAM tables need 64 B-aligned reads, so each face row comes in as the aligned 64 B span holding it
    // (one 64 B slot per user, table, D tile and half); its 32 B half is then stored from L1.
    constexpr uint32_t kSpan = 64;
    const uint32_t l1_rows = (get_write_ptr(cb_rows) + kSpan - 1) & ~(kSpan - 1);
    uint32_t slot = l1_rows;
    for (uint32_t u = 0; u < users; ++u) {
        const uint32_t b = b0 + u;
        const uint32_t r = static_cast<uint32_t>(p[(b >= 16 ? 256 : 0) + (b & 15)]);
        const uint32_t tr = r >> 5;
        for (uint32_t t = 0; t < 2; ++t) {
            const auto& tab = t == 0 ? cos_tab : sin_tab;
            for (uint32_t d = 0; d < Dt; ++d) {
                for (uint32_t half = 0; half < 2; ++half) {
                    const uint32_t off = face_row_offset(r & 31, half);
                    noc_async_read(tab.get_noc_addr(tr * Dt + d, off & ~(kSpan - 1)), slot, kSpan);
                    slot += kSpan;
                }
            }
        }
    }
    noc_async_read_barrier();

    // Scatter into row b of the outputs.
    slot = l1_rows;
    for (uint32_t u = 0; u < users; ++u) {
        const uint32_t b = b0 + u;
        const uint32_t r = static_cast<uint32_t>(p[(b >= 16 ? 256 : 0) + (b & 15)]);
        for (uint32_t t = 0; t < 2; ++t) {
            const auto& out = t == 0 ? cos_out : sin_out;
            for (uint32_t d = 0; d < Dt; ++d) {
                for (uint32_t half = 0; half < 2; ++half) {
                    const uint32_t src = slot + (face_row_offset(r & 31, half) & (kSpan - 1));
                    noc_async_write(src, out.get_noc_addr(d, face_row_offset(b, half)), kFaceRowBytes);
                    slot += kSpan;
                }
            }
        }
    }
    noc_async_write_barrier();
}

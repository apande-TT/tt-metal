// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The writer of the C++ codec residual add + RMS norm (see tt/cpp_addnorm.py): per unit it writes the WT
// tiles of the new residual stream h + r as the compute packs them, fills the row scale across its tile
// (each row's column-0 value into all 32 columns, as binary_ng's column-broadcast reader does), then
// writes the WT normalised tiles.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// fill_tile_with_first_column (binary_ng), from a source float32 tile into a separate destination tile.
inline void fill_from_first_column(uint32_t src_addr, uint32_t dst_addr) {
    auto* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_addr);
    auto* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_addr);
    for (uint32_t k = 0, face_offset = 0; k < 2; ++k, face_offset += 512) {
        for (uint32_t row = 0, row_offset = 0; row < 16; ++row, row_offset += 16) {
            const uint32_t left = face_offset + row_offset;
            const uint32_t right = left + 256;
            const uint32_t v = src[left];
            for (uint32_t col = 0; col < 16; ++col) {
                dst[left + col] = v;
                dst[right + col] = v;
            }
        }
    }
}

void kernel_main() {
    const uint32_t hn_addr = get_arg_val<uint32_t>(0);
    const uint32_t y_addr = get_arg_val<uint32_t>(1);
    const uint32_t u0 = get_arg_val<uint32_t>(2);
    const uint32_t nu = get_arg_val<uint32_t>(3);

    constexpr uint32_t WT = get_compile_time_arg_val(0);
    constexpr auto ahn = TensorAccessorArgs<1>();
    constexpr auto ay = TensorAccessorArgs<ahn.next_compile_time_args_offset()>();
    const auto shn = TensorAccessor(ahn, hn_addr);
    const auto sy = TensorAccessor(ay, y_addr);

    constexpr uint32_t cb_inv = 3;
    constexpr uint32_t cb_invf = 4;
    constexpr uint32_t cb_hn = 16;
    constexpr uint32_t cb_y = 17;

    for (uint32_t u = u0; u < u0 + nu; ++u) {
        for (uint32_t w = 0; w < WT; ++w) {
            cb_wait_front(cb_hn, 1);
            noc_async_write_page(u * WT + w, shn, get_read_ptr(cb_hn));
            noc_async_write_barrier();
            cb_pop_front(cb_hn, 1);
        }
        cb_wait_front(cb_inv, 1);
        cb_reserve_back(cb_invf, 1);
        fill_from_first_column(get_read_ptr(cb_inv), get_write_ptr(cb_invf));
        cb_push_back(cb_invf, 1);
        cb_pop_front(cb_inv, 1);
        for (uint32_t w = 0; w < WT; ++w) {
            cb_wait_front(cb_y, 1);
            noc_async_write_page(u * WT + w, sy, get_read_ptr(cb_y));
            noc_async_write_barrier();
            cb_pop_front(cb_y, 1);
        }
    }
}

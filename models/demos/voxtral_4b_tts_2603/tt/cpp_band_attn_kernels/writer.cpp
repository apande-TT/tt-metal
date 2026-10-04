// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The writer of the C++ banded codec attention (see tt/cpp_band_attn.py): fills each unit's row max and
// row sum across their tiles (binary_ng's reader-side column broadcast, as tt/cpp_softmax's writer
// does) and writes the unit's DT context tiles (into the merged [B, 1, T, H * D] layout when MERGED).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// fill_tile_with_first_column (binary_ng), from a source tile into a separate destination tile.
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

inline void fill_cb(uint32_t cb_in, uint32_t cb_out) {
    cb_wait_front(cb_in, 1);
    cb_reserve_back(cb_out, 1);
    fill_from_first_column(get_read_ptr(cb_in), get_write_ptr(cb_out));
    cb_push_back(cb_out, 1);
    cb_pop_front(cb_in, 1);
}

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t u0 = get_arg_val<uint32_t>(1);
    const uint32_t nu = get_arg_val<uint32_t>(2);

    constexpr uint32_t DT = get_compile_time_arg_val(0);
    constexpr uint32_t H = get_compile_time_arg_val(1);
    constexpr uint32_t RT = get_compile_time_arg_val(2);
    constexpr uint32_t MERGED = get_compile_time_arg_val(3);  // 1: write the merged [B, 1, T, H * D]
    constexpr auto ay = TensorAccessorArgs<4>();
    const auto sy = TensorAccessor(ay, y_addr);

    constexpr uint32_t cb_max = 7;
    constexpr uint32_t cb_maxf = 8;
    constexpr uint32_t cb_sum = 10;
    constexpr uint32_t cb_sumf = 11;
    constexpr uint32_t cb_out = 16;
    const uint32_t tile_bytes = get_tile_size(cb_out);

    for (uint32_t u = u0; u < u0 + nu; ++u) {
        fill_cb(cb_max, cb_maxf);
        fill_cb(cb_sum, cb_sumf);
        cb_wait_front(cb_out, DT);
        uint32_t l1 = get_read_ptr(cb_out);
        for (uint32_t d = 0; d < DT; ++d) {
            const uint32_t pg = MERGED ? ((u / RT / H) * RT + u % RT) * (H * DT) + ((u / RT) % H) * DT + d
                                       : u * DT + d;
            noc_async_write_page(pg, sy, l1);
            l1 += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, DT);
    }
}

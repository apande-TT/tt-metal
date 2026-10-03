// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The gather of the C++ waveform flatten (see tt/cpp_wave.py): per unit (one output row b and one
// 32-row tile row tt of the codec's channels-last [B, L, C] float32 TILE output) it reads each valid
// row's 16-float face-row runs straight from DRAM into their row-major place in a [rows, C] block --
// out[r * C + c] = x[tt * 32 + r, c] -- from the part-chain tensor (rows [0, B1)) or the whole-section
// tensor (rows [B1, B)).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// Whole tiles come from DRAM in one burst each; the face-row runs are then gathered L1 -> L1 over the
// core's own NoC address (no RISC load / store loop).
template <typename Acc>
inline void gather(const Acc& src, uint32_t t0, uint32_t rows, uint32_t tiles_l1, uint32_t l1, uint32_t C, uint32_t CT) {
    for (uint32_t i = 0; i < CT; ++i) {
        noc_async_read_page(t0 + i, src, tiles_l1 + i * 4096);
    }
    noc_async_read_barrier();
    for (uint32_t r = 0; r < rows; ++r) {
        const uint32_t face_row = (r >> 4) * 2;
        const uint32_t in_face = (r & 15) * 64;
        uint32_t dst = l1 + r * C * 4;
        for (uint32_t col0 = 0; col0 < C; col0 += 16) {
            const uint32_t face = face_row + ((col0 >> 4) & 1);
            noc_async_read(get_noc_addr(tiles_l1 + (col0 >> 5) * 4096 + face * 1024 + in_face), dst, 64);
            dst += 64;
        }
    }
}

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t u0 = get_arg_val<uint32_t>(2);
    const uint32_t nu = get_arg_val<uint32_t>(3);

    constexpr uint32_t LT = get_compile_time_arg_val(0);  // tile rows of L
    constexpr uint32_t CT = get_compile_time_arg_val(1);  // tiles of C
    constexpr uint32_t B1 = get_compile_time_arg_val(2);  // rows from the first tensor
    constexpr uint32_t L = get_compile_time_arg_val(3);   // valid rows of L
    constexpr uint32_t C = get_compile_time_arg_val(4);   // valid columns (a multiple of 16)
    constexpr auto aa = TensorAccessorArgs<5>();
    constexpr auto ab = TensorAccessorArgs<aa.next_compile_time_args_offset()>();
    const auto sa = TensorAccessor(aa, a_addr);
    const auto sb = TensorAccessor(ab, b_addr);

    constexpr uint32_t cb_rm = 1;
    constexpr uint32_t cb_tiles = 0;
    const uint32_t tiles_l1 = get_write_ptr(cb_tiles);
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t row = u / LT;
        const uint32_t tt = u % LT;
        const uint32_t rows = (L - tt * 32) < 32 ? (L - tt * 32) : 32;
        cb_reserve_back(cb_rm, 1);
        const uint32_t l1 = get_write_ptr(cb_rm);
        if (row < B1) {
            gather(sa, (row * LT + tt) * CT, rows, tiles_l1, l1, C, CT);
        } else {
            gather(sb, ((row - B1) * LT + tt) * CT, rows, tiles_l1, l1, C, CT);
        }
        noc_async_read_barrier();
        cb_push_back(cb_rm, 1);
    }
}

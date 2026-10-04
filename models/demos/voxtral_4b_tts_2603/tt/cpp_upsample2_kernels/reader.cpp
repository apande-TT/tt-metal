// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The "now" stream of the C++ stride-2 transposed convolution (see tt/cpp_upsample2.py). Unit u = output tile
// (sample b, output tile row R, column tile c): output rows p = 32R .. 32R + 31 are m = 16R + (p >> 1) with
// parity j = p & 1, and their "now" term is Y row m of tap block j -- rows 16h .. 16h + 15 (h = R & 1) of Y
// tile row R >> 1, i.e. the two contiguous faces 2h, 2h + 1 of that tile (one 2 KB half-tile read per block).
// The 32 rows are gathered into one tile in output order by LOCAL NoC reads, one 64 B face row at a time
// (rows alternate between the two blocks, so no run is longer than a row); m >= L (Y's per-sample tile
// padding) and output rows past out_len read zeros.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t u0 = get_arg_val<uint32_t>(1);
    const uint32_t nu = get_arg_val<uint32_t>(2);

    constexpr uint32_t L = get_compile_time_arg_val(0);    // real rows a sample
    constexpr uint32_t RTY = get_compile_time_arg_val(1);  // Y tile rows a sample (ceil(L / 32))
    constexpr uint32_t OL = get_compile_time_arg_val(2);   // output rows a sample
    constexpr uint32_t OR = get_compile_time_arg_val(3);   // output tile rows a sample
    constexpr uint32_t CT = get_compile_time_arg_val(4);   // column tiles a tap block (= output column tiles)
    constexpr auto ay = TensorAccessorArgs<5>();
    const auto sy = TensorAccessor(ay, y_addr);
    constexpr uint32_t YCT = 4 * CT;

    constexpr uint32_t cb_now = 0;
    constexpr uint32_t cb_scratch = 2;
    constexpr uint32_t face = 1024;  // fp32 16 x 16
    constexpr uint32_t half = 2048;  // two faces: 16 rows of a tile
    constexpr uint32_t seg = 64;     // one face row

    cb_reserve_back(cb_scratch, 2);
    const uint32_t scratch = get_write_ptr(cb_scratch);  // [block 0 half | block 1 half | zeros]
    const uint32_t zeros = scratch + 2 * half;
    volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zeros);
    for (uint32_t w = 0; w < seg / 4; ++w) {
        z[w] = 0;
    }

    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t c = u % CT;
        const uint32_t R = (u / CT) % OR;
        const uint32_t b = u / (CT * OR);
        const uint32_t ty = R >> 1;
        const uint32_t h = R & 1;
        const uint32_t m0 = 16 * R;
        if (m0 < L) {
            for (uint32_t j = 0; j < 2; ++j) {
                noc_async_read(sy.get_noc_addr((b * RTY + ty) * YCT + j * CT + c, h * half), scratch + j * half, half);
            }
            noc_async_read_barrier();
        }
        cb_reserve_back(cb_now, 1);
        const uint32_t dst0 = get_write_ptr(cb_now);
        noc_async_read_one_packet_set_state(get_noc_addr(scratch), seg);
        for (uint32_t p = 0; p < 32; ++p) {
            const uint32_t i = p >> 1;
            const uint32_t j = p & 1;
            const bool real = m0 + i < L && 32 * R + p < OL;
            const uint32_t dst = dst0 + (p >> 4) * 2 * face + (p & 15) * seg;
            for (uint32_t f = 0; f < 2; ++f) {
                const uint32_t src = real ? scratch + j * half + f * face + i * seg : zeros;
                noc_async_read_one_packet_with_state(src, dst + f * face);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_now, 1);
    }
}

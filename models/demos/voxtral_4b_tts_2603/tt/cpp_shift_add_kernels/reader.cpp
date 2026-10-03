// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The input stream of the C++ convolution shift-add (see tt/cpp_shift_add.py). Unit u = (sample b, output
// tile row r, output column tile c). For each tap k the unit needs rows 32r + k .. 32r + k + 31 of the
// sample's block of Y, in tap k's column block: the two tile rows those span are burst-read whole into
// local scratch, and the 32 rows are gathered out of them -- as contiguous face-row runs, by LOCAL NoC
// reads -- into one shifted tile, pushed in tap order.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t u0 = get_arg_val<uint32_t>(1);
    const uint32_t nu = get_arg_val<uint32_t>(2);

    constexpr uint32_t K = get_compile_time_arg_val(0);    // taps
    constexpr uint32_t RPT = get_compile_time_arg_val(1);  // Y tile rows a sample
    constexpr uint32_t RT = get_compile_time_arg_val(2);   // output tile rows a sample
    constexpr uint32_t OT = get_compile_time_arg_val(3);   // output column tiles
    constexpr uint32_t CT = get_compile_time_arg_val(4);   // Y column tiles a tap block
    constexpr uint32_t YCT = get_compile_time_arg_val(5);  // Y column tiles in all
    constexpr uint32_t YRT = get_compile_time_arg_val(6);  // Y tile rows in all
    constexpr auto ay = TensorAccessorArgs<7>();
    const auto sy = TensorAccessor(ay, y_addr);

    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_scratch = 1;
    constexpr uint32_t tile_bytes = 4096;  // fp32
    constexpr uint32_t seg = 64;           // 16 fp32 values: one face row
    constexpr uint32_t face = 1024;

    cb_reserve_back(cb_scratch, 2);
    const uint32_t scratch = get_write_ptr(cb_scratch);

    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t c = u % OT;
        const uint32_t r = (u / OT) % RT;
        const uint32_t b = u / (OT * RT);
        const uint32_t tr = b * RPT + r;
        for (uint32_t k = 0; k < K; ++k) {
            const uint32_t col = k * CT + c;
            // The two tile rows rows k .. k + 31 span (the second clamped to the tensor: past the last
            // sample's block it only feeds padding rows of the output).
            const uint32_t tr1 = tr + 1 < YRT ? tr + 1 : tr;
            noc_async_read_page(tr * YCT + col, sy, scratch);
            noc_async_read_page(tr1 * YCT + col, sy, scratch + tile_bytes);
            noc_async_read_barrier();
            cb_reserve_back(cb_in, 1);
            const uint32_t dst0 = get_write_ptr(cb_in);
            for (uint32_t f = 0; f < 2; ++f) {
                uint32_t i = 0;
                while (i < 32) {
                    const uint32_t src_row = k + i;
                    const uint32_t ts = src_row >> 5;
                    const uint32_t ri = src_row & 31;
                    uint32_t n = 16 - (ri & 15);
                    if (16 - (i & 15) < n) {
                        n = 16 - (i & 15);
                    }
                    if (32 - i < n) {
                        n = 32 - i;
                    }
                    const uint32_t src = scratch + ts * tile_bytes + ((ri >> 4) * 2 + f) * face + (ri & 15) * seg;
                    const uint32_t dst = dst0 + ((i >> 4) * 2 + f) * face + (i & 15) * seg;
                    noc_async_read(get_noc_addr(src), dst, n * seg);
                    i += n;
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_in, 1);
        }
    }
}

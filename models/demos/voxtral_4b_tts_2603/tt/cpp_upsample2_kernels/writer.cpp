// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The "delayed" stream and the output writer of the C++ stride-2 transposed convolution (see
// tt/cpp_upsample2.py). For output tile (b, R, c), row p = 2i + j (m = 16R + i) takes Y row m - 1 of tap block
// 2 + j: rows i >= 1 are rows i - 1 of the same half-tile the "now" term reads (faces 2h, 2h + 1 of Y tile row
// R >> 1), and row i = 0 is the row just before it -- row 15 of the top half (h = 1) or row 31 of the previous
// Y tile row (h = 0), two 64 B face-row reads per block; m - 1 < 0 or m - 1 >= L reads zeros. The gathered
// tile goes to the compute kernel, which adds it to the "now" tile; the sum comes back here and is written.
// The next unit's delayed tile is gathered before waiting on this unit's sum, so the three cores overlap.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t o_addr = get_arg_val<uint32_t>(1);
    const uint32_t u0 = get_arg_val<uint32_t>(2);
    const uint32_t nu = get_arg_val<uint32_t>(3);

    constexpr uint32_t RTY = get_compile_time_arg_val(0);
    constexpr uint32_t OR = get_compile_time_arg_val(1);
    constexpr uint32_t CT = get_compile_time_arg_val(2);
    constexpr auto ay = TensorAccessorArgs<3>();
    constexpr auto ao = TensorAccessorArgs<ay.next_compile_time_args_offset()>();
    const auto sy = TensorAccessor(ay, y_addr);
    const auto so = TensorAccessor(ao, o_addr);
    constexpr uint32_t YCT = 4 * CT;

    constexpr uint32_t cb_delayed = 1;
    constexpr uint32_t cb_scratch = 3;
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t face = 1024;
    constexpr uint32_t half = 2048;
    constexpr uint32_t seg = 64;

    cb_reserve_back(cb_scratch, 2);
    // [block 2 half | block 3 half | prev rows: block 2 (left, right), block 3 (left, right) | zeros]
    const uint32_t scratch = get_write_ptr(cb_scratch);
    const uint32_t prev = scratch + 2 * half;
    const uint32_t zeros = prev + 4 * seg;
    volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zeros);
    for (uint32_t w = 0; w < seg / 4; ++w) {
        z[w] = 0;
    }

    auto gather = [&](uint32_t u) {
        const uint32_t c = u % CT;
        const uint32_t R = (u / CT) % OR;
        const uint32_t b = u / (CT * OR);
        const uint32_t ty = R >> 1;
        const uint32_t h = R & 1;
        const bool valid = ty < RTY;  // rows m - 1 = 16R .. 16R + 14 are real
        const bool has_prev = R > 0;  // row m - 1 = 16R - 1 (always < L once R > 0: 16R - 1 < 16 * OR - 1 <= L)
        for (uint32_t j = 0; j < 2; ++j) {
            const uint32_t col = (2 + j) * CT + c;
            if (valid) {
                noc_async_read(sy.get_noc_addr((b * RTY + ty) * YCT + col, h * half), scratch + j * half, half);
            }
            if (has_prev) {
                // Row 16R - 1: row 15 of the top half of tile row ty (h = 1), or row 31 of tile row ty - 1 (h = 0).
                const uint32_t pt = h ? ty : ty - 1;
                const uint32_t pf = h ? 0 : 2;
                for (uint32_t f = 0; f < 2; ++f) {
                    noc_async_read(
                        sy.get_noc_addr((b * RTY + pt) * YCT + col, (pf + f) * face + 15 * seg),
                        prev + (2 * j + f) * seg,
                        seg);
                }
            }
        }
        noc_async_read_barrier();
        cb_reserve_back(cb_delayed, 1);
        const uint32_t dst0 = get_write_ptr(cb_delayed);
        noc_async_read_one_packet_set_state(get_noc_addr(scratch), seg);
        for (uint32_t p = 0; p < 32; ++p) {
            const uint32_t i = p >> 1;
            const uint32_t j = p & 1;
            const uint32_t dst = dst0 + (p >> 4) * 2 * face + (p & 15) * seg;
            for (uint32_t f = 0; f < 2; ++f) {
                uint32_t src;
                if (i == 0) {
                    src = has_prev ? prev + (2 * j + f) * seg : zeros;
                } else {
                    src = valid ? scratch + j * half + f * face + (i - 1) * seg : zeros;
                }
                noc_async_read_one_packet_with_state(src, dst + f * face);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_delayed, 1);
    };

    if (nu > 0) {
        gather(u0);
    }
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        if (u + 1 < u0 + nu) {
            gather(u + 1);
        }
        cb_wait_front(cb_out, 1);
        noc_async_write_page(u, so, get_read_ptr(cb_out));
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}

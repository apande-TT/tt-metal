// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The input stream of the C++ tail-row regroup (see tt/cpp_tail_rows.py). Unit u = (pair p, head h,
// sample b): the sample's R rows of head h, out of the compact bf16 TILE tensor [1, H, B * R, D] --
// the input tile rows they span are burst-read whole, then two 32-byte face-row segments a row and
// column tile are gathered locally into rows 0..R-1 of a tile row; rows R..31 stay zero (both CB
// slots are zeroed once and those rows are never written).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t a0 = get_arg_val<uint32_t>(0);  // pair 0 input
    const uint32_t a1 = get_arg_val<uint32_t>(1);  // pair 1 input (unused when NP == 1)
    const uint32_t u0 = get_arg_val<uint32_t>(2);
    const uint32_t nu = get_arg_val<uint32_t>(3);

    constexpr uint32_t DT = get_compile_time_arg_val(0);   // column tiles
    constexpr uint32_t R = get_compile_time_arg_val(1);    // real rows a sample
    constexpr uint32_t B = get_compile_time_arg_val(2);    // samples
    constexpr uint32_t H = get_compile_time_arg_val(3);    // heads
    constexpr uint32_t RTI = get_compile_time_arg_val(4);  // input tile rows a head (B * R / 32, rounded up)
    constexpr auto acc0 = TensorAccessorArgs<5>();
    constexpr auto acc1 = TensorAccessorArgs<acc0.next_compile_time_args_offset()>();
    const auto s0 = TensorAccessor(acc0, a0);
    const auto s1 = TensorAccessor(acc1, a1);

    constexpr uint32_t cb_in = 0;
    constexpr uint32_t tile_bytes = 2048;  // bf16
    constexpr uint32_t seg = 32;           // 16 bf16 values: one face row

    {
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        const uint32_t base = get_write_ptr(cb_in);
        for (uint32_t off = 0; off < 2 * DT * tile_bytes; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, base + off, MEM_ZEROS_SIZE);
        }
        noc_async_read_barrier();
    }

    // Per unit: burst the (at most two) input tile rows the sample's R rows span into local scratch,
    // then gather the rows out of it with LOCAL NoC reads (a remote 32-byte read per row segment is
    // latency-bound: 160 of them a unit).
    constexpr uint32_t cb_scratch = 1;
    cb_reserve_back(cb_scratch, 2 * DT);
    const uint32_t scratch = get_write_ptr(cb_scratch);
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t p = u / (H * B);
        const uint32_t hb = u % (H * B);
        const uint32_t h = hb / B;
        const uint32_t b = hb % B;
        const uint32_t t0 = (b * R) >> 5;
        const uint32_t t1 = (b * R + R - 1) >> 5;
        for (uint32_t t = t0; t <= t1; ++t) {
            for (uint32_t c = 0; c < DT; ++c) {
                const uint32_t page = (h * RTI + t) * DT + c;
                const uint32_t dst = scratch + ((t - t0) * DT + c) * tile_bytes;
                if (p == 0) {
                    noc_async_read_page(page, s0, dst);
                } else {
                    noc_async_read_page(page, s1, dst);
                }
            }
        }
        noc_async_read_barrier();
        cb_reserve_back(cb_in, DT);
        const uint32_t l1 = get_write_ptr(cb_in);
        // Rows that stay inside one 16-row face on both sides are contiguous in both: copy them as runs
        // (at most 4 a face half), not one 32-byte segment a row.
        for (uint32_t c = 0; c < DT; ++c) {
            for (uint32_t f = 0; f < 2; ++f) {
                uint32_t i = 0;
                while (i < R) {
                    const uint32_t src_row = b * R + i;
                    const uint32_t ts = (src_row >> 5) - t0;
                    const uint32_t ri = src_row & 31;
                    uint32_t n = 16 - (ri & 15);
                    if (16 - (i & 15) < n) {
                        n = 16 - (i & 15);
                    }
                    if (R - i < n) {
                        n = R - i;
                    }
                    const uint32_t src = scratch + (ts * DT + c) * tile_bytes + ((ri >> 4) * 2 + f) * 512 + (ri & 15) * seg;
                    const uint32_t dst = l1 + c * tile_bytes + ((i >> 4) * 2 + f) * 512 + (i & 15) * seg;
                    noc_async_read(get_noc_addr(src), dst, n * seg);
                    i += n;
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_in, DT);
    }
}

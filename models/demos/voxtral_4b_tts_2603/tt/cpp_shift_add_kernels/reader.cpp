// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The input stream of the C++ convolution shift-add (see tt/cpp_shift_add.py). Unit u = (sample b, output
// tile row r, output column tile c). For each tap k the unit needs padded rows p = 32r + k .. 32r + k + 31
// of the sample, in tap k's column block. A padded row p is Y row src(p) of the sample's block:
//   MODE 0: Y already holds the padded rows -- src = p;
//   MODE 1: Y holds the L unpadded rows and the padding is REFLECT (P rows mirrored about row 0 in front, the
//           tail mirrored about row L - 1) -- the conv is linear per row, so a padded row's product IS the
//           product of the row it copies;
//   MODE 2: the same with REPLICATE padding (row 0 in front, row L - 1 behind).
// The (at most two) Y tile rows those rows fall in are burst-read whole into local scratch, and the 32 rows
// are gathered out of them -- as contiguous face-row runs, by LOCAL NoC reads -- into one shifted tile,
// pushed in tap order.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

template <uint32_t MODE, uint32_t P, uint32_t L, uint32_t LAST>
inline uint32_t src_row(uint32_t p) {
    if constexpr (MODE == 0) {
        return p < LAST ? p : LAST;
    } else {
        if (p < P) {
            return MODE == 1 ? P - p : 0;
        }
        const uint32_t q = p - P;
        if (q < L) {
            return q;
        }
        if constexpr (MODE == 1) {
            const uint32_t back = q - L + 1;  // 1, 2, ... rows past the end
            return back < L ? L - 1 - back : 0;
        } else {
            return L - 1;
        }
    }
}

// The 32 padded rows p .. p + 31 all map linearly (src = p - P, or src = p for MODE 0): no padding row among them.
template <uint32_t MODE, uint32_t P, uint32_t L, uint32_t LAST>
inline bool interior(uint32_t p) {
    if constexpr (MODE == 0) {
        return p + 31 <= LAST;
    } else {
        return p >= P && p + 31 - P < L;
    }
}

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t u0 = get_arg_val<uint32_t>(1);
    const uint32_t nu = get_arg_val<uint32_t>(2);

    constexpr uint32_t K = get_compile_time_arg_val(0);     // taps
    constexpr uint32_t RPT = get_compile_time_arg_val(1);   // Y tile rows a sample
    constexpr uint32_t RT = get_compile_time_arg_val(2);    // output tile rows a sample
    constexpr uint32_t OT = get_compile_time_arg_val(3);    // output column tiles
    constexpr uint32_t CT = get_compile_time_arg_val(4);    // Y column tiles a tap block
    constexpr uint32_t YCT = get_compile_time_arg_val(5);   // Y column tiles in all
    constexpr uint32_t MODE = get_compile_time_arg_val(6);  // 0 pre-padded, 1 reflect, 2 replicate
    constexpr uint32_t P = get_compile_time_arg_val(7);     // front padding rows (MODE 1 / 2)
    constexpr uint32_t L = get_compile_time_arg_val(8);     // real rows a sample (MODE 1 / 2)
    constexpr uint32_t ES = get_compile_time_arg_val(9);    // Y element bytes: 4 float32, 2 bfloat16
    constexpr auto ay = TensorAccessorArgs<10>();
    const auto sy = TensorAccessor(ay, y_addr);
    // The last Y row a source may be (MODE 0 clamps into the sample's block; padding rows of the output only).
    constexpr uint32_t LAST = RPT * 32 - 1;

    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_scratch = 1;
    constexpr uint32_t tile_bytes = 1024 * ES;
    constexpr uint32_t seg = 16 * ES;  // 16 values: one face row (32 B for bf16 -- L1 reads stay 16 B aligned)
    constexpr uint32_t face = 256 * ES;

    // structural: a unit's 2 K source tiles are read in ONE batch (one barrier), then its K shifted tiles are
    // gathered in one batch (one barrier) -- not a read / barrier / gather / barrier round trip per tap.
    cb_reserve_back(cb_scratch, 2 * K);
    const uint32_t scratch = get_write_ptr(cb_scratch);

    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t c = u % OT;
        const uint32_t r = (u / OT) % RT;
        const uint32_t b = u / (OT * RT);
        uint32_t first[K];
        for (uint32_t k = 0; k < K; ++k) {
            const uint32_t col = k * CT + c;
            const uint32_t p0 = 32 * r + k;
            // The source rows of this tap's 32 rows span at most two Y tile rows: find the first (an interior
            // run of rows maps linearly, so its first row is the minimum; only an edge run is scanned).
            uint32_t lo;
            if (interior<MODE, P, L, LAST>(p0)) {
                lo = p0 - (MODE == 0 ? 0 : P);
            } else {
                lo = src_row<MODE, P, L, LAST>(p0);
                for (uint32_t i = 1; i < 32; ++i) {
                    const uint32_t s = src_row<MODE, P, L, LAST>(p0 + i);
                    lo = s < lo ? s : lo;
                }
            }
            const uint32_t t0 = lo >> 5;
            const uint32_t t1 = t0 + 1 < RPT ? t0 + 1 : t0;
            first[k] = t0;
            noc_async_read_page((b * RPT + t0) * YCT + col, sy, scratch + (2 * k) * tile_bytes);
            noc_async_read_page((b * RPT + t1) * YCT + col, sy, scratch + (2 * k + 1) * tile_bytes);
        }
        noc_async_read_barrier();
        cb_reserve_back(cb_in, K);
        const uint32_t base = get_write_ptr(cb_in);
        for (uint32_t k = 0; k < K; ++k) {
            const uint32_t p0 = 32 * r + k;
            const uint32_t t0 = first[k];
            const uint32_t dst0 = base + k * tile_bytes;
            const uint32_t tap = scratch + (2 * k) * tile_bytes;
            const bool linear = interior<MODE, P, L, LAST>(p0);
            const uint32_t s_first = linear ? p0 - (MODE == 0 ? 0 : P) - 32 * t0 : 0;
            uint32_t i = 0;
            while (i < 32) {
                uint32_t s, n;
                if (linear) {
                    // Interior rows are contiguous: a run ends only at a 16-row face boundary on either side.
                    s = s_first + i;
                    const uint32_t a = 16 - (i & 15), bnd = 16 - (s & 15);
                    n = a < bnd ? a : bnd;
                } else {
                    s = src_row<MODE, P, L, LAST>(p0 + i) - 32 * t0;
                    // Extend the run while both sides stay contiguous inside one 16-row face.
                    n = 1;
                    while (i + n < 32 && ((i + n) & 15) != 0 && ((s + n) & 15) != 0 &&
                           src_row<MODE, P, L, LAST>(p0 + i + n) - 32 * t0 == s + n) {
                        ++n;
                    }
                }
                const uint32_t ts = s >> 5;
                const uint32_t ri = s & 31;
                for (uint32_t f = 0; f < 2; ++f) {
                    const uint32_t src = tap + ts * tile_bytes + ((ri >> 4) * 2 + f) * face + (ri & 15) * seg;
                    const uint32_t dst = dst0 + ((i >> 4) * 2 + f) * face + (i & 15) * seg;
                    noc_async_read(get_noc_addr(src), dst, n * seg);
                }
                i += n;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_in, K);
    }
}

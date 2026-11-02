// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Served prefill: the SSD states of n prompts, src (n, H, N, P) fp32 tiles, into their users' rows of the decode
// state dst (G N, 32, CPG * 32) (layout in ssm_step_reader.cpp). Row slot[i] (runtime arg 4 + i) of dst tile
// (g N + n) * CPG + c is prompt i's S[h, n, 32 (c % CPH) + j] over j, h = g HPG + c / CPH: row n % 32 of src tile
// ((i H + h) N / 32 + n / 32) * CPH + c % CPH, moved as its two 64 B face rows. The n G N CPG rows are dealt in
// contiguous ranges over the cores; every other user's row is untouched.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t FACE_BYTES = 1024, FACE_ROW_BYTES = 64;  // fp32

uint32_t row_offset(uint32_t r, uint32_t half) {
    return ((r / 16) * 2 + half) * FACE_BYTES + (r % 16) * FACE_ROW_BYTES;
}
}  // namespace

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);    // heads (this chip's)
    constexpr uint32_t N = get_compile_time_arg_val(1);    // ssm_state_size
    constexpr uint32_t CPH = get_compile_time_arg_val(2);  // column tiles per head (P / 32)
    constexpr uint32_t HPG = get_compile_time_arg_val(3);  // heads per group
    constexpr uint32_t G = get_compile_time_arg_val(4);    // groups (this chip's)
    constexpr uint32_t SLOTS = get_compile_time_arg_val(5);
    constexpr auto a_src = TensorAccessorArgs<6>();
    constexpr auto a_dst = TensorAccessorArgs<a_src.next_compile_time_args_offset()>();
    const auto src = TensorAccessor(a_src, get_arg_val<uint32_t>(0));
    const auto dst = TensorAccessor(a_dst, get_arg_val<uint32_t>(1));
    const uint32_t start = get_arg_val<uint32_t>(2), end = get_arg_val<uint32_t>(3);
    constexpr uint32_t CPG = HPG * CPH, NT = N / 32, PER = G * N * CPG;  // dst tiles (rows per prompt)

    constexpr uint32_t cb = 0;
    cb_reserve_back(cb, 1);
    const uint32_t l1 = get_write_ptr(cb);

    for (uint32_t w0 = start; w0 < end; w0 += SLOTS) {
        const uint32_t m = end - w0 < SLOTS ? end - w0 : SLOTS;
        for (uint32_t k = 0; k < m; ++k) {
            const uint32_t w = w0 + k, i = w / PER, t = w % PER;
            const uint32_t c = t % CPG, gn = t / CPG, n = gn % N, g = gn / N;
            const uint32_t h = g * HPG + c / CPH;
            const uint32_t s_tile = ((i * H + h) * NT + n / 32) * CPH + c % CPH;
            for (uint32_t half = 0; half < 2; ++half) {
                noc_async_read(
                    src.get_noc_addr(s_tile, row_offset(n % 32, half)),
                    l1 + (2 * k + half) * FACE_ROW_BYTES,
                    FACE_ROW_BYTES);
            }
        }
        noc_async_read_barrier();
        for (uint32_t k = 0; k < m; ++k) {
            const uint32_t w = w0 + k, i = w / PER, t = w % PER;
            const uint32_t slot = get_arg_val<uint32_t>(4 + i);
            for (uint32_t half = 0; half < 2; ++half) {
                noc_async_write(
                    l1 + (2 * k + half) * FACE_ROW_BYTES, dst.get_noc_addr(t, row_offset(slot, half)), FACE_ROW_BYTES);
            }
        }
        noc_async_write_barrier();
    }
}

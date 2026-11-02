// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Mamba2 decode step, reader (ssm_step_compute.cpp has the math). Users are tile rows throughout:
//   state S  (G N, 32, 512) fp32: tile (g N + n) * 16 + c holds S[b, h, n, p] at row b, column 32 c + j,
//            h = 8 g + c / 2 (8 heads per group), p = 32 (c % 2) + j
//   hc       (1, 32, conv_dim) fp32, the conv output: x (16 c-tiles per group) | B (G N) | C (G N)
//   dt, dec  (1, 32, H) fp32: softplus(dt + dt_bias) and exp(dt A), column h
// This core owns group g, column tiles [c0, c0 + NC) and state rows n in [n0, n1). The per-user scalars a tile
// needs (dt and decay of its head, B and C of its n) are columns of the natural-layout tiles; each is copied along
// its row of a tile the compute side reads elementwise: dt across whole rows, the decay across the left faces, B
// across the left faces and C across the right ones of one tile.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t TILE_BYTES = 4096;  // fp32: four 16x16 faces, row-major within a face

// value at row r, column col of the 32x32 fp32 tile at src
inline uint32_t at(uint32_t src, uint32_t r, uint32_t col) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        src)[((r / 16) * 2 + col / 16) * 256 + (r % 16) * 16 + col % 16];
}

// every row r of the left (half 0) or right (half 1) faces of the tile at dst <- column col of the tile at src
inline void spread(uint32_t src, uint32_t col, uint32_t dst, uint32_t half) {
    auto* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    for (uint32_t r = 0; r < 32; ++r) {
        const uint32_t v = at(src, r, col);
        auto* p = d + ((r / 16) * 2 + half) * 256 + (r % 16) * 16;
        for (uint32_t j = 0; j < 16; ++j) {
            p[j] = v;
        }
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t NC = get_compile_time_arg_val(0);
    constexpr uint32_t N = get_compile_time_arg_val(1);    // ssm_state_size
    constexpr uint32_t XT = get_compile_time_arg_val(2);   // x tiles in hc (= the B offset in tiles)
    constexpr uint32_t BCT = get_compile_time_arg_val(3);  // B tiles in hc (= C offset - B offset)
    constexpr uint32_t HPG = get_compile_time_arg_val(4);  // heads per group
    constexpr uint32_t CPH = get_compile_time_arg_val(5);  // column tiles per head
    constexpr auto a_s = TensorAccessorArgs<6>();
    constexpr auto a_hc = TensorAccessorArgs<a_s.next_compile_time_args_offset()>();
    constexpr auto a_dt = TensorAccessorArgs<a_hc.next_compile_time_args_offset()>();
    constexpr auto a_dec = TensorAccessorArgs<a_dt.next_compile_time_args_offset()>();
    const auto S = TensorAccessor(a_s, get_arg_val<uint32_t>(0));
    const auto hc = TensorAccessor(a_hc, get_arg_val<uint32_t>(1));
    const auto dt = TensorAccessor(a_dt, get_arg_val<uint32_t>(2));
    const auto dec = TensorAccessor(a_dec, get_arg_val<uint32_t>(3));
    const uint32_t g = get_arg_val<uint32_t>(4), c0 = get_arg_val<uint32_t>(5);
    const uint32_t n0 = get_arg_val<uint32_t>(6), n1 = get_arg_val<uint32_t>(7);
    constexpr uint32_t CPG = HPG * CPH;  // column tiles per group

    constexpr uint32_t cb_s = 0, cb_dec = 1, cb_x = 2, cb_dt = 3, cb_bc = 4, cb_scr = 7;

    // scratch: dt, dec, then B and C tiles covering [n0, n1) (at most two each: n1 - n0 <= 32)
    cb_reserve_back(cb_scr, 6);
    const uint32_t scr = get_write_ptr(cb_scr);
    const uint32_t row0 = g * N + n0, row_last = g * N + n1 - 1;
    const uint32_t bt0 = row0 / 32, nbt = row_last / 32 - bt0 + 1;
    noc_async_read(dt.get_noc_addr(0), scr, TILE_BYTES);
    noc_async_read(dec.get_noc_addr(0), scr + TILE_BYTES, TILE_BYTES);
    for (uint32_t t = 0; t < nbt; ++t) {
        noc_async_read(hc.get_noc_addr(XT + bt0 + t), scr + (2 + t) * TILE_BYTES, TILE_BYTES);
        noc_async_read(hc.get_noc_addr(XT + BCT + bt0 + t), scr + (4 + t) * TILE_BYTES, TILE_BYTES);
    }
    cb_reserve_back(cb_x, NC);
    const uint32_t x_l1 = get_write_ptr(cb_x);
    for (uint32_t i = 0; i < NC; ++i) {
        noc_async_read(hc.get_noc_addr(g * CPG + c0 + i), x_l1 + i * TILE_BYTES, TILE_BYTES);
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, NC);

    cb_reserve_back(cb_dt, NC);
    cb_reserve_back(cb_dec, NC);
    const uint32_t dt_l1 = get_write_ptr(cb_dt), dec_l1 = get_write_ptr(cb_dec);
    for (uint32_t i = 0; i < NC; ++i) {
        const uint32_t h = g * HPG + (c0 + i) / CPH;
        spread(scr, h, dt_l1 + i * TILE_BYTES, 0);
        spread(scr, h, dt_l1 + i * TILE_BYTES, 1);
        spread(scr + TILE_BYTES, h, dec_l1 + i * TILE_BYTES, 0);
    }
    cb_push_back(cb_dt, NC);
    cb_push_back(cb_dec, NC);

    for (uint32_t n = n0; n < n1; ++n) {
        const uint32_t row = g * N + n, t = row / 32 - bt0, col = row % 32;
        cb_reserve_back(cb_s, NC);
        const uint32_t s_l1 = get_write_ptr(cb_s);
        for (uint32_t i = 0; i < NC; ++i) {
            noc_async_read(S.get_noc_addr(row * CPG + c0 + i), s_l1 + i * TILE_BYTES, TILE_BYTES);
        }
        cb_reserve_back(cb_bc, 1);
        const uint32_t bc = get_write_ptr(cb_bc);
        spread(scr + (2 + t) * TILE_BYTES, col, bc, 0);
        spread(scr + (4 + t) * TILE_BYTES, col, bc, 1);
        cb_push_back(cb_bc, 1);
        noc_async_read_barrier();
        cb_push_back(cb_s, NC);
    }
}

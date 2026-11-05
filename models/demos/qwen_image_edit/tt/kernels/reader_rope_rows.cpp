// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of rope_swap.cpp (negate = 0) for a head width D whose half is NOT tile aligned (only a multiple of 8, e.g.
// the vision attention's D = 80): x [B, H, S, Dt * 32] tiled (columns >= D zero), one head row group (the Dt tiles of
// one tile row of one head) at a time. It reads the group's x tiles and their cos / sin table tiles ([1, 1, S, Dt *
// 32]: the sin table carries rotate_half's sign, [-sin | sin]) and the half-swapped tiles x'[r][j] =
// x[r][(j + D / 2) % D] (0 for j >= D) by 32-byte NOC reads of 8-column runs (a run never crosses a 16-column face
// row), all in flight before one barrier. Per output
// tile c the cb holds x, x', cos, sin (slots 4c .. 4c + 3), pushed as one group of 4 * Dt tiles.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

// word offset of element (r, j) in a float32 tile (four 16 x 16 faces, face = 2 * face row + face column)
inline uint32_t tile_word(uint32_t r, uint32_t j) { return (((r >> 4) << 1) + (j >> 4)) * 256 + (r & 15) * 16 + (j & 15); }

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t cos_addr = get_arg_val<uint32_t>(1);
    const uint32_t sin_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_groups = get_arg_val<uint32_t>(3);
    const uint32_t start_group = get_arg_val<uint32_t>(4);

    constexpr uint32_t Dt = get_compile_time_arg_val(0);  // tiles per head row
    constexpr uint32_t St = get_compile_time_arg_val(1);  // tile rows per head
    constexpr uint32_t D = get_compile_time_arg_val(2);   // head width (columns), D / 2 a multiple of 8
    constexpr auto x_args = TensorAccessorArgs<3>();
    constexpr auto cos_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr uint32_t half = D / 2;

    constexpr uint32_t cb_id_in = 0;
    const uint32_t page = get_local_cb_interface(cb_id_in).fifo_page_size;
    const auto x = TensorAccessor(x_args, x_addr);
    const auto cs = TensorAccessor(cos_args, cos_addr);
    const auto sn = TensorAccessor(sin_args, sin_addr);

    Noc noc;
    DataflowBuffer dfb(cb_id_in);
    for (uint32_t g = start_group; g < start_group + num_groups; ++g) {
        const uint32_t tab = (g % St) * Dt;  // the group's tile row of the tables
        dfb.reserve_back(4 * Dt);
        const uint32_t base = dfb.get_write_ptr();
        for (uint32_t c = 0; c < Dt; ++c) {
            noc.async_read(x, dfb, page, {.page_id = g * Dt + c}, {.offset_bytes = 4 * c * page});
            noc.async_read(cs, dfb, page, {.page_id = tab + c}, {.offset_bytes = (4 * c + 2) * page});
            noc.async_read(sn, dfb, page, {.page_id = tab + c}, {.offset_bytes = (4 * c + 3) * page});
        }
        // x' straight from x's pages by 8-column (32-byte) runs, zero past D (each page's NOC address once)
        uint64_t src_page[Dt];
        for (uint32_t c = 0; c < Dt; ++c) {
            src_page[c] = x.get_noc_addr(g * Dt + c);
        }
        for (uint32_t c = 0; c < Dt; ++c) {
            for (uint32_t j0 = 0; j0 < 32; j0 += 8) {
                const uint32_t J = 32 * c + j0;
                if (J < D) {
                    const uint32_t S = (J + half) % D;
                    const uint64_t src = src_page[S / 32];
                    const uint32_t dst = base + (4 * c + 1) * page;
                    for (uint32_t r = 0; r < 32; ++r) {
                        noc_async_read(src + 4 * tile_word(r, S % 32), dst + 4 * tile_word(r, j0), 32);
                    }
                } else {
                    volatile tt_l1_ptr uint32_t* dst =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + (4 * c + 1) * page);
                    for (uint32_t r = 0; r < 32; ++r) {
                        const uint32_t d = tile_word(r, j0);
                        for (uint32_t w = 0; w < 8; ++w) {
                            dst[d + w] = 0;
                        }
                    }
                }
            }
        }
        noc.async_read_barrier();
        noc_async_read_barrier();
        dfb.push_back(4 * Dt);
    }
}

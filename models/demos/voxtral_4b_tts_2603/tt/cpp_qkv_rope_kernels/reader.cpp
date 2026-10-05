// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The input stream of tt/cpp_qkv_rope.py. Feeds the STOCK RoPE compute kernel (rotary_embedding.cpp, the multi-
// tile path) exactly what the stock reader feeds it -- per head tile row, for each column tile j: the rotated
// tile (j + HALF) % WT, the sin tile, the input tile j, the cos tile -- but straight out of the fused qkv
// projection `[1, 1, S, (NQ + 2 NKV) * D]` (head hh of tile row r is tiles r * ROWW + hh * WT ..), so the head
// split and its slices are gone. v tile rows are passed through to the writer untouched.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t cos_addr = get_arg_val<uint32_t>(1);
    const uint32_t sin_addr = get_arg_val<uint32_t>(2);
    const uint32_t u0 = get_arg_val<uint32_t>(3);  // first (row, q / k head) unit
    const uint32_t nqk = get_arg_val<uint32_t>(4);
    const uint32_t w0 = get_arg_val<uint32_t>(5);  // first (row, v head) unit
    const uint32_t nv = get_arg_val<uint32_t>(6);

    constexpr uint32_t WT = get_compile_time_arg_val(0);
    constexpr uint32_t HALF = get_compile_time_arg_val(1);
    constexpr uint32_t NQ = get_compile_time_arg_val(2);
    constexpr uint32_t NKV = get_compile_time_arg_val(3);
    constexpr uint32_t ROWW = get_compile_time_arg_val(4);  // tiles a qkv tile row
    constexpr uint32_t SCALAR = get_compile_time_arg_val(5);
    constexpr auto ax = TensorAccessorArgs<6>();
    constexpr auto ac = TensorAccessorArgs<ax.next_compile_time_args_offset()>();
    constexpr auto as = TensorAccessorArgs<ac.next_compile_time_args_offset()>();
    const auto sx = TensorAccessor(ax, x_addr);
    const auto sc = TensorAccessor(ac, cos_addr);
    const auto ss = TensorAccessor(as, sin_addr);

    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_rot = 1;
    constexpr uint32_t cb_cos = 2;
    constexpr uint32_t cb_sin = 3;
    constexpr uint32_t cb_scalar = 4;
    constexpr uint32_t cb_v = 17;
    const uint32_t x_bytes = get_tile_size(cb_in);
    const uint32_t c_bytes = get_tile_size(cb_cos);
    const uint32_t s_bytes = get_tile_size(cb_sin);

    // The scalar -1 tile, as the stock reader writes it (only datum 0 is read by the scalar broadcast).
    cb_reserve_back(cb_scalar, 1);
    volatile tt_l1_ptr uint16_t* scalar = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_scalar));
    scalar[0] = static_cast<uint16_t>(SCALAR);
    cb_push_back(cb_scalar, 1);

    uint32_t row = 0xFFFFFFFF;
    for (uint32_t u = u0; u < u0 + nqk; ++u) {
        const uint32_t r = u / (NQ + NKV);
        const uint32_t hh = u % (NQ + NKV);  // q heads, then k heads: the qkv's own head order
        const uint32_t base = r * ROWW + hh * WT;
        if (r != row) {
            // cos / sin once per tile row (the compute keeps them until the row changes)
            row = r;
            cb_reserve_back(cb_sin, WT);
            cb_reserve_back(cb_cos, WT);
            const uint32_t ls = get_write_ptr(cb_sin);
            const uint32_t lc = get_write_ptr(cb_cos);
            for (uint32_t j = 0; j < WT; ++j) {
                noc_async_read_page(r * WT + j, ss, ls + j * s_bytes);
                noc_async_read_page(r * WT + j, sc, lc + j * c_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb_sin, WT);
            cb_push_back(cb_cos, WT);
        }
        cb_reserve_back(cb_rot, WT);
        cb_reserve_back(cb_in, WT);
        const uint32_t lr = get_write_ptr(cb_rot);
        const uint32_t li = get_write_ptr(cb_in);
        for (uint32_t j = 0; j < WT; ++j) {
            noc_async_read_page(base + (j + HALF) % WT, sx, lr + j * x_bytes);
            noc_async_read_page(base + j, sx, li + j * x_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_rot, WT);
        cb_push_back(cb_in, WT);
    }
    for (uint32_t w = w0; w < w0 + nv; ++w) {
        const uint32_t r = w / NKV;
        const uint32_t base = r * ROWW + (NQ + NKV + w % NKV) * WT;
        cb_reserve_back(cb_v, WT);
        const uint32_t lv = get_write_ptr(cb_v);
        for (uint32_t j = 0; j < WT; ++j) {
            noc_async_read_page(base + j, sx, lv + j * x_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_v, WT);
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The guarded tail's reader with its tolerance formed from the two norms instead of an output-sized tol tensor:
// ex, dn, nn, lo (same-shape float32 tiles) into c_0, c_1, c_2, c_4, tile by tile; the slack-scaled row norms
// a [.., M, 1] into c_3 as a column-broadcast tile, one per tile row (pushed when the output's tile row starts);
// the column norms b [.., 1, N] into c_5 as a row-broadcast tile, one per output tile, only its first row read
// from DRAM (two 64 B face rows, not a 4 KB tile).
// Output tile i = (batch, r, c) with i = batch * Mt * Nt + r * Nt + c; a's page batch * a_bstride + r, b's
// page batch * b_bstride + c (a stride 0 broadcasts that norm over the batch).
// Compile-time: ex, dn, nn, lo, a, b accessor args. Runtime: addr ex, dn, nn, lo, a, b, count, start, Mt, Nt,
// a_bstride, b_bstride.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

// The first column of a float32 tile (faces 0 and 2) across every column of its row (as binary_ng's reader).
FORCE_INLINE void fill_first_column(uint32_t l1) {
    auto* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    for (uint32_t k = 0, fo = 0; k < 2; ++k, fo += 512) {
        for (uint32_t row = 0, ro = 0; row < 16; ++row, ro += 16) {
            const uint32_t l = fo + ro, r = l + 256;
            const uint32_t v = p[l];
            for (uint32_t col = 0; col < 16; ++col) {
                p[l + col] = v;
                p[r + col] = v;
            }
        }
    }
}

// The first row of a float32 tile (row 0 of faces 0 and 1) down every row (as binary_ng's reader).
FORCE_INLINE void fill_first_row(uint32_t l1) {
    constexpr uint32_t kFace = 1024, kRow = 64;
    const uint32_t f0 = l1, f1 = l1 + kFace;
    for (uint32_t n = kRow; n < kFace; n <<= 1) {
        noc_async_read(get_noc_addr(f0), f0 + n, n);
        noc_async_read(get_noc_addr(f1), f1 + n, n);
        noc_async_read_barrier();
    }
    noc_async_read(get_noc_addr(f0), f0 + 2 * kFace, 2 * kFace);
    noc_async_read_barrier();
}

}  // namespace

void kernel_main() {
    const uint32_t addr_ex = get_arg_val<uint32_t>(0);
    const uint32_t addr_dn = get_arg_val<uint32_t>(1);
    const uint32_t addr_nn = get_arg_val<uint32_t>(2);
    const uint32_t addr_lo = get_arg_val<uint32_t>(3);
    const uint32_t addr_a = get_arg_val<uint32_t>(4);
    const uint32_t addr_b = get_arg_val<uint32_t>(5);
    const uint32_t count = get_arg_val<uint32_t>(6);
    const uint32_t start = get_arg_val<uint32_t>(7);
    const uint32_t mt = get_arg_val<uint32_t>(8);
    const uint32_t nt = get_arg_val<uint32_t>(9);
    const uint32_t a_bstride = get_arg_val<uint32_t>(10);
    const uint32_t b_bstride = get_arg_val<uint32_t>(11);

    constexpr auto args_ex = TensorAccessorArgs<0>();
    constexpr auto args_dn = TensorAccessorArgs<args_ex.next_compile_time_args_offset()>();
    constexpr auto args_nn = TensorAccessorArgs<args_dn.next_compile_time_args_offset()>();
    constexpr auto args_lo = TensorAccessorArgs<args_nn.next_compile_time_args_offset()>();
    constexpr auto args_a = TensorAccessorArgs<args_lo.next_compile_time_args_offset()>();
    constexpr auto args_b = TensorAccessorArgs<args_a.next_compile_time_args_offset()>();
    const auto s_ex = TensorAccessor(args_ex, addr_ex);
    const auto s_dn = TensorAccessor(args_dn, addr_dn);
    const auto s_nn = TensorAccessor(args_nn, addr_nn);
    const auto s_lo = TensorAccessor(args_lo, addr_lo);
    const auto s_a = TensorAccessor(args_a, addr_a);
    const auto s_b = TensorAccessor(args_b, addr_b);
    // 0: no negated-input product (the tail takes dn where the guard trips); nn's slot is never read
    constexpr uint32_t has_nn = get_compile_time_arg_val(args_b.next_compile_time_args_offset());

    constexpr uint32_t cb_ex = 0, cb_dn = 1, cb_nn = 2, cb_a = 3, cb_lo = 4, cb_b = 5;
    constexpr uint32_t bytes = 4096;
    const uint32_t per_batch = mt * nt;

    for (uint32_t i = start; i < start + count; ++i) {
        const uint32_t bt = i / per_batch;
        const uint32_t rem = i - bt * per_batch;
        const uint32_t r = rem / nt;
        const uint32_t c = rem - r * nt;
        const bool new_row = (i == start) || (c == 0);

        uint32_t l1_a = 0;
        if (new_row) {
            cb_reserve_back(cb_a, 1);
            l1_a = get_write_ptr(cb_a);
            noc_async_read(s_a.get_noc_addr(bt * a_bstride + r), l1_a, bytes);
        }
        cb_reserve_back(cb_ex, 1);
        noc_async_read(s_ex.get_noc_addr(i), get_write_ptr(cb_ex), bytes);
        cb_reserve_back(cb_dn, 1);
        noc_async_read(s_dn.get_noc_addr(i), get_write_ptr(cb_dn), bytes);
        if constexpr (has_nn) {
            cb_reserve_back(cb_nn, 1);
            noc_async_read(s_nn.get_noc_addr(i), get_write_ptr(cb_nn), bytes);
        }
        cb_reserve_back(cb_lo, 1);
        noc_async_read(s_lo.get_noc_addr(i), get_write_ptr(cb_lo), bytes);
        cb_reserve_back(cb_b, 1);
        const uint32_t l1_b = get_write_ptr(cb_b);
        const uint32_t pb = bt * b_bstride + c;
        noc_async_read(s_b.get_noc_addr(pb, 0), l1_b, 64);        // row 0 of face 0
        noc_async_read(s_b.get_noc_addr(pb, 1024), l1_b + 1024, 64);  // row 0 of face 1
        noc_async_read_barrier();

        if (new_row) {
            fill_first_column(l1_a);
            cb_push_back(cb_a, 1);
        }
        fill_first_row(l1_b);
        cb_push_back(cb_ex, 1);
        cb_push_back(cb_dn, 1);
        if constexpr (has_nn) {
            cb_push_back(cb_nn, 1);
        }
        cb_push_back(cb_lo, 1);
        cb_push_back(cb_b, 1);
    }
}

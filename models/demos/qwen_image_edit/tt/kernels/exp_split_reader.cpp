// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// exp_split's reader: the float32 scores s into c_0 tile by tile, and the row maxima m [.., M, 1] into c_1 as a
// column-broadcast tile, one per output tile row (pushed when a tile row starts, and at the first tile).
// Output tile i = (batch, r, c) with i = batch * Mt * Nt + r * Nt + c; m's page is batch * Mt + r.
// Compile-time: s, m accessor args. Runtime: addr s, m, count, start, Nt.

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

}  // namespace

void kernel_main() {
    const uint32_t addr_s = get_arg_val<uint32_t>(0);
    const uint32_t addr_m = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    const uint32_t start = get_arg_val<uint32_t>(3);
    const uint32_t nt = get_arg_val<uint32_t>(4);

    constexpr auto args_s = TensorAccessorArgs<0>();
    constexpr auto args_m = TensorAccessorArgs<args_s.next_compile_time_args_offset()>();
    const auto s_s = TensorAccessor(args_s, addr_s);
    const auto s_m = TensorAccessor(args_m, addr_m);

    constexpr uint32_t cb_s = 0, cb_m = 1;
    constexpr uint32_t bytes = 4096;

    for (uint32_t i = start; i < start + count; ++i) {
        const uint32_t row = i / nt;  // the global tile row (batch * Mt + r): m's page
        if (i == start || i - row * nt == 0) {
            cb_reserve_back(cb_m, 1);
            const uint32_t l1_m = get_write_ptr(cb_m);
            noc_async_read(s_m.get_noc_addr(row), l1_m, bytes);
            noc_async_read_barrier();
            fill_first_column(l1_m);
            cb_push_back(cb_m, 1);
        }
        cb_reserve_back(cb_s, 1);
        noc_async_read(s_s.get_noc_addr(i), get_write_ptr(cb_s), bytes);
        noc_async_read_barrier();
        cb_push_back(cb_s, 1);
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The weight stream of tt/cpp_swiglu_mm.py: this core's W fused (gate, up) column tiles of every K row of the
// interleaved [K, 2N] weight, one K block (KB rows x W tiles, row-major) at a time -- in minimal_matmul's K
// order for these columns (REV: last block first, as its snake order runs every second N block).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t c0 = get_arg_val<uint32_t>(1);  // first fused column tile
    const uint32_t rev = get_arg_val<uint32_t>(2);  // K blocks last to first

    constexpr uint32_t KB = get_compile_time_arg_val(0);
    constexpr uint32_t NB = get_compile_time_arg_val(1);
    constexpr uint32_t W = get_compile_time_arg_val(2);
    constexpr uint32_t NTF = get_compile_time_arg_val(3);  // fused column tiles a K row
    constexpr auto aw = TensorAccessorArgs<4>();
    const auto sw = TensorAccessor(aw, w_addr);

    constexpr uint32_t cb_w = 1;
    const uint32_t w_bytes = get_tile_size(cb_w);
    for (uint32_t i = 0; i < NB; ++i) {
        const uint32_t b = rev ? NB - 1 - i : i;
        uint32_t row = b * KB * NTF + c0;
        cb_reserve_back(cb_w, KB * W);
        uint32_t l1 = get_write_ptr(cb_w);
        for (uint32_t k = 0; k < KB; ++k) {
            for (uint32_t t = 0; t < W; ++t) {
                noc_async_read_page(row + t, sw, l1);
                l1 += w_bytes;
            }
            row += NTF;
        }
        noc_async_read_barrier();
        cb_push_back(cb_w, KB * W);
    }
}

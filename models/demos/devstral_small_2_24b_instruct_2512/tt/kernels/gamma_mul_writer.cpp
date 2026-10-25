// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer of tt/gamma_mul.py: for each tile row th0 .. th0 + nrows - 1 the row's GW output tiles (columns tw0 ..).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t o_addr = get_arg_val<uint32_t>(0);
    const uint32_t th0 = get_arg_val<uint32_t>(1);
    const uint32_t nrows = get_arg_val<uint32_t>(2);
    const uint32_t tw0 = get_arg_val<uint32_t>(3);

    constexpr uint32_t WT = get_compile_time_arg_val(0);
    constexpr uint32_t GW = get_compile_time_arg_val(1);
    constexpr auto a_o = TensorAccessorArgs<2>();
    const auto s_o = TensorAccessor(a_o, o_addr);
    constexpr uint32_t cb_out = 2;
    const uint32_t page = get_tile_size(cb_out);

    for (uint32_t r = 0; r < nrows; ++r) {
        cb_wait_front(cb_out, GW);
        const uint32_t l1 = get_read_ptr(cb_out);
        const uint32_t p0 = (th0 + r) * WT + tw0;
        for (uint32_t j = 0; j < GW; ++j) {
            noc_async_write(l1 + j * page, s_o.get_noc_addr(p0 + j), page);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_out, GW);
    }
    noc_async_write_barrier();
}

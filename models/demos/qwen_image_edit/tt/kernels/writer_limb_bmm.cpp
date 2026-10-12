// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer of the precise limb batched matmul (limb_bmm.cpp): the single float32 output from cb 16, one tile at
// a time as the stock writer_unary_interleaved_start_id.cpp does, in the compute order of this core's units
// (unit u = (batch bt, rows [r0, r0 + nr)), see reader_limb_bmm.cpp): per column nt, tile (bt * Mt + mt) * Nt + nt
// of each row mt.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_unit = get_arg_val<uint32_t>(1);
    const uint32_t num_units = get_arg_val<uint32_t>(2);

    constexpr uint32_t Mt = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);
    constexpr uint32_t blocks = get_compile_time_arg_val(2);
    constexpr uint32_t R = get_compile_time_arg_val(3);
    constexpr auto dst_args = TensorAccessorArgs<4>();

    constexpr uint32_t cb_out = 16;
    const uint32_t page = get_local_cb_interface(cb_out).fifo_page_size;
    const auto s = TensorAccessor(dst_args, dst_addr);

    Noc noc;
    DataflowBuffer dfb(cb_out);
    for (uint32_t u = start_unit; u < start_unit + num_units; ++u) {
        const uint32_t bt = u / blocks;
        const uint32_t r0 = (u % blocks) * R;
        const uint32_t r1 = r0 + R < Mt ? r0 + R : Mt;
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            for (uint32_t mt = r0; mt < r1; ++mt) {
                dfb.wait_front(1);
                noc.async_write(dfb, s, page, {}, {.page_id = (bt * Mt + mt) * Nt + nt});
                noc.async_writes_flushed();
                dfb.pop_front(1);
            }
        }
    }
    noc.async_write_barrier();
}

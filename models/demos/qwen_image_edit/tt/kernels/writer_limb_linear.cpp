// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer of the two-limb linear (limb_linear.cpp): each mb x nb float32 output block of cb 16 (tile r * nb + c)
// to y[row0 + r, nblk * nb + c], per work unit in the compute kernel's order.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t first_unit = get_arg_val<uint32_t>(1);

    constexpr uint32_t Nt = get_compile_time_arg_val(0);
    constexpr uint32_t mb = get_compile_time_arg_val(1);
    constexpr uint32_t nb = get_compile_time_arg_val(2);
    constexpr uint32_t units = get_compile_time_arg_val(3);
    constexpr uint32_t unit_stride = get_compile_time_arg_val(4);
    constexpr auto y_args = TensorAccessorArgs<5>();

    constexpr uint32_t cb_out = 16;
    const uint32_t page = get_local_cb_interface(cb_out).fifo_page_size;
    const auto y = TensorAccessor(y_args, y_addr);

    Noc noc;
    DataflowBuffer dout(cb_out);
    for (uint32_t i = 0; i < units; ++i) {
        const uint32_t row0 = (first_unit + i * unit_stride) * mb;
        for (uint32_t nblk = 0; nblk < Nt / nb; ++nblk) {
            dout.wait_front(mb * nb);
            for (uint32_t r = 0; r < mb; ++r) {
                for (uint32_t c = 0; c < nb; ++c) {
                    const uint32_t id = (row0 + r) * Nt + nblk * nb + c;
                    noc.async_write(dout, y, page, {.offset_bytes = (r * nb + c) * page}, {.page_id = id});
                }
            }
            noc.async_write_barrier();
            dout.pop_front(mb * nb);
        }
    }
}

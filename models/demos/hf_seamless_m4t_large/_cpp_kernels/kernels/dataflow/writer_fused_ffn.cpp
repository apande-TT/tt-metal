// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// C++ Metalium writer for the fused-FFN kernel. Writes the fc2 output tile
// stream from L1 (cb_out) to DRAM. Adapted from tt_metal/programming_examples/
// matmul/matmul_multi_core/kernels/dataflow/writer_unary_interleaved_start_id.cpp.
//
// The intermediate `hidden` activation is NOT written by this kernel — it
// lives only in L1 (cb_hidden) between the compute kernel's fc1 and fc2
// stages, which is the whole point of the fusion.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = tt::CBIndex::c_17;
    constexpr uint32_t onetile = 1;

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, dst_addr);

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_wait_front(cb_out, onetile);
        uint32_t l1_read_addr = get_read_ptr(cb_out);
        noc_async_write_page(i, out, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_out, onetile);
    }
}

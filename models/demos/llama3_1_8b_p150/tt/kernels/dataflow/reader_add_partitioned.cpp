// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Partitioned eltwise-add reader. Adapted from tt_metal/programming_examples/eltwise_binary/
// kernels/dataflow/read_tiles.cpp, which walks page_id 0..n_tiles on a single core; this variant
// takes a [start_id, start_id + num_tiles) SLICE so the host can spread one add across the whole
// compute grid, matching the partitioning reader_mm_partitioned.cpp already uses in this model.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);
    uint32_t num_tiles = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;
    constexpr uint32_t onetile = 1;

    // Host gives the CBs the same dtype/page size as the tensors, so the accessors can share them.
    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, src0_addr);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, src1_addr);

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_reserve_back(cb_id_in0, onetile);
        cb_reserve_back(cb_id_in1, onetile);

        // Both reads are issued before the barrier so they overlap on the NoC.
        noc_async_read_page(i, a, get_write_ptr(cb_id_in0));
        noc_async_read_page(i, b, get_write_ptr(cb_id_in1));
        noc_async_read_barrier();

        cb_push_back(cb_id_in0, onetile);
        cb_push_back(cb_id_in1, onetile);
    }
}

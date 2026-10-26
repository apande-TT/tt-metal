// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Data movement of the sampler's cross-device top-k (pipeline.py MergeTopK, kernels/topk_merge4.cpp).
//
// READER (default): the gathered values and ids a PAIR of tiles at a time -- the pair's four DRAM reads are in
// flight together under one barrier (the stock reader_create_index_local_topk.cpp issues one tile per barrier,
// 8 serial round trips, and the compute waits for a whole pair before it starts).
// WRITER (-DMERGE_TOPK_WRITER): the top-K values and ids, both writes under one barrier.
// 2-core split (MergeTopK.SPLIT): the reader reads the pair at runtime arg 2; on the root (-DMERGE_ROOT) it then
// waits for the leaf's semaphore (runtime arg 3) and hands the leaf's winner (already in cb 8 / 9) to the compute. The
// leaf's writer (-DMERGE_LEAF_SEND) copies its winner straight into the root's cb 8 / 9 and increments that semaphore.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t addr0 = get_arg_val<uint32_t>(0);  // values (reader: gathered input, writer: output)
    const uint32_t addr1 = get_arg_val<uint32_t>(1);  // ids

    constexpr uint32_t dfb0_id = get_compile_time_arg_val(0);
    constexpr uint32_t dfb1_id = get_compile_time_arg_val(1);
    constexpr uint32_t tiles = get_compile_time_arg_val(2);  // reader: gathered tiles, writer: K tiles (1)

    constexpr auto args0 = TensorAccessorArgs<3>();
    constexpr auto args1 = TensorAccessorArgs<decltype(args0)::next_compile_time_args_offset()>();
    const auto acc0 = TensorAccessor(args0, addr0);
    const auto acc1 = TensorAccessor(args1, addr1);

    const Noc noc;
    DataflowBuffer dfb0(dfb0_id);
    DataflowBuffer dfb1(dfb1_id);
    const uint32_t bytes0 = dfb0.get_entry_size();
    const uint32_t bytes1 = dfb1.get_entry_size();

#if defined(MERGE_LEAF_SEND)
    // addr0 / addr1 carry the root's NOC coordinates; the winner lands at the same cb addresses there
    (void)acc0;
    (void)acc1;
    DataflowBuffer rmt0(8);
    DataflowBuffer rmt1(9);
    const uint64_t root_val = get_noc_addr(addr0, addr1, rmt0.get_write_ptr());
    const uint64_t root_idx = get_noc_addr(addr0, addr1, rmt1.get_write_ptr());
    const uint64_t root_sem = get_noc_addr(addr0, addr1, get_semaphore(get_arg_val<uint32_t>(2)));
    dfb0.wait_front(1);
    dfb1.wait_front(1);
    noc_async_write(dfb0.get_read_ptr(), root_val, bytes0);
    noc_async_write(dfb1.get_read_ptr(), root_idx, bytes1);
    noc_async_write_barrier();
    noc_semaphore_inc(root_sem, 1);
    noc_async_atomic_barrier();
    dfb0.pop_front(1);
    dfb1.pop_front(1);
#elif !defined(MERGE_TOPK_WRITER)
#ifdef MERGE_SPLIT
    const uint32_t t0 = get_arg_val<uint32_t>(2);  // this core's pair
#else
    const uint32_t t0 = 0;
#endif
    for (uint32_t t = t0; t < t0 + tiles; t += 2) {
        dfb0.reserve_back(2);
#ifdef PACKED_IN
        // one tensor: the packed key tiles (acc1 / dfb1 unused)
        for (uint32_t i = 0; i < 2; ++i) {
            noc.async_read(acc0, dfb0, bytes0, {.page_id = t + i}, {.offset_bytes = i * bytes0});
        }
        noc.async_read_barrier();
        dfb0.push_back(2);
        (void)acc1;
        (void)bytes1;
#else
        dfb1.reserve_back(2);
        for (uint32_t i = 0; i < 2; ++i) {
            noc.async_read(acc0, dfb0, bytes0, {.page_id = t + i}, {.offset_bytes = i * bytes0});
            noc.async_read(acc1, dfb1, bytes1, {.page_id = t + i}, {.offset_bytes = i * bytes1});
        }
        noc.async_read_barrier();
        dfb0.push_back(2);
        dfb1.push_back(2);
#endif
    }
#ifdef MERGE_ROOT
    // the leaf's winner: wait until it has landed in cb 8 / 9, then hand it to the compute
    DataflowBuffer rmt0(8);
    DataflowBuffer rmt1(9);
    rmt0.reserve_back(1);
    rmt1.reserve_back(1);
    volatile tt_l1_ptr uint32_t* sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(3)));
    noc_semaphore_wait(sem, 1);
    noc_semaphore_set(sem, 0);
    rmt0.push_back(1);
    rmt1.push_back(1);
#endif
#else
#ifdef WRITE_ONE
    // one output (VocabTopK.PACKED_OUT's packed key tiles): dfb1 / acc1 unused
    (void)acc1;
    (void)bytes1;
    for (uint32_t t = 0; t < tiles; ++t) {
        dfb0.wait_front(1);
        noc.async_write(dfb0, acc0, bytes0, {.offset_bytes = 0}, {.page_id = t});
        noc.async_write_barrier();
        dfb0.pop_front(1);
    }
#else
    for (uint32_t t = 0; t < tiles; ++t) {
        dfb0.wait_front(1);
        dfb1.wait_front(1);
        noc.async_write(dfb0, acc0, bytes0, {.offset_bytes = 0}, {.page_id = t});
        noc.async_write(dfb1, acc1, bytes1, {.offset_bytes = 0}, {.page_id = t});
        noc.async_write_barrier();
        dfb0.pop_front(1);
        dfb1.pop_front(1);
    }
#endif
#endif
}

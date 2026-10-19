// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode projection, weight reader (a few cores per DRAM bank, next to that bank's optimal worker).
// The host lays the weight out so that this bank holds, contiguously from the buffer's bank offset, exactly the
// tiles its workers consume in order, each worker's part contiguous (group by group, K-major, group_w tiles per
// K row) and starting at runtime arg 3. The reader streams them in
// fixed blocks of txns_per_block one-packet transactions, two blocks in flight behind transaction ids (the
// pipeline of deepseek matmul_wo's dm0.cpp), into a CB of cb_blocks blocks: a deep CB lets the stream run ahead
// while compute is still busy with a fused prologue.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

#define NUM_SLOTS 3

void kernel_main() {
    constexpr uint32_t cb_in1 = get_compile_time_arg_val(0);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t txns_per_block = get_compile_time_arg_val(2);
    constexpr uint32_t txn_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t cb_blocks = get_compile_time_arg_val(4);  // blocks the weight CB holds (>= NUM_SLOTS)
    constexpr uint32_t start_sem = get_compile_time_arg_val(5);  // local semaphore to wait on first, or ~0u
    constexpr uint32_t start_count = get_compile_time_arg_val(6);  // ... until it reaches this

    const uint32_t bank_id = get_arg_val<uint32_t>(0);
    const uint32_t vchannel = get_arg_val<uint32_t>(1);
    uint32_t w_offset = get_common_arg_val<uint32_t>(0) + get_arg_val<uint32_t>(3);  // this worker's part
    const uint32_t num_blocks = get_arg_val<uint32_t>(5);

    const uint64_t dram_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, 0);
    if constexpr (start_sem != 0xFFFFFFFF) {
        // A fused prologue's small DRAM reads go first: queued behind this stream they would stall it.
        auto* start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(start_sem));
        noc_semaphore_wait(start, start_count);
        noc_semaphore_set(start, 0);
    }
    const uint32_t cb_base = get_write_ptr(cb_in1);

    constexpr uint32_t block_bytes = txns_per_block * txn_bytes;

    noc_async_read_one_packet_set_state<true>(dram_noc_addr, txn_bytes, vchannel);

    uint32_t trid_to_issue = 1, trid_to_wait = 1, slot = 0;
    bool in_flight = false;
    cb_reserve_back(cb_in1, block_tiles);
    for (uint32_t b = 0; b < num_blocks; ++b) {
        noc_async_read_set_trid(trid_to_issue);
        uint32_t l1 = cb_base + slot * block_bytes;
        for (uint32_t t = 0; t < txns_per_block; ++t) {
            noc_async_read_one_packet_with_state_with_trid<false, true>(dram_noc_addr, w_offset, l1, trid_to_issue);
            w_offset += txn_bytes;
            l1 += txn_bytes;
        }
        if (++slot == cb_blocks) {
            slot = 0;
        }
        if (++trid_to_issue > NUM_SLOTS) {
            trid_to_issue = 1;
        }
        if (in_flight) {
            noc_async_read_barrier_with_trid(trid_to_wait);
            cb_push_back(cb_in1, block_tiles);
            if (++trid_to_wait > NUM_SLOTS) {
                trid_to_wait = 1;
            }
            // reserve_back is not incremental: one block is already reserved for the read in flight.
            cb_reserve_back(cb_in1, 2 * block_tiles);
        }
        in_flight = true;
    }
    noc_async_read_barrier_with_trid(trid_to_wait);
    cb_push_back(cb_in1, block_tiles);
}

#undef NUM_SLOTS

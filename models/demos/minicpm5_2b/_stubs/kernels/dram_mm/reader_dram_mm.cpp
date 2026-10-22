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


void kernel_main() {
    constexpr uint32_t cb_in1 = get_compile_time_arg_val(0);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t txns_per_block = get_compile_time_arg_val(2);
    constexpr uint32_t txn_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t cb_blocks = get_compile_time_arg_val(4);  // blocks the weight CB holds (>= NUM_SLOTS)
    constexpr uint32_t start_sem = get_compile_time_arg_val(5);  // local semaphore to wait on first, or ~0u
    constexpr uint32_t start_count = get_compile_time_arg_val(6);  // ... until it reaches this
    // Relay (a partner core computes this worker's last groups): blocks from relay_from on are not pushed to this
    // core's compute; each one, once landed, bumps the partner's relay semaphore and the partner reads it from here.
    constexpr uint32_t relay_sem = get_compile_time_arg_val(7);  // ~0u: no relay on this program
    constexpr uint32_t depth = get_compile_time_arg_val(8);  // weight blocks in flight (<= cb_blocks, <= 15)

    const uint32_t bank_id = get_arg_val<uint32_t>(0);
    const uint32_t vchannel = get_arg_val<uint32_t>(1);
    uint32_t w_offset = get_common_arg_val<uint32_t>(0) + get_arg_val<uint32_t>(3);  // this worker's part
    const uint32_t num_blocks = get_arg_val<uint32_t>(5);
    uint32_t relay_from = num_blocks;  // == num_blocks: no relay
    uint64_t relay_noc = 0;
    if constexpr (relay_sem != 0xFFFFFFFF) {
        relay_from = get_arg_val<uint32_t>(8);
        relay_noc = get_noc_addr(get_arg_val<uint32_t>(6), get_arg_val<uint32_t>(7), get_semaphore(relay_sem));
    }

    if constexpr (relay_sem != 0xFFFFFFFF) {
        if (num_blocks == 0) {
            // Relay partner: no DRAM stream. The worker (args 6, 7) lands this core's weight blocks in its own CB
            // slots from..from + count - 1 (never reused: its whole part fits its CB) and bumps relay_sem once per
            // landed block; copy every run of landed blocks from the worker's L1 (same CB address on every core of
            // the program) into this CB in one read and push them.
            constexpr uint32_t block_bytes = txns_per_block * txn_bytes;
            const uint32_t src_x = get_arg_val<uint32_t>(6);
            const uint32_t src_y = get_arg_val<uint32_t>(7);
            const uint32_t from = get_arg_val<uint32_t>(8);
            const uint32_t count = get_arg_val<uint32_t>(9);
            auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(relay_sem));
            const uint32_t cb_base = get_write_ptr(cb_in1);
            uint32_t done = 0;
            while (done < count) {
                noc_semaphore_wait_min(sem, done + 1);
                const uint32_t landed = *sem;
                const uint32_t n = (landed < count ? landed : count) - done;
                cb_reserve_back(cb_in1, n * block_tiles);
                noc_async_read(
                    get_noc_addr(src_x, src_y, cb_base + (from + done) * block_bytes),
                    get_write_ptr(cb_in1),
                    n * block_bytes);
                noc_async_read_barrier();
                cb_push_back(cb_in1, n * block_tiles);
                done += n;
            }
            noc_semaphore_set(sem, 0);
            return;
        }
    }
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

    // depth blocks in flight (one transaction id each); a block is pushed (or relayed) once it has landed and the
    // next one is issued only into a free slot (reserve through it: reserve_back is not incremental).
    uint32_t slot = 0, landed = 0;
    auto land = [&]() {
        noc_async_read_barrier_with_trid(1 + landed % depth);
        if (landed < relay_from) {
            cb_push_back(cb_in1, block_tiles);
        } else {
            noc_semaphore_inc(relay_noc, 1);
        }
        ++landed;
    };
    for (uint32_t b = 0; b < num_blocks; ++b) {
        if (b - landed == depth) {
            land();
        }
        // Relayed blocks (num_blocks <= cb_blocks then, host-checked) sit in slots this core's compute never reaches.
        if (b < relay_from) {
            cb_reserve_back(cb_in1, (b - landed + 1) * block_tiles);
        }
        const uint32_t trid = 1 + b % depth;
        noc_async_read_set_trid(trid);
        uint32_t l1 = cb_base + slot * block_bytes;
        for (uint32_t t = 0; t < txns_per_block; ++t) {
            noc_async_read_one_packet_with_state_with_trid<false, true>(dram_noc_addr, w_offset, l1, trid);
            w_offset += txn_bytes;
            l1 += txn_bytes;
        }
        if (++slot == cb_blocks) {
            slot = 0;
        }
    }
    while (landed < num_blocks) {
        land();
    }
    if (relay_from < num_blocks) {
        noc_async_atomic_barrier();
    }
}


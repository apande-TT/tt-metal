// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Data movement of tt/local_cast.py. Input and output are L1-interleaved with the same page count, so page p of both
// sits in bank p % NUM_L1_BANKS at the same slot p / NUM_L1_BANKS: every core casts the pages of its OWN bank, each
// read and write a local (loopback) NOC transfer. Every bank reserves NMAX slots in both buffers, so a bank holding
// fewer pages runs its last slot as a padding tile (allocated, never read as a page). ROLE 0 reads the input slots
// into cb_in in CHUNK-page batches; ROLE 1 writes cb_out's pages back to the output slots.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);  // the input (ROLE 0) or output (ROLE 1) buffer

    constexpr uint32_t ROLE = get_compile_time_arg_val(0);
    constexpr uint32_t CB = get_compile_time_arg_val(1);
    constexpr uint32_t PAGE = get_compile_time_arg_val(2);
    constexpr uint32_t NMAX = get_compile_time_arg_val(3);
    constexpr uint32_t CHUNK = get_compile_time_arg_val(4);
    constexpr auto a_t = TensorAccessorArgs<5>();
    const auto s_t = TensorAccessor(a_t, addr);

    uint64_t base = 0;
    bool found = false;
    for (uint32_t bank = 0; bank < NUM_L1_BANKS && !found; ++bank) {
        base = s_t.get_noc_addr(bank);  // page `bank` is the bank's first slot
        found = noc_address_backend::is_local(base, noc_index);
    }

    for (uint32_t k = 0; k < NMAX; k += CHUNK) {
        const uint32_t n = NMAX - k < CHUNK ? NMAX - k : CHUNK;
        if constexpr (ROLE == 0) {
            cb_reserve_back(CB, n);
            if (found) {
                const uint32_t wp = get_write_ptr(CB);
                for (uint32_t j = 0; j < n; ++j) {
                    noc_async_read(base + (k + j) * PAGE, wp + j * PAGE, PAGE);
                }
                noc_async_read_barrier();
            }
            cb_push_back(CB, n);
        } else {
            cb_wait_front(CB, n);
            if (found) {
                const uint32_t rp = get_read_ptr(CB);
                for (uint32_t j = 0; j < n; ++j) {
                    noc_async_write(rp + j * PAGE, base + (k + j) * PAGE, PAGE);
                }
                noc_async_writes_flushed();
            }
            cb_pop_front(CB, n);
        }
    }
    if constexpr (ROLE == 1) {
        noc_async_write_barrier();
    }
}

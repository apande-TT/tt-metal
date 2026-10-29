// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// x sender of tt/qkv_decode.py (one spare core, both data-movement RISCs): RISC r reads chunks r, r + 2, ... (CH
// tiles each) of the decode activation row out of the input norm's width shards into its own x CB and multicasts each
// to the compute cores' x CBs (the same L1 address on every core of the program) over its own NOC, then multicasts
// the count of its chunks sent so far into their valid semaphore r (the data and the count go out on the same NOC
// and VC, so a core never sees the count before the chunk). The next own chunk is read while that one is in flight.
// Both RISCs first wait until every compute core has reserved its x CB (ready count).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);  // K tiles
    constexpr uint32_t CH = get_compile_time_arg_val(1);  // tiles a chunk
    constexpr uint32_t x_tile = get_compile_time_arg_val(2);
    constexpr uint32_t n_cores = get_compile_time_arg_val(3);  // compute cores (= multicast destinations)
    constexpr uint32_t sem_ready = get_compile_time_arg_val(4);
    constexpr uint32_t sem_valid = get_compile_time_arg_val(5);  // this RISC's count
    constexpr auto x_args = TensorAccessorArgs<6>();
    constexpr uint32_t cb_x = 0, NCH = Kt / CH;
    static_assert(Kt % CH == 0, "qkv x chunking");

    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(0));
    const uint32_t r = get_arg_val<uint32_t>(1);
    // the compute rectangle, NOC 0 order corners (NOC 1 multicasts from the far corner)
    const uint32_t x0 = get_arg_val<uint32_t>(2), y0 = get_arg_val<uint32_t>(3);
    const uint32_t x1 = get_arg_val<uint32_t>(4), y1 = get_arg_val<uint32_t>(5);
    auto mcast = [&](uint32_t l1) {
        return noc_index == 0 ? get_noc_multicast_addr(x0, y0, x1, y1, l1) : get_noc_multicast_addr(x1, y1, x0, y0, l1);
    };

    cb_reserve_back(cb_x, Kt);
    const uint32_t x_l1 = get_write_ptr(cb_x);
    auto read_chunk = [&](uint32_t c) {
        for (uint32_t k = c * CH; k < (c + 1) * CH; ++k) {
            noc_async_read_page(k, x, x_l1 + k * x_tile);
        }
    };
    if (r < NCH) {
        read_chunk(r);
    }
    noc_async_read_barrier();
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_ready)), n_cores);
    const uint32_t valid_l1 = get_semaphore(sem_valid);
    volatile tt_l1_ptr uint32_t* count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_l1);
    uint32_t sent = 0;
    for (uint32_t c = r; c < NCH; c += 2) {
        const uint32_t off = c * CH * x_tile;
        noc_async_write_multicast(x_l1 + off, mcast(x_l1 + off), CH * x_tile, n_cores);
        *count = ++sent;
        noc_semaphore_set_multicast(valid_l1, mcast(valid_l1), n_cores);
        if (c + 2 < NCH) {
            read_chunk(c + 2);
            noc_async_read_barrier();
        }
        noc_async_writes_flushed();  // the count word has been read out before it changes
    }
    noc_async_write_barrier();
}

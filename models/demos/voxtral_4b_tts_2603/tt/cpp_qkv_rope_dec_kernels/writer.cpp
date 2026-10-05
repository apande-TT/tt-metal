// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The output stream of tt/cpp_qkv_rope_dec.py: user b's RoPE'd q tile row to the interleaved q `[B, 1, NQ, D]`
// (pages b * DT ..), then its RoPE'd k tile row into this core's own shard of the height-sharded k.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);  // this core's k shard (local L1)
    const uint32_t b = get_arg_val<uint32_t>(2);

    constexpr uint32_t DT = get_compile_time_arg_val(0);
    constexpr auto aq = TensorAccessorArgs<1>();
    const auto sq = TensorAccessor(aq, q_addr);

    constexpr uint32_t cb_y = 16;
    constexpr uint32_t bytes = 4096;

    cb_wait_front(cb_y, DT);
    uint32_t l1 = get_read_ptr(cb_y);
    for (uint32_t j = 0; j < DT; ++j) {
        noc_async_write_page(b * DT + j, sq, l1);
        l1 += bytes;
    }
    noc_async_writes_flushed();
    cb_pop_front(cb_y, DT);

    cb_wait_front(cb_y, DT);
    noc_async_write(get_read_ptr(cb_y), get_noc_addr(k_addr), DT * bytes);
    noc_async_write_barrier();
    cb_pop_front(cb_y, DT);
}

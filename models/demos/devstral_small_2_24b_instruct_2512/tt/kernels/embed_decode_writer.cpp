// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer of tt/embed_decode.py: this core's WT fp32 tiles into its own width shard of the decode residual (a sharded
// buffer sits at the same L1 address on every core of its grid).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t shard_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t WT = get_compile_time_arg_val(0);
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(1);

    const uint32_t tile = get_tile_size(CB_OUT);
    cb_wait_front(CB_OUT, WT);
    noc_async_write(get_read_ptr(CB_OUT), get_noc_addr(shard_addr), WT * tile);
    noc_async_write_barrier();
    cb_pop_front(CB_OUT, WT);
}

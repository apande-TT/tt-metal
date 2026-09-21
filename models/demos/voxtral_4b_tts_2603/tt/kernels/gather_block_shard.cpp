// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Reads ONE core's rectangular block of tiles out of an interleaved tensor and lands it in that
// core's L1 shard -- the interleaved-to-sharded reshard, with the block's ROWS split between the
// two data-movement processors so both NoCs pull at once.
//
// The stock reader issues every tile of the block from a single RISCV, which is what leaves the
// op at a small fraction of DRAM bandwidth on a 64-core block shard: the limit is how fast one
// processor can issue reads, not how fast the banks can serve them. Each instance of this kernel
// is given a HALF-OPEN row range of the same block, so the two copies together cover it exactly
// once and neither needs to know about the other -- no semaphore, no shared cursor.
//
// The destination is the output tensor's own L1 shard, addressed directly. A sharded buffer sits
// at the same L1 address on every core, so one runtime arg names it for all of them, and there is
// no circular buffer to reserve or push: nothing on this core consumes the data, the next op reads
// the tensor.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_l1_addr = get_arg_val<uint32_t>(1);
    // Tile id of the block's top-left corner in the interleaved source.
    const uint32_t start_tile_id = get_arg_val<uint32_t>(2);
    const uint32_t row_start = get_arg_val<uint32_t>(3);
    const uint32_t row_count = get_arg_val<uint32_t>(4);

    constexpr uint32_t tile_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t input_width_tiles = get_compile_time_arg_val(2);
    constexpr auto src_args = TensorAccessorArgs<3>();

    const auto s = TensorAccessor(src_args, src_addr, tile_bytes);

    uint32_t tile_id = start_tile_id + row_start * input_width_tiles;
    uint32_t l1_addr = dst_l1_addr + row_start * block_width_tiles * tile_bytes;
    for (uint32_t h = 0; h < row_count; ++h) {
        for (uint32_t w = 0; w < block_width_tiles; ++w) {
            noc_async_read_page(tile_id + w, s, l1_addr);
            l1_addr += tile_bytes;
        }
        tile_id += input_width_tiles;
    }
    noc_async_read_barrier();
}

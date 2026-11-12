// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// SUM_PARTS' reader, blocked: the n slabs of one stacked float32 tensor g = [n, ...] into c_0 .. c_{n-1}, BLK tiles
// per slab per barrier (slab k's tile i is page i + k * stride).
// Compile time: TensorAccessorArgs(g), n, BLK.  Runtime: addr, num_tiles, start_id, stride.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);
    const uint32_t stride = get_arg_val<uint32_t>(3);

    constexpr auto args = TensorAccessorArgs<0>();
    constexpr uint32_t n = get_compile_time_arg_val(args.next_compile_time_args_offset());
    constexpr uint32_t BLK = get_compile_time_arg_val(args.next_compile_time_args_offset() + 1);
    const auto src = TensorAccessor(args, addr);
    const uint32_t bytes = get_local_cb_interface(0).fifo_page_size;

    const uint32_t end = start_id + num_tiles;
    for (uint32_t i = start_id; i < end; i += BLK) {
        const uint32_t blk = (end - i) < BLK ? (end - i) : BLK;
        for (uint32_t k = 0; k < n; ++k) {
            cb_reserve_back(k, blk);
            uint32_t l1 = get_write_ptr(k);
            for (uint32_t t = 0; t < blk; ++t) {
                noc_async_read(src.get_noc_addr(i + t + k * stride), l1, bytes);
                l1 += bytes;
            }
        }
        noc_async_read_barrier();
        for (uint32_t k = 0; k < n; ++k) {
            cb_push_back(k, blk);
        }
    }
}

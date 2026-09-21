// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The attention head MERGE, as a pure TILE PERMUTATION -- the mirror of `split_qkv_heads.cpp`.
//
// `nlp_concat_heads` folds `[B, heads, S, hd]` back to `[B, 1, S, heads*hd]`. When `hd` is a whole
// number of tiles -- 128 here, four of them -- not one element crosses a tile boundary: output
// tile (b, s, h*hd_t + d) IS input tile (b, h, s, d). So the whole op is "copy the tiles to
// different page numbers", and its cost is how many copies are in flight at once.
//
// The roofline tags the library op dispatch-bound on a partial grid, which is the signature of the
// wrong UNIT OF WORK rather than of a shape problem: its stride arithmetic walks (batch, head), so
// at 32 users it has 32 units to spread over 110 cores however long the sequence is. This kernel's
// unit is the OUTPUT TILE, of which there are thousands, so every core gets work and both of a
// core's data-movement processors carry an independent half-open range -- no semaphore, no shared
// cursor. Each processor owns a private scratch region and runs BATCH tiles at a time, so the NoC
// round trip is paid once per batch rather than once per tile.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t start = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t batch_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t num_heads = get_compile_time_arg_val(3);
    constexpr uint32_t seq_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t head_dim_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t out_width_tiles = num_heads * head_dim_tiles;

    constexpr auto src_args = TensorAccessorArgs<6>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    const auto src = TensorAccessor(src_args, src_addr, tile_bytes);
    const auto dst = TensorAccessor(dst_args, dst_addr, tile_bytes);

    const uint32_t scratch = get_write_ptr(cb_id);

    uint32_t done = 0;
    while (done < count) {
        uint32_t run = count - done;
        if (run > batch_tiles) {
            run = batch_tiles;
        }
        for (uint32_t i = 0; i < run; ++i) {
            const uint32_t page = start + done + i;
            const uint32_t w = page % out_width_tiles;
            const uint32_t rest = page / out_width_tiles;
            const uint32_t s = rest % seq_tiles;
            const uint32_t b = rest / seq_tiles;
            const uint32_t h = w / head_dim_tiles;
            const uint32_t d = w - h * head_dim_tiles;
            noc_async_read_page(((b * num_heads + h) * seq_tiles + s) * head_dim_tiles + d, src,
                                scratch + i * tile_bytes);
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < run; ++i) {
            noc_async_write_page(start + done + i, dst, scratch + i * tile_bytes, tile_bytes);
        }
        noc_async_write_barrier();
        done += run;
    }
}

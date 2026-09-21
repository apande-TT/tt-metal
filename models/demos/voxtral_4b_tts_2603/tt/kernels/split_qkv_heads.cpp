// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The prefill QKV head split, as a pure TILE PERMUTATION.
//
// `nlp_create_qkv_heads` turns `[B, 1, S, (nq + 2*nkv)*hd]` into q/k/v of `[B, heads, S, hd]`.
// When `hd` is a whole number of tiles -- 128 here, four of them -- not one element crosses a tile
// boundary: output tile (b, h, s, d) IS input tile (b, s, head_offset + h*hd_t + d). So the whole
// op is "copy 12288 tiles to different page numbers", and its cost is how many copies are in
// flight at once.
//
// The library op parallelises over (batch, sequence tile) -- 32 users x 2 tile rows = 64 units on
// a 110-core grid -- because that is the unit its stride arithmetic is written in. This kernel
// parallelises over OUTPUT TILES instead, which is 12288 units, so every core gets work and both
// of a core's data-movement processors get an independent range. Each processor owns a private
// scratch region and runs BATCH tiles at a time: BATCH reads issued back to back, one barrier,
// BATCH writes, one barrier -- so the NoC round trip is paid once per batch rather than per tile.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// One contiguous run of output tiles for ONE of the three destinations.
//
// `head_base` is where this destination's heads start along the fused tensor's width, in tiles;
// everything else about the mapping is the same for q, k and v.
template <typename SrcAcc, typename DstAcc>
FORCE_INLINE void copy_section(
    const SrcAcc& src,
    const DstAcc& dst,
    uint32_t scratch,
    uint32_t start,
    uint32_t count,
    uint32_t head_base,
    uint32_t heads,
    uint32_t seq_tiles,
    uint32_t head_dim_tiles,
    uint32_t fused_width_tiles,
    uint32_t tile_bytes,
    uint32_t batch_tiles) {
    uint32_t done = 0;
    while (done < count) {
        uint32_t run = count - done;
        if (run > batch_tiles) {
            run = batch_tiles;
        }
        for (uint32_t i = 0; i < run; ++i) {
            const uint32_t page = start + done + i;
            const uint32_t d = page % head_dim_tiles;
            const uint32_t rest = page / head_dim_tiles;
            const uint32_t s = rest % seq_tiles;
            const uint32_t bh = rest / seq_tiles;
            const uint32_t h = bh % heads;
            const uint32_t b = bh / heads;
            const uint32_t src_page = (b * seq_tiles + s) * fused_width_tiles + (head_base + h * head_dim_tiles + d);
            noc_async_read_page(src_page, src, scratch + i * tile_bytes);
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < run; ++i) {
            noc_async_write_page(start + done + i, dst, scratch + i * tile_bytes, tile_bytes);
        }
        noc_async_write_barrier();
        done += run;
    }
}

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t q_addr = get_arg_val<uint32_t>(1);
    const uint32_t k_addr = get_arg_val<uint32_t>(2);
    const uint32_t v_addr = get_arg_val<uint32_t>(3);
    const uint32_t q_start = get_arg_val<uint32_t>(4);
    const uint32_t q_count = get_arg_val<uint32_t>(5);
    const uint32_t k_start = get_arg_val<uint32_t>(6);
    const uint32_t k_count = get_arg_val<uint32_t>(7);
    const uint32_t v_start = get_arg_val<uint32_t>(8);
    const uint32_t v_count = get_arg_val<uint32_t>(9);

    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t batch_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t num_heads = get_compile_time_arg_val(3);
    constexpr uint32_t num_kv_heads = get_compile_time_arg_val(4);
    constexpr uint32_t seq_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t head_dim_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t fused_width_tiles = get_compile_time_arg_val(7);
    constexpr auto src_args = TensorAccessorArgs<8>();
    constexpr auto q_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();

    const auto src = TensorAccessor(src_args, src_addr, tile_bytes);
    const auto q = TensorAccessor(q_args, q_addr, tile_bytes);
    const auto k = TensorAccessor(k_args, k_addr, tile_bytes);
    const auto v = TensorAccessor(v_args, v_addr, tile_bytes);

    const uint32_t scratch = get_write_ptr(cb_id);

    copy_section(
        src, q, scratch, q_start, q_count, 0, num_heads, seq_tiles, head_dim_tiles, fused_width_tiles, tile_bytes,
        batch_tiles);
    copy_section(
        src, k, scratch, k_start, k_count, num_heads * head_dim_tiles, num_kv_heads, seq_tiles, head_dim_tiles,
        fused_width_tiles, tile_bytes, batch_tiles);
    copy_section(
        src, v, scratch, v_start, v_count, (num_heads + num_kv_heads) * head_dim_tiles, num_kv_heads, seq_tiles,
        head_dim_tiles, fused_width_tiles, tile_bytes, batch_tiles);
}

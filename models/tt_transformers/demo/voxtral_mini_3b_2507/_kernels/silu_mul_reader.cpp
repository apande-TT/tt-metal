// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for the SwiGLU product kernel: stream this core's run of gate/up tiles into two circular
// buffers, BLOCK tiles at a time.
//
// The tile stream is FLAT.  gate, up and the product have identical shape, layout and dtype and are
// all interleaved, so a page is a tile and the whole op is "tile i of a, tile i of b, tile i of y"
// for a contiguous run of i.  That is why this kernel needs no notion of the tensor's rank or of
// where the row boundaries fall: the host hands each core (start, n_tiles) and nothing else.
//
// BLOCK RATHER THAN ONE TILE AT A TIME, because that is what the ttl attempt at this op got wrong.
// A per-tile pipeline pays a cb handshake and a read barrier per tile, which measured 33% over
// ttnn's binary_ng on exactly this shape; binary_ng amortises both over a block, so this does too.

#include <cstdint>

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t start = get_arg_val<uint32_t>(2);
    const uint32_t n_tiles = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t block = get_compile_time_arg_val(0);

    constexpr auto a_args = TensorAccessorArgs<1>();
    const auto a = TensorAccessor(a_args, a_addr);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr);

    const uint32_t tile_bytes = get_tile_size(cb_a);

    // The host sizes every core's run as a whole number of blocks, so there is no tail to guard.
    for (uint32_t i = 0; i < n_tiles; i += block) {
        cb_reserve_back(cb_a, block);
        cb_reserve_back(cb_b, block);
        uint32_t wa = get_write_ptr(cb_a);
        uint32_t wb = get_write_ptr(cb_b);
        for (uint32_t j = 0; j < block; ++j) {
            noc_async_read_page(start + i + j, a, wa);
            noc_async_read_page(start + i + j, b, wb);
            wa += tile_bytes;
            wb += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_a, block);
        cb_push_back(cb_b, block);
    }
}

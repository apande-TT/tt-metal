// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of the joint head split: the output [B, H, Lt + It tile rows, Dt] (head-major, the text rows first)
// takes head h's column block (col0 + h * Dt ..) of the text projection [B, Lt rows, Wt] for its first Lt tile
// rows and of the image projection [B, It rows, Wt] for the rest. Each output tile is a plain copy of one
// source tile, streamed in output order into the writer's cb, up to `batch` tiles per barrier.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t txt_addr = get_arg_val<uint32_t>(0);
    const uint32_t img_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);
    const uint32_t start_tile = get_arg_val<uint32_t>(3);
    // shapes as runtime args: every call shares one program binary
    const uint32_t Lt = get_arg_val<uint32_t>(4);    // text tile rows per batch entry
    const uint32_t It = get_arg_val<uint32_t>(5);    // image tile rows per batch entry
    const uint32_t Wt = get_arg_val<uint32_t>(6);    // source width in tiles (both projections)
    const uint32_t H = get_arg_val<uint32_t>(7);     // heads
    const uint32_t Dt = get_arg_val<uint32_t>(8);    // head width in tiles
    const uint32_t col0 = get_arg_val<uint32_t>(9);  // first source column tile of head 0

    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr auto txt_args = TensorAccessorArgs<1>();
    constexpr auto img_args = TensorAccessorArgs<txt_args.next_compile_time_args_offset()>();
    const uint32_t St = Lt + It;

    constexpr uint32_t cb_id = 16;
    const uint32_t page_bytes = get_local_cb_interface(cb_id).fifo_page_size;
    const auto txt = TensorAccessor(txt_args, txt_addr);
    const auto img = TensorAccessor(img_args, img_addr);

    Noc noc;
    DataflowBuffer dfb(cb_id);
    const uint32_t end = start_tile + num_tiles;
    for (uint32_t t0 = start_tile; t0 < end; t0 += batch) {
        const uint32_t n = t0 + batch < end ? batch : end - t0;
        dfb.reserve_back(n);
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t t = t0 + j;
            const uint32_t dc = t % Dt;
            const uint32_t r = t / Dt;
            const uint32_t sr = r % St;
            const uint32_t bh = r / St;
            const uint32_t col = col0 + (bh % H) * Dt + dc;
            const uint32_t b = bh / H;
            if (sr < Lt) {
                noc.async_read(txt, dfb, page_bytes, {.page_id = (b * Lt + sr) * Wt + col}, {.offset_bytes = j * page_bytes});
            } else {
                noc.async_read(
                    img, dfb, page_bytes, {.page_id = (b * It + sr - Lt) * Wt + col}, {.offset_bytes = j * page_bytes});
            }
        }
        noc.async_read_barrier();
        dfb.push_back(n);
    }
}

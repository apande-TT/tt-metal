// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of the head-merging limb split (limb_split.fused_split_merge_heads): the split kernel's float32 input
// tiles in the MERGED layout [B, 1, S, H * D] read straight from the head-major [B, H, S, D] attention output, so
// the per-head slices + concat that would build the merged tensor are never run. Output tile t (St tile rows, H * Dt
// tiles per row) is input tile ((b * H + h) * St + r) * Dt + c % Dt with b = t / (St * H * Dt), r its tile row,
// c its tile column and h = c / Dt -- the tile the slice + concat would have copied there.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_tile = get_arg_val<uint32_t>(2);

    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t St = get_compile_time_arg_val(1);
    constexpr uint32_t Dt = get_compile_time_arg_val(2);
    constexpr auto src_args = TensorAccessorArgs<3>();

    constexpr uint32_t cb_id_in = 0;
    constexpr uint32_t row_tiles = H * Dt;
    const uint32_t page = get_local_cb_interface(cb_id_in).fifo_page_size;
    const auto src = TensorAccessor(src_args, src_addr);

    Noc noc;
    DataflowBuffer dfb(cb_id_in);
    for (uint32_t t = start_tile; t < start_tile + num_tiles; ++t) {
        const uint32_t b = t / (St * row_tiles);
        const uint32_t r = t / row_tiles % St;
        const uint32_t c = t % row_tiles;
        const uint32_t in_tile = ((b * H + c / Dt) * St + r) * Dt + c % Dt;
        dfb.reserve_back(1);
        noc.async_read(src, dfb, page, {.page_id = in_tile}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb.push_back(1);
    }
}

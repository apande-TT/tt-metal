// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of the face-aligned head merge (the inverse of reader_face_heads.cpp): the output [N, 1, St tile rows, Wt]
// takes head h's D columns at h * D from the head-major attention output [N, H, St tile rows, Dto] (D a multiple of
// 16, not necessarily of 32; H * D whole tiles). A block of 16 output columns is one face column of a tile and lies
// in one head, so each output tile is assembled from whole 16 x 16 faces of the source tiles (face f = 2 * face row +
// face column): the permute + reshape [N, S, H, D] -> [N, 1, S, H * D] without its transpose / reshape programs.
// Streamed in output order into the writer's cb, up to `batch` tiles per barrier.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_tile = get_arg_val<uint32_t>(2);
    // shapes as runtime args: every call shares one program binary
    const uint32_t St = get_arg_val<uint32_t>(3);   // tile rows per batch entry
    const uint32_t Wt = get_arg_val<uint32_t>(4);   // output width in tiles (H * D / 32)
    const uint32_t H = get_arg_val<uint32_t>(5);    // heads
    const uint32_t Df = get_arg_val<uint32_t>(6);   // head width in face columns (D / 16)
    const uint32_t Dto = get_arg_val<uint32_t>(7);  // source head width in tiles (ceil(D / 32))

    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr auto src_args = TensorAccessorArgs<1>();

    constexpr uint32_t cb_id = 16;
    const uint32_t page_bytes = get_local_cb_interface(cb_id).fifo_page_size;
    const uint32_t face_bytes = page_bytes / 4;
    const auto src = TensorAccessor(src_args, src_addr);

    Noc noc;
    DataflowBuffer dfb(cb_id);
    const uint32_t end = start_tile + num_tiles;
    for (uint32_t t0 = start_tile; t0 < end; t0 += batch) {
        const uint32_t n = t0 + batch < end ? batch : end - t0;
        dfb.reserve_back(n);
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t t = t0 + j;
            const uint32_t ct = t % Wt;
            const uint32_t r = t / Wt;
            const uint32_t sr = r % St;
            const uint32_t nb = r / St;
            for (uint32_t q = 0; q < 2; ++q) {
                const uint32_t g = 2 * ct + q;  // the output's face column
                const uint32_t h = g / Df;
                const uint32_t fc = g % Df;  // its face column within head h
                const uint32_t page = ((nb * H + h) * St + sr) * Dto + fc / 2;
                for (uint32_t fr = 0; fr < 2; ++fr) {
                    noc.async_read(
                        src,
                        dfb,
                        face_bytes,
                        {.page_id = page, .offset_bytes = (2 * fr + fc % 2) * face_bytes},
                        {.offset_bytes = j * page_bytes + (2 * fr + q) * face_bytes});
                }
            }
        }
        noc.async_read_barrier();
        dfb.push_back(n);
    }
}

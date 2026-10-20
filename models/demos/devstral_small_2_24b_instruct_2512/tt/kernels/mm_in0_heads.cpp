// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// in0 reader of tt/minimal_mm.py when in0 is an attention output in heads layout (B, NH, S, HD) multiplied as its
// head concat (B, 1, S, NH * HD): the stock dm_in0_sender.cpp with only its block read remapped. Tile (i, j) of the
// (B * ST) x (NH * HDT) in0 view is tile ((b * NH + h) * ST + s) * HDT + d of the heads tensor, with b = i / ST,
// s = i % ST, h = j / HDT, d = j % HDT -- the tile nlp_concat_heads would have copied there. IN0_NH / IN0_ST / IN0_HDT:
// heads, sequence tiles and head-dim tiles.

#include "ttnn/operations/experimental/minimal_matmul/device/kernels/matmul_dataflow_common.hpp"

template <uint32_t M_block_tiles, uint32_t K_block_tiles, typename TensorAccessorType>
void read_in0_heads_block_sync(
    const TensorAccessorType& tensor_accessor,
    const TensorShape2D& shape,
    uint32_t cb_id,
    uint32_t tile_size_bytes,
    uint32_t d0_start,
    uint32_t d0_end,
    uint32_t d1_start,
    uint32_t d1_end) {
    Noc noc;
    CircularBuffer cb(cb_id);
    const uint32_t cb_base_write_ptr = cb.get_write_ptr();
    uint32_t write_ptr = cb_base_write_ptr;
    for (uint32_t i = d0_start; i < d0_end; i++) {
        if (i >= shape.logical_d0) {
            break;
        }
        const uint32_t row_base = (i / IN0_ST) * IN0_NH * IN0_ST + (i % IN0_ST);  // (b * NH) * ST + s
        for (uint32_t j = d1_start; j < d1_end; j++) {
            if (j < shape.logical_d1) {
                const uint32_t tile_id = (row_base + (j / IN0_HDT) * IN0_ST) * IN0_HDT + (j % IN0_HDT);
                noc.async_read(
                    tensor_accessor, CoreLocalMem<uint32_t>(write_ptr), tile_size_bytes, {.page_id = tile_id}, {});
            } else {
                fill_zeros_async(noc, cb, tile_size_bytes, write_ptr - cb_base_write_ptr);
            }
            write_ptr += tile_size_bytes;
        }
        write_ptr += (K_block_tiles - (d1_end - d1_start)) * tile_size_bytes;
    }
    noc.async_read_barrier();
    noc.write_zeros_l1_barrier();
}

#define read_in0_block_sync read_in0_heads_block_sync
#include "ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in0_sender.cpp"

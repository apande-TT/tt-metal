// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// tt/minimal_mm.py: a copy of minimal_matmul's matmul_dataflow_common.hpp (included by kernels/mm_dm_in0.cpp and
// mm_dm_in1_out.cpp) with two tile-index remaps for matmul(..., swapped=True), off unless defined:
//   IN1_TRANSPOSED: in1 tile (k, n) is read from tile (n, k) of a stored (N x K) tensor (the compute unpacks it
//                   transposed), so in1 = x^T without a transpose op;
//   OUT_TRANSPOSED: output block tile (m, n) is written to tile (n, m) (the compute packs each tile transposed).
// and one output remap for matmul(..., heads_out=) (the prefill QKV written in nlp_create_qkv_heads' layouts):
//   OUT_HEADS: the N_chunks = 3 outputs are q (OUT_HEADS_NQ heads), k and v (OUT_HEADS_NKV heads each), each
//              (B, H, S, HD); output tile (m, n) with m = b * OUT_HEADS_ST + st and n = h * OUT_HEADS_HDT + d (h over
//              the q, then k, then v heads) goes to page ((b * H + h') * OUT_HEADS_ST + st) * OUT_HEADS_HDT + d of
//              its tensor -- where nlp_create_qkv_heads would have copied it.

#pragma once

#include <cstdint>
#include <algorithm>
#include <tuple>
#include <utility>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

// in0's K is read from TWO source buffers via the in3 path: either the AG local pre-gather slice
// (READ_FROM_LOCAL_INPUT) or a plain fused concatenation of two operands (IN0_VIRTUAL_CONCAT).
// IN0_HAS_SECOND_SOURCE gates the shared read plumbing; the two cases differ only in how the
// local K-range / widths are derived (see dm_in0_sender.cpp).
#if defined(READ_FROM_LOCAL_INPUT) || defined(IN0_VIRTUAL_CONCAT)
#define IN0_HAS_SECOND_SOURCE 1
#endif

namespace detail {
template <typename... Args, uint32_t... Indexes>
auto make_tensor_accessor_tuple_impl(
    const std::tuple<Args...>& args_tuple,
    uint32_t address_rt_arg_index_start,
    uint32_t page_size,
    std::integer_sequence<uint32_t, Indexes...>) {
    // Third argument page_size from runtime args overrides TensorAccessorArgs::AlignedPageSize, which may be stale on
    // program cache hits.
    return std::make_tuple(TensorAccessor(
        std::get<Indexes>(args_tuple), get_arg_val<uint32_t>(address_rt_arg_index_start + Indexes), page_size)...);
}
}  // namespace detail

/**
 * Create a tuple of TensorAccessors from a tuple of TensorAccessorArgs.
 * Each tensor gets its address from consecutive RT args starting at address_rt_arg_index_start.
 */
template <typename... Args>
auto make_tensor_accessor_tuple_uniform_page_size(
    const std::tuple<Args...>& args_tuple, uint32_t address_rt_arg_index_start, uint32_t page_size) {
    return detail::make_tensor_accessor_tuple_impl(
        args_tuple, address_rt_arg_index_start, page_size, std::make_integer_sequence<uint32_t, sizeof...(Args)>());
}
inline void fill_zeros_async(const Noc& noc, const CircularBuffer& cb, uint32_t bytes, uint32_t offset_bytes = 0) {
    noc.async_write_zeros(cb, bytes, {.offset_bytes = offset_bytes});
}

struct TensorShape2D {
    uint32_t logical_d0;
    uint32_t logical_d1;
    uint32_t padded_d0;
    uint32_t padded_d1;
    // Constructor to initialize with 2D shape
    TensorShape2D(uint32_t _d0, uint32_t _d1, uint32_t _padded_d0, uint32_t _padded_d1) :
        logical_d0(_d0), logical_d1(_d1), padded_d0(_padded_d0), padded_d1(_padded_d1) {
        ASSERT(_d0 > 0);
        ASSERT(_d1 > 0);
        ASSERT(_d0 <= _padded_d0);
        ASSERT(_d1 <= _padded_d1);
    }
};

/**
 * Read a block of in0 from a potentially padded tensor.
 * Since this is for matmul, no need to read when M >= logical_M
 * Otherwise, if K >= logical_K, fill with zeros.
 */
template <
    uint32_t M_block_tiles,
    uint32_t K_block_tiles,
    typename TensorAccessorType
#ifdef IN0_HAS_SECOND_SOURCE
    ,
    typename LocalTensorAccessorType
#endif
    >
void read_in0_block_sync(
    const TensorAccessorType& tensor_accessor,
    const TensorShape2D& shape,
    uint32_t cb_id,
    uint32_t tile_size_bytes,
#ifdef IN0_HAS_SECOND_SOURCE
    const LocalTensorAccessorType& in3_accessor,
    uint32_t local_k_start,
    uint32_t local_k_end,
    uint32_t input_tensor_Wt,
    // Row width (in tiles) of the MAIN (tensor_accessor) buffer. AG path: full logical K (in0 is the
    // full gathered tensor). Two-input K-split: the main tensor's own (narrow) width so its tile_id
    // indexes the narrow buffer correctly.
    uint32_t main_Wt,
#endif
    uint32_t d0_start,
    uint32_t d0_end,
    uint32_t d1_start,
    uint32_t d1_end) {
    ASSERT(d0_end > d0_start);
    ASSERT(d1_end > d1_start);

    Noc noc;
    CircularBuffer cb(cb_id);
    const uint32_t cb_base_write_ptr = cb.get_write_ptr();
    uint32_t write_ptr = cb_base_write_ptr;
    for (uint32_t i = d0_start; i < d0_end; i++) {
        if (i >= shape.logical_d0) {
            break;
        }
        for (uint32_t j = d1_start; j < d1_end; j++) {
            if (j < shape.logical_d1) {
#ifdef IN0_HAS_SECOND_SOURCE
                if (local_k_start <= j && j <= local_k_end) {
                    // read from self_tensor_accessor
                    uint32_t tile_id = i * input_tensor_Wt + (j - local_k_start);
                    noc.async_read(
                        in3_accessor, CoreLocalMem<uint32_t>(write_ptr), tile_size_bytes, {.page_id = tile_id}, {});
                } else {
                    // Non-local K-tile: read from the main buffer. main_Wt generalizes the row
                    // stride so the main tensor can be either the full gathered K (AG) or a narrow
                    // K-split half (two-input). j is the logical K index; main covers a prefix so no
                    // offset is needed (local range is the suffix for the two-input case).
                    uint32_t tile_id = i * main_Wt + j;
                    noc.async_read(
                        tensor_accessor, CoreLocalMem<uint32_t>(write_ptr), tile_size_bytes, {.page_id = tile_id}, {});
                }
#else
                {
                    uint32_t tile_id = i * shape.logical_d1 + j;
                    noc.async_read(
                        tensor_accessor, CoreLocalMem<uint32_t>(write_ptr), tile_size_bytes, {.page_id = tile_id}, {});
                }
#endif
            } else {
                fill_zeros_async(noc, cb, tile_size_bytes, write_ptr - cb_base_write_ptr);
            }
            write_ptr += tile_size_bytes;
        }
        // finish up incrementing write_ptr if (d1_end - d1_start) < K_block_tiles
        write_ptr += (K_block_tiles - (d1_end - d1_start)) * tile_size_bytes;
    }
    noc.async_read_barrier();
    noc.write_zeros_l1_barrier();
}

/**
 * Read a block of in1 from a potentially padded tensor.
 * Since this is for matmul, no need to read when N >= logical_N
 * Otherwise, if K >= logical_K, fill with zeros.
 */
template <uint32_t K_block_tiles, uint32_t N_block_tiles, typename TensorAccessorType>
void read_in1_block_sync(
    const TensorAccessorType& tensor_accessor,
    const TensorShape2D& shape,
    uint32_t cb_id,
    uint32_t tile_size_bytes,
    uint32_t d0_start,
    uint32_t d0_end,
    uint32_t d1_start,
    uint32_t d1_end) {
    ASSERT(d0_end > d0_start);
    ASSERT(d1_end > d1_start);
    Noc noc;
    CircularBuffer cb(cb_id);
    const uint32_t cb_base_write_ptr = cb.get_write_ptr();
    uint32_t write_ptr = cb_base_write_ptr;
    for (uint32_t i = d0_start; i < d0_end; i++) {
        for (uint32_t j = d1_start; j < d1_end; j++) {
            if (j >= shape.logical_d1) {
                write_ptr += tile_size_bytes;
                continue;
            }
            if (i < shape.logical_d0) {
#ifdef IN1_TRANSPOSED
                uint32_t tile_id = j * shape.logical_d0 + i;  // in1 (k, n) is tile (n, k) of the stored (N x K) tensor
#else
                uint32_t tile_id = i * shape.logical_d1 + j;
#endif
                noc.async_read(
                    tensor_accessor, CoreLocalMem<uint32_t>(write_ptr), tile_size_bytes, {.page_id = tile_id}, {});
            } else {
                fill_zeros_async(noc, cb, tile_size_bytes, write_ptr - cb_base_write_ptr);
            }
            write_ptr += tile_size_bytes;
        }
        // finish up incrementing write_ptr if (d1_end - d1_start) < K_block_tiles
        write_ptr += (N_block_tiles - (d1_end - d1_start)) * tile_size_bytes;
    }
    noc.async_read_barrier();
    noc.write_zeros_l1_barrier();
}

#ifdef CHUNKED_FWD
// CHUNKED_FWD (tt/minimal_mm.py CHUNKED_FWD): the injector reads a block a few rows at a time and every core of the
// forwarding chain passes each row chunk on as soon as it lands, instead of storing and forwarding the whole block --
// the same tiles in the same CB slots, so the compute is unchanged. These issue the reads of rows [r0, r1) of a block
// (into the CB slots read_in0_block_sync / read_in1_block_sync would put them in) tagged with NOC transaction id trid,
// so the injector issues a whole block at once and waits for it chunk by chunk (wait_rows).

// in0 block rows [r0, r1) (tile rows m0 + r0 .. of the M x K view; K_block_tiles tiles a row). IN0_NH: in0 is a heads-
// layout (B, NH, S, HD) attention output read as its head concat (tile (i, j) at ((b * NH + h) * ST + s) * HDT + d).
template <uint32_t K_block_tiles, typename TensorAccessorType>
void issue_in0_rows(
    const TensorAccessorType& tensor_accessor,
    const TensorShape2D& shape,
    uint32_t cb_id,
    uint32_t tile_size_bytes,
    uint32_t m0,
    uint32_t r0,
    uint32_t r1,
    uint32_t d1_start,
    uint32_t d1_end,
    uint32_t trid) {
    Noc noc;
    CircularBuffer cb(cb_id);
    const uint32_t cb_base_write_ptr = cb.get_write_ptr();
    uint32_t write_ptr = cb_base_write_ptr + r0 * K_block_tiles * tile_size_bytes;
    for (uint32_t i = m0 + r0; i < m0 + r1; i++) {
        if (i >= shape.logical_d0) {
            break;
        }
#ifdef IN0_NH
        const uint32_t row_base = (i / IN0_ST) * IN0_NH * IN0_ST + (i % IN0_ST);  // (b * NH) * ST + s
#endif
        for (uint32_t j = d1_start; j < d1_end; j++) {
            if (j < shape.logical_d1) {
#ifdef IN0_NH
                const uint32_t tile_id = (row_base + (j / IN0_HDT) * IN0_ST) * IN0_HDT + (j % IN0_HDT);
#else
                const uint32_t tile_id = i * shape.logical_d1 + j;
#endif
                noc.async_read<NocOptions::TXN_ID>(
                    tensor_accessor,
                    CoreLocalMem<uint32_t>(write_ptr),
                    tile_size_bytes,
                    {.page_id = tile_id},
                    {},
                    {.trid = trid});
            } else {
                fill_zeros_async(noc, cb, tile_size_bytes, write_ptr - cb_base_write_ptr);
            }
            write_ptr += tile_size_bytes;
        }
        write_ptr += (K_block_tiles - (d1_end - d1_start)) * tile_size_bytes;
    }
}

// in1 block rows [r0, r1) (K tiles k0 + r0 ..; N_block_tiles tiles a row)
template <uint32_t N_block_tiles, typename TensorAccessorType>
void issue_in1_rows(
    const TensorAccessorType& tensor_accessor,
    const TensorShape2D& shape,
    uint32_t cb_id,
    uint32_t tile_size_bytes,
    uint32_t k0,
    uint32_t r0,
    uint32_t r1,
    uint32_t d1_start,
    uint32_t d1_end,
    uint32_t trid) {
    Noc noc;
    CircularBuffer cb(cb_id);
    const uint32_t cb_base_write_ptr = cb.get_write_ptr();
    uint32_t write_ptr = cb_base_write_ptr + r0 * N_block_tiles * tile_size_bytes;
    for (uint32_t i = k0 + r0; i < k0 + r1; i++) {
        for (uint32_t j = d1_start; j < d1_end; j++) {
            if (j >= shape.logical_d1) {
                write_ptr += tile_size_bytes;
                continue;
            }
            if (i < shape.logical_d0) {
#ifdef IN1_TRANSPOSED
                uint32_t tile_id = j * shape.logical_d0 + i;
#else
                uint32_t tile_id = i * shape.logical_d1 + j;
#endif
                noc.async_read<NocOptions::TXN_ID>(
                    tensor_accessor,
                    CoreLocalMem<uint32_t>(write_ptr),
                    tile_size_bytes,
                    {.page_id = tile_id},
                    {},
                    {.trid = trid});
            } else {
                fill_zeros_async(noc, cb, tile_size_bytes, write_ptr - cb_base_write_ptr);
            }
            write_ptr += tile_size_bytes;
        }
        write_ptr += (N_block_tiles - (d1_end - d1_start)) * tile_size_bytes;
    }
}

// The chunk-tagged reads above: wait for chunk `trid`'s tiles (and any zero fills) before it is forwarded / pushed.
inline void wait_rows(uint32_t trid) {
    Noc noc;
    noc.async_read_barrier<NocOptions::TXN_ID>({.trid = trid});
    noc.write_zeros_l1_barrier();
}
#endif  // CHUNKED_FWD

/**
 * Write a block of output to a potentially padded tensor.
 * Skip writing when M >= logical_M or N >= logical_N
 */
template <uint32_t M_block_tiles, uint32_t N_block_tiles, typename TensorAccessorType>
void write_block_sync(
    const TensorAccessorType& tensor_accessor,
    const TensorShape2D& shape,
    uint32_t read_ptr,
    uint32_t tile_size_bytes,
    uint32_t d0_start,
    uint32_t d0_end,
    uint32_t d1_start,
    uint32_t d1_end) {
    ASSERT(d0_end > d0_start);
    ASSERT(d1_end > d1_start);

    Noc noc;
    for (uint32_t i = d0_start; i < d0_end; i++) {
        if (i >= shape.logical_d0) {
            break;
        }
        for (uint32_t j = d1_start; j < d1_end; j++) {
            if (j >= shape.logical_d1) {
                read_ptr += tile_size_bytes;
                continue;
            }
#ifdef OUT_TRANSPOSED
            uint32_t tile_id = j * shape.logical_d0 + i;  // block tile (m, n) is out tile (n, m)
#else
            uint32_t tile_id = i * shape.logical_d1 + j;
#endif
            noc.async_write(
                CoreLocalMem<uint32_t>(read_ptr), tensor_accessor, tile_size_bytes, {}, {.page_id = tile_id});
            read_ptr += tile_size_bytes;
        }
        // finish up incrementing read_ptr if (d1_end - d1_start) < N_block_tiles
        read_ptr += (N_block_tiles - (d1_end - d1_start)) * tile_size_bytes;
    }
    noc.async_writes_flushed();
}

/**
 * Read ternary inputs (ternary_a and ternary_b) and write data to CB
 *
 * For ternary_a: read M_block_tiles * N_block_tiles tiles (full block), pushed one row at a time.
 * For ternary_b:
 *   - broadcast_ternary_b=1: read 1 row of tiles (N_block_tiles), compute broadcasts across M rows.
 *   - broadcast_ternary_b=0: read M rows of tiles, pushed one row at a time (matches ternary_a pattern).
 * Performance optimization: Unlike read_in0_block_sync and read_in1_block_sync, pushes ternary_a
 * tiles one row at a time. This allows the compute kernel to begin processing addcmul operations
 * as soon as the first row is ready, rather than waiting for the entire block. This overlapping
 * of data movement and compute improves overall throughput.
 */
template <uint32_t M_block_tiles, uint32_t N_block_tiles, typename TensorAccessorType>
void read_ternary_blocks_sync(
    const TensorAccessorType& ternary_a_accessor,
    const TensorAccessorType& ternary_b_accessor,
    const TensorShape2D& shape,
    uint32_t ternary_a_cb,
    uint32_t ternary_b_cb,
    uint32_t a_tile_size_bytes,
    uint32_t b_tile_size_bytes,
    uint32_t broadcast_ternary_b,
    uint32_t d0_start,
    uint32_t d0_end,
    uint32_t d1_start,
    uint32_t d1_end) {
    ASSERT(d0_end > d0_start);
    ASSERT(d1_end > d1_start);

    Noc noc;
    CircularBuffer cb_ternary_a(ternary_a_cb);
    CircularBuffer cb_ternary_b(ternary_b_cb);

    if (broadcast_ternary_b) {
        // Broadcast: read single row, push all at once
        cb_ternary_b.reserve_back(N_block_tiles);
        uint32_t ternary_b_write_ptr = cb_ternary_b.get_write_ptr();
        for (uint32_t n_tile_id = d1_start; n_tile_id < d1_end; n_tile_id++) {
            if (n_tile_id >= shape.logical_d1) {
                break;
            }
            noc.async_read(
                ternary_b_accessor,
                CoreLocalMem<uint32_t>(ternary_b_write_ptr),
                b_tile_size_bytes,
                {.page_id = n_tile_id},
                {});
            ternary_b_write_ptr += b_tile_size_bytes;
        }
        noc.async_read_barrier();
        cb_ternary_b.push_back(N_block_tiles);
    } else {
        // No broadcast: read row-by-row (matches ternary_a pattern)
        uint32_t b_m_id = 0;
        uint32_t b_i = d0_start;
        for (; b_i < d0_end; b_i++, b_m_id++) {
            cb_ternary_b.reserve_back(N_block_tiles);
            uint32_t ternary_b_write_ptr = cb_ternary_b.get_write_ptr();
            for (uint32_t j = d1_start; j < d1_end; j++) {
                if (j >= shape.logical_d1) {
                    break;
                }
                if (b_i < shape.logical_d0) {
                    uint32_t tile_id = b_i * shape.logical_d1 + j;
                    noc.async_read(
                        ternary_b_accessor,
                        CoreLocalMem<uint32_t>(ternary_b_write_ptr),
                        b_tile_size_bytes,
                        {.page_id = tile_id},
                        {});
                }
                ternary_b_write_ptr += b_tile_size_bytes;
            }
            noc.async_read_barrier();
            cb_ternary_b.push_back(N_block_tiles);
        }
        for (; b_m_id < M_block_tiles; b_m_id++) {
            cb_ternary_b.reserve_back(N_block_tiles);
            cb_ternary_b.push_back(N_block_tiles);
        }
    }

    uint32_t m_id = 0;
    uint32_t i = d0_start;
    for (; i < d0_end; i++, m_id++) {
        cb_ternary_a.reserve_back(N_block_tiles);

        uint32_t ternary_a_write_ptr = cb_ternary_a.get_write_ptr();
        for (uint32_t j = d1_start; j < d1_end; j++) {
            if (j >= shape.logical_d1) {
                // Do not move tile data into CB if tile is outside ternary/output tensor.
                // This can happen when ternary/output tensor shape is not a multiple of block sizes:
                // For instance, if tensor shape is (M_tiles=7, N_tiles=3), but block sizes are (M_block_tiles=4,
                // N_block_tiles=4)
                break;
            }
            if (i < shape.logical_d0) {
                uint32_t tile_id = i * shape.logical_d1 + j;
                noc.async_read(
                    ternary_a_accessor,
                    CoreLocalMem<uint32_t>(ternary_a_write_ptr),
                    a_tile_size_bytes,
                    {.page_id = tile_id},
                    {});
            }
            ternary_a_write_ptr += a_tile_size_bytes;
        }
        noc.async_read_barrier();

        cb_ternary_a.push_back(N_block_tiles);
    }
    for (; m_id < M_block_tiles; m_id++) {
        cb_ternary_a.reserve_back(N_block_tiles);
        cb_ternary_a.push_back(N_block_tiles);
    }
}

/**
 * This write method is more granular, waiting on a row of output tiles
 * in the output CB before writing those out, rather than waiting on the entire block.
 */
template <uint32_t M_block_tiles, uint32_t N_block_tiles, typename TensorAccessorType>
void write_block_sync_granular(
    const TensorAccessorType& tensor_accessor,
    const TensorShape2D& shape,
    uint32_t cb_id_out,
    uint32_t tile_size_bytes,
    uint32_t d0_start,
    uint32_t d0_end,
    uint32_t d1_start,
    uint32_t d1_end) {
    Noc noc;
    CircularBuffer cb_out(cb_id_out);
    for (uint32_t m_id = 0; m_id < M_block_tiles; m_id++) {
        cb_out.wait_front(N_block_tiles);
        uint32_t m_tile = d0_start + m_id;
        if (m_tile < d0_end && m_tile < shape.logical_d0) {
            uint32_t out_read_ptr = cb_out.get_read_ptr();
            for (uint32_t n_tile_id = d1_start; n_tile_id < d1_end; n_tile_id++) {
                if (n_tile_id >= shape.logical_d1) {
                    break;
                }
#ifdef OUT_TRANSPOSED
                uint32_t tile_id = n_tile_id * shape.logical_d0 + m_tile;  // block tile (m, n) is out tile (n, m)
#else
                uint32_t tile_id = m_tile * shape.logical_d1 + n_tile_id;
#endif
                noc.async_write(
                    CoreLocalMem<uint32_t>(out_read_ptr), tensor_accessor, tile_size_bytes, {}, {.page_id = tile_id});
                out_read_ptr += tile_size_bytes;
            }
        }
        // Drain this row's outstanding write-source reads out of the cb_out L1 slot BEFORE pop_front
        // releases it back to the compute producer. Otherwise the producer can repack the freed slot while
        // the async writes are still in flight (WAR on the output CB source) -> corrupt output.
        noc.async_writes_flushed();
        cb_out.pop_front(N_block_tiles);
    }
}

/**
 * Helper: dispatch to correct tuple element using fold expression.
 * Each branch calls noc.async_write with the concrete TensorAccessor type.
 */
template <typename Tuple, size_t... Is>
FORCE_INLINE void write_tile_to_chunk(
    Noc noc,
    const Tuple& accessors,
    uint32_t chunk_idx,
    uint32_t tile_id,
    uint32_t read_ptr,
    uint32_t tile_size_bytes,
    std::index_sequence<Is...>) {
    // Fold expression: expands to if/else chain at compile time
    // Each branch calls noc.async_write with the concrete type (TensorAccessor<DSpec> dispatch preserved)
    ((chunk_idx == Is
          ? (noc.async_write(
                 CoreLocalMem<uint32_t>(read_ptr), std::get<Is>(accessors), tile_size_bytes, {}, {.page_id = tile_id}),
             void())
          : void()),
     ...);
}

#ifdef OUT_HEADS
// (OUT_HEADS) the output tensor (0 = q, 1 = k, 2 = v) and page of output tile (m, n)
FORCE_INLINE void heads_target(uint32_t m, uint32_t n, uint32_t& chunk, uint32_t& page) {
    const uint32_t b = m / OUT_HEADS_ST, st = m % OUT_HEADS_ST;
    const uint32_t h = n / OUT_HEADS_HDT, d = n % OUT_HEADS_HDT;
    uint32_t hc = h, nh = OUT_HEADS_NQ;
    chunk = 0;
    if (h >= OUT_HEADS_NQ + OUT_HEADS_NKV) {
        chunk = 2;
        hc = h - OUT_HEADS_NQ - OUT_HEADS_NKV;
        nh = OUT_HEADS_NKV;
    } else if (h >= OUT_HEADS_NQ) {
        chunk = 1;
        hc = h - OUT_HEADS_NQ;
        nh = OUT_HEADS_NKV;
    }
    page = ((b * nh + hc) * OUT_HEADS_ST + st) * OUT_HEADS_HDT + d;
}
#endif

/**
 * Write a block of output to a potentially padded tensor.
 * Skip writing when M >= logical_M or N >= logical_N
 *
 * Note: Unlike write_block_sync, this function takes a tuple of accessors, rather than a single accessor.
 */
template <
    uint32_t M_block_tiles,
    uint32_t N_block_tiles,
    uint32_t N_chunks,
    uint32_t N_tiles_per_chunk,
    typename... Accessors>
void write_block_sync_split(
    const std::tuple<Accessors...>& accessors,
    const TensorShape2D& chunk_shape,
    uint32_t read_ptr,
    uint32_t tile_size_bytes,
    uint32_t d0_start,
    uint32_t d0_end,
    uint32_t d1_start,
    uint32_t d1_end) {
    ASSERT(d0_end > d0_start);
    ASSERT(d1_end > d1_start);

    Noc noc;
    const uint32_t chunk_idx_start = d1_start / N_tiles_per_chunk;
    const uint32_t tile_idx_in_chunk_start = d1_start % N_tiles_per_chunk;

    for (uint32_t i = d0_start; i < d0_end; i++) {
        // Assumes that all chunks have same number of tiles on the M-axis
        if (i >= chunk_shape.logical_d0) {
            break;
        }

        uint32_t chunk_idx = chunk_idx_start;
        uint32_t tile_idx_in_chunk = tile_idx_in_chunk_start;

        for (uint32_t j = d1_start; j < d1_end; j++, tile_idx_in_chunk++) {
#ifdef OUT_HEADS
            if (j < OUT_HEADS_NT) {
                uint32_t hchunk, hpage;
                heads_target(i, j, hchunk, hpage);
                write_tile_to_chunk(
                    noc, accessors, hchunk, hpage, read_ptr, tile_size_bytes, std::index_sequence_for<Accessors...>{});
            }
            read_ptr += tile_size_bytes;
            continue;
#endif
            // If we've reached the end of the current chunk, move to the next one
            if (tile_idx_in_chunk >= chunk_shape.logical_d1) {
                tile_idx_in_chunk = 0;
                chunk_idx++;  // Move to next chunk; if chunk is past last one then next branch will skip padding
            }

            // Skip padding
            if (chunk_idx >= N_chunks) {
                read_ptr += tile_size_bytes;
                continue;
            }

            uint32_t tile_id_in_chunk = i * chunk_shape.logical_d1 + tile_idx_in_chunk;

            // Compile-time dispatch preserving concrete types
            write_tile_to_chunk(
                noc,
                accessors,
                chunk_idx,
                tile_id_in_chunk,
                read_ptr,
                tile_size_bytes,
                std::index_sequence_for<Accessors...>{});
            read_ptr += tile_size_bytes;
        }
        // finish up incrementing read_ptr if (d1_end - d1_start) < N_block_tiles
        read_ptr += (N_block_tiles - (d1_end - d1_start)) * tile_size_bytes;
    }
    noc.async_writes_flushed();
}

/**
 * Variadic write method for split operation with N output tensors.
 * Takes the tuple directly, preserving concrete TensorAccessor<DSpec> types for noc_async_write_tile.
 */
template <
    uint32_t M_block_tiles,
    uint32_t N_block_tiles,
    uint32_t N_chunks,
    uint32_t N_tiles_per_chunk,
    typename... Accessors>
void write_block_sync_granular_split(
    const std::tuple<Accessors...>& accessors,
    const TensorShape2D& chunk_shape,
    uint32_t cb_id_out,
    uint32_t tile_size_bytes,
    uint32_t d0_start,
    uint32_t d0_end,
    uint32_t d1_start,
    uint32_t d1_end) {
    Noc noc;
    CircularBuffer cb_out(cb_id_out);
    const uint32_t chunk_idx_start = d1_start / N_tiles_per_chunk;
    const uint32_t tile_idx_in_chunk_start = d1_start % N_tiles_per_chunk;

    for (uint32_t m_id = 0; m_id < M_block_tiles; m_id++) {
        cb_out.wait_front(N_block_tiles);
        uint32_t m_tile = d0_start + m_id;
        if (m_tile < d0_end && m_tile < chunk_shape.logical_d0) {
            uint32_t out_read_ptr = cb_out.get_read_ptr();

            uint32_t chunk_idx = chunk_idx_start;
            uint32_t tile_idx_in_chunk = tile_idx_in_chunk_start;

            for (uint32_t n_tile_id = d1_start; n_tile_id < d1_end; n_tile_id++, tile_idx_in_chunk++) {
#ifdef OUT_HEADS
                if (n_tile_id >= OUT_HEADS_NT) {
                    break;
                }
                uint32_t hchunk, hpage;
                heads_target(m_tile, n_tile_id, hchunk, hpage);
                write_tile_to_chunk(
                    noc,
                    accessors,
                    hchunk,
                    hpage,
                    out_read_ptr,
                    tile_size_bytes,
                    std::index_sequence_for<Accessors...>{});
                out_read_ptr += tile_size_bytes;
                continue;
#endif
                // If we've reached the end of the current chunk, move to the next one
                if (tile_idx_in_chunk >= chunk_shape.logical_d1) {
                    tile_idx_in_chunk = 0;
                    chunk_idx++;  // Move to next chunk; if chunk is past last one then next branch will skip padding
                }

                if (chunk_idx >= N_chunks) {
                    break;
                }

                uint32_t tile_id = m_tile * chunk_shape.logical_d1 + tile_idx_in_chunk;
                // Compile-time dispatch preserving concrete types
                write_tile_to_chunk(
                    noc,
                    accessors,
                    chunk_idx,
                    tile_id,
                    out_read_ptr,
                    tile_size_bytes,
                    std::index_sequence_for<Accessors...>{});

                out_read_ptr += tile_size_bytes;
            }
        }
        // Flush this row's write-source reads out of the cb_out slot before releasing it (same WAR on the
        // output CB as write_block_sync_granular above).
        noc.async_writes_flushed();
        cb_out.pop_front(N_block_tiles);
    }
}

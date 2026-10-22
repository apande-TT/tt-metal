// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B mm_swiglu: copy of the legacy minimal_matmul in0 kernel
// (ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in0_sender.cpp), trimmed ONLY of
// the code paths that are compiled out for this op (no FUSE_AG, SRS_FUSE_OP_SIGNALER, MM_WINDOW_BLOCKS,
// FUSE_TERNARY, FUSE_BIAS, READ_FROM_LOCAL_INPUT / IN0_VIRTUAL_CONCAT), which also drops the include of the CCL
// fused_receiver_utils.hpp they needed. Everything else -- compile-time / runtime arg layout, the in0 read and
// forwarding chain, the deferred output writes, the FUSE_SWIGLU half-width output geometry -- is unchanged.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "matmul_dataflow_common.hpp"

void kernel_main() {
    Noc noc;
    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t padded_M_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t padded_K_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t N_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t padded_N_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t M_blocks_per_core = get_compile_time_arg_val(9);
    constexpr uint32_t N_blocks_per_core = get_compile_time_arg_val(10);
    constexpr uint32_t in0_tile_size = get_compile_time_arg_val(11);
    constexpr uint32_t out_tile_size = get_compile_time_arg_val(12);
    constexpr uint32_t in2_tile_size = get_compile_time_arg_val(13);
    Semaphore<> in0_sender_semaphore(get_compile_time_arg_val(14));
    Semaphore<> in0_receiver_semaphore(get_compile_time_arg_val(15));
    Semaphore<> in0_valid_semaphore(get_compile_time_arg_val(16));
    constexpr uint32_t is_output_writer = get_compile_time_arg_val(17);
    constexpr uint32_t is_injector_core = get_compile_time_arg_val(18);
    constexpr uint32_t N_chunks = get_compile_time_arg_val(19);
    constexpr uint32_t N_tiles_per_chunk = get_compile_time_arg_val(20);
    // get_compile_time_arg_val(21): in3_tile_size (unused without a second in0 source; kept for the arg layout)

    // Load input/output addresses and range parameters
    uint32_t argidx = 0;
    const uint32_t in0_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t in2_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t in3_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t is_sink_core = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_dest_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_dest_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_sender_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in0_sender_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t defer_write_k_block = get_arg_val<uint32_t>(argidx++);
    const uint32_t max_defer_write_k_block = get_arg_val<uint32_t>(argidx++);
    (void)in2_addr;
    (void)in3_addr;
    (void)max_defer_write_k_block;

    const uint32_t out_addr_rt_arg_idx = argidx;  // Output addresses start here

    // Tensor accessor for input tensor
    constexpr auto in0_args = TensorAccessorArgs<22>();
    const auto in0_reader = TensorAccessor(in0_args, in0_addr);

    // Always create tuple of output accessors (size = N_chunks)
    constexpr uint32_t out_tensor_args_cta_offset = in0_args.next_compile_time_args_offset();
    constexpr auto outputs_args = make_tensor_accessor_args_tuple<N_chunks, out_tensor_args_cta_offset>();
    auto outputs_tuple = make_tensor_accessor_tuple_uniform_page_size(outputs_args, out_addr_rt_arg_idx, out_tile_size);

    const TensorShape2D in0_shape(M_tiles, K_tiles, padded_M_tiles, padded_K_tiles);
    const TensorShape2D out_shape(M_tiles, N_tiles, padded_M_tiles, padded_N_tiles);
    const TensorShape2D out0_shape(M_tiles, N_tiles_per_chunk, padded_M_tiles, N_tiles_per_chunk);

    constexpr uint32_t K_num_blocks = padded_K_tiles / K_block_tiles;
    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

#ifdef FUSE_SWIGLU
    // SwiGLU emits one output tile per interleaved gate/up pair, so the output along N
    // is half the matmul (weight) N. The weight-space n ranges are halved at each write.
    constexpr uint32_t out_N_block_tiles = N_block_tiles / 2;
    constexpr uint32_t out_block_num_tiles_swiglu = M_block_tiles * out_N_block_tiles;
    const TensorShape2D out_shape_swiglu(M_tiles, N_tiles / 2, padded_M_tiles, padded_N_tiles / 2);
    // Split (chunks>1): each output chunk is half the weight per-chunk width.
    constexpr uint32_t out_N_tiles_per_chunk = N_tiles_per_chunk / 2;
    const TensorShape2D out0_shape_swiglu(M_tiles, out_N_tiles_per_chunk, padded_M_tiles, out_N_tiles_per_chunk);
#endif

    constexpr uint32_t cb_in0_id = tt::CBIndex::c_0;
    constexpr uint32_t cb_out_id = tt::CBIndex::c_2;

    CircularBuffer cb_in0(cb_in0_id);
    CircularBuffer cb_out(cb_out_id);

    in0_valid_semaphore.set(VALID);

    /**
     * This is a Row-Major output block ordering.
     * It enables reuse of the last in0 block when striding the output block N dimension.
     */

    bool k_forward = true;
    bool reuse_block = false;

    uint32_t defer_write_m_tile = 0;
    uint32_t defer_write_m_tile_end = 0;
    uint32_t defer_write_n_tile = 0;
    uint32_t defer_write_n_tile_end = 0;
    bool defer_write = false;

    for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
        uint32_t m_tile = M_start_tile + m_block_iter * M_block_tiles;
        uint32_t m_tile_end = std::min(m_tile + M_block_tiles, M_end_tile);
        uint32_t out_m_tile = m_tile;
        uint32_t out_m_tile_end = m_tile_end;
        uint32_t current_M_block_tiles = m_tile_end - m_tile;
        uint32_t current_block_bytes = current_M_block_tiles * K_block_tiles * in0_tile_size;

        // When striding M block, in0 gets no reuse
        reuse_block = false;
        k_forward = true;
        for (uint32_t n_block_iter = 0; n_block_iter < N_blocks_per_core; n_block_iter++) {
            uint32_t n_tile = N_start_tile + n_block_iter * N_block_tiles;
            uint32_t n_tile_end = std::min(n_tile + N_block_tiles, N_end_tile);
            bool is_last_block = (m_block_iter == M_blocks_per_core - 1) && (n_block_iter == (N_blocks_per_core - 1));

            for (uint32_t k_block_iter = 0; k_block_iter < K_num_blocks; k_block_iter++) {
#ifndef CHUNK_FWD
                if (defer_write && k_block_iter == defer_write_k_block) {
                    if constexpr (is_output_writer) {
#ifdef FUSE_SWIGLU
                        cb_out.wait_front(out_block_num_tiles_swiglu);
                        uint32_t out_read_ptr_swiglu = cb_out.get_read_ptr();
                        if constexpr (N_chunks == 1) {
                            write_block_sync<M_block_tiles, out_N_block_tiles>(
                                std::get<0>(outputs_tuple),
                                out_shape_swiglu,
                                out_read_ptr_swiglu,
                                out_tile_size,
                                defer_write_m_tile,
                                defer_write_m_tile_end,
                                defer_write_n_tile / 2,
                                defer_write_n_tile_end / 2);
                        } else {
                            write_block_sync_split<M_block_tiles, out_N_block_tiles, N_chunks, out_N_tiles_per_chunk>(
                                outputs_tuple,
                                out0_shape_swiglu,
                                out_read_ptr_swiglu,
                                out_tile_size,
                                defer_write_m_tile,
                                defer_write_m_tile_end,
                                defer_write_n_tile / 2,
                                defer_write_n_tile_end / 2);
                        }
                        cb_out.pop_front(out_block_num_tiles_swiglu);
#else
                        cb_out.wait_front(out_block_num_tiles);
                        uint32_t out_read_ptr = cb_out.get_read_ptr();

                        // write_block_sync_split is more generic (support multiple output tensors)
                        // But for N_chunks == 1 (non-split minimal_matmul), write_block_sync should be faster
                        if constexpr (N_chunks == 1) {
                            write_block_sync<M_block_tiles, N_block_tiles>(
                                std::get<0>(outputs_tuple),
                                out_shape,
                                out_read_ptr,
                                out_tile_size,
                                defer_write_m_tile,
                                defer_write_m_tile_end,
                                defer_write_n_tile,
                                defer_write_n_tile_end);
                        } else {
                            write_block_sync_split<M_block_tiles, N_block_tiles, N_chunks, N_tiles_per_chunk>(
                                outputs_tuple,
                                out0_shape,
                                out_read_ptr,
                                out_tile_size,
                                defer_write_m_tile,
                                defer_write_m_tile_end,
                                defer_write_n_tile,
                                defer_write_n_tile_end);
                        }
                        cb_out.pop_front(out_block_num_tiles);
#endif  // FUSE_SWIGLU
                    }
                }

#endif  // !CHUNK_FWD
                if (reuse_block && k_block_iter == 0) {
                    // We strided an N block and this is the first k block, so we get reuse and do not need to read in0
                    reuse_block = false;
#ifdef CHUNK_FWD
                    goto deferred_write;
#else
                    continue;
#endif
                }
#ifdef CHUNK_FWD
                {
                    // CHUNK_FWD (DIRECT_OUT and SwiGLU): the in0 block moves down the row's forwarding chain in IN0_CHUNK_TILES-tile chunks
                    // (row-major within the block), each forwarded as soon as it lands and counted on the next
                    // core's receiver semaphore, instead of store-and-forward of the whole block per hop.
                    const uint32_t k_block = k_forward ? k_block_iter : (K_num_blocks - 1) - k_block_iter;
                    constexpr uint32_t chunk_tiles = IN0_CHUNK_TILES;
                    constexpr uint32_t chunks_per_row = K_block_tiles / chunk_tiles;
                    const uint32_t chunk_bytes = chunk_tiles * in0_tile_size;
#ifdef IN0_STRIP_ROWS
                    // KSPLIT strips: the block moves as subblock-row strips (one CB slot each, the compute consumes
                    // them in order), so the CB holds two strips instead of two whole blocks.
                    constexpr uint32_t strip_rows = IN0_STRIP_ROWS;
#else
                    constexpr uint32_t strip_rows = M_block_tiles;
#endif
                    constexpr uint32_t strip_tiles = strip_rows * K_block_tiles;
                    for (uint32_t strip_m = 0; strip_m < current_M_block_tiles; strip_m += strip_rows) {
                    const uint32_t strip_m_tiles = std::min(strip_rows, current_M_block_tiles - strip_m);
                    const uint32_t n_chunks = strip_m_tiles * chunks_per_row;
                    cb_in0.reserve_back(strip_tiles);
                    const uint32_t in0_base = cb_in0.get_write_ptr();
                    if constexpr (!is_injector_core) {
                        in0_receiver_semaphore.set(0);
                        in0_sender_semaphore.up(noc, in0_sender_noc_x, in0_sender_noc_y, 1);
                    }
                    bool downstream_ready = false;
                    for (uint32_t c = 0; c < n_chunks; c++) {
                        const uint32_t row = c / chunks_per_row;
                        const uint32_t k0 = (c % chunks_per_row) * chunk_tiles;
                        const uint32_t chunk_addr = in0_base + (row * K_block_tiles + k0) * in0_tile_size;
                        if constexpr (is_injector_core) {
                            const uint32_t m = m_tile + strip_m + row;
                            if (m < in0_shape.logical_d0) {
                                for (uint32_t t = 0; t < chunk_tiles; t++) {
                                    noc.async_read(
                                        in0_reader,
                                        CoreLocalMem<uint32_t>(chunk_addr + t * in0_tile_size),
                                        in0_tile_size,
                                        {.page_id = m * in0_shape.logical_d1 + k_block * K_block_tiles + k0 + t},
                                        {});
                                }
                                noc.async_read_barrier();
                            }
                        } else {
                            in0_receiver_semaphore.wait_min(c + 1);
                        }
                        if (!is_sink_core) {
                            if (!downstream_ready) {
                                in0_sender_semaphore.wait(1);
                                in0_sender_semaphore.set(0);
                                downstream_ready = true;
                            }
                            noc.async_write(
                                CoreLocalMem<uint32_t>(chunk_addr),
                                UnicastEndpoint{},
                                chunk_bytes,
                                {},
                                {.noc_x = in0_dest_noc_x, .noc_y = in0_dest_noc_y, .addr = chunk_addr});
                            noc.async_writes_flushed();
                            in0_receiver_semaphore.up(noc, in0_dest_noc_x, in0_dest_noc_y, 1);
                        }
                    }
                    cb_in0.push_back(strip_tiles);
                    }
                }
#else
                {
                uint32_t k_block = k_forward ? k_block_iter : (K_num_blocks - 1) - k_block_iter;
                cb_in0.reserve_back(in0_block_num_tiles);

                uint32_t in0_start_address = cb_in0.get_write_ptr();
                if constexpr (is_injector_core) {
                    read_in0_block_sync<M_block_tiles, K_block_tiles>(
                        in0_reader,
                        in0_shape,
                        cb_in0_id,
                        in0_tile_size,
                        m_tile,
                        m_tile_end,
                        k_block * K_block_tiles,
                        (k_block + 1) * K_block_tiles);
                } else {
                    // Get from previous device
                    in0_receiver_semaphore.set(INVALID);
                    in0_sender_semaphore.up(noc, in0_sender_noc_x, in0_sender_noc_y, 1);
                    in0_receiver_semaphore.wait(VALID);
                }

                // Critical to performance for sender to push data to compute before mcasting
                // This frees sender to start next read earlier
                cb_in0.push_back(in0_block_num_tiles);

                if (!is_sink_core) {
                    in0_sender_semaphore.wait(1);
                    in0_sender_semaphore.set(0);

                    /**
                     * in0 is M_block_tiles x K_block_tiles. When M block is partial, we don't need to write the
                     * padded tiles. Use `current_block_bytes`.
                     */
                    noc.async_write(
                        CoreLocalMem<uint32_t>(in0_start_address),
                        UnicastEndpoint{},
                        current_block_bytes,
                        {},
                        {.noc_x = in0_dest_noc_x, .noc_y = in0_dest_noc_y, .addr = in0_start_address});

#ifdef ARCH_BLACKHOLE
                    noc.async_writes_flushed();
#endif

                    in0_valid_semaphore.relay_unicast(noc, in0_receiver_semaphore, in0_dest_noc_x, in0_dest_noc_y);
                }
                }
#endif  // CHUNK_FWD
#ifdef CHUNK_FWD
            // CHUNK_FWD: the previous block's deferred output write comes AFTER this block's in0 read / forward, so
            // a new M block's in0 streams in while compute is still finishing the previous block (writing first
            // blocked the read -- and the row's forwarding chain behind it -- on that block's compute).
            deferred_write:
                if (defer_write && k_block_iter == defer_write_k_block) {
                    if constexpr (is_output_writer) {
#ifdef FUSE_SWIGLU
                        cb_out.wait_front(out_block_num_tiles_swiglu);
                        uint32_t out_read_ptr_swiglu = cb_out.get_read_ptr();
                        if constexpr (N_chunks == 1) {
                            write_block_sync<M_block_tiles, out_N_block_tiles>(
                                std::get<0>(outputs_tuple),
                                out_shape_swiglu,
                                out_read_ptr_swiglu,
                                out_tile_size,
                                defer_write_m_tile,
                                defer_write_m_tile_end,
                                defer_write_n_tile / 2,
                                defer_write_n_tile_end / 2);
                        } else {
                            write_block_sync_split<M_block_tiles, out_N_block_tiles, N_chunks, out_N_tiles_per_chunk>(
                                outputs_tuple,
                                out0_shape_swiglu,
                                out_read_ptr_swiglu,
                                out_tile_size,
                                defer_write_m_tile,
                                defer_write_m_tile_end,
                                defer_write_n_tile / 2,
                                defer_write_n_tile_end / 2);
                        }
                        cb_out.pop_front(out_block_num_tiles_swiglu);
#else
                        cb_out.wait_front(out_block_num_tiles);
                        uint32_t out_read_ptr = cb_out.get_read_ptr();

                        // write_block_sync_split is more generic (support multiple output tensors)
                        // But for N_chunks == 1 (non-split minimal_matmul), write_block_sync should be faster
                        if constexpr (N_chunks == 1) {
                            write_block_sync<M_block_tiles, N_block_tiles>(
                                std::get<0>(outputs_tuple),
                                out_shape,
                                out_read_ptr,
                                out_tile_size,
                                defer_write_m_tile,
                                defer_write_m_tile_end,
                                defer_write_n_tile,
                                defer_write_n_tile_end);
                        } else {
                            write_block_sync_split<M_block_tiles, N_block_tiles, N_chunks, N_tiles_per_chunk>(
                                outputs_tuple,
                                out0_shape,
                                out_read_ptr,
                                out_tile_size,
                                defer_write_m_tile,
                                defer_write_m_tile_end,
                                defer_write_n_tile,
                                defer_write_n_tile_end);
                        }
                        cb_out.pop_front(out_block_num_tiles);
#endif  // FUSE_SWIGLU
                    }
                }

#endif  // CHUNK_FWD
            }

            k_forward = !k_forward;
            // We get reuse on in0 when striding N block
            reuse_block = true;

            defer_write_m_tile = out_m_tile;
            defer_write_m_tile_end = out_m_tile_end;
            defer_write_n_tile = n_tile;
            defer_write_n_tile_end = n_tile_end;
            /**
             * If this isn't the last output block, defer writing until the defer_k_write_block iteration
             * of the next output block.
             */
            defer_write = !is_last_block;
#ifndef CHUNK_FWD
            defer_write = defer_write && !is_injector_core;
#endif

            if (!defer_write) {
                if constexpr (is_output_writer) {
#ifdef FUSE_SWIGLU
                    if constexpr (N_chunks == 1) {
                        write_block_sync_granular<M_block_tiles, out_N_block_tiles>(
                            std::get<0>(outputs_tuple),
                            out_shape_swiglu,
                            cb_out_id,
                            out_tile_size,
                            out_m_tile,
                            out_m_tile_end,
                            n_tile / 2,
                            n_tile_end / 2);
                    } else {
                        write_block_sync_granular_split<
                            M_block_tiles,
                            out_N_block_tiles,
                            N_chunks,
                            out_N_tiles_per_chunk>(
                            outputs_tuple,
                            out0_shape_swiglu,
                            cb_out_id,
                            out_tile_size,
                            out_m_tile,
                            out_m_tile_end,
                            n_tile / 2,
                            n_tile_end / 2);
                    }
#else
                    // write_block_sync_granular_split is more generic (support multiple output tensors)
                    // But for N_chunks == 1 (non-split minimal_matmul), write_block_sync_granular should be faster
                    if constexpr (N_chunks == 1) {
                        write_block_sync_granular<M_block_tiles, N_block_tiles>(
                            std::get<0>(outputs_tuple),
                            out_shape,
                            cb_out_id,
                            out_tile_size,
                            out_m_tile,
                            out_m_tile_end,
                            n_tile,
                            n_tile_end);
                    } else {
                        write_block_sync_granular_split<M_block_tiles, N_block_tiles, N_chunks, N_tiles_per_chunk>(
                            outputs_tuple,
                            out0_shape,
                            cb_out_id,
                            out_tile_size,
                            out_m_tile,
                            out_m_tile_end,
                            n_tile,
                            n_tile_end);
                    }
#endif  // FUSE_SWIGLU
                }
            }
        }
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}

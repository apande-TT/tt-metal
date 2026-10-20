// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B mm_swiglu: copy of the legacy minimal_matmul in1 kernel
// (ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/dm_in1_sender_out.cpp), trimmed ONLY of
// the code paths that are compiled out for this op (no FUSE_AG, SRS_FUSE_OP_SIGNALER, MM_WINDOW_BLOCKS,
// FUSE_TERNARY, FUSE_BIAS), which also drops the include of the CCL fused_receiver_utils.hpp they needed.
// Everything else -- compile-time / runtime arg layout, the in1 read and forwarding chain, the (transposed-grid)
// output writes, the FUSE_SWIGLU half-width output geometry -- is unchanged. (With M < N the grid is not
// transposed, so this kernel is not the output writer here; its write paths are compiled out by is_output_writer.)

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
    constexpr uint32_t in1_tile_size = get_compile_time_arg_val(11);
    constexpr uint32_t out_tile_size = get_compile_time_arg_val(12);
    constexpr uint32_t in2_tile_size = get_compile_time_arg_val(13);
    Semaphore<> in1_sender_semaphore(get_compile_time_arg_val(14));
    Semaphore<> in1_receiver_semaphore(get_compile_time_arg_val(15));
    Semaphore<> in1_valid_semaphore(get_compile_time_arg_val(16));
    constexpr uint32_t is_output_writer = get_compile_time_arg_val(17);
    constexpr uint32_t is_injector_core = get_compile_time_arg_val(18);
    constexpr uint32_t N_chunks = get_compile_time_arg_val(19);
    constexpr uint32_t N_tiles_per_chunk = get_compile_time_arg_val(20);

    // Load input/output addresses and range parameters
    uint32_t argidx = 0;
    const uint32_t in1_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t in2_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t is_sink_core = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_dest_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_dest_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_sender_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t in1_sender_noc_y = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t defer_write_k_block = get_arg_val<uint32_t>(argidx++);
    const uint32_t max_defer_write_k_block = get_arg_val<uint32_t>(argidx++);
    (void)in2_addr;
    (void)max_defer_write_k_block;

    const uint32_t out_addr_rt_arg_idx = argidx;  // Output addresses start here

    // Tensor accessor for input tensor
    constexpr auto in1_args = TensorAccessorArgs<21>();
    const auto in1_reader = TensorAccessor(in1_args, in1_addr);

    // Always create tuple of output accessors (size = N_chunks)
    constexpr uint32_t out_tensor_args_cta_offset = in1_args.next_compile_time_args_offset();
    constexpr auto outputs_args = make_tensor_accessor_args_tuple<N_chunks, out_tensor_args_cta_offset>();
    auto outputs_tuple = make_tensor_accessor_tuple_uniform_page_size(outputs_args, out_addr_rt_arg_idx, out_tile_size);

    const TensorShape2D in1_shape(K_tiles, N_tiles, padded_K_tiles, padded_N_tiles);
    const TensorShape2D out_shape(M_tiles, N_tiles, padded_M_tiles, padded_N_tiles);
    const TensorShape2D out0_shape(M_tiles, N_tiles_per_chunk, padded_M_tiles, N_tiles_per_chunk);

    constexpr uint32_t K_num_blocks = padded_K_tiles / K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

#ifdef FUSE_SWIGLU
    // SwiGLU emits one output tile per interleaved gate/up pair, so the output along N
    // is half the matmul (weight) N. Compute the halved output geometry once here; the
    // weight-space n ranges (n_tile, N_tiles, ...) are halved at each write call site.
    constexpr uint32_t out_N_block_tiles = N_block_tiles / 2;
    constexpr uint32_t out_block_num_tiles_swiglu = M_block_tiles * out_N_block_tiles;
    const TensorShape2D out_shape_swiglu(M_tiles, N_tiles / 2, padded_M_tiles, padded_N_tiles / 2);
    // Split (chunks>1): each output chunk is half the weight per-chunk width.
    constexpr uint32_t out_N_tiles_per_chunk = N_tiles_per_chunk / 2;
    const TensorShape2D out0_shape_swiglu(M_tiles, out_N_tiles_per_chunk, padded_M_tiles, out_N_tiles_per_chunk);
#endif

    constexpr uint32_t cb_in1_id = tt::CBIndex::c_1;
    constexpr uint32_t cb_out_id = tt::CBIndex::c_2;

    CircularBuffer cb_in1(cb_in1_id);
    CircularBuffer cb_out(cb_out_id);

    in1_valid_semaphore.set(VALID);

    constexpr uint32_t full_N_tiles_bytes = N_block_tiles * in1_tile_size;

    bool k_forward = true;

    uint32_t defer_write_m_tile = 0;
    uint32_t defer_write_m_tile_end = 0;
    uint32_t defer_write_n_tile = 0;
    uint32_t defer_write_n_tile_end = 0;
    bool defer_write = false;

    for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
        uint32_t m_tile = M_start_tile + m_block_iter * M_block_tiles;
        uint32_t m_tile_end = std::min(m_tile + M_block_tiles, M_end_tile);

        k_forward = true;

        for (uint32_t n_block_iter = 0; n_block_iter < N_blocks_per_core; n_block_iter++) {
            uint32_t n_tile = N_start_tile + n_block_iter * N_block_tiles;
            uint32_t n_tile_end = std::min(n_tile + N_block_tiles, N_end_tile);
            uint32_t current_N_block_tiles = n_tile_end - n_tile;
            uint32_t current_N_tiles_bytes = current_N_block_tiles * in1_tile_size;
            bool is_last_block = (m_block_iter == M_blocks_per_core - 1) && (n_block_iter == (N_blocks_per_core - 1));
            for (uint32_t k_block_iter = 0; k_block_iter < K_num_blocks; k_block_iter++) {
                if (defer_write && k_block_iter == defer_write_k_block) {
                    if constexpr (is_output_writer) {
#ifdef FUSE_SWIGLU
                        cb_out.wait_front(out_block_num_tiles_swiglu);
                        uint32_t out_read_ptr = cb_out.get_read_ptr();
                        if constexpr (N_chunks == 1) {
                            write_block_sync<M_block_tiles, out_N_block_tiles>(
                                std::get<0>(outputs_tuple),
                                out_shape_swiglu,
                                out_read_ptr,
                                out_tile_size,
                                defer_write_m_tile,
                                defer_write_m_tile_end,
                                defer_write_n_tile / 2,
                                defer_write_n_tile_end / 2);
                        } else {
                            write_block_sync_split<M_block_tiles, out_N_block_tiles, N_chunks, out_N_tiles_per_chunk>(
                                outputs_tuple,
                                out0_shape_swiglu,
                                out_read_ptr,
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

#ifdef DIRECT_OUT
                {
                    // DIRECT_OUT: the in1 block moves down the column's forwarding chain in IN1_CHUNK_K-row chunks
                    // (K_block x N_block row-major, so a chunk is contiguous), each forwarded as soon as it lands and
                    // counted on the next core's receiver semaphore, instead of store-and-forward of the whole block.
                    const uint32_t k_block = k_forward ? k_block_iter : (K_num_blocks - 1) - k_block_iter;
                    constexpr uint32_t chunk_rows = IN1_CHUNK_K;
                    constexpr uint32_t n_chunks = K_block_tiles / chunk_rows;
                    cb_in1.reserve_back(in1_block_num_tiles);
                    const uint32_t in1_base = cb_in1.get_write_ptr();
                    if constexpr (!is_injector_core) {
                        in1_receiver_semaphore.set(0);
                        in1_sender_semaphore.up(noc, in1_sender_noc_x, in1_sender_noc_y, 1);
                    }
                    bool downstream_ready = false;
                    for (uint32_t c = 0; c < n_chunks; c++) {
                        const uint32_t chunk_addr = in1_base + c * chunk_rows * full_N_tiles_bytes;
                        if constexpr (is_injector_core) {
                            uint32_t addr = chunk_addr;
                            for (uint32_t r = 0; r < chunk_rows; r++) {
                                const uint32_t k = k_block * K_block_tiles + c * chunk_rows + r;
                                for (uint32_t n = n_tile; n < n_tile + N_block_tiles; n++) {
                                    if (n < in1_shape.logical_d1) {
                                        noc.async_read(
                                            in1_reader,
                                            CoreLocalMem<uint32_t>(addr),
                                            in1_tile_size,
                                            {.page_id = k * in1_shape.logical_d1 + n},
                                            {});
                                    }
                                    addr += in1_tile_size;
                                }
                            }
                            noc.async_read_barrier();
                        } else {
                            in1_receiver_semaphore.wait_min(c + 1);
                        }
                        if (!is_sink_core) {
                            if (!downstream_ready) {
                                in1_sender_semaphore.wait(1);
                                in1_sender_semaphore.set(0);
                                downstream_ready = true;
                            }
                            noc.async_write(
                                CoreLocalMem<uint32_t>(chunk_addr),
                                UnicastEndpoint{},
                                chunk_rows * full_N_tiles_bytes,
                                {},
                                {.noc_x = in1_dest_noc_x, .noc_y = in1_dest_noc_y, .addr = chunk_addr});
                            noc.async_writes_flushed();
                            in1_receiver_semaphore.up(noc, in1_dest_noc_x, in1_dest_noc_y, 1);
                        }
                    }
                    cb_in1.push_back(in1_block_num_tiles);
                }
#else
                uint32_t k_block = k_forward ? k_block_iter : (K_num_blocks - 1) - k_block_iter;
                cb_in1.reserve_back(in1_block_num_tiles);

                uint32_t in1_start_address = cb_in1.get_write_ptr();
                if constexpr (is_injector_core) {
                    read_in1_block_sync<K_block_tiles, N_block_tiles>(
                        in1_reader,
                        in1_shape,
                        cb_in1_id,
                        in1_tile_size,
                        k_block * K_block_tiles,
                        (k_block + 1) * K_block_tiles,
                        n_tile,
                        n_tile_end);
                } else {
                    in1_receiver_semaphore.set(INVALID);
                    in1_sender_semaphore.up(noc, in1_sender_noc_x, in1_sender_noc_y, 1);
                    in1_receiver_semaphore.wait(VALID);
                }

                // Critical to performance for sender to push data to compute before mcasting
                // This frees sender to start next read earlier
                cb_in1.push_back(in1_block_num_tiles);

                if (!is_sink_core) {
                    in1_sender_semaphore.wait(1);
                    in1_sender_semaphore.set(0);

                    /**
                     * in1 is K_block_tiles x N_block_tiles. When N block is partial, we don't need to write the
                     * padded tiles. For each tile in the K block, write only the non-padded N tiles. Use
                     * `current_N_tiles_bytes`.
                     */
                    for (uint32_t i = 0; i < K_block_tiles; i++) {
                        noc.async_write(
                            CoreLocalMem<uint32_t>(in1_start_address),
                            UnicastEndpoint{},
                            current_N_tiles_bytes,
                            {},
                            {.noc_x = in1_dest_noc_x, .noc_y = in1_dest_noc_y, .addr = in1_start_address});
                        in1_start_address += full_N_tiles_bytes;
                    }

#ifdef ARCH_BLACKHOLE
                    noc.async_writes_flushed();
#endif

                    in1_valid_semaphore.relay_unicast(noc, in1_receiver_semaphore, in1_dest_noc_x, in1_dest_noc_y);
                }
#endif  // DIRECT_OUT
            }

            k_forward = !k_forward;
            // We have an output block to write out

            defer_write_m_tile = m_tile;
            defer_write_m_tile_end = m_tile_end;
            defer_write_n_tile = n_tile;
            defer_write_n_tile_end = n_tile_end;
            /**
             * If this isn't the last output block, defer writing until the defer_k_write_block iteration
             * of the next output block.
             */
            defer_write = !is_last_block;
            defer_write = defer_write && !is_injector_core;

            if (!defer_write) {
                if constexpr (is_output_writer) {
#ifdef FUSE_SWIGLU
                    if constexpr (N_chunks == 1) {
                        write_block_sync_granular<M_block_tiles, out_N_block_tiles>(
                            std::get<0>(outputs_tuple),
                            out_shape_swiglu,
                            cb_out_id,
                            out_tile_size,
                            m_tile,
                            m_tile_end,
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
                            m_tile,
                            m_tile_end,
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
                            m_tile,
                            m_tile_end,
                            n_tile,
                            n_tile_end);
                    } else {
                        write_block_sync_granular_split<M_block_tiles, N_block_tiles, N_chunks, N_tiles_per_chunk>(
                            outputs_tuple,
                            out0_shape,
                            cb_out_id,
                            out_tile_size,
                            m_tile,
                            m_tile_end,
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

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode LM head reader (M = one tile row).
// Adapted from programming_examples/matmul/matmul_multi_core reader_mm_output_tiles_partitioned.cpp.
//   1. Reads the whole activation row x[0, 0..Kt) into cb_in0 once (kept resident by compute).
//   2. For each group of `group_w` output columns owned by this core, streams the weight tiles
//      w[kt, n] in K-major blocks of `k_block` K rows x `group_w` columns (constant block size,
//      so the CB never wraps mid-block).
// Output column n is global over the concatenated vocab: chunk = n / Nt_chunk, col = n % Nt_chunk.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Nt_chunk = get_compile_time_arg_val(1);
    constexpr uint32_t group_w = get_compile_time_arg_val(2);
    constexpr uint32_t k_block = get_compile_time_arg_val(3);
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(4);
    constexpr uint32_t cb_in1 = get_compile_time_arg_val(5);
    constexpr uint32_t in0_tile_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t in1_tile_bytes = get_compile_time_arg_val(7);
    constexpr auto in0_args = TensorAccessorArgs<8>();
    // All weight chunks share one spec (same shape/dtype/DRAM interleaved): one args block, per-chunk address.
    constexpr auto in1_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();

    const uint32_t n_start = get_arg_val<uint32_t>(0);
    const uint32_t num_groups = get_arg_val<uint32_t>(1);
    // common args: [in0_addr, w_addr[0], w_addr[1], ...]
    const uint32_t in0_addr = get_common_arg_val<uint32_t>(0);

    constexpr uint32_t block_tiles = k_block * group_w;

    // Activations: Kt tiles, read once.
    {
        const auto in0 = TensorAccessor(in0_args, in0_addr);
        cb_reserve_back(cb_in0, Kt);
        uint32_t l1_addr = get_write_ptr(cb_in0);
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            noc_async_read_page(kt, in0, l1_addr);
            l1_addr += in0_tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_in0, Kt);
    }

    uint32_t n = n_start;
    for (uint32_t g = 0; g < num_groups; ++g) {
        uint32_t w_addr[group_w];
        uint32_t w_col[group_w];
        for (uint32_t j = 0; j < group_w; ++j) {
            const uint32_t chunk = (n + j) / Nt_chunk;
            w_col[j] = (n + j) - chunk * Nt_chunk;
            w_addr[j] = get_common_arg_val<uint32_t>(1 + chunk);
        }
        for (uint32_t kb = 0; kb < Kt; kb += k_block) {
            cb_reserve_back(cb_in1, block_tiles);
            uint32_t l1_addr = get_write_ptr(cb_in1);
            for (uint32_t k = 0; k < k_block; ++k) {
                const uint32_t row_page = (kb + k) * Nt_chunk;
                for (uint32_t j = 0; j < group_w; ++j) {
                    const auto in1 = TensorAccessor(in1_args, w_addr[j]);
                    noc_async_read_page(row_page + w_col[j], in1, l1_addr);
                    l1_addr += in1_tile_bytes;
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_in1, block_tiles);
        }
        n += group_w;
    }
}

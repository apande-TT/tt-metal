// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Middle level of the sampler's three-level vocab top-k (pipeline.py VocabTopK).
//
// The stock multi-core ttnn.topk sends every local core's K candidates to ONE final core, which then
// runs log2(#local cores) bitonic merge iterations over all of them serially (64 tiles -> 6 iterations,
// ~100 us at 32 x 32768). Here each middle core gathers the candidates of its own group of local cores and
// merges them; the final core merges only the middle cores' results, so the serial merge work per core
// drops from 63 tile merges to 7 + 7.
//
// This kernel is the stock final-merge loop (topk_final.cpp) with two changes:
//   * the result keeps the transposed (column) layout -- it is copied out as is (topk_local.cpp's tail)
//     because it travels on to the final core instead of to DRAM;
//   * its sort direction comes from the runtime arg direction_init, alternating across middle cores
//     exactly as the stock op alternates its local cores, so that consecutive inputs of the final merge
//     form bitonic pairs.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/topk.h"
#include "api/compute/transpose.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"

#include "topk_common_funcs.hpp"

void kernel_main() {
    constexpr std::uint32_t input_dfb_index = get_compile_time_arg_val(0);             // gathered values
    constexpr std::uint32_t index_dfb_index = get_compile_time_arg_val(1);             // gathered indices
    constexpr std::uint32_t input_transposed_dfb_index = get_compile_time_arg_val(2);  // values workspace
    constexpr std::uint32_t index_transposed_dfb_index = get_compile_time_arg_val(3);  // indices workspace
    constexpr std::uint32_t values_dfb_index = get_compile_time_arg_val(4);            // result values (to final)
    constexpr std::uint32_t output_ind_dfb_index = get_compile_time_arg_val(5);        // result indices (to final)
    constexpr std::uint32_t Ht = get_compile_time_arg_val(6);
    constexpr std::uint32_t Wt = get_compile_time_arg_val(7);  // gathered tiles (group size x Kt)
    constexpr std::uint32_t K = get_compile_time_arg_val(8);
    constexpr std::uint32_t Kt = get_compile_time_arg_val(9);
    constexpr std::uint32_t logk = get_compile_time_arg_val(10);
    constexpr std::uint32_t logWt = get_compile_time_arg_val(11);
    constexpr bool largest = get_compile_time_arg_val(12) == 1;

    const std::uint32_t direction_init = get_arg_val<std::uint32_t>(0);

    constexpr std::uint32_t input_dest_start = 0;
    constexpr std::uint32_t index_dest_start = 2;
    constexpr std::uint32_t input_dest_end = 1;
    constexpr std::uint32_t index_dest_end = 3;

    constexpr std::uint32_t tiles_per_seq = (K + 31) / 32;
    const bool switch_dir = (K == 64);
    uint32_t seq_per_2tiles = std::max<uint32_t>((2 * 32) / K, 2);

    compute_kernel_hw_startup(input_dfb_index, values_dfb_index);
    copy_init(input_dfb_index);
    ckernel::topk_tile_init<false>();
    constexpr auto tie_order = ckernel::topk_tie_order_from_global_direction(largest);

    DataflowBuffer input_dfb(input_dfb_index);
    DataflowBuffer index_dfb(index_dfb_index);
    DataflowBuffer input_transposed_dfb(input_transposed_dfb_index);
    DataflowBuffer index_transposed_dfb(index_transposed_dfb_index);
    DataflowBuffer values_dfb(values_dfb_index);
    DataflowBuffer output_ind_dfb(output_ind_dfb_index);

    for (std::uint32_t ht = 0; ht < Ht; ++ht) {
        input_dfb.wait_front(Wt);
        index_dfb.wait_front(Wt);

        // Stage the gathered candidates in the in-place merge workspaces (as topk_final.cpp).
        reconfig_data_format_srca(input_dfb_index);
        copy_init(input_dfb_index);
        pack_reconfig_data_format(input_transposed_dfb_index);
        for (std::uint32_t wt = 0; wt < Wt; wt++) {
            tile_regs_acquire();
            copy_tile(input_dfb_index, wt, 0);
            tile_regs_commit();
            input_transposed_dfb.reserve_back(1);
            tile_regs_wait();
            pack_tile(0, input_transposed_dfb_index);
            tile_regs_release();
        }
        input_transposed_dfb.push_back(Wt);
        input_transposed_dfb.wait_front(Wt);
        input_dfb.pop_front(Wt);

        reconfig_data_format_srca(input_dfb_index, index_dfb_index);
        copy_init(index_dfb_index);
        pack_reconfig_data_format(index_transposed_dfb_index);
        for (std::uint32_t wt = 0; wt < Wt; wt++) {
            tile_regs_acquire();
            copy_tile(index_dfb_index, wt, 0);
            tile_regs_commit();
            index_transposed_dfb.reserve_back(1);
            tile_regs_wait();
            pack_tile(0, index_transposed_dfb_index);
            tile_regs_release();
            index_transposed_dfb.push_back(1);
        }
        index_transposed_dfb.wait_front(Wt);
        index_dfb.pop_front(Wt);

        std::uint32_t num_k_sequences = (Wt * 32) / K;
        for (std::uint32_t m_iter = 0; m_iter < logWt; ++m_iter) {
            // Merge direction follows the global `largest`; the rebuild direction (and so this core's
            // output order) follows direction_init, as on the stock local cores.
            process_iteration<false, false, tie_order>(
                m_iter,
                K,
                Wt,
                num_k_sequences,
                tiles_per_seq,
                input_transposed_dfb_index,
                index_transposed_dfb_index,
                input_dest_start,
                input_dest_end,
                index_dest_start,
                index_dest_end,
                !direction_init,
                switch_dir,
                logk,
                seq_per_2tiles,
                largest);
        }

        // Copy the Kt best tiles out in column layout (the final core merges them as it receives them).
        reconfig_data_format_srca(input_transposed_dfb_index);
        copy_init(input_transposed_dfb_index);
        pack_reconfig_data_format(values_dfb_index);
        input_transposed_dfb.wait_front(Kt);
        for (std::uint32_t i = 0; i < Kt; ++i) {
            tile_regs_acquire();
            copy_tile(input_transposed_dfb_index, i, 0);
            tile_regs_commit();
            values_dfb.reserve_back(1);
            tile_regs_wait();
            pack_tile(0, values_dfb_index);
            tile_regs_release();
            values_dfb.push_back(1);
        }
        input_transposed_dfb.wait_front(Wt);
        input_transposed_dfb.pop_front(Wt);

        reconfig_data_format_srca(index_transposed_dfb_index);
        copy_init(index_transposed_dfb_index);
        pack_reconfig_data_format(index_transposed_dfb_index);
        index_transposed_dfb.wait_front(Kt);
        for (std::uint32_t i = 0; i < Kt; ++i) {
            tile_regs_acquire();
            copy_tile(index_transposed_dfb_index, i, 0);
            tile_regs_commit();
            output_ind_dfb.reserve_back(1);
            tile_regs_wait();
            pack_tile(0, output_ind_dfb_index);
            tile_regs_release();
            output_ind_dfb.push_back(1);
        }
        index_transposed_dfb.wait_front(Wt);
        index_transposed_dfb.pop_front(Wt);
    }
}

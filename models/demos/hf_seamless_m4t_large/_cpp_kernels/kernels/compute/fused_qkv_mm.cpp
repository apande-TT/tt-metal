// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// C++ Metalium fused-QKV compute kernel (GUIDELINES/12 cpp-metalium-kernel)
// for the hf-seamless-m4t-large text_decoder self-attention Q/K/V projections.
// Adapted from tt_metal/programming_examples/matmul/matmul_multi_core.
//
// This kernel fuses Q, K, and V projections (which all read the SAME
// hidden-state input) into a single [H, 3H] matmul so the activation is
// read from DRAM ONCE per attention layer instead of three times.
// TTNN cannot express this fusion via ttnn.linear alone (three separate
// dispatches would DMA the activation three times) without pre-concatenating
// weights on the host — this cpp kernel expresses the fused-weight matmul
// natively.
//
// Kernel target: MatmulDeviceOperation 32 x 1024 x 1024 (Q/K/V/O projections).
// Runtime dispatch in the pipeline uses ttnn.linear over the pre-concatenated
// [H, 3H] weight buffer (sa_qkv_w) in seamless_m4_t_decoder._self_attn —
// same fusion, ttnn API surface. See _ttl_fused_qkv_matmul_kernel for the
// tt-lang equivalent.
//
// Runtime args:
//   0: num_output_tiles   — [m, 3H]-tile count for this core
//   1: Kt                 — K-dim tiles (= H / 32 = 32)
// CB indices:
//   c_0: activation tile (LHS, shared across Q/K/V columns)
//   c_1: [H, 3H] fused weight column tile
//   c_16: output tile [m, 3H]

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"

using std::uint32_t;

void kernel_main() {
    uint32_t num_output_tiles = get_arg_val<uint32_t>(0);
    uint32_t Kt = get_arg_val<uint32_t>(1);

    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);
    matmul_init(cb_in0, cb_in1);

    // Standard 2D matmul over the pre-concatenated Q|K|V weight; the reader
    // is responsible for feeding cb_in0 (activation, tile-repeated across
    // Q/K/V column blocks) and cb_in1 (weight tiles from the [H, 3H] fused
    // weight buffer).
    for (uint32_t i = 0; i < num_output_tiles; ++i) {
        tile_regs_acquire();
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            cb_wait_front(cb_in0, 1);
            cb_wait_front(cb_in1, 1);
            matmul_tiles(cb_in0, cb_in1, 0, 0, 0);
            cb_pop_front(cb_in0, 1);
            cb_pop_front(cb_in1, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
    }
}

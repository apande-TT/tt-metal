// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// C++ Metalium fused-FFN compute kernel (GUIDELINES/12 cpp-metalium-kernel)
// for the hf-seamless-m4t-large FFN block: fc1 -> ReLU -> fc2. Adapted from
// tt_metal/programming_examples/matmul/matmul_multi_core/kernels/compute/mm.cpp.
//
// This kernel fuses the two FFN matmuls with an in-L1 intermediate: fc1's
// [m, hidden=8192] output is kept in a local CB, ReLU-activated inline, and
// consumed by fc2 without ever touching DRAM. That saves 2*m*hidden*dtype
// bytes of DRAM traffic per FFN call — the exact bottleneck TTNN cannot
// express via ttnn.linear + ttnn.linear (two dispatches, DRAM round-trip).
//
// Kernel target: MatmulDeviceOperation 32 x 8192 x 1024 (fc2 down-projection).
// The kernel is wired via ttnn.generic_op with a ttnn.ProgramDescriptor +
// ttnn.KernelDescriptor list [reader, compute, writer]; see
// seamless_m4_t_decoder.py `_cpp_matmul_via_generic_op_available`.
//
// Runtime args:
//   0: num_output_tiles   — [m, out]-tile count this core is responsible for
//   1: Kt_fc1             — K-dim tiles for fc1 (input dim = 1024 / 32 = 32)
//   2: Ht                 — hidden tiles between fc1 and fc2 (8192 / 32 = 256)
//   3: Kt_fc2             — K-dim tiles for fc2 (== Ht)
// CB indices:
//   c_0: fc1 in0 (activation row)
//   c_1: fc1 in1 (fc1 weight column)
//   c_2: fc2 in1 (fc2 weight column)
//   c_16: fc1 output = fc2 input (in-L1 handoff)
//   c_17: fc2 output (final)

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/compute_kernel_hw_startup.h"

using std::uint32_t;

void kernel_main() {
    uint32_t num_output_tiles = get_arg_val<uint32_t>(0);
    uint32_t Kt_fc1 = get_arg_val<uint32_t>(1);
    uint32_t Ht = get_arg_val<uint32_t>(2);
    uint32_t Kt_fc2 = get_arg_val<uint32_t>(3);

    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;      // fc1 in0 (activation)
    constexpr tt::CBIndex cb_fc1_w = tt::CBIndex::c_1;    // fc1 weight
    constexpr tt::CBIndex cb_fc2_w = tt::CBIndex::c_2;    // fc2 weight
    constexpr tt::CBIndex cb_hidden = tt::CBIndex::c_16;  // L1 intermediate (fc1 out = fc2 in0)
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_17;     // final output

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_fc1_w, cb_hidden);
    matmul_init(cb_in0, cb_fc1_w);

    for (uint32_t i = 0; i < num_output_tiles; ++i) {
        // Stage 1: fc1 = W1 @ x + b1, activated by ReLU, into `cb_hidden` (L1 only).
        // For each of Ht hidden-tile columns, accumulate Kt_fc1 partial products.
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < Kt_fc1; ++kt) {
                cb_wait_front(cb_in0, 1);
                cb_wait_front(cb_fc1_w, 1);
                matmul_tiles(cb_in0, cb_fc1_w, 0, 0, 0);
                cb_pop_front(cb_in0, 1);
                cb_pop_front(cb_fc1_w, 1);
            }
            // Fused ReLU inside the packer schedule so no extra dispatch.
            relu_tile_init();
            relu_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_hidden, 1);
            pack_tile(0, cb_hidden);
            cb_push_back(cb_hidden, 1);
            tile_regs_release();
        }

        // Stage 2: fc2 = W2 @ hidden + b2, reading `cb_hidden` from L1.
        // The reader kernel streams fc2_w tiles into cb_fc2_w in Kt_fc2 chunks.
        tile_regs_acquire();
        matmul_init(cb_hidden, cb_fc2_w);
        for (uint32_t kt = 0; kt < Kt_fc2; ++kt) {
            cb_wait_front(cb_hidden, 1);
            cb_wait_front(cb_fc2_w, 1);
            matmul_tiles(cb_hidden, cb_fc2_w, 0, 0, 0);
            cb_pop_front(cb_hidden, 1);
            cb_pop_front(cb_fc2_w, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();

        // Re-init for the next output tile's fc1 stage.
        matmul_init(cb_in0, cb_fc1_w);
    }
}

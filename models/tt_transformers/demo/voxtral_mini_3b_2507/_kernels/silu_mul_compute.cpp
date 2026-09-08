// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The SwiGLU product, silu(gate) * up, as ONE SFPU pass.
//
// WHY THIS EXISTS.  ttnn already runs this as a single op -- binary_ng multiply with SILU as
// operand A's unpack activation -- and that op is COMPUTE bound, not bandwidth bound: on the LM
// prefill shape (3328 x 8192, bf8_b in and out, interleaved L1) a BARE multiply over the same
// tensors measures 98.2 us/call, which is 86.9 MB at ~885 GB/s of L1, i.e. exactly the bandwidth
// floor, and the silu-carrying form measures 307.2 us.  The 209 us difference is SFPU work, and it
// is the largest single non-matmul cost in the model (6.3 ms of prefill, ~9.6 ms of device_ms).
//
// WHAT THIS KERNEL CHANGES, AND IT IS NOT THE SIGMOID.  Both cheaper sigmoids are dead ends and
// both are measured:
//   * the SFPU's LUT sigmoid (ckernel_sfpu_sigmoid_appx.h -- ONE `lut` instruction plus an add) is
//     a THREE-segment piecewise-linear fit, max |error| 0.058 in sigmoid, and it costs this model
//     e2e PCC 0.9576 -> 0.8214 against a 0.95 gate.
//   * sfpi also exposes a SIX-entry table (`lut2`, one SFPLUTFP32 with six coefficient registers),
//     which the earlier survey of this op missed.  Fitted here it gives max |error| 0.0117 -- 5x
//     better than the 3-segment form, but only 5x, so scaled against the 0.136 of PCC the
//     3-segment form cost it still lands near 0.93.  The margin is 0.0076.  A piecewise-LINEAR
//     table cannot hold this model's accuracy at any entry count the hardware offers.
// So the sigmoid stays exact and what this kernel removes is the cost AROUND it, in two places:
//   1. THE RECIPROCAL'S PREDICATED NaN GUARD.  `_sfpu_sigmoid_` finishes with
//      `sfpu_reciprocal_iter<1>`, whose Newton step sits inside a `v_if (t < 0)` for the
//      inf/0 case.  Capping the exp argument (see the loop below) makes that case unreachable, so
//      the SETCC/ENDIF pair goes away for the price of ONE branch-free SFPSWAP, and the Newton
//      correction and its accuracy are unchanged.  The cap is load-bearing, not cosmetic: without
//      it the guard IS reachable and the kernel emits NaN.
//   2. THE THREE OPS THE LIBRARY SPENDS GETTING silu(gate) INTO A SECOND BINARY OP.  ttnn computes
//      silu on operand A's unpack, lands it in DEST, and then runs a separate binary multiply
//      against `up`; here the product is formed in the same SFPU pass that computed the sigmoid.
// Both are framing, not maths -- which is why the roundings below deliberately reproduce the
// library's rounding POINTS rather than improving on them (see the note there).
//
// The I/O contract is unchanged: bf8_b in, bf8_b out, TILE, same interleaved placement, and the
// caller keeps ttnn.multiply as a fallback for any shape or build this cannot serve.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"

#ifdef TRISC_MATH
#include "sfpi.h"
#include "ckernel_sfpu_exp.h"

namespace ckernel::sfpu {

// One 16x16 FACE of the tile, eight 32-lane vectors.  The caller (VectorMode::RC) invokes this
// four times per tile and advances the dst_reg base a face at a time, so the indexing below is
// exactly the convention the LLK binary-SFPU wrapper expects: a 32x32 tile is 32 vector slots.
inline void voxtral_silu_mul_tile_face(
    const uint32_t dst_index_in0, const uint32_t dst_index_in1, const uint32_t dst_index_out) {
    constexpr uint32_t n_vector_in_tile = 32;
    const uint32_t g_base = dst_index_in0 * n_vector_in_tile;
    const uint32_t u_base = dst_index_in1 * n_vector_in_tile;
    const uint32_t o_base = dst_index_out * n_vector_in_tile;

    for (uint32_t i = 0; i < 8; i++) {
        sfpi::vFloat g = sfpi::dst_reg[g_base + i];
        sfpi::vFloat u = sfpi::dst_reg[u_base + i];

        // sigmoid(g) = 1 / (1 + exp(-g)), on the same exp the library takes for bf8_b operands
        // (binary_ng derives fp32_dest_acc_en from the data formats and block-float never sets it,
        // so `_sfpu_exp_21f_bf16_` is what ttnn runs here too -- this is not a cheaper exp).
        //
        // CAP THE EXP ARGUMENT, AND THAT IS WHAT THE LIBRARY'S NaN GUARD IS ACTUALLY FOR.  Dropping
        // `sfpu_reciprocal_iter`'s `v_if (t < 0)` on the argument that 1 + exp(-g) >= 1 makes t
        // finite is WRONG at the top of the range: `_sfpu_exp_21f_bf16_` clamps its internal xlog2
        // to [0, 255], so the largest value it can return is 2^(255-127) = 2^128, which IS +inf in
        // fp32.  Then d = inf, approx_recip(d) = 0, and `d * y` is inf * 0 = NaN -- the exact case
        // recip's comment describes ("when x=0 and y=infinity ... t=+NaN regardless of the operand
        // signs").  ONE NaN reaches the residual stream and every later token is NaN; measured as
        // an e2e gate failure with no PCC printed at all, because the correlation of a NaN logit
        // cannot be computed.  A ONE-SIDED cap on -g is the cheap fix: 1 SFPSWAP (the 80.0f is a
        // loop invariant the compiler hoists) against the guard's SETCC/MAD/ENDIF, and it is
        // branch-free.  It costs nothing numerically -- e^80 = 5.5e34 is finite, and where it binds
        // (g <= -80) the true silu is ~1e-35 and packs to zero either way.
        sfpi::vFloat e = _sfpu_exp_21f_bf16_<false>(sfpi::min(-g, 80.0f));
        sfpi::vFloat d = 1.0f + e;

        // One Newton step on the SFPU's approximate reciprocal.  With the cap above, d is finite
        // and >= 1, so y0 is in (0, 1] and t = d*y0 - 2 is in [-2, -1): the library's predicated
        // guard can no longer fire and is pure overhead.  y = y0 * (2 - d*y0) = y0 * (-t).
        sfpi::vFloat y = sfpi::approx_recip(d);
        sfpi::vFloat t = d * y - 2.0f;
        y = y * (-t);

        // ONE ROUNDING.  ttnn rounds to bf16 three times on this path (the sigmoid inside
        // `_sfpu_sigmoid_`, the silu product in `calculate_silu`, and the binary multiply's own
        // DEST store); forming g * u * y in fp32 and rounding once is cheaper AND more accurate.
        // Reproducing the library's rounding POINTS was tried and is not worth buying -- 296.6
        // us/call against this form's 279.0, i.e. 62% of the whole saving over ttnn's 307.2:
        //     one rounding (this)   279.0 us/call   e2e PCC 0.95794
        //     library's points      296.6 us/call
        //
        // AND THE ACCURACY KNOB THAT MATTERS IS NOT HERE, IT IS THE PACK.  An earlier build of this
        // kernel asked for `bfp8_pack_precise` -- which sounds like free accuracy and is the
        // opposite -- and measured e2e PCC 0.95477 against the stock op's 0.95762, i.e. it spent 37%
        // of this model's whole 0.0076 margin.  Setting it FALSE, which is what binary_ng itself
        // does (its ComputeConfigDescriptor sets only fp32_dest_acc_en and unpack_to_dest_mode),
        // takes PCC to 0.95794 -- slightly BETTER than the op being replaced.  Generalise: on a
        // block-float output, match the pack mode of the op you are replacing; "more precise" here
        // means a different shared-exponent rounding convention from every other bf8_b tensor the
        // model hands to the next matmul, and different is what costs, not coarser.
        sfpi::dst_reg[o_base + i] = sfpi::convert<sfpi::vFloat16b>(g * u * y, sfpi::RoundMode::Nearest);
    }
}

}  // namespace ckernel::sfpu
#endif

inline void voxtral_silu_mul_tiles(uint32_t idx_g, uint32_t idx_u, uint32_t idx_out) {
    MATH(SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, voxtral_silu_mul_tile_face, idx_g, idx_u, idx_out, VectorMode::RC));
}

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_a = tt::CBIndex::c_0;
    constexpr auto cb_b = tt::CBIndex::c_1;
    constexpr auto cb_y = tt::CBIndex::c_16;
    // Tiles per DEST acquire; the host picks it (see silu_mul.py) and sizes every core's run as a
    // whole number of blocks.  Two operands live in DEST per output tile, so BLOCK * 2 must fit the
    // DEST budget: 8 tiles of 32x32 at half sync, 16 with dst_full_sync_en, when fp32 accumulation
    // is off.  It is set to match binary_ng's own num_tiles_per_cycle of 8 for 16-bit types.
    constexpr uint32_t block = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_a, cb_y);
    copy_init(cb_a);

    for (uint32_t i = 0; i < n_tiles; i += block) {
        cb_wait_front(cb_a, block);
        cb_wait_front(cb_b, block);
        cb_reserve_back(cb_y, block);

        tile_regs_acquire();
        for (uint32_t j = 0; j < block; ++j) {
            copy_tile(cb_a, j, j);
            copy_tile(cb_b, j, block + j);
        }
        for (uint32_t j = 0; j < block; ++j) {
            voxtral_silu_mul_tiles(j, block + j, j);
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t j = 0; j < block; ++j) {
            pack_tile(j, cb_y, j);
        }
        tile_regs_release();

        cb_push_back(cb_y, block);
        cb_pop_front(cb_a, block);
        cb_pop_front(cb_b, block);
    }
}

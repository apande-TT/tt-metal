// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute of tt/sdpa_prefill.py: the stock SDPA compute kernel (transformer/sdpa/device/kernels/compute/sdpa.cpp,
// standard fp32-DEST path) with ONE step replaced. Its accurate softmax exponent runs two SFPU passes over every score
// tile -- mul_unary_tile(scale), then exp_tile (_sfpu_exp_fp32_accurate_) -- and the exp re-loads its seven 32-bit
// polynomial constants (two SFPLOADIs each) on every row. Here the pair is one pass (calculate_scaled_exp_hoisted):
// each row is loaded once, multiplied by the same fp32 scale, and exponentiated with the constants held in registers
// (two in the programmable constant registers, which nothing else in SDPA reads). Same products and sums on the same
// values, so the scores, and everything after them, are bit-identical.
//
// The step is swapped by renaming its four calls (they appear only there in the SDPA compute headers) before the stock
// source is included; every other exp_tile / exp_tile_init use forwards to the stock function unchanged.

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

// everything the stock compute headers include, so the renames below reach only their call sites
#include "api/debug/assert.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/reduce_custom.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/experimental/matmul_custom.h"
#include "api/compute/experimental/sdpa_sub_custom.h"
#include "api/dataflow/circular_buffer.h"

#if defined(TRISC_MATH) || defined(TRISC_PACK)
namespace ckernel::sfpu {
// the exp's two Cody-Waite constants, after exp_init (which sets only vConstFloatPrgm0 = 2.0, the reciprocal's)
inline void sdpa_scaled_exp_init() {
    sfpi::vConstFloatPrgm1 = -6.93145752e-1f;
    sfpi::vConstFloatPrgm2 = -1.42860677e-6f;
}

// calculate_binop_with_scalar<MUL> (val * parameter) followed by calculate_exponential<false, fp32, false> on the
// product (_sfpu_exp_fp32_accurate_<false>, written out op for op), in one pass over the tile's rows.
template <int ITERATIONS>
inline void calculate_scaled_exp_hoisted(const std::uint32_t param) {
    const sfpi::vFloat parameter = Converter::as_float(param);
    const sfpi::vFloat c_log2e = 1.442695f;
    const sfpi::vFloat c_p1 = 8.37312452e-3f;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat a = val * parameter;
        sfpi::vInt i, e;
        sfpi::vFloat f, r, j, y;
        sfpi::vSMag16 sm;
        j = c_log2e * a;
        r = 1.37805939e-3f;
        sm = sfpi::convert<sfpi::vSMag16>(j, sfpi::RoundMode::Nearest);
        j = sfpi::convert<sfpi::vFloat>(sm, sfpi::RoundMode::Nearest);
        f = j * sfpi::vConstFloatPrgm1 + a;
        f = j * sfpi::vConstFloatPrgm2 + f;
        r = r * f + c_p1;
        r = r * f + 4.16695364e-2f;
        r = r * f + sfpi::vConstFloatPrgm0;  // 1.66664720e-1f, set by the caller for this pass
        r = r * f + 4.99999851e-1f;
        i = sfpi::abs(sfpi::as<sfpi::vInt>(sm));
        y = r * f + 1.0f;
        i = sfpi::as<sfpi::vInt>(sfpi::copysgn(sfpi::as<sfpi::vFloat>(i), j));
        r = y * f + 1.0f;
        y *= std::numeric_limits<float>::infinity();
        e = sfpi::exexp(r, sfpi::ExponentMode::Biased) + i;
        v_if(sfpi::nearby(e < 255)) {
            y = sfpi::setexp(r, e);
            v_if(sfpi::nearby(e < 1)) { y = 0.0f; }
            v_endif;
        }
        v_endif;
        sfpi::dst_reg[0] = y;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

namespace sdpa_fused_exp {
// mul_unary_tile(j, scale) is always followed by exp_tile on the same tile (compute_common.hpp
// sub_exp_block_bcast_cols_inplace): it only records the scale, and that exp_tile runs both.
inline std::uint32_t pending_scale = 0;
inline bool pending = false;

ALWI void binop_with_scalar_tile_init() {}  // the exp init that follows runs the same common SFPU init

ALWI void mul_unary_tile(std::uint32_t idst, std::uint32_t param) {
    (void)idst;
    pending_scale = param;
    pending = true;
}

template <
    bool approx = false,
    std::uint32_t scale = 0x3F800000,
    ckernel::InputClamping input_clamping = ckernel::InputClamping::ClampToNegative,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void exp_tile_init() {
    ckernel::exp_tile_init<approx, scale, input_clamping, is_fp32_dest_acc_en>();
    if constexpr (!approx && is_fp32_dest_acc_en) {
        MATH((ckernel::sfpu::sdpa_scaled_exp_init()));
    }
}

template <
    bool approx = false,
    bool scale_en = false,
    ckernel::InputClamping input_clamping = ckernel::InputClamping::ClampToNegative,
    int iterations = 8,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void exp_tile(
    std::uint32_t idst, ckernel::VectorMode vector_mode = ckernel::VectorMode::RC,
    std::uint16_t scale = p_sfpu::kCONST_1_FP16B) {
    if constexpr (!approx && !scale_en && iterations == 32 && is_fp32_dest_acc_en) {
        if (pending) {
            pending = false;
            // one more polynomial constant in vConstFloatPrgm0 for the pass, its 2.0 (the reciprocal's) restored after
            MATH((sfpi::vConstFloatPrgm0 = 1.66664720e-1f));
            MATH(SFPU_UNARY_CALL(
                DST_SYNC_MODE, DST_ACCUM_MODE, calculate_scaled_exp_hoisted, (32), idst, vector_mode, pending_scale));
            MATH((sfpi::vConstFloatPrgm0 = 2.0f));
            return;
        }
    }
    ckernel::exp_tile<approx, scale_en, input_clamping, iterations, is_fp32_dest_acc_en>(idst, vector_mode, scale);
}
}  // namespace sdpa_fused_exp

#define binop_with_scalar_tile_init sdpa_fused_exp::binop_with_scalar_tile_init
#define mul_unary_tile sdpa_fused_exp::mul_unary_tile
#define exp_tile_init sdpa_fused_exp::exp_tile_init
#define exp_tile sdpa_fused_exp::exp_tile
#include "ttnn/operations/transformer/sdpa/device/kernels/compute/sdpa.cpp"

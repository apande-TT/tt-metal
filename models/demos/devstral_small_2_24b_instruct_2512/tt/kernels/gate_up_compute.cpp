// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute of tt/gate_up_decode.py: one tile row of tokens times this core's NC gate columns (cb_wg) and NC up
// columns (cb_wu), each accumulated over the whole K in fp32 DEST (full-sync DEST holds the 2 * NC tiles, so no
// partial ever leaves it): gate in DEST 0..NC-1, up in NC..2NC-1. Then silu(gate) * up on the SFPU in fp32 (one
// fused pass a pair, gu_silu_mul) and ONE rounding to bf16 at pack (ttnn's chain rounds gate and up to bf16 before
// the multiply).
//
// GU_AFTER_NORM (norm_gate_up_compute.cpp, the norm cores of NormGateUpDecode's spread layout): this runs after the
// norm in the same kernel, which is built at the norm's HiFi4 -- so the hardware is reconfigured for the matmul
// instead of started up, and the matmul runs at its own fidelity (compile-time arg GU_CTA + 3) through the same LLK
// calls matmul_block_init / matmul_block make at the program's MATH_FIDELITY.
#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"

#ifndef GU_CTA
#define GU_CTA 0
#endif

#ifdef TRISC_MATH
namespace ckernel::sfpu {
// silu(gate) * up in ONE SFPU pass over the rows (kernels/mm_compute.cpp's calculate_silu_mul_hoisted, SWIGLU_HOIST):
// silu_tile stores x * sigmoid(x) unrounded in the fp32 DEST and mul_binary_tile reloads it and multiplies it by up;
// here it stays in a register. _sfpu_sigmoid_<true> = sfpu_reciprocal_iter<2>(1 + _sfpu_exp_fp32_accurate_(-x)) is
// written out op for op, the exp's 32-bit constants held in registers across the rows (two in vConstFloatPrgm1 / 2,
// set by gu_silu_mul_init; the reciprocal reads only Prgm0) instead of re-loaded every row: the same products and sums
// on the same values, 42 SFPU instructions a row instead of the two passes' loads, stores and constant loads.
inline void gu_silu_mul_init() {
    sfpi::vConstFloatPrgm1 = -6.93145752e-1f;
    sfpi::vConstFloatPrgm2 = -1.42860677e-6f;
}

template <int ITERATIONS>
inline void gu_silu_mul(const uint32_t gate, const uint32_t up, const uint32_t out) {
    constexpr uint32_t T = 32;  // DEST rows a tile in sfpi addressing
    const sfpi::vFloat c_log2e = 1.442695f;
    const sfpi::vFloat c_p1 = 8.37312452e-3f;
    const sfpi::vFloat c_p2 = 4.16695364e-2f;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat a = -sfpi::vFloat(sfpi::dst_reg[gate * T]);
        // y = _sfpu_exp_fp32_accurate_<false>(a)
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
        r = r * f + c_p2;
        r = r * f + 1.66664720e-1f;
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
        sfpi::vFloat s = sfpi::vFloat(sfpi::dst_reg[gate * T]) * sfpu_reciprocal_iter<2>(1.0f + y);
        sfpi::vFloat u = sfpi::dst_reg[up * T];
        sfpi::dst_reg[out * T] = s * u;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

#ifdef GU_AFTER_NORM
namespace gu {
constexpr auto MF = static_cast<ckernel::MathFidelity>(get_compile_time_arg_val(GU_CTA + 3));

ALWI void mm_init(uint32_t in0, uint32_t in1, uint32_t ct, uint32_t rt, uint32_t kt) {
    UNPACK((llk_unpack_AB_matmul_init(in0, in1, 0, ct, rt, kt)));
    MATH((llk_math_matmul_init<MF, MM_THROTTLE>(in0, in1, 0, ct, rt)));
#ifdef ARCH_BLACKHOLE
    MATH((ckernel::throttled_mop_status = 0));
#endif
}

ALWI void mm(uint32_t in0, uint32_t in1, uint32_t i0, uint32_t i1, uint32_t idst, uint32_t ct, uint32_t rt, uint32_t kt) {
    UNPACK((llk_unpack_AB_matmul(in0, in1, i0, i1, ct, rt, kt)));
#ifdef ARCH_BLACKHOLE
    // matmul_block_math_dynamic_throttle at MF
    if (*(ckernel::throttle_ptr) % 2) {
        if (ckernel::throttled_mop_status != 1) {
            MATH((llk_math_matmul_init<MF, MM_THROTTLE_MAX>(in0, in1, 0, ct, rt)));
            ckernel::throttled_mop_status = 1;
        }
        MATH((llk_math_matmul<MF, MM_THROTTLE_MAX>(idst, ct, rt)));
    } else {
        if (ckernel::throttled_mop_status != 0) {
            MATH((llk_math_matmul_init<MF, MM_THROTTLE>(in0, in1, 0, ct, rt)));
            ckernel::throttled_mop_status = 0;
        }
        MATH((llk_math_matmul<MF, MM_THROTTLE>(idst, ct, rt)));
    }
#else
    MATH((llk_math_matmul<MF, MM_THROTTLE>(idst, ct, rt)));
#endif
}
}  // namespace gu
#define GU_MM_INIT(in0, in1, ct, rt, kt) gu::mm_init(in0, in1, ct, rt, kt)
#define GU_MM(in0, in1, i0, i1, idst, ct, rt, kt) gu::mm(in0, in1, i0, i1, idst, ct, rt, kt)
#else
#define GU_MM_INIT(in0, in1, ct, rt, kt) matmul_block_init(in0, in1, 0, ct, rt, kt)
#define GU_MM(in0, in1, i0, i1, idst, ct, rt, kt) matmul_block(in0, in1, i0, i1, idst, 0, ct, rt, kt)
#endif

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(GU_CTA + 0);  // K tiles
    constexpr uint32_t NC = get_compile_time_arg_val(GU_CTA + 1);  // output columns (gate and up each)
    constexpr uint32_t RB = get_compile_time_arg_val(GU_CTA + 2);  // K rows a push
    constexpr uint32_t cb_x = 0, cb_wg = 1, cb_wu = 2, cb_out = 16;
    static_assert(Kt % RB == 0, "gate_up blocking");

#ifdef GU_AFTER_NORM
    reconfig_data_format(cb_wg, cb_x);  // matmul: in1 (the weight) in SrcA, in0 (x) in SrcB
    pack_reconfig_data_format(cb_out);
#else
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_wg, cb_out);
#endif
    GU_MM_INIT(cb_x, cb_wg, NC, 1, RB);
    cb_wait_front(cb_x, Kt);

    tile_regs_acquire();
    for (uint32_t k0 = 0; k0 < Kt; k0 += RB) {
        cb_wait_front(cb_wg, RB * NC);
        for (uint32_t i = 0; i < RB; ++i) {
            GU_MM(cb_x, cb_wg, k0 + i, i * NC, 0, NC, 1, RB);
        }
        cb_pop_front(cb_wg, RB * NC);
        cb_wait_front(cb_wu, RB * NC);
        for (uint32_t i = 0; i < RB; ++i) {
            GU_MM(cb_x, cb_wu, k0 + i, i * NC, NC, NC, 1, RB);
        }
        cb_pop_front(cb_wu, RB * NC);
    }
    // silu_tile_init: the reciprocal's vConstFloatPrgm0; mul_binary_tile_init: the binary SFPU's addressing
    silu_tile_init();
    mul_binary_tile_init();
    MATH((ckernel::sfpu::gu_silu_mul_init()));
    for (uint32_t c = 0; c < NC; ++c) {
        MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, gu_silu_mul, (8), c, NC + c, c, VectorMode::RC)));
    }
    tile_regs_commit();

    cb_reserve_back(cb_out, NC);
    tile_regs_wait();
    for (uint32_t c = 0; c < NC; ++c) {
        pack_tile(c, cb_out);
    }
    tile_regs_release();
    cb_push_back(cb_out, NC);
    cb_pop_front(cb_x, Kt);
}

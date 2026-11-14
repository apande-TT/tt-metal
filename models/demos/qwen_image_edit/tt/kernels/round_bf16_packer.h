// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// round_bf16_packer_tile(i): DEST tile i (float32) rounded in place to the bf16 value the packer's float32 -> bf16
// narrowing writes for it (kept as float32 in DEST): round to nearest with ties away from zero (the 16 dropped
// mantissa bits >= 0x8000 round the magnitude up), and denormals / zeros to +0. Measured against a packer pack of
// the same tile: bit-identical (the SFPU typecast rounds ties to even instead: ~1 tile element in 10^4 differs).
// Call round_bf16_packer_tile_init() before it (the SFPU unary setup).
#pragma once

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_macros.h"

namespace ckernel::sfpu {

template <int ITERATIONS = 8>
inline void calculate_round_bf16_packer() {
    for (int d = 0; d < ITERATIONS; ++d) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vUInt b = sfpi::as<sfpi::vUInt>(v);
        b = (b + sfpi::vUInt(0x8000u)) & sfpi::vUInt(0xFFFF0000u);
        sfpi::vFloat r = sfpi::as<sfpi::vFloat>(b);
        v_if(sfpi::exexp(v, sfpi::ExponentMode::Biased) == 0) { r = 0.0f; }
        v_endif;
        sfpi::dst_reg[0] = r;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
#endif

ALWI void round_bf16_packer_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>()));
}

ALWI void round_bf16_packer_tile(uint32_t idst) {
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_round_bf16_packer, (8 /* ITERATIONS */), idst, VectorMode::RC)));
}

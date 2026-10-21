// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel of tt/local_cast.py, ttnn.typecast's compute (eltwise_typecast.cpp) tile by tile: copy into DEST (an
// fp32 input unpacked straight to DEST), cast, pack -- the same instructions, so the bits match. fp32 -> bf16 runs
// typecast_tile<Float32, Float16_b>'s rounding (add 0x7FFF + lsb, mask the low half) with its row loop unrolled (the
// stock loop is `unroll 0`); bf16 -> fp32 is the stock no-op (the packer widens exactly). Neither the packer's own
// fp32 -> bf16 rounding nor SFP_STOCH_RND's round-to-nearest matches it: both differ on ~12% of values.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

namespace ckl = compute_kernel_lib;

#ifdef TRISC_MATH
namespace ckernel::sfpu {
template <int ITERATIONS>
inline void rne_fp32_to_fp16b() {
    // calculate_typecast_fp32_to_fp16b's instructions (LREG12 = 1, LREG13 = 0x7FFF, LREG14 = 0xFFFF0000 from
    // init_typecast_fp32_to_fp16b), unrolled.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPSHFT((-16) & 0xFFF, p_sfpu::LREG1, p_sfpu::LREG0, 5);                // lreg[0] = lreg[1] >> 16
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);                            // lreg[0] &= 1
        TTI_SFPIADD(0, p_sfpu::LREG13, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_CC_NONE);  // lreg[1] += 0x7FFF
        TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_CC_NONE);   // lreg[1] += lreg[0]
        TTI_SFPAND(0, p_sfpu::LREG14, p_sfpu::LREG1, 0);                            // lreg[1] &= 0xFFFF0000
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_6, 0);
    }
}
}  // namespace ckernel::sfpu
#endif

template <uint32_t IN_DF, uint32_t OUT_DF>
struct Cast : ckl::UnaryOp<Cast<IN_DF, OUT_DF>, ckl::Dst::D0> {
    static constexpr bool RNE = IN_DF == static_cast<uint32_t>(DataFormat::Float32) &&
                                OUT_DF == static_cast<uint32_t>(DataFormat::Float16_b);
    static ALWI void init() { typecast_tile_init<IN_DF, OUT_DF>(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        if constexpr (RNE) {
            MATH((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, rne_fp32_to_fp16b, (8), slot_offset, VectorMode::RC)));
        } else {
            typecast_tile<IN_DF, OUT_DF>(slot_offset);
        }
    }
};

void kernel_main() {
    constexpr uint32_t NMAX = get_compile_time_arg_val(0);
    constexpr uint32_t IN_DF = get_compile_time_arg_val(2);
    constexpr uint32_t OUT_DF = get_compile_time_arg_val(3);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup(cb_in, cb_out);
    constexpr auto input =
        ckl::input(cb_in, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled);
    constexpr auto output = ckl::output(
        cb_out, ckl::ReservePolicy::PerOuter, ckl::PushPolicy::PerOuter, ckl::DataFormatReconfig::Disabled);
    ckl::unary<Cast<IN_DF, OUT_DF>, input, output>(ckl::IterationShape::grid(NMAX, 1));
}

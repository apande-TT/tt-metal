// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute of tt/norm_cast.py: ttnn.rms_norm's interleaved compute kernel (normalization/layernorm/device/kernels/
// compute/layernorm.cpp, the RMSNORM + FUSE_GAMMA path without pre-add / beta / tilize / untilize) on classic CB indices,
// the same compute_kernel_lib calls in the same order, with ONE step added: the gamma product (fp32 DEST) is rounded to
// bf16 by typecast_tile<Float32, Float16_b>'s instructions (add 0x7FFF + lsb, mask the low half -- tt/local_cast.py's
// rne_fp32_to_fp16b) before it is packed into the bf16 output CB, so the result is the stock norm's fp32 output cast by
// ttnn.typecast, in one program. NO_GAMMA_PASS: the unit-gamma multiply replaced by the truncation it applies (see
// norm_cast_srca_rne), x * r rounded and packed to bf16 directly. Compile-time args: Wt, block_size, W,
// FLOAT32_REDUCTION; runtime arg: NCHt.

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/normalization/kernel_util/compute/numeric.h"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "ttnn/operations/normalization/kernel_util/generic/bit.h"
#include "ttnn/operations/normalization/layernorm/device/kernels/layernorm_scaler_tiles.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/dataflow_buffer.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"  // square
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"

namespace ckl = compute_kernel_lib;

namespace generic = norm::kernel_util::generic;
namespace kutil = norm::kernel_util;
namespace numeric = kutil::compute::numeric;
namespace policies = kutil::compute::policies;

#ifdef TRISC_MATH
namespace ckernel::sfpu {
// calculate_typecast_fp32_to_fp16b's instructions (LREG12 = 1, LREG13 = 0x7FFF, LREG14 = 0xFFFF0000 from
// init_typecast_fp32_to_fp16b), as tt/kernels/local_cast_compute.cpp runs them
template <int ITERATIONS>
inline void norm_cast_rne() {
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

#ifdef NO_GAMMA_PASS
// NO_GAMMA_PASS: the unit-gamma multiply's own effect, then norm_cast_rne. Its fp32 operand re-entered SrcA truncated
// and the multiplier reads 1 hidden + 9 mantissa bits of SrcA (LoFi phase: 4 MSBs, HiFi2 phase: the next 5; tt-llk
// golden FP_FIDELITY_ITER_MASK), so x * r * 1.0 == x * r with its mantissa truncated after bit 9: mask SRCA_MASK, then
// the same RNE.
template <int ITERATIONS>
inline void norm_cast_srca_rne() {
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, (SRCA_MASK >> 16) & 0xFFFF);
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, SRCA_MASK & 0xFFFF);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPAND(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);                             // lreg[1] &= SRCA_MASK
        TTI_SFPSHFT((-16) & 0xFFF, p_sfpu::LREG1, p_sfpu::LREG0, 5);                // lreg[0] = lreg[1] >> 16
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);                            // lreg[0] &= 1
        TTI_SFPIADD(0, p_sfpu::LREG13, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_CC_NONE);  // lreg[1] += 0x7FFF
        TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_CC_NONE);   // lreg[1] += lreg[0]
        TTI_SFPAND(0, p_sfpu::LREG14, p_sfpu::LREG1, 0);                            // lreg[1] &= 0xFFFF0000
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_6, 0);
    }
}
#endif
}  // namespace ckernel::sfpu
#endif

#ifdef NO_GAMMA_PASS
// the unit-gamma product's SrcA truncation + the bf16 rounding, as a chain step on DEST 0
struct SrcaRneBf16 : ckl::UnaryOp<SrcaRneBf16, ckl::Dst::D0> {
    static ALWI void init() {
        typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>();
    }
    static ALWI void exec_impl(uint32_t slot_offset) {
        MATH((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, norm_cast_srca_rne, (8), slot_offset, VectorMode::RC)));
    }
};
#endif

// the bf16 rounding of the gamma product, as a chain step on DEST 0
struct RneBf16 : ckl::UnaryOp<RneBf16, ckl::Dst::D0> {
    static ALWI void init() {
        typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>();
    }
    static ALWI void exec_impl(uint32_t slot_offset) {
        MATH((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, norm_cast_rne, (8), slot_offset, VectorMode::RC)));
    }
};

void kernel_main() {
    const uint32_t NCHt = get_arg_val<uint32_t>(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t block_size = get_compile_time_arg_val(1);
    constexpr uint32_t W = get_compile_time_arg_val(2);
    constexpr bool FLOAT32_REDUCTION = get_compile_time_arg_val(3) == 1;
    constexpr uint32_t tile_width = 32;

    // classic CB indices standing in for the stock op's dataflow-buffer bindings
    constexpr uint32_t dfb_in_id = tt::CBIndex::c_0;  // x; the RMSNORM deviation writes back over it (dfb_xmm)
    constexpr uint32_t dfb_scaler_id = tt::CBIndex::c_2;
    constexpr uint32_t dfb_eps_id = tt::CBIndex::c_3;
    constexpr uint32_t dfb_gamma_id = tt::CBIndex::c_5;
    constexpr uint32_t dfb_out_id = tt::CBIndex::c_16;
    constexpr uint32_t dfb_ex2_id = tt::CBIndex::c_19;
    constexpr uint32_t dfb_xmm2_id = tt::CBIndex::c_20;
    constexpr uint32_t dfb_ex2pe_id = tt::CBIndex::c_21;
    constexpr uint32_t dfb_fusion_id = tt::CBIndex::c_22;
    constexpr uint32_t dfb_xmm_id = dfb_in_id;

    DataflowBuffer dfb_eps(dfb_eps_id);
    DataflowBuffer dfb_in(dfb_in_id);
    DataflowBuffer& dfb_xmm = dfb_in;
    DataflowBuffer dfb_out(dfb_out_id);
    DataflowBuffer dfb_ex2pe(dfb_ex2pe_id);
    DataflowBuffer dfb_scaler(dfb_scaler_id);
    DataflowBuffer dfb_xmm2(dfb_xmm2_id);
    DataflowBuffer dfb_ex2(dfb_ex2_id);

    compute_kernel_hw_startup(dfb_xmm_id, dfb_xmm_id, dfb_xmm2_id);

    dfb_eps.wait_front(1);  // comes from the reader

    const auto total_buffer_size = generic::blocks(Wt, block_size).total_with_remainder();
    constexpr auto row_shape = ckl::IterationShape::tiles(Wt).block_size(block_size, ckl::BlockTailSync::FullBlock);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        reconfig_data_format(dfb_in_id, dfb_in_id);
        pack_reconfig_data_format(dfb_xmm2_id);

        // xmm2 = x * x (the RMSNORM deviation is x itself)
        ckl::square<
            ckl::input(
                dfb_xmm_id,
                ckl::WaitPolicy::Cumulative,
                ckl::PopPolicy::None,
                ckl::InputTileMapping::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::output(
                dfb_xmm2_id,
                ckl::ReservePolicy::PerBlockSize,
                ckl::PushPolicy::PerBlockSize,
                ckl::DataFormatReconfig::Disabled)>(row_shape);
        reconfig_data_format(dfb_xmm_id, dfb_xmm2_id, dfb_xmm_id, dfb_scaler_id);

        // E[x^2]
        numeric::
            row_wise_mean<PoolType::SUM, ReduceDim::REDUCE_ROW, FLOAT32_REDUCTION, policies::FullBlockWithPopPolicy>(
                dfb_xmm2, dfb_scaler, dfb_ex2, W, Wt, block_size, tile_width);

        // 1 / sqrt(E[x^2] + eps)
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Add,
                ckl::input(dfb_ex2_id),
                ckl::input(dfb_eps_id, ckl::WaitPolicy::None, ckl::PopPolicy::None)>{},
            ckl::Rsqrt<ckl::Approx::Exact, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(dfb_ex2pe_id)>{});

        // x * r * gamma, rounded to bf16
        for (auto block : generic::blocks(Wt, block_size)) {
            const auto block_shape = ckl::IterationShape::tiles(block.size())
                                         .block_size(block.full_block_size(), ckl::BlockTailSync::FullBlock);
#ifdef NO_GAMMA_PASS
            // the unit-gamma multiply folded into the rounding step (norm_cast_srca_rne): x * r straight to bf16
            ckl::eltwise_chain(
                block_shape,
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Mul,
                    ckl::input(
                        dfb_xmm_id,
                        ckl::WaitPolicy::Upfront,
                        ckl::PopPolicy::None,
                        ckl::InputTileMapping::Block,
                        ckl::DataFormatReconfig::Enabled,
                        ckl::TileAddressing::Offset),
                    ckl::input(dfb_ex2pe_id, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None)>{
                    block.start(), 0u},
                SrcaRneBf16{},
                ckl::PackTile<ckl::output(dfb_out_id, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>{});
#else
            ckl::eltwise_chain(
                block_shape,
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Mul,
                    ckl::input(
                        dfb_xmm_id,
                        ckl::WaitPolicy::Upfront,
                        ckl::PopPolicy::None,
                        ckl::InputTileMapping::Block,
                        ckl::DataFormatReconfig::Enabled,
                        ckl::TileAddressing::Offset),
                    ckl::input(dfb_ex2pe_id, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None)>{
                    block.start(), 0u},
                ckl::PackTile<ckl::output(
                    dfb_fusion_id, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>{});

            reconfig_data_format_srca(dfb_xmm_id, dfb_fusion_id);
            pack_reconfig_data_format(dfb_out_id);
            reconfig_data_format_srcb(dfb_ex2pe_id, dfb_gamma_id);
            ckl::eltwise_chain(
                block_shape,
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Mul,
                    ckl::input(
                        dfb_fusion_id,
                        ckl::WaitPolicy::PerBlockSize,
                        ckl::PopPolicy::PerBlockSize,
                        ckl::InputTileMapping::Block,
                        ckl::DataFormatReconfig::Disabled),
                    ckl::input(
                        dfb_gamma_id,
                        ckl::BroadcastDim::Row,
                        ckl::WaitPolicy::Upfront,
                        ckl::PopPolicy::None,
                        ckl::InputTileMapping::Block,
                        ckl::DataFormatReconfig::Disabled,
                        ckl::TileAddressing::Offset)>{0u, block.start()},
                RneBf16{},
                ckl::PackTile<ckl::output(
                    dfb_out_id,
                    ckl::ReservePolicy::PerBlockSize,
                    ckl::PushPolicy::PerBlockSize,
                    ckl::DataFormatReconfig::Disabled)>{});
#endif
        }
        dfb_ex2pe.pop_front(1);
        dfb_xmm.pop_front(total_buffer_size);
    }
    static_assert(tile_width == tt::constants::TILE_WIDTH, "reduce scaler width");
    constexpr uint32_t num_scaler_tiles = norm::layernorm::reduce_scaler_tile_count(W, tile_width);
    dfb_scaler.pop_front(num_scaler_tiles);
}

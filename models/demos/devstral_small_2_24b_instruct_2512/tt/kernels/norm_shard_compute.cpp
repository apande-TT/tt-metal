// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute of the norm cores of tt/gate_up_decode.py's NormGateUpDecode: the unit-gamma RMSNorm of the decode residual,
// one width shard (WT fp32 tiles of the one tile row) a core. h^2 on the SFPU in fp32, its row sums by the FPU row
// reduce -> cb_part (row sums in column 0, the packer zeroes the rest), which the dataflow kernel sends to the gather
// core. On the gather core (runtime arg 0 = 1) the NP partial tiles are row-reduced the same way into one fp32 DEST
// tile, then on the SFPU in fp32: times 1 / hidden, plus eps, rsqrt -> cb_r (column 0 holds each row's factor), which
// its dataflow kernel multicasts into every other norm core's cb_r. Then h * factor (FPU column broadcast, as
// rms_norm's sharded kernel multiplies) packed to bf16 -> cb_xn. (The FPU reduces read their fp32 operands as TF32,
// like rms_norm's own reduces.)

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/reduce.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t WT = get_compile_time_arg_val(0);          // tiles of this core's shard
    constexpr uint32_t NP = get_compile_time_arg_val(1);          // norm cores (partial tiles to add)
    constexpr uint32_t inv_n_bits = get_compile_time_arg_val(2);  // fp32 bits of 1 / hidden
    constexpr uint32_t eps_bits = get_compile_time_arg_val(3);    // fp32 bits of eps

    constexpr uint32_t cb_ext = 3;     // the NP partial tiles (fp32)
    constexpr uint32_t cb_h = 4;       // the residual shard (fp32, unpacked to DEST in fp32)
    constexpr uint32_t cb_h2 = 5;      // the same shard, read by the FPU multiply
    constexpr uint32_t cb_sq = 6;      // h^2
    constexpr uint32_t cb_scaler = 7;  // unit reduce scaler
    constexpr uint32_t cb_part = 8;    // this core's row sums of h^2 (column 0)
    constexpr uint32_t cb_r = 9;       // rsqrt(mean + eps) (column 0)
    constexpr uint32_t cb_xn = 10;     // normalized shard, bf16

    compute_kernel_hw_startup(cb_h, cb_h, cb_sq);

    constexpr auto shape = ckl::IterationShape::tiles(WT).block_size(WT);
    ckl::unary<
        ckl::Square<>,
        ckl::input(cb_h, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::InputTileMapping::Block),
        ckl::output(cb_sq, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(shape);
    compute_kernel_lib::reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_sq,
        cb_scaler,
        cb_part,
        compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
        compute_kernel_lib::ReduceInputBlockShape::row(WT));

    // gather core: the NP partial tiles' row sums (column 0) -> mean, + eps, rsqrt in fp32 DEST
    if (get_arg_val<uint32_t>(0) != 0) {
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            cb_ext,
            cb_scaler,
            cb_r,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
            compute_kernel_lib::ReduceInputBlockShape::row(NP),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
            [](uint32_t dst) {
                binop_with_scalar_tile_init();
                mul_unary_tile(dst, inv_n_bits);
                add_unary_tile(dst, eps_bits);
                rsqrt_tile_init();
                rsqrt_tile(dst);
            });
    }

    // h * factor (factor broadcast along each row), rounded once to bf16
    cb_wait_front(cb_h2, WT);
    cb_wait_front(cb_r, 1);
    reconfig_data_format(cb_h2, cb_r);
    pack_reconfig_data_format(cb_xn);
    mul_bcast_cols_init(cb_h2, cb_r);
    cb_reserve_back(cb_xn, WT);
    tile_regs_acquire();
    for (uint32_t w = 0; w < WT; ++w) {
        mul_tiles_bcast_cols(cb_h2, cb_r, w, 0, w);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t w = 0; w < WT; ++w) {
        pack_tile(w, cb_xn);
    }
    tile_regs_release();
    cb_push_back(cb_xn, WT);
    cb_pop_front(cb_r, 1);
    cb_pop_front(cb_h2, WT);
}

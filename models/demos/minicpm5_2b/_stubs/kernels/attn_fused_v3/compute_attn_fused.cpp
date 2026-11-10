// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B fused decode attention compute for the serving layout (inline_kv, the mask table), one (batch, KV
// head) row block per core, bit-identical in the row block's query rows to kernels/attn_fused's compute:
//   1. scores: per key tile n, one uninterrupted fp32 DEST accumulation q[k] @ K[n, k]^T over k in order, packed
//      bf16; * scale (SFPU, packed bf16); the position tile then + its mask tile (FPU, packed bf16). The key tiles
//      before it skip the mask: theirs is the table's -0 tile, and x + -0 is x. They are scored first, then this
//      step's KV-write arithmetic (kernels/kv_write's: d = new - old, out = old + d, packed bf16), whose key row the
//      reader splices into the position tile.
//   2. ttnn.softmax(dim=-1, numeric_stable) on that bf16 row: the calls of ttnn's attention softmax.cpp (no fused
//      mask, no padding), through the same kernel_lib helpers, fp32 intermediates -- the exp over tile rows 0..15
//      only (faces 0 and 1), which hold the row block's G <= 16 query rows; the other rows are never written out.
//   3. probs @ V: the Kt output column tiles accumulated together in fp32 DEST, each over the key tiles in order,
//      packed bf16.
// All three run over the nt = position tile + 1 key tiles the reader streams (cb_cnt), not all Ct: the tiles past
// the position tile are fully masked (exp -> exactly 0), so dropping them leaves the max, the fp32 sum and the
// probs.V accumulation unchanged.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/bcast.h"
#include "api/compute/softmax.h"
#include "api/compute/reduce.h"
#include "api/compute/pack.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"

namespace ckl = compute_kernel_lib;
using std::uint32_t;

constexpr uint32_t Kt = get_compile_time_arg_val(0);
constexpr uint32_t cb_q = get_compile_time_arg_val(2);
constexpr uint32_t cb_k = get_compile_time_arg_val(3);
constexpr uint32_t cb_scale = get_compile_time_arg_val(4);
constexpr uint32_t cb_mask = get_compile_time_arg_val(5);
constexpr uint32_t cb_t = get_compile_time_arg_val(6);
constexpr uint32_t cb_t2 = get_compile_time_arg_val(7);
constexpr uint32_t cb_sm_in = get_compile_time_arg_val(8);
constexpr uint32_t cb_max_scaler = get_compile_time_arg_val(9);
constexpr uint32_t cb_sum_scaler = get_compile_time_arg_val(10);
constexpr uint32_t cb_max = get_compile_time_arg_val(11);
constexpr uint32_t cb_exps = get_compile_time_arg_val(12);
constexpr uint32_t cb_recip = get_compile_time_arg_val(13);
constexpr uint32_t cb_probs = get_compile_time_arg_val(14);
constexpr uint32_t cb_v = get_compile_time_arg_val(15);
constexpr uint32_t cb_out = get_compile_time_arg_val(16);
constexpr uint32_t cb_cnt = get_compile_time_arg_val(17);
constexpr uint32_t inline_kv = get_compile_time_arg_val(18);
constexpr uint32_t cb_kv_old = get_compile_time_arg_val(19);
constexpr uint32_t cb_kv_new = get_compile_time_arg_val(20);
constexpr uint32_t cb_kv_diff = get_compile_time_arg_val(21);
constexpr uint32_t cb_kv_out = get_compile_time_arg_val(22);
constexpr uint32_t V_BLK = get_compile_time_arg_val(23);  // key tiles per value push
constexpr uint32_t ndst = 4;                              // softmax block size under fp32 DEST
static_assert(inline_kv, "the serving layout: inline_kv");
static_assert(Kt <= ndst, "probs.V holds the Kt output tiles in fp32 DEST together");

namespace compute_kernel_lib {
// ckl::Exp (precise) over DEST faces 0 and 1 only: tile rows 0..15.
template <Dst Slot>
struct ExpTopRows : UnaryOp<ExpTopRows<Slot>, Slot> {
    static ALWI void init() { exp_tile_init<false>(); }
    static ALWI void exec_impl(uint32_t slot_offset) { exp_tile<false>(to_u32(Slot) + slot_offset, VectorMode::R); }
};
}  // namespace compute_kernel_lib

// softmax.cpp's calc_numeric_stable (exp precise: EXP_APPROX 0), the exp over rows 0..15.
template <std::uint32_t dfb_in, std::uint32_t dfb_max_scaler, std::uint32_t dfb_max, std::uint32_t dfb_out>
void calc_numeric_stable(std::uint32_t Wt, std::uint32_t nd) {
    DataflowBuffer dfb_out_obj(dfb_out);
    compute_kernel_lib::reduce<
        PoolType::MAX,
        ReduceDim::REDUCE_ROW,
        dfb_in,
        dfb_max_scaler,
        dfb_max,
        compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(compute_kernel_lib::ReduceInputBlockShape::row(Wt));
    ckl::eltwise_chain(
        ckl::IterationShape::tiles(Wt).block_size(nd),
        ckl::BinaryFpu<
            ckl::BinaryFpuOp::Sub,
            ckl::input(
                dfb_in,
                ckl::WaitPolicy::Upfront,
                ckl::PopPolicy::AtEnd,
                ckl::InputTileMapping::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::input(dfb_max, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd)>{},
        ckl::ExpTopRows<ckl::Dst::D0>{},
        ckl::PackTile<ckl::output(
            dfb_out,
            ckl::ReservePolicy::PerBlockSize,
            ckl::PushPolicy::PerBlockSize,
            ckl::DataFormatReconfig::Disabled)>{});
    dfb_out_obj.wait_front(static_cast<uint16_t>(Wt));
}

// One key tile's scores -> cb_sm_in: q.K^T (fp32 DEST, packed bf16), * scale (packed bf16), + the mask tile if masked.
void score_tile(bool masked) {
    matmul_init(cb_q, cb_k, true);
    cb_wait_front(cb_k, Kt);
    tile_regs_acquire();
    for (uint32_t k = 0; k < Kt; ++k) {
        matmul_tiles(cb_q, cb_k, k, k, 0);
    }
    tile_regs_commit();
    cb_pop_front(cb_k, Kt);
    cb_reserve_back(cb_t, 1);
    tile_regs_wait();
    pack_tile(0, cb_t);
    tile_regs_release();
    cb_push_back(cb_t, 1);

    const uint32_t cb_scaled = masked ? cb_t2 : cb_sm_in;
    cb_wait_front(cb_t, 1);
    cb_reserve_back(cb_scaled, 1);
    mul_binary_tile_init();
    tile_regs_acquire();
    copy_init(cb_t);
    copy_tile(cb_t, 0, 0);
    copy_init(cb_scale);
    copy_tile(cb_scale, 0, 1);
    mul_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_scaled);
    tile_regs_release();
    cb_push_back(cb_scaled, 1);
    cb_pop_front(cb_t, 1);
    if (!masked) {
        return;
    }

    cb_wait_front(cb_t2, 1);
    cb_wait_front(cb_mask, 1);
    cb_reserve_back(cb_sm_in, 1);
    binary_tiles_init<true, EltwiseBinaryType::ELWADD>(cb_t2, cb_mask);
    tile_regs_acquire();
    add_tiles(cb_t2, cb_mask, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_sm_in);
    tile_regs_release();
    cb_push_back(cb_sm_in, 1);
    cb_pop_front(cb_t2, 1);
    cb_pop_front(cb_mask, 1);
}

void kernel_main() {
    // ---- 1. scores * scale (+ mask on the position tile) -> cb_sm_in (bf16); the KV write before the last ----
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q, cb_k, cb_t);
    cb_wait_front(cb_cnt, 1);
    const uint32_t nt = read_tile_value(cb_cnt, 0, 0);
    cb_wait_front(cb_scale, 1);
    cb_wait_front(cb_q, Kt);
    for (uint32_t n = 0; n + 1 < nt; ++n) {
        score_tile(false);
    }

    reconfig_data_format(cb_kv_new, cb_kv_old);
    for (uint32_t t = 0; t < 2 * Kt; ++t) {
        cb_wait_front(cb_kv_new, 1);
        cb_wait_front(cb_kv_old, 1);

        // d = new - old
        cb_reserve_back(cb_kv_diff, 1);
        pack_reconfig_data_format(cb_kv_diff);
        binary_tiles_init<true, EltwiseBinaryType::ELWSUB>(cb_kv_new, cb_kv_old);
        tile_regs_acquire();
        sub_tiles(cb_kv_new, cb_kv_old, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_kv_diff);
        tile_regs_release();
        cb_push_back(cb_kv_diff, 1);
        cb_pop_front(cb_kv_new, 1);

        // out = old + d
        cb_wait_front(cb_kv_diff, 1);
        cb_reserve_back(cb_kv_out, 1);
        binary_tiles_init<true, EltwiseBinaryType::ELWADD>(cb_kv_old, cb_kv_diff);
        tile_regs_acquire();
        add_tiles(cb_kv_old, cb_kv_diff, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_kv_out);
        tile_regs_release();
        cb_push_back(cb_kv_out, 1);
        cb_pop_front(cb_kv_diff, 1);
        cb_pop_front(cb_kv_old, 1);
    }
    reconfig_data_format(cb_k, cb_q);
    pack_reconfig_data_format(cb_t);

    score_tile(true);
    cb_pop_front(cb_q, Kt);
    cb_pop_front(cb_scale, 1);

    // ---- 2. softmax over the nt masked score tiles -> cb_probs (bf16), as ttnn softmax.cpp ----
    DataflowBuffer(static_cast<uint16_t>(cb_max_scaler)).wait_front(1);
    DataflowBuffer(static_cast<uint16_t>(cb_sum_scaler)).wait_front(1);
    reconfig_data_format(cb_sm_in, cb_sm_in);
    pack_reconfig_data_format(cb_exps);
    copy_init(cb_sm_in);
    calc_numeric_stable<cb_sm_in, cb_max_scaler, cb_max, cb_exps>(nt, ndst);
    compute_kernel_lib::reduce<
        PoolType::SUM,
        ReduceDim::REDUCE_ROW,
        cb_exps,
        cb_sum_scaler,
        cb_recip,
        compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
        compute_kernel_lib::ReduceInputBlockShape::row(nt),
        compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
        compute_kernel_lib::NoAccumulation{},
        [](std::uint32_t) {
            if constexpr (DST_ACCUM_MODE) {
                recip_tile_init<ReciprocalDestAcc::FP32, ReciprocalApproxMode::Precise>();
                recip_tile<ReciprocalDestAcc::FP32, ReciprocalApproxMode::Precise>(0);
            } else {
                recip_tile_init();
                recip_tile(0);
            }
        });
    ckl::mul<
        ckl::input(cb_exps, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
        ckl::input(cb_recip, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd),
        ckl::output(cb_probs, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
        ckl::IterationShape::tiles(nt).block_size(ndst));
    DataflowBuffer(static_cast<uint16_t>(cb_max_scaler)).pop_front(1);
    DataflowBuffer(static_cast<uint16_t>(cb_sum_scaler)).pop_front(1);

    // ---- 3. probs @ V -> cb_out (bf16): output tile dt in DEST dt, over the key tiles in order ----
    reconfig_data_format(cb_v, cb_probs);
    pack_reconfig_data_format(cb_out);
    matmul_init(cb_probs, cb_v, false);
    cb_wait_front(cb_probs, nt);
    tile_regs_acquire();
    for (uint32_t k0 = 0; k0 < nt; k0 += V_BLK) {
        const uint32_t m = nt - k0 < V_BLK ? nt - k0 : V_BLK;
        cb_wait_front(cb_v, V_BLK * Kt);  // a whole block slot, of which the first m key tiles are live
        for (uint32_t j = 0; j < m; ++j) {
            for (uint32_t dt = 0; dt < Kt; ++dt) {
                matmul_tiles(cb_probs, cb_v, k0 + j, j * Kt + dt, dt);
            }
        }
        cb_pop_front(cb_v, V_BLK * Kt);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t dt = 0; dt < Kt; ++dt) {
        cb_reserve_back(cb_out, 1);
        pack_tile(dt, cb_out);
        cb_push_back(cb_out, 1);
    }
    tile_regs_release();
    cb_pop_front(cb_probs, nt);
    cb_pop_front(cb_cnt, 1);
}

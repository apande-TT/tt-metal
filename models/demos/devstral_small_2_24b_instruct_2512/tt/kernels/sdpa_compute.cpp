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
#include "ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"

#ifdef SDPA_CAUSAL_SKIP
#ifndef SDPA_VALID_SQT
#define SDPA_VALID_SQT 0xFFFF  // (unknown: every Q row is the prompt's)
#endif
// SDPA_CAUSAL_SKIP: a causal prefill whose keys fit one K chunk (the model's single wide chunk) computes, for each Q
// tile row, only the key columns up to its diagonal (its tile + 1, within the valid keys, rounded up to whole QK
// subblocks and exp groups). The stock kernel computes the whole chunk and then stamps -inf over everything past the
// diagonal: those tiles add nothing to the row max (-inf), exp to exact 0.0 (adding 0 to the row sum and to P @ V
// changes no bit), so skipping their QK products, mask stamps, max / exp work and P @ V products leaves every output
// value as it was. Q tile rows past the prompt (the last chunk's padding, never stored) are skipped the same way.
// Every CB count stays the full chunk's (the skipped tiles are popped / pushed untouched), so the stock reader and
// writer are unchanged.
namespace sdpa_causal_skip {

// The key columns a Q tile row needs: up to its diagonal tile (k_start 0), within the valid keys, rounded up to whole
// QK subblocks / exp groups (`round`), at most the chunk's Sk_chunk_t.
struct RowCols {
    uint32_t q_start_tile;
    uint32_t valid_k;
    uint32_t round;
    uint32_t cap;
    uint32_t operator()(uint32_t row) const {
        uint32_t w = q_start_tile + row + 1;
        w = w < valid_k ? w : valid_k;
        w = ((w + round - 1) / round) * round;
        return w < cap ? w : cap;
    }
};

// apply_causal_mask_lightweight (k_start_tile 0, no straddle) on each row's needed columns, rows `stride` tiles apart
template <uint32_t dst_batch>
void causal_mask_cols(
    uint32_t mask_cb,
    uint32_t neginf_idx,
    uint32_t diag_idx,
    uint32_t out_cb,
    uint32_t num_rows,
    const RowCols& cols_of,
    uint32_t stride) {
    reconfig_data_format_srca(mask_cb);
    pack_reconfig_data_format(out_cb);
    copy_init(mask_cb);
    PACK((llk_pack_reconfig_l1_acc(1)));
    for (uint32_t row = 0; row < num_rows; row++) {
        const uint32_t row_offset = row * stride;
        const uint32_t diag_col = cols_of.q_start_tile + row;
        const uint32_t cols = cols_of(row);
        if (diag_col < cols) {
            tile_regs_acquire();
            copy_tile(mask_cb, diag_idx, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, out_cb, row_offset + diag_col);
            tile_regs_release();
            const uint32_t neginf_start = diag_col + 1;
            if (neginf_start < cols) {
                stamp_tile_range_l1_acc<dst_batch>(
                    mask_cb, neginf_idx, out_cb, row_offset + neginf_start, cols - neginf_start);
            }
        }
    }
    PACK((llk_pack_reconfig_l1_acc(0)));
}

// reduce_c (no eltwise max with a previous chunk) over each row's needed tiles, rows `stride` tiles apart
template <PoolType pool_type, ReduceDim reduce_dim, uint32_t in0_cb, uint32_t scale_cb, uint32_t rows>
void reduce_c_cols(uint32_t out_cb, const RowCols& cols_of, uint32_t stride, uint32_t real_rows) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_scale(scale_cb);
    CircularBuffer cb_out(out_cb);
    cb_scale.wait_front(1);
    cb_in0.wait_front(rows * stride);
    cb_out.reserve_back(rows);
    pack_reconfig_data_format(out_cb);
    binary_max_tile_init();
    constexpr uint32_t reduce_dst_idx = 0;
    for (uint32_t i = 0; i < real_rows; i++) {
        const uint32_t cols = cols_of(i);
        reconfig_data_format_srca(in0_cb);
        tile_regs_acquire();
        reduce_init<pool_type, reduce_dim>(in0_cb, scale_cb, out_cb);
        for (uint32_t j = 0; j < cols; j++) {
            reduce_tile<pool_type, reduce_dim>(in0_cb, scale_cb, i * stride + j, 0, reduce_dst_idx);
        }
        reduce_uninit();
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(reduce_dst_idx, out_cb);
        tile_regs_release();
    }
    cb_out.push_back(rows);
}

// sub_exp_block_bcast_cols_inplace (accurate exp, RC, in place, with the row-sum reduce) on each row's needed columns
// of the `cols`-wide rows; the remaining column groups are popped and pushed back untouched (the FIFO walks every tile)
template <uint32_t in0_cb, uint32_t rows, uint32_t scale_fp32>
void sub_exp_cols(uint32_t in1_cb, uint32_t reduce_cb, uint32_t cols, const RowCols& cols_of, uint32_t real_rows) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_reduce(reduce_cb);
    reconfig_data_format(in0_cb, in1_cb);
    sub_bcast_cols_init(in0_cb, in1_cb);
    PACK((llk_pack_relu_config(ReluConfig::zero())));
    cb_in0.wait_front(rows * cols);
    cb_in1.wait_front(rows);
    cb_reduce.reserve_back(rows);
#ifdef SUB_EXP_GRANULARITY
    uint32_t dst_tiles = (cols < SUB_EXP_GRANULARITY) ? cols : SUB_EXP_GRANULARITY;
    uint32_t granularity = (cols >= SUB_EXP_GRANULARITY) ? (cols / SUB_EXP_GRANULARITY) : 1;
#else
    uint32_t dst_tiles = cols;
    uint32_t granularity = 1;
#endif
    for (uint32_t i = 0; i < rows; ++i) {
        const uint32_t w = i < real_rows ? cols_of(i) : 0;
        for (uint32_t u = 0; u < granularity; u++) {
            if (u * dst_tiles >= w) {
                cb_in0.pop_front(dst_tiles);
                cb_in0.reserve_back(dst_tiles);
                cb_in0.push_back(dst_tiles);
                continue;
            }
            tile_regs_acquire();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                sub_tiles_bcast_cols(in0_cb, in1_cb, j, i, j);
                binop_with_scalar_tile_init();
                mul_unary_tile(j, scale_fp32);
                exp_tile_init<false, 0x3F800000, InputClamping::ClampToNegative>();
                exp_tile<false, false, InputClamping::ClampToNegative, 32>(j, VectorMode::None);
            }
            tile_regs_commit();
            cb_in0.pop_front(dst_tiles);
            cb_in0.reserve_back(dst_tiles);
            tile_regs_wait();
            pack_reconfig_data_format(in0_cb);
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                pack_tile(j, in0_cb);
            }
            cb_in0.push_back(dst_tiles);
            pack_reconfig_data_format(reduce_cb);
            if (u > 0) {
                PACK((llk_pack_reconfig_l1_acc(1)));
            }
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                pack_tile<true>(j, reduce_cb, i);
                if (u == 0 && j == 0) {
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
            }
            tile_regs_release();
            PACK((llk_pack_reconfig_l1_acc(0)));
        }
    }
    cb_reduce.push_back(rows);
    PACK((llk_pack_relu_config(ReluConfig::none())));
}

// matmul_blocks (QK^T: transpose) over the first `real_rows` rows, each subblock row computing its rows' needed columns
// (the last row of the subblock's); output rows stay N tiles apart and every row of the M x N block is pushed
ALWI void matmul_qk_rows(
    const uint32_t in0_cb,
    const uint32_t in1_cb,
    const uint32_t out_cb,
    const uint32_t M,
    const uint32_t N,
    const uint32_t K,
    const uint32_t in0_block_w,
    const uint32_t subblock_h,
    const uint32_t subblock_w,
    const uint32_t real_rows,
    const RowCols& cols_of) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_out(out_cb);
    matmul_block_init(in0_cb, in1_cb, true, subblock_w, subblock_h, in0_block_w);
    const uint32_t in0_subblock_num_tiles = subblock_h * in0_block_w;
    uint32_t in0_wait_tiles = in0_subblock_num_tiles;
    uint32_t in0_index_offset = 0;
    reconfig_data_format(in1_cb, in0_cb);
    cb_in1.wait_front(K * N);
    cb_out.reserve_back(M * N);
    for (uint32_t m0 = 0; m0 < real_rows; m0 += subblock_h) {
        cb_in0.wait_front(in0_wait_tiles);
        const uint32_t cols = cols_of(m0 + subblock_h - 1);
        uint32_t in1_index_offset = 0;
        for (uint32_t n0 = 0; n0 < cols; n0 += subblock_w) {
            tile_regs_acquire();
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;
            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                matmul_block(in0_cb, in1_cb, in0_index, in1_index, 0, true, subblock_w, subblock_h, in0_block_w);
                in0_index++;
                in1_index += N;
            }
            tile_regs_commit();
            tile_regs_wait();
            uint32_t dst_idx = 0;
            for (uint32_t r = 0; r < subblock_h; r++) {
                for (uint32_t c = 0; c < subblock_w; c++) {
                    pack_tile<true>(dst_idx, out_cb, r * N + n0 + c);
                    dst_idx++;
                }
            }
            tile_regs_release();
            in1_index_offset += subblock_w;
        }
        in0_index_offset += subblock_h * in0_block_w;
        in0_wait_tiles += in0_subblock_num_tiles;
        cb_out.push_back(subblock_h * N);
    }
    if (real_rows < M) {
        cb_out.push_back((M - real_rows) * N);  // the padding rows' tiles, untouched
    }
    cb_in1.pop_front(K * N);
}

// matmul_blocks (P @ V, no transpose) over the first `real_rows` rows, each subblock row's inner loop running its rows'
// needed key tiles (the last row of the subblock's) of the in0_block_w; every row of the M x N output is pushed
ALWI void matmul_pv_rows(
    const uint32_t in0_cb,
    const uint32_t in1_cb,
    const uint32_t out_cb,
    const uint32_t M,
    const uint32_t N,
    const uint32_t K,
    const uint32_t in1_num_subblocks,
    const uint32_t in0_block_w,
    const uint32_t subblock_h,
    const uint32_t subblock_w,
    const uint32_t real_rows,
    const RowCols& cols_of) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_out(out_cb);
    matmul_block_init(in0_cb, in1_cb, false, subblock_w, subblock_h, in0_block_w);
    const uint32_t in0_subblock_num_tiles = subblock_h * in0_block_w;
    uint32_t in0_wait_tiles = in0_subblock_num_tiles;
    uint32_t in0_index_offset = 0;
    reconfig_data_format(in1_cb, in0_cb);
    cb_in1.wait_front(K * N);
    cb_out.reserve_back(M * N);
    for (uint32_t m0 = 0; m0 < real_rows; m0 += subblock_h) {
        cb_in0.wait_front(in0_wait_tiles);
        const uint32_t k_count = cols_of(m0 + subblock_h - 1);
        uint32_t in1_index_offset = 0;
        for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; ++in1_subblock) {
            tile_regs_acquire();
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;
            for (uint32_t inner_dim = 0; inner_dim < k_count; inner_dim++) {
                matmul_block(in0_cb, in1_cb, in0_index, in1_index, 0, false, subblock_w, subblock_h, in0_block_w);
                in0_index++;
                in1_index += N;
            }
            tile_regs_commit();
            tile_regs_wait();
            uint32_t dst_idx = 0;
            const uint32_t out_col_offset = in1_subblock * subblock_w;
            for (uint32_t r = 0; r < subblock_h; r++) {
                for (uint32_t c = 0; c < subblock_w; c++) {
                    pack_tile<true>(dst_idx, out_cb, r * N + out_col_offset + c);
                    dst_idx++;
                }
            }
            tile_regs_release();
            in1_index_offset += subblock_w;
        }
        in0_index_offset += subblock_h * in0_block_w;
        in0_wait_tiles += in0_subblock_num_tiles;
        cb_out.push_back(subblock_h * N);
    }
    if (real_rows < M) {
        cb_out.push_back((M - real_rows) * N);  // the padding rows' tiles, untouched
    }
    cb_in1.pop_front(K * N);
}

// sdpa_inner_loop<STANDARD> for a causal, lightweight-masked prefill whose keys are one K chunk (k_num_chunks == 1:
// no online-softmax rescale), with the per-row column truncation above.
template <
    uint32_t cb_qk_im,
    uint32_t cb_identity_scale_in,
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t DHt,
    uint32_t vDHt,
    uint32_t scale_fp32,
    uint32_t valid_Skt,
    uint32_t valid_Sqt>
void causal_one_chunk(
    const uint32_t Skt,
    const uint32_t qk_in0_block_w,
    const uint32_t qk_subblock_w,
    const uint32_t qk_subblock_h,
    const uint32_t out_in0_block_w,
    const uint32_t out_subblock_w,
    const uint32_t out_subblock_h,
    const uint32_t out_in1_num_subblocks,
    const uint32_t iter_q_start,
    const uint32_t iter_q_end,
    const uint32_t q_num_chunks,
    const uint32_t local_q_start,
    const uint32_t q_chunk_tiles,
    const uint32_t qk_chunk_tiles,
    const uint32_t cb_q_in,
    const uint32_t cb_k_in,
    const uint32_t cb_v_in,
    const uint32_t cb_mask_in,
    const uint32_t cb_col_identity,
    const uint32_t cb_out_im_A,
    const uint32_t cb_out_im_B,
    const uint32_t cb_max_A,
    const uint32_t cb_max_B,
    const uint32_t cb_sum_A,
    const uint32_t cb_sum_B,
    const uint32_t cb_out,
    const LightweightMaskContext& lw_mask,
    const bool use_zigzag_balancing) {
    CircularBuffer cb_q_in_obj(cb_q_in);
    CircularBuffer cb_qk_im_obj(cb_qk_im);
    constexpr uint32_t dst_size = compute_kernel_lib::DEST_AUTO_LIMIT;
#ifdef SUB_EXP_GRANULARITY
    constexpr uint32_t exp_group = (Sk_chunk_t < SUB_EXP_GRANULARITY) ? Sk_chunk_t : SUB_EXP_GRANULARITY;
#else
    constexpr uint32_t exp_group = Sk_chunk_t;
#endif
    // column rounding: whole QK subblocks and whole exp groups
    const uint32_t round = (exp_group % qk_subblock_w == 0)   ? exp_group
                           : (qk_subblock_w % exp_group == 0) ? qk_subblock_w
                                                              : qk_subblock_w * exp_group;
    for (uint32_t q_iter = iter_q_start; q_iter < iter_q_end; ++q_iter) {
        const uint32_t linear_q_chunk = local_q_start + (q_iter - iter_q_start);
        const uint32_t q_chunk = remap_q_index(linear_q_chunk, q_num_chunks, use_zigzag_balancing) % q_num_chunks;
        const uint32_t q_start_tile = q_chunk * Sq_chunk_t;
        const RowCols cols_of{q_start_tile, valid_Skt, round, Sk_chunk_t};
        // the Q tile rows this chunk holds of the prompt (past them: padding rows the writer never stores)
        uint32_t rr = q_start_tile < valid_Sqt ? valid_Sqt - q_start_tile : 0;
        if (rr > Sq_chunk_t || rr == 0 || rr % qk_subblock_h != 0 || rr % out_subblock_h != 0) {
            rr = Sq_chunk_t;
        }

        uint32_t alias_prev_sum = cb_sum_A;
        uint32_t alias_cur_sum = cb_sum_B;
        uint32_t alias_prev_max = cb_max_A;
        uint32_t alias_cur_max = cb_max_B;
        uint32_t alias_mm2_prev_out = cb_out_im_A;
        uint32_t alias_mm2_cur_out = cb_out_im_B;

        /* QK = Q_CHUNK @ K_CHUNK, each row's needed columns (rows stay Sk_chunk_t tiles apart) */
        reconfig_data_format(cb_k_in, cb_q_in);
        pack_reconfig_data_format(cb_qk_im);
        matmul_qk_rows(
            cb_q_in,
            cb_k_in,
            cb_qk_im,
            Sq_chunk_t,
            Sk_chunk_t,
            DHt,
            qk_in0_block_w,
            qk_subblock_h,
            qk_subblock_w,
            rr,
            cols_of);

        /* QK += MASK (k chunk 0 always overlaps the causal diagonal) */
        reconfig_data_format(cb_qk_im, cb_mask_in);
        cb_qk_im_obj.wait_front(Sk_chunk_t * Sq_chunk_t);
        cb_qk_im_obj.pop_front(Sk_chunk_t * Sq_chunk_t);
        cb_qk_im_obj.reserve_back(Sk_chunk_t * Sq_chunk_t);
        causal_mask_cols<dst_size>(
            cb_mask_in, lw_mask.neginf_tile_idx, lw_mask.causal_diag_tile_idx, cb_qk_im, rr, cols_of, Sk_chunk_t);
        cb_qk_im_obj.push_back(Sk_chunk_t * Sq_chunk_t);

        reconfig_data_format(cb_qk_im, cb_identity_scale_in);
        reduce_c_cols<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, Sq_chunk_t>(
            alias_cur_max, cols_of, Sk_chunk_t, rr);

        sub_exp_cols<cb_qk_im, Sq_chunk_t, scale_fp32>(alias_cur_max, alias_cur_sum, Sk_chunk_t, cols_of, rr);

        reconfig_data_format(cb_v_in, cb_qk_im);
        pack_reconfig_data_format(alias_mm2_cur_out);
        /* OUT_IM = QK @ V_CHUNK over each row's needed keys */
        matmul_pv_rows(
            cb_qk_im,
            cb_v_in,
            alias_mm2_cur_out,
            Sq_chunk_t,
            vDHt,
            Sk_chunk_t,
            out_in1_num_subblocks,
            out_in0_block_w,
            out_subblock_h,
            out_subblock_w,
            rr,
            cols_of);

        cb_qk_im_obj.pop_front(qk_chunk_tiles);
        reconfig_data_format(alias_prev_max, alias_cur_max);
        std::swap(alias_prev_sum, alias_cur_sum);
        std::swap(alias_mm2_prev_out, alias_mm2_cur_out);
        std::swap(alias_prev_max, alias_cur_max);

        matmul_reduce<Sq_chunk_t>(cb_col_identity, alias_prev_sum);
        recip_block_inplace(alias_prev_sum, Sq_chunk_t);
        pack_reconfig_data_format(cb_out);
        mul_block_bcast_cols<Sq_chunk_t, vDHt, false, false>(alias_mm2_prev_out, alias_prev_sum, cb_out);
        CircularBuffer(alias_prev_max).pop_front(Sq_chunk_t);
        cb_q_in_obj.pop_front(q_chunk_tiles);
    }
}

template <
    uint32_t cb_qk_im,
    uint32_t cb_identity_scale_in,
    uint32_t cb_attention_sink,
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t DHt,
    uint32_t vDHt,
    bool use_attention_sink,
    bool is_causal,
    bool use_provided_mask,
    bool use_padded_mask,
    bool is_chunked,
    uint32_t scale_fp32,
    uint32_t sliding_window_size,
    bool lightweight_mask_enabled = false,
    bool use_windowed_narrowing = false,
    uint32_t cb_windowed_k_range = 0>
void sdpa_standard(
    const uint32_t Skt,
    const uint32_t qk_in0_block_w,
    const uint32_t qk_subblock_w,
    const uint32_t qk_subblock_h,
    const uint32_t qk_in0_num_subblocks,
    const uint32_t qk_in1_num_subblocks,
    const uint32_t out_in0_block_w,
    const uint32_t out_subblock_w,
    const uint32_t out_subblock_h,
    const uint32_t out_in0_num_subblocks,
    const uint32_t out_in1_num_subblocks,
    const uint32_t iter_q_start,
    const uint32_t iter_q_end,
    const uint32_t q_num_chunks,
    const uint32_t local_q_start,
    const uint32_t chunked_q_chunk_offset,
    const uint32_t k_num_chunks,
    const uint32_t q_chunk_tiles,
    const uint32_t k_chunk_tiles,
    const uint32_t v_chunk_tiles,
    const uint32_t qk_chunk_tiles,
    const uint32_t out_chunk_tiles,
    const uint32_t cb_q_in,
    const uint32_t cb_k_in,
    const uint32_t cb_v_in,
    const uint32_t cb_mask_in,
    const uint32_t cb_col_identity,
    const uint32_t cb_out_im_A,
    const uint32_t cb_out_im_B,
    const uint32_t cb_max_A,
    const uint32_t cb_max_B,
    const uint32_t cb_sum_A,
    const uint32_t cb_sum_B,
    const uint32_t cb_exp_max_diff,
    const uint32_t cb_out,
    const LightweightMaskContext& lw_mask = {},
    const bool use_zigzag_balancing = false) {
    constexpr uint32_t valid_Skt = get_compile_time_arg_val(25);
    if constexpr (
        is_causal && lightweight_mask_enabled && !use_attention_sink && !use_provided_mask && !use_padded_mask &&
        !is_chunked && sliding_window_size == 0 && !use_windowed_narrowing) {
        if (k_num_chunks == 1 && Skt == Sk_chunk_t) {
            causal_one_chunk<
                cb_qk_im,
                cb_identity_scale_in,
                Sq_chunk_t,
                Sk_chunk_t,
                DHt,
                vDHt,
                scale_fp32,
                valid_Skt,
                SDPA_VALID_SQT>(
                Skt,
                qk_in0_block_w,
                qk_subblock_w,
                qk_subblock_h,
                out_in0_block_w,
                out_subblock_w,
                out_subblock_h,
                out_in1_num_subblocks,
                iter_q_start,
                iter_q_end,
                q_num_chunks,
                local_q_start,
                q_chunk_tiles,
                qk_chunk_tiles,
                cb_q_in,
                cb_k_in,
                cb_v_in,
                cb_mask_in,
                cb_col_identity,
                cb_out_im_A,
                cb_out_im_B,
                cb_max_A,
                cb_max_B,
                cb_sum_A,
                cb_sum_B,
                cb_out,
                lw_mask,
                use_zigzag_balancing);
            return;
        }
    }
    ::sdpa_standard<
        cb_qk_im,
        cb_identity_scale_in,
        cb_attention_sink,
        Sq_chunk_t,
        Sk_chunk_t,
        DHt,
        vDHt,
        use_attention_sink,
        is_causal,
        use_provided_mask,
        use_padded_mask,
        is_chunked,
        scale_fp32,
        sliding_window_size,
        lightweight_mask_enabled,
        use_windowed_narrowing,
        cb_windowed_k_range>(
        Skt,
        qk_in0_block_w,
        qk_subblock_w,
        qk_subblock_h,
        qk_in0_num_subblocks,
        qk_in1_num_subblocks,
        out_in0_block_w,
        out_subblock_w,
        out_subblock_h,
        out_in0_num_subblocks,
        out_in1_num_subblocks,
        iter_q_start,
        iter_q_end,
        q_num_chunks,
        local_q_start,
        chunked_q_chunk_offset,
        k_num_chunks,
        q_chunk_tiles,
        k_chunk_tiles,
        v_chunk_tiles,
        qk_chunk_tiles,
        out_chunk_tiles,
        cb_q_in,
        cb_k_in,
        cb_v_in,
        cb_mask_in,
        cb_col_identity,
        cb_out_im_A,
        cb_out_im_B,
        cb_max_A,
        cb_max_B,
        cb_sum_A,
        cb_sum_B,
        cb_exp_max_diff,
        cb_out,
        lw_mask,
        use_zigzag_balancing);
}
}  // namespace sdpa_causal_skip

#define sdpa_standard sdpa_causal_skip::sdpa_standard
#endif  // SDPA_CAUSAL_SKIP

#include "ttnn/operations/transformer/sdpa/device/kernels/compute/sdpa.cpp"

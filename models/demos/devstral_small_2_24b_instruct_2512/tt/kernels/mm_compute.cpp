// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute of tt/minimal_mm.py: ttnn.experimental.minimal_matmul's compute kernel (minimal_matmul/device/kernels/
// compute.cpp; no bias / ternary / fused activation, fp32 DEST) with its output stage regrouped. The matmul loop is the
// stock one. Each output block's fp32 L1 partials (intermediate_cb) leave through an epilogue that moves EPI_TILES tiles
// (SWIGLU_PAIRS gate/up pairs) a DEST acquire with its inits run once per block; the SwiGLU pair runs silu and the up
// multiply as ONE SFPU pass (calculate_silu_mul), issued from the PACK thread under SWIGLU_PACK_SFPU (pack_silu_mul;
// MATH copies the next pair group meanwhile; bit-identical, A/B on the vision gate/up). Measured on the 416-row text
// gate/up: the SFPU is ~1.4 of its 9.8 ms (vision 7 x 24-tile blocks: 98 of 232 us),
// unpack + pack of the partials ~0.05. Every tile goes through the same unpack, datacopy, SFPU products and pack as in
// the stock kernel, so the output is bit-identical.
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/circular_buffer.h"

#ifndef EPI_TILES
#define EPI_TILES 1
#endif
#ifndef SWIGLU_PAIRS
#define SWIGLU_PAIRS 1
#endif

// Copy the block's fp32 partials out, EPI_TILES tiles a DEST acquire (EPI_TILES divides N_block_tiles).
void copy_and_pack_block(uint32_t in_cb, uint32_t out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
    CircularBuffer cb_out(out_cb);
    reconfig_data_format_srca(in_cb);
    pack_reconfig_data_format(out_cb);
    copy_init(in_cb);
    uint32_t tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < N_block_tiles; n += EPI_TILES) {
            tile_regs_acquire();
            for (uint32_t t = 0; t < EPI_TILES; t++) {
                copy_tile(in_cb, tile_id + t, t);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t t = 0; t < EPI_TILES; t++) {
                pack_tile(t, out_cb);
            }
            tile_regs_release();
            tile_id += EPI_TILES;
        }
        cb_out.push_back(N_block_tiles);
    }
}

#ifdef FUSE_RESIDUAL
// out = residual + the block's fp32 partials, RES_TILES tiles a DEST acquire: the residual tiles (the stock reader's
// FUSE_TERNARY "a" block, unpacked to DEST unrounded: UnpackToDestFp32) in DEST 0.., the partials copied exactly as
// copy_and_pack_block copies them in DEST RES_TILES.., then the SFPU add a fp32 ttnn.add(residual, out) runs
// (binary_ng: residual in the left operand, add_binary_tile<NearestEven>) -- the same values through the same add.
constexpr uint32_t RES_TILES = 2;
void residual_add_block(
    uint32_t in_cb, uint32_t res_cb, uint32_t bcast_cb, uint32_t out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
    CircularBuffer cb_out(out_cb);
    CircularBuffer cb_res(res_cb);
    CircularBuffer cb_bcast(bcast_cb);
    // the stock reader's broadcast "b" row (FUSE_TERNARY reads one a block); unused
    cb_bcast.wait_front(N_block_tiles);
    cb_bcast.pop_front(N_block_tiles);
    pack_reconfig_data_format(out_cb);
    add_binary_tile_init();
    uint32_t tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        cb_res.wait_front(N_block_tiles);
        for (uint32_t n = 0; n < N_block_tiles; n += RES_TILES) {
            const uint32_t nt = (N_block_tiles - n) < RES_TILES ? (N_block_tiles - n) : RES_TILES;
            tile_regs_acquire();
            reconfig_data_format_srca(in_cb);
            copy_init(in_cb);
            for (uint32_t t = 0; t < nt; t++) {
                copy_tile(in_cb, tile_id + n + t, RES_TILES + t);
            }
            reconfig_data_format_srca(in_cb, res_cb);
            copy_init(res_cb);
            for (uint32_t t = 0; t < nt; t++) {
                copy_tile(res_cb, n + t, t);
            }
            for (uint32_t t = 0; t < nt; t++) {
                add_binary_tile<ckernel::DstRoundingMode::NearestEven>(t, RES_TILES + t, t);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t t = 0; t < nt; t++) {
                pack_tile(t, out_cb);
            }
            tile_regs_release();
        }
        tile_id += N_block_tiles;
        cb_res.pop_front(N_block_tiles);
        cb_out.push_back(N_block_tiles);
    }
}
#endif  // FUSE_RESIDUAL

#ifdef FUSE_SWIGLU
#if defined(TRISC_MATH) || defined(TRISC_PACK)
namespace ckernel::sfpu {
// silu_tile followed by mul_binary_tile in ONE pass over the rows (fp32 DEST): silu_tile stores x * sigmoid(x)
// unrounded and mul_binary_tile reloads it and multiplies it by up, so keeping it in a register runs the same two
// products on the same values.
template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_silu_mul(const uint32_t gate, const uint32_t up, const uint32_t out) {
    static_assert(is_fp32_dest_acc_en, "the fused SwiGLU pass replays the fp32-DEST silu / mul only");
    constexpr uint32_t T = 32;  // DEST rows a tile in sfpi addressing
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[gate * T];
        sfpi::vFloat s = x * _sfpu_sigmoid_<is_fp32_dest_acc_en>(x);
        sfpi::vFloat u = sfpi::dst_reg[up * T];
        sfpi::dst_reg[out * T] = s * u;
        sfpi::dst_reg++;
    }
}

#ifdef SWIGLU_HOIST
// calculate_silu_mul with the fp32 exp's 32-bit polynomial constants held in registers across the rows instead of
// re-loaded (two SFPLOADIs each) every row: _sfpu_sigmoid_ = sfpu_reciprocal_iter<2>(1 + _sfpu_exp_fp32_accurate_(-x))
// written out op for op, the same products and sums on the same values. Two constants ride in the programmable
// constant registers (silu_mul_hoist_init, after silu_init's 2.0 in Prgm0), three in LREGs, which x leaves free by
// being re-read from DEST for the final product (the row is unchanged there until the store). 42 SFPU instructions a
// row instead of 52; the SFPU is issue-bound (~1 instruction a cycle), so the pass shrinks by the same fraction.
inline void silu_mul_hoist_init() {
    sfpi::vConstFloatPrgm1 = -6.93145752e-1f;
    sfpi::vConstFloatPrgm2 = -1.42860677e-6f;
}

template <int ITERATIONS>
inline void calculate_silu_mul_hoisted(const uint32_t gate, const uint32_t up, const uint32_t out) {
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
#endif  // SWIGLU_HOIST
}  // namespace ckernel::sfpu
#endif

#if defined(SWIGLU_PACK_SFPU) && defined(TRISC_PACK)
// calculate_silu_mul over the four faces of the DEST half the packer owns, issued by the PACK thread: the face loop of
// _llk_math_eltwise_binary_sfpu_params_ (16 rows a face) without its wait for the FPU (MATH committed this half; the
// FPU is busy with the next one). The caller points DEST_TARGET_REG_CFG_MATH_Offset at the packer's half.
inline void pack_silu_mul(const uint32_t gate, const uint32_t up, const uint32_t out) {
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
#pragma GCC unroll 0
    for (int face = 0; face < 4; face++) {
#ifdef SWIGLU_HOIST
        ckernel::sfpu::calculate_silu_mul_hoisted<8>(gate, up, out);
#else
        ckernel::sfpu::calculate_silu_mul<DST_ACCUM_MODE, 8>(gate, up, out);
#endif
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
}
#endif

// out = silu(gate) * up per interleaved gate/up tile pair (column tile 2p gate, 2p + 1 up), SWIGLU_PAIRS pairs a DEST
// acquire (pair j: gate in DEST 2j, up in 2j + 1). The copy / silu / mul inits run once per block (the stock kernel
// re-runs them per pair): the SFPU inits program ADDR_MOD_6/7, the SFPU config and vConstFloatPrgm0 (the reciprocal's
// 2.0) only, the datacopy uses ADDR_MOD_0/2, and the mul init sets no constant, so re-running them changes nothing.
void swiglu_block(uint32_t in_cb, uint32_t out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
    CircularBuffer cb_out(out_cb);
    reconfig_data_format_srca(in_cb);
    pack_reconfig_data_format(out_cb);
    const uint32_t out_N_block_tiles = N_block_tiles >> 1;
    copy_init(in_cb);
#ifdef SWIGLU_PACK_SFPU
    silu_tile_init_pack();  // the SFPU pass runs on the PACK thread (pack_silu_mul), the datacopy stays on MATH
#ifdef SWIGLU_HOIST
    PACK((ckernel::sfpu::silu_mul_hoist_init()));
#endif
#else
    silu_tile_init();
    mul_binary_tile_init();
#endif
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        const uint32_t row_base = m * N_block_tiles;
        for (uint32_t p = 0; p < out_N_block_tiles; p += SWIGLU_PAIRS) {
            const uint32_t np = (out_N_block_tiles - p) < SWIGLU_PAIRS ? (out_N_block_tiles - p) : SWIGLU_PAIRS;
            tile_regs_acquire();
            for (uint32_t j = 0; j < np; j++) {
                const uint32_t gate_tile_id = row_base + ((p + j) << 1);
                copy_tile(in_cb, gate_tile_id, 2 * j);
                copy_tile(in_cb, gate_tile_id + 1, 2 * j + 1);
            }
#ifndef SWIGLU_PACK_SFPU
            for (uint32_t j = 0; j < np; j++) {
                MATH((SFPU_BINARY_CALL(
                    DST_SYNC_MODE,
                    DST_ACCUM_MODE,
                    calculate_silu_mul,
                    (DST_ACCUM_MODE, 8),
                    2 * j,
                    2 * j + 1,
                    2 * j,
                    VectorMode::RC)));
            }
#endif
            tile_regs_commit();
#ifdef SWIGLU_PACK_SFPU
            // the moe_fused_swiglu recipe (add_silu_elementwise): wait for MATH's commit, aim the SFPU at the packer's
            // DEST half, run the pass from PACK, wait for the SFPU before packing
            PACK((TTI_SEMWAIT(
                p_stall::STALL_TDMA | p_stall::STALL_CFG,
                semaphore::t6_sem(semaphore::MATH_PACK),
                p_stall::STALL_ON_ZERO)));
            PACK((TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset())));
            for (uint32_t j = 0; j < np; j++) {
                PACK((pack_silu_mul(2 * j, 2 * j + 1, 2 * j)));
            }
            PACK((TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU)));
#else
            tile_regs_wait();
#endif
            for (uint32_t j = 0; j < np; j++) {
                pack_tile(2 * j, out_cb);
            }
            tile_regs_release();
        }
        cb_out.push_back(out_N_block_tiles);
    }
}
#endif  // FUSE_SWIGLU

// The stock kernel's matmul_blocks: subblocks of the M x N block, K_block_tiles deep, packed out of order into the
// fp32 accumulation block (L1 accumulation after the first K block).
void matmul_blocks(
    const uint32_t in0_cb,
    const uint32_t in1_cb,
    const uint32_t out_cb,
    const uint32_t M_block_tiles,
    const uint32_t N_block_tiles,
    const uint32_t full_N_block_tiles,
    const uint32_t K_block_tiles,
    const uint32_t subblock_h,
    const uint32_t subblock_w) {
    uint32_t in0_index_offset = 0;
    for (uint32_t M_start = 0; M_start < M_block_tiles; M_start += subblock_h) {
        uint32_t in1_index_offset = 0;
        for (uint32_t N_start = 0; N_start < N_block_tiles; N_start += subblock_w) {
            tile_regs_acquire();
            uint32_t dst_index = 0;
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;
            for (uint32_t inner_dim = 0; inner_dim < K_block_tiles; inner_dim++) {
                matmul_block(
                    in0_cb,
                    in1_cb,
                    in0_index,
                    in1_index,
                    dst_index,
                    false /*transpose*/,
                    subblock_w,
                    subblock_h,
                    K_block_tiles);
                in0_index++;
                in1_index += full_N_block_tiles;
            }
            tile_regs_commit();
            tile_regs_wait();
            uint32_t write_dst_index = 0;
            for (uint32_t h = 0; h < subblock_h; h++) {
                uint32_t h_tile_id = M_start + h;
                for (uint32_t w = 0; w < subblock_w; w++) {
                    uint32_t w_tile_id = N_start + w;
                    uint32_t out_tile_id = h_tile_id * full_N_block_tiles + w_tile_id;
                    pack_tile<true>(write_dst_index, out_cb, out_tile_id);
                    write_dst_index++;
                    dst_index++;
                }
            }
            tile_regs_release();
            in1_index_offset += subblock_w;
        }
        in0_index_offset += subblock_h * K_block_tiles;
    }
}

void kernel_main() {
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t M_blocks_per_core = get_compile_time_arg_val(4);
    constexpr uint32_t N_blocks_per_core = get_compile_time_arg_val(5);
    constexpr uint32_t subblock_h = get_compile_time_arg_val(6);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(7);
    static_assert(N_block_tiles % EPI_TILES == 0, "EPI_TILES must divide the N block");

    uint32_t argidx = 0;
    const uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_end_tile = get_arg_val<uint32_t>(argidx++);

    constexpr uint32_t in0_cb = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    constexpr uint32_t intermediate_cb = tt::CBIndex::c_3;

    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_out(out_cb);
    CircularBuffer cb_intermediate(intermediate_cb);

    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_cb, in1_cb, intermediate_cb);
    matmul_init(in0_cb, in1_cb);

    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    bool reuse_in0_block = false;

    for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
        uint32_t m_tile = M_start_tile + m_block_iter * M_block_tiles;
        uint32_t m_tile_end = std::min(m_tile + M_block_tiles, M_end_tile);
        uint32_t current_M_block_tiles = m_tile_end - m_tile;
        uint32_t current_subblock_h = std::min(current_M_block_tiles, subblock_h);

        for (uint32_t n_block_iter = 0; n_block_iter < N_blocks_per_core; n_block_iter++) {
            uint32_t n_tile = N_start_tile + n_block_iter * N_block_tiles;
            uint32_t n_tile_end = std::min(n_tile + N_block_tiles, N_end_tile);
            uint32_t current_N_block_tiles = n_tile_end - n_tile;
            uint32_t current_subblock_w = std::min(current_N_block_tiles, subblock_w);

            reconfig_data_format(in1_cb, in0_cb);
            pack_reconfig_data_format(intermediate_cb);
            matmul_block_init(
                in0_cb,
                in1_cb,
                false /*transpose*/,
                current_subblock_w /*ct_dim*/,
                current_subblock_h /*rt_dim*/,
                K_block_tiles /*kt_dim*/);
            cb_intermediate.reserve_back(out_block_num_tiles);
            for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {
                cb_in0.wait_front(in0_block_num_tiles);
                cb_in1.wait_front(in1_block_num_tiles);

                matmul_blocks(
                    in0_cb,
                    in1_cb,
                    intermediate_cb,
                    current_M_block_tiles,
                    current_N_block_tiles,
                    N_block_tiles,
                    K_block_tiles,
                    current_subblock_h,
                    current_subblock_w);

                if (k_block == K_num_blocks - 1) {
                    if (n_block_iter < N_blocks_per_core - 1) {
                        reuse_in0_block = true;  // striding N next: in0 is reused
                    }
                }
                if (!reuse_in0_block) {
                    cb_in0.pop_front(in0_block_num_tiles);
                }
                cb_in1.pop_front(in1_block_num_tiles);
                reuse_in0_block = false;
                if (k_block == 0) {
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
            }

            cb_intermediate.push_back(out_block_num_tiles);
            PACK((llk_pack_reconfig_l1_acc(0)));

#ifdef FUSE_SWIGLU
            cb_out.reserve_back(out_block_num_tiles >> 1);
            cb_intermediate.wait_front(out_block_num_tiles);
            swiglu_block(intermediate_cb, out_cb, M_block_tiles, N_block_tiles);
            cb_intermediate.pop_front(out_block_num_tiles);
#elif defined(FUSE_RESIDUAL)
            cb_out.reserve_back(out_block_num_tiles);
            cb_intermediate.wait_front(out_block_num_tiles);
            residual_add_block(
                intermediate_cb, tt::CBIndex::c_5, tt::CBIndex::c_6, out_cb, M_block_tiles, N_block_tiles);
            cb_intermediate.pop_front(out_block_num_tiles);
#else
            cb_out.reserve_back(out_block_num_tiles);
            cb_intermediate.wait_front(out_block_num_tiles);
            copy_and_pack_block(intermediate_cb, out_cb, M_block_tiles, N_block_tiles);
            cb_intermediate.pop_front(out_block_num_tiles);
#endif
        }
    }
}

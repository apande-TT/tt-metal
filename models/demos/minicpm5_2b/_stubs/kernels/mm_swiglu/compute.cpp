// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B mm_swiglu compute: copy of the legacy minimal_matmul compute kernel
// (ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/compute.cpp; its Metal 2.0 fork
// compute_metal2.cpp, which the stand-alone ttnn.experimental.minimal_matmul binds today, has identical arithmetic:
// same calls in matmul_blocks / copy_and_pack_block, and its descriptor gives the fp32 intermediate the default
// UnpackToSrc mode, which is what this legacy-CB kernel gets with unpack_to_dest_mode left at Default).
//
// Kept verbatim: kernel_main's matmul loop (one full-K block, same subblocks, same k order, same packer L1-acc
// handling, same reconfig / init calls), matmul_blocks, copy_and_pack_block.
// Removed (compiled out for this op): FUSE_BIAS / FUSE_TERNARY / fused-activation paths.
// Replaced: the FUSE_SWIGLU output stage (swiglu_block -> swiglu_block_exact below), so the output is
// bit-identical to the unfused reference
//     gate = minimal_matmul(x, Wg); up = minimal_matmul(x, Wu)          (bf16 outputs, fp32 DEST, HiFi2)
//     h    = ttnn.multiply(gate, up, input_tensor_a_activations=[SILU])  (binary_ng SFPU kernel, 16-bit DEST)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/tilize.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/circular_buffer.h"

// Verbatim from the reference: the non-fused minimal_matmul output stage (fp32 intermediate -> unpack to SrcA ->
// datacopy to DEST -> pack to the bf16 output format).
void copy_and_pack_block(uint32_t in_cb, uint32_t out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
    CircularBuffer cb_out(out_cb);
    // Data formats must be reconfigured before the op init: llk_unpack_A_init asserts that the
    // unpacker is already configured for in_cb (see #55052), and the preceding matmul left it
    // configured for its own operands.
    reconfig_data_format_srca(in_cb);
    pack_reconfig_data_format(out_cb);
    copy_init(in_cb);
    uint32_t fused_act_dst_id = 0;

    uint32_t tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            tile_regs_acquire();
            tile_regs_wait();
            copy_tile(in_cb, tile_id, fused_act_dst_id /*dst*/);
#ifdef SFPU_OP_INIT_ACTIVATION
            SFPU_OP_FUNC_ACTIVATION
#endif
            pack_tile(fused_act_dst_id, out_cb);
            tile_regs_commit();
            tile_regs_release();
            tile_id++;
        }
        cb_out.push_back(N_block_tiles);
    }
}

#ifdef FUSE_SWIGLU
// Bit-exact SwiGLU output stage. The matmul produced an interleaved gate/up block in `in_cb` (the fp32
// intermediate accumulator): within each M row, column tile 2p is the gate projection and 2p+1 the up projection
// (tile-pair interleaved weight). Each pair yields one output tile, reproducing the reference ops' roundings
// exactly and nothing else:
//
// Step 1 -- G / U = the bf16 tiles the reference minimal_matmul would have written for the gate / up tile.
//   The block goes through copy_and_pack_block, the reference's own (non-fused) output stage, unchanged: same
//   reconfig_data_format_srca(in_cb) / pack_reconfig_data_format / copy_init, same unpack of the fp32
//   intermediate to SrcA (unpack_to_dest_mode Default, as the reference: legacy factory default, Metal 2.0
//   descriptor UnpackToSrc), same datacopy to the fp32 DEST, same packer fp32 -> bf16 conversion -- only the
//   destination CB is the bf16 scratch `scratch_cb` (same data format as the reference output CB, so the packer
//   is configured identically). The gate/up accumulators themselves are bit-identical to the reference's (every
//   output tile is the same full-K, in-order HiFi2 fp32-DEST accumulation; its column neighbour in the subblock
//   does not influence it), hence G and U are the reference gate / up tensors' tiles bit for bit.
//
// Step 2 -- h = silu16(G) * U, reproducing binary_ng's eltwise_binary_sfpu_no_bcast.cpp (MUL is an SFPU op in
//   binary_ng; bf16 operands/output -> fp32_dest_acc_en = false, i.e. a 16-bit DEST), whose sequence is
//   (a) LHS preprocess: copy G -> DEST, silu_tile (DST_ACCUM_MODE = false), pack bf16 to cb_post;
//   (b) copy post -> DEST0, copy U -> DEST1, mul_binary_tile(0, 1, 0) (DST_ACCUM_MODE = false), pack bf16.
//   This kernel runs with a 32-bit DEST (the matmul needs it), so it calls the SAME SFPU routines instantiated
//   with is_fp32_dest_acc_en = false (silu_tile<false>, mul_binary_tile<false>; the template argument only
//   selects the ckernel::sfpu functor instantiation -- SFPU_UNARY_CALL / SFPU_BINARY_CALL do not use DST_ACCUM
//   for addressing, so DEST tile indexing is unchanged):
//   * DEST loads: G and U are bf16 values; in the reference they sit in a 16-bit DEST, here copy_tile places the
//     same values exactly in the fp32 DEST (bf16 -> fp32 widening is exact), so SFPLOAD yields the same fp32
//     LREG values in both cases.
//   * silu (ckernel_sfpu_silu.h, calculate_silu<false, 8>): result = x * _sfpu_sigmoid_<false>(x), where the
//     16-bit instantiation uses exp_21f (_sfpu_exp_21f_bf16_<true>) and sfpu_reciprocal_iter<1> (the fp32 one
//     would use _sfpu_exp_accurate_ and 2 Newton steps), then
//     result = convert<vFloat16b>(result, RoundMode::Nearest) (SFPSTOCHRND fp32 -> fp16b, a bf16 value held
//     in fp32 format). In the reference SFPSTORE to the 16-bit DEST then truncates -- a no-op on a value that is
//     already bf16 -- and cb_post's bf16 pack / re-unpack is likewise exact. Here SFPSTORE writes that same
//     bf16-valued fp32 into the fp32 DEST, so DEST0 holds exactly the reference's `post` value without the
//     pack/unpack round trip. silu_tile_init (silu_init<APPROX>: sigmoid_init<false> -> vConstFloatPrgm0 = 2.0
//     for the reciprocal) does not depend on the DEST mode.
//   * mul (ckernel_sfpu_binary.h, calculate_sfpu_binary_mul<APPROX, MUL, 8, false>): result = in0 * in1 (fp32
//     SFPMAD), then float32_to_bf16_rne(result), then result = 0 where in0 == 0 or in1 == 0. Operand order is
//     the reference's (DEST0 = silu(G) = binary_ng's LHS, DEST1 = U = RHS). In the reference SFPSTORE truncates
//     the (already bf16) result to the 16-bit DEST and the packer writes it out exactly; here the bf16-valued
//     fp32 lands in the fp32 DEST and the packer's fp32 -> bf16 conversion of a bf16-representable value is
//     exact. mul_binary_tile_init (sfpu_binary_init<MUL>) does nothing DEST-mode dependent.
//   So every value the reference rounds to bf16 (G, U, silu(G), the product) is rounded identically here, with
//   the same rounding mode at the same point, and nothing else is rounded.
//
// N_block_tiles must be even (enforced host-side). Like the reference's swiglu_block / copy_and_pack_block, the
// full M_block_tiles x N_block_tiles block is processed even when it is partial (the writer skips padding).
void swiglu_block_exact(
    uint32_t in_cb, uint32_t scratch_cb, uint32_t out_cb, uint32_t M_block_tiles, uint32_t N_block_tiles) {
    CircularBuffer cb_scratch(scratch_cb);
    CircularBuffer cb_out(out_cb);
    const uint32_t block_tiles = M_block_tiles * N_block_tiles;

    // Step 1: G / U tiles through the reference output stage, into the bf16 scratch CB.
    cb_scratch.reserve_back(block_tiles);
    copy_and_pack_block(in_cb, scratch_cb, M_block_tiles, N_block_tiles);
    cb_scratch.wait_front(block_tiles);

    // Step 2: silu16(G) * U per pair.
    reconfig_data_format_srca(in_cb, scratch_cb);
    pack_reconfig_data_format(scratch_cb, out_cb);

    constexpr uint32_t GATE_DST = 0;
    constexpr uint32_t UP_DST = 1;
    const uint32_t out_N_block_tiles = N_block_tiles >> 1;

    for (uint32_t m = 0; m < M_block_tiles; m++) {
        const uint32_t row_base = m * N_block_tiles;
        for (uint32_t p = 0; p < out_N_block_tiles; p++) {
            const uint32_t gate_tile_id = row_base + (p << 1);
            const uint32_t up_tile_id = gate_tile_id + 1;

            tile_regs_acquire();
            copy_init(scratch_cb);
            copy_tile(scratch_cb, gate_tile_id, GATE_DST);
            copy_tile(scratch_cb, up_tile_id, UP_DST);
            silu_tile_init();
            silu_tile<false>(GATE_DST);  // 16-bit-DEST instantiation: binary_ng's LHS activation
            mul_binary_tile_init();
            mul_binary_tile<false>(GATE_DST, UP_DST, GATE_DST);  // 16-bit-DEST instantiation: RNE + 0*x = 0
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(GATE_DST, out_cb);
            tile_regs_release();
        }
        cb_out.push_back(out_N_block_tiles);
    }
    cb_scratch.pop_front(block_tiles);
}
#endif  // FUSE_SWIGLU

// Slightly modified from compute_common.hpp
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

#ifdef KSPLIT
// KSPLIT (DIRECT_OUT with K in several blocks): one K block of every subblock of the output block. Blocks after the
// first reload the subblock's fp32 partials from `interm_cb` straight into DEST (the CB is UnpackToDestFp32, so the
// reload is exact) and keep accumulating; blocks before the last pack fp32 partials to `interm_cb`, the last packs the
// bf16 output. This is the accumulation sequence of ttnn's 2-D multicast matmul with fp32 DEST and exact partial
// reload, so every output tile is the same full-K in-order fp32 accumulation, rounded once to bf16 by the packer.
void matmul_blocks_ksplit(
    const uint32_t in0_cb,
    const uint32_t in1_cb,
    const uint32_t interm_cb,
    const uint32_t out_cb,
    const bool reload,
    const bool last,
    const uint32_t M_block_tiles,
    const uint32_t N_block_tiles,
    const uint32_t full_N_block_tiles,
    const uint32_t K_block_tiles,
    const uint32_t subblock_h,
    const uint32_t subblock_w) {
    const uint32_t dst_cb = last ? out_cb : interm_cb;
    uint32_t in0_index_offset = 0;
    for (uint32_t M_start = 0; M_start < M_block_tiles; M_start += subblock_h) {
        uint32_t in1_index_offset = 0;
        for (uint32_t N_start = 0; N_start < N_block_tiles; N_start += subblock_w) {
            tile_regs_acquire();
            if (reload) {
                reconfig_data_format_srca(in1_cb, interm_cb);
                copy_init(interm_cb);
                uint32_t dst = 0;
                for (uint32_t h = 0; h < subblock_h; h++) {
                    for (uint32_t w = 0; w < subblock_w; w++) {
                        copy_tile(interm_cb, (M_start + h) * full_N_block_tiles + N_start + w, dst++);
                    }
                }
                reconfig_data_format_srca(interm_cb, in1_cb);
                matmul_block_init(in0_cb, in1_cb, false /*transpose*/, subblock_w, subblock_h, K_block_tiles);
            }
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;
            for (uint32_t inner_dim = 0; inner_dim < K_block_tiles; inner_dim++) {
                matmul_block(
                    in0_cb, in1_cb, in0_index, in1_index, 0, false /*transpose*/, subblock_w, subblock_h, K_block_tiles);
                in0_index++;
                in1_index += full_N_block_tiles;
            }
            tile_regs_commit();
            tile_regs_wait();
            uint32_t dst = 0;
            for (uint32_t h = 0; h < subblock_h; h++) {
                for (uint32_t w = 0; w < subblock_w; w++) {
                    pack_tile<true>(dst++, dst_cb, (M_start + h) * full_N_block_tiles + N_start + w);
                }
            }
            tile_regs_release();
            in1_index_offset += subblock_w;
        }
        in0_index_offset += subblock_h * K_block_tiles;
    }
}
#endif  // KSPLIT

void kernel_main() {
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t M_blocks_per_core = get_compile_time_arg_val(4);
    constexpr uint32_t N_blocks_per_core = get_compile_time_arg_val(5);
    constexpr uint32_t subblock_h = get_compile_time_arg_val(6);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(7);

    uint32_t argidx = 0;
    const uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t N_end_tile = get_arg_val<uint32_t>(argidx++);

    constexpr uint32_t in0_cb = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb = tt::CBIndex::c_1;
    constexpr uint32_t out_cb = tt::CBIndex::c_2;
    constexpr uint32_t intermediate_cb = tt::CBIndex::c_3;
    // mm_swiglu addition: bf16 scratch for the reference-rounded G / U tiles (one M_block x N_block block).
    constexpr uint32_t scratch_cb = tt::CBIndex::c_7;

    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_out(out_cb);
    CircularBuffer cb_intermediate(intermediate_cb);

    // compute_kernel_hw_startup must be the first compute API call (before SFPU/op inits).
#ifdef DIRECT_OUT
    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_cb, in1_cb, out_cb);
#else
    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_cb, in1_cb, intermediate_cb);
#endif

    matmul_init(in0_cb, in1_cb);

    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    bool reuse_in0_block = false;

    uint32_t current_M_block_tiles = M_block_tiles;
    uint32_t current_N_block_tiles = N_block_tiles;
    uint32_t current_subblock_h = subblock_h;
    uint32_t current_subblock_w = subblock_w;

    for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
        uint32_t m_tile = M_start_tile + m_block_iter * M_block_tiles;
        uint32_t m_tile_end = std::min(m_tile + M_block_tiles, M_end_tile);
        current_M_block_tiles = m_tile_end - m_tile;
        current_subblock_h = std::min(current_M_block_tiles, subblock_h);

        for (uint32_t n_block_iter = 0; n_block_iter < N_blocks_per_core; n_block_iter++) {
            uint32_t n_tile = N_start_tile + n_block_iter * N_block_tiles;
            uint32_t n_tile_end = std::min(n_tile + N_block_tiles, N_end_tile);
            current_N_block_tiles = n_tile_end - n_tile;
            current_subblock_w = std::min(current_N_block_tiles, subblock_w);

#if defined(DIRECT_OUT) && defined(KSPLIT)
            {
                CircularBuffer cb_interm(intermediate_cb);
                matmul_block_init(
                    in0_cb, in1_cb, false /*transpose*/, current_subblock_w, current_subblock_h, K_block_tiles);
                cb_out.reserve_back(out_block_num_tiles);
                for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {
                    const bool first = k_block == 0;
                    const bool last = k_block == K_num_blocks - 1;
                    cb_in0.wait_front(in0_block_num_tiles);
                    cb_in1.wait_front(in1_block_num_tiles);
                    // The partials CB holds one block and is updated in place: each subblock's partials are
                    // reloaded (unpacker) before the same positions are repacked (packer, through DEST).
                    if (!first) {
                        cb_interm.wait_front(out_block_num_tiles);
                    } else if (!last) {
                        cb_interm.reserve_back(out_block_num_tiles);
                    }
                    pack_reconfig_data_format(last ? out_cb : intermediate_cb);
                    matmul_blocks_ksplit(
                        in0_cb,
                        in1_cb,
                        intermediate_cb,
                        out_cb,
                        !first,
                        last,
                        current_M_block_tiles,
                        current_N_block_tiles,
                        N_block_tiles,
                        K_block_tiles,
                        current_subblock_h,
                        current_subblock_w);
                    if (!first) {
                        cb_interm.pop_front(out_block_num_tiles);
                    }
                    if (!last) {
                        if (!first) {
                            cb_interm.reserve_back(out_block_num_tiles);
                        }
                        cb_interm.push_back(out_block_num_tiles);
                    }
                    cb_in0.pop_front(in0_block_num_tiles);
                    cb_in1.pop_front(in1_block_num_tiles);
                }
                cb_out.push_back(out_block_num_tiles);
            }
            continue;
#elif defined(DIRECT_OUT)
            // Plain matmul, K in one block (host-asserted): every output tile is one full-K accumulation in the fp32
            // DEST, packed straight to the bf16 output. Bit-identical to ttnn's 2-D multicast matmul with fp32 DEST,
            // whose cross-K-block partial reload is exact (UnpackToDestFp32), and whose last block packs DEST to bf16.
            matmul_block_init(
                in0_cb, in1_cb, false /*transpose*/, current_subblock_w, current_subblock_h, K_block_tiles);
            cb_in0.wait_front(in0_block_num_tiles);
            cb_in1.wait_front(in1_block_num_tiles);
            cb_out.reserve_back(out_block_num_tiles);
            matmul_blocks(
                in0_cb,
                in1_cb,
                out_cb,
                current_M_block_tiles,
                current_N_block_tiles,
                N_block_tiles,
                K_block_tiles,
                current_subblock_h,
                current_subblock_w);
            cb_out.push_back(out_block_num_tiles);
            if (n_block_iter == N_blocks_per_core - 1) {
                cb_in0.pop_front(in0_block_num_tiles);
            }
            cb_in1.pop_front(in1_block_num_tiles);
            continue;
#endif
            // Reconfig before init: on all but the first block the unpackers are still
            // configured for the previous output stage's operands (see #55052).
            reconfig_data_format(in1_cb, in0_cb);
            pack_reconfig_data_format(intermediate_cb);
            matmul_block_init(
                in0_cb,
                in1_cb,
                false /*transpose*/,
                current_subblock_w /*ct_dim*/,
                current_subblock_h /*rt_dim*/,
                K_block_tiles /*kt_dim*/);
            // Accumulation buffer
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
                    /**
                     * On next iteration we might get reuse on in0
                     *
                     */
                    if (n_block_iter < N_blocks_per_core - 1) {
                        // going to stride on N, so reuse in0
                        reuse_in0_block = true;
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
            // SwiGLU collapses the interleaved gate/up block to half its N width.
            cb_out.reserve_back(out_block_num_tiles >> 1);
            cb_intermediate.wait_front(out_block_num_tiles);
            swiglu_block_exact(intermediate_cb, scratch_cb, out_cb, M_block_tiles, N_block_tiles);
            cb_intermediate.pop_front(out_block_num_tiles);
#else
            cb_out.reserve_back(out_block_num_tiles);
            cb_intermediate.wait_front(out_block_num_tiles);
            copy_and_pack_block(intermediate_cb, out_cb, M_block_tiles, N_block_tiles);
            cb_intermediate.pop_front(out_block_num_tiles);
#endif  // FUSE_SWIGLU
        }
    }
}

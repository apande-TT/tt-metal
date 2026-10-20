// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute of tt/minimal_mm.py: ttnn.experimental.minimal_matmul's compute kernel (minimal_matmul/device/kernels/
// compute.cpp; no bias / ternary / fused activation, fp32 DEST) with its output stage regrouped. The matmul loop is the
// stock one. Each output block's fp32 L1 partials (intermediate_cb) leave through an epilogue that moves EPI_TILES tiles
// (SWIGLU_PAIRS gate/up pairs) a DEST acquire with its inits run once per block; the SwiGLU pair runs silu and the up
// multiply as ONE SFPU pass (calculate_silu_mul). Measured on the 416-row text gate/up: the SFPU is ~1.4 of its 9.8 ms,
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

#ifdef FUSE_SWIGLU
#ifdef TRISC_MATH
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
}  // namespace ckernel::sfpu
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
    silu_tile_init();
    mul_binary_tile_init();
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
            tile_regs_commit();
            tile_regs_wait();
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
#else
            cb_out.reserve_back(out_block_num_tiles);
            cb_intermediate.wait_front(out_block_num_tiles);
            copy_and_pack_block(intermediate_cb, out_cb, M_block_tiles, N_block_tiles);
            cb_intermediate.pop_front(out_block_num_tiles);
#endif
        }
    }
}

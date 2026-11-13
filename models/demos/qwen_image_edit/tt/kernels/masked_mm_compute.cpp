// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The masked-lane linear: y = sum_r lane_r(x) @ w, lane r of x keeping the K columns k % 8 == r -- the exact-lane sum
// of the precise linears without the 8x input copy of the K-stacked product. Per K tile k: each of this core's x
// tiles is masked into its 8 lanes in L1 (c_3; x * 0/1 on the FPU), then every output tile (m, n) of the N chunk
// takes the 8 lane products lane_r(x[m, k]) @ w[k, n] in the float32 DST (each tile product still a lane's exact
// one: <= 4 nonzero K terms per 32-group) and the packer adds them into the chunk's float32 block in L1 (c_16, k > 0).
// Compile-time: kt, nt, nc, the subblock width (2 x it <= the 4 float32 DST tiles), the most rows any core takes.
// Runtime: row count.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"

void kernel_main() {
    const uint32_t rows = get_arg_val<uint32_t>(0);
    constexpr uint32_t kt = get_compile_time_arg_val(0);
    constexpr uint32_t nt = get_compile_time_arg_val(1);
    constexpr uint32_t nc = get_compile_time_arg_val(2);
    constexpr uint32_t sbw = get_compile_time_arg_val(3);
    constexpr uint32_t lanes = get_compile_time_arg_val(5);  // column lanes of x (8 or 4; a multiple of 4)
    // 1: c_0 carries [x, lo] per row and the mask tile `lanes` is all ones -- lo's product is lane `lanes`
    constexpr uint32_t extra = get_compile_time_arg_val(6);
    constexpr uint32_t S = 1 + extra, L = lanes + extra;
    // 1: the reader pre-fills each output block with the bias (signalled on c_5): accumulate onto it from k = 0
    constexpr uint32_t has_bias = get_compile_time_arg_val(7);
    constexpr uint32_t cb_sig = 5;
    constexpr uint32_t cb_x = 0, cb_w = 1, cb_m = 2, cb_lane = 3, cb_out = 16;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_lane, cb_w, cb_out);
    cb_wait_front(cb_m, L);
    for (uint32_t n0 = 0; n0 < nt; n0 += nc) {
        cb_reserve_back(cb_out, rows * nc);
        if constexpr (has_bias) {
            cb_wait_front(cb_sig, 1);
        }
        for (uint32_t k = 0; k < kt; ++k) {
            cb_wait_front(cb_x, rows * S);
            cb_wait_front(cb_w, nc);

            // the 8 lanes of each x tile: x * mask_r (exact), packed bf16 to c_3 at m * 8 + r
            mul_init(cb_x, cb_m, false);
            pack_reconfig_data_format(cb_lane);
            pack_reconfig_l1_acc(0);
            cb_reserve_back(cb_lane, rows * L);
            for (uint32_t m = 0; m < rows; ++m) {
                for (uint32_t r0 = 0; r0 < lanes; r0 += 4) {
                    tile_regs_acquire();
                    for (uint32_t r = 0; r < 4; ++r) {
                        mul_tiles(cb_x, cb_m, m * S, r0 + r, r);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t r = 0; r < 4; ++r) {
                        pack_tile<true>(r, cb_lane, m * L + r0 + r);
                    }
                    tile_regs_release();
                }
                if constexpr (extra) {  // lo * ones: lo's tile into its lane slot
                    tile_regs_acquire();
                    mul_tiles(cb_x, cb_m, m * S + 1, lanes, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile<true>(0, cb_lane, m * L + lanes);
                    tile_regs_release();
                }
            }
            cb_push_back(cb_lane, rows * L);
            cb_pop_front(cb_x, rows * S);

            // the lane products, accumulated over the 8 lanes in DST and over k by the packer in L1: rt x 2 output
            // subblocks, one matmul_block per lane (its rt lane tiles of row stride 8 against the same 2 w tiles),
            // so each lane tile is unpacked once per subblock row and each w tile once per subblock column
            cb_wait_front(cb_lane, rows * L);
            pack_reconfig_data_format(cb_out);
            pack_reconfig_l1_acc((k > 0 || has_bias) ? 1 : 0);
            for (uint32_t m0 = 0; m0 < rows; m0 += 2) {
                const uint32_t rt = rows - m0 >= 2 ? 2 : 1;
                matmul_block_init(cb_lane, cb_w, 0, sbw, rt, L);
                for (uint32_t j0 = 0; j0 < nc; j0 += sbw) {
                    tile_regs_acquire();
                    for (uint32_t r = 0; r < L; ++r) {
                        matmul_block(cb_lane, cb_w, m0 * L + r, j0, 0, 0, sbw, rt, L);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t i = 0; i < rt; ++i) {
                        for (uint32_t j = 0; j < sbw; ++j) {
                            pack_tile<true>(i * sbw + j, cb_out, (m0 + i) * nc + j0 + j);
                        }
                    }
                    tile_regs_release();
                }
            }
            cb_pop_front(cb_lane, rows * L);
            cb_pop_front(cb_w, nc);
        }
        pack_reconfig_l1_acc(0);
        cb_push_back(cb_out, rows * nc);
        if constexpr (has_bias) {
            cb_pop_front(cb_sig, 1);
        }
    }
    cb_pop_front(cb_m, L);
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The compute kernel of the C++ decode down projection (see tt/cpp_down_dec.py): y = x @ w2, accumulated
// EXACTLY as the stock 1D-multicast matmul accumulates it (bmm_large_block_zm_fused_bias_activation with
// PACKER_L1_ACC + FP32_DEST_ACC_EN, no bias):
//   - every K block of KB tiles is summed into a FRESH DEST, k = 0 .. KB-1 in order;
//   - blocks 0 .. NB-2 are packed into the fp32 partials CB, block 0 overwriting and the later ones
//     accumulated by the packer (packer L1 acc);
//   - the LAST block first reloads the partials into DEST (cb_p is unpack-to-dest fp32, so the reload is
//     exact), sums its KB products on top, and packs the result without L1 acc.
// Same K block, same per-tile product order, same packer accumulation: the stock bits. Holding one DEST
// accumulation across every K block (compute.cpp) rounds differently, and that drift was enough to send a
// free-running greedy decode into a repeat loop on one demo row.
//
// The MT x PN output block is computed as PN / CT subblocks of MT x CT; cb_p holds exactly MT x PN tiles so
// every block's subblocks land on the same partials (the packer accumulates in place).
//
// The readers push CH K rows at a time and the first subblock waits for them cumulatively inside the K
// block, so the math follows the weight stream instead of waiting for a whole KB-row block (at KB = 32 a
// block is 104 KB a core, and a block-granular wait left the first and last blocks serial). The K block
// the accumulation is grouped by is still KB. One-tile-row activations only (MT == 1): the x reader's
// chunks are then the block's own tile order.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t MT = get_compile_time_arg_val(0);
    constexpr uint32_t KB = get_compile_time_arg_val(1);
    constexpr uint32_t NB = get_compile_time_arg_val(2);
    constexpr uint32_t PN = get_compile_time_arg_val(3);
    constexpr uint32_t CT = get_compile_time_arg_val(4);
    constexpr uint32_t CH = get_compile_time_arg_val(5);
    constexpr uint32_t NSUB = PN / CT;
    constexpr uint32_t SUB = MT * CT;
    static_assert(MT == 1, "chunked K waits assume a one-tile-row activation");
    static_assert(KB % CH == 0, "the stream chunk must divide the K block");

    constexpr auto cb_x = tt::CBIndex::c_0;
    constexpr auto cb_w = tt::CBIndex::c_1;
    constexpr auto cb_y = tt::CBIndex::c_16;
    constexpr auto cb_p = tt::CBIndex::c_24;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_p);
    matmul_block_init(cb_x, cb_w, 0, CT, MT, KB);

    bool reload = false;
    for (uint32_t b = 0; b < NB; ++b) {
        const bool last = b == NB - 1;
        for (uint32_t s = 0; s < NSUB; ++s) {
            tile_regs_acquire();
            if (reload) {
                reconfig_data_format_srca(cb_w, cb_p);
                copy_init(cb_p);
                cb_wait_front(cb_p, SUB);
                copy_block(cb_p, 0, 0, SUB);
                cb_pop_front(cb_p, SUB);
                reconfig_data_format_srca(cb_p, cb_w);
                matmul_block_init(cb_x, cb_w, 0, CT, MT, KB);
            }
            uint32_t i1 = s * CT;
            for (uint32_t k = 0; k < KB; ++k) {
                if (s == 0 && k % CH == 0) {
                    cb_wait_front(cb_x, MT * (k + CH));
                    cb_wait_front(cb_w, (k + CH) * PN);
                }
                matmul_block(cb_x, cb_w, k, i1, 0, 0, CT, MT, KB);
                i1 += PN;
            }
            tile_regs_commit();
            if (last) {
                cb_reserve_back(cb_y, SUB);
                tile_regs_wait();
                pack_reconfig_data_format(cb_y);
                pack_reconfig_l1_acc(0);
                for (uint32_t i = 0; i < SUB; ++i) {
                    pack_tile(i, cb_y);
                }
                tile_regs_release();
                cb_push_back(cb_y, SUB);
            } else {
                cb_reserve_back(cb_p, SUB);
                tile_regs_wait();
                if (b == 0) {
                    pack_reconfig_l1_acc(0);  // no accumulation for the first block
                } else if (b == 1) {
                    pack_reconfig_l1_acc(1);
                }
                for (uint32_t i = 0; i < SUB; ++i) {
                    pack_tile(i, cb_p);
                }
                tile_regs_release();
                cb_push_back(cb_p, SUB);
            }
        }
        // As stock: the partials of blocks 0 .. NB-3 are consumed in place by the next block's packer
        // accumulation; those of block NB-2 stay for the last block's reload.
        if (b + 2 < NB) {
            for (uint32_t s = 0; s < NSUB; ++s) {
                cb_wait_front(cb_p, SUB);
                cb_pop_front(cb_p, SUB);
            }
        }
        if (b + 2 == NB) {
            reload = true;
        }
        cb_pop_front(cb_x, MT * KB);
        cb_pop_front(cb_w, KB * PN);
    }
}

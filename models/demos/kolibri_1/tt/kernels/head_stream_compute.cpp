// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// LM head, one tile row of tokens (compute): the same arithmetic as ttnn's 1D-multicast matmul with in0_block_w = IB,
// 1 x SW output subblocks, fp32 dest and packer L1 accumulation (bmm_large_block_zm_fused_bias_activation.cpp), so
// the logits are bit-identical to ttnn.linear's. Per IB-tile K block each subblock is accumulated in dest and packed
// onto the fp32 partials with L1 accumulation (overwrite on the first block); for the last block the partials are
// reloaded into dest (unpacked as fp32), the block's products added on top, and the result packed to the output.
#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);  // K tiles
    constexpr uint32_t NC = get_compile_time_arg_val(1);  // output columns a core
    constexpr uint32_t SW = get_compile_time_arg_val(2);  // output subblock width
    constexpr uint32_t IB = get_compile_time_arg_val(3);  // K tiles a dest pass
    constexpr uint32_t cb_x = 0, cb_w = 1, cb_out = 16, cb_part = 24;
    constexpr uint32_t NBLK = Kt / IB, NSB = NC / SW;
    static_assert(Kt % IB == 0 && NC % SW == 0 && NBLK > 1, "head_stream blocking");

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_part);
    matmul_block_init(cb_x, cb_w, 0, SW, 1, IB);
    cb_wait_front(cb_x, Kt);

    bool reload = false;
    for (uint32_t blk = 0; blk < NBLK; ++blk) {
        const bool last = blk == NBLK - 1;
        cb_wait_front(cb_w, IB * NC);
        for (uint32_t s = 0; s < NSB; ++s) {
            tile_regs_acquire();
            if (reload) {
                reconfig_data_format_srca(cb_w, cb_part);
                copy_init(cb_part);
                cb_wait_front(cb_part, SW);
                copy_block(cb_part, 0, 0, SW);
                cb_pop_front(cb_part, SW);
                reconfig_data_format_srca(cb_part, cb_w);
                matmul_block_init(cb_x, cb_w, 0, SW, 1, IB);
            }
            uint32_t in0_index = blk * IB, in1_index = s * SW;
            for (uint32_t i = 0; i < IB; ++i) {
                matmul_block(cb_x, cb_w, in0_index, in1_index, 0, 0, SW, 1, IB);
                ++in0_index;
                in1_index += NC;
            }
            tile_regs_commit();
            if (last) {
                cb_reserve_back(cb_out, SW);
                tile_regs_wait();
                pack_reconfig_data_format(cb_out);
                pack_reconfig_l1_acc(0);
                pack_block(0, cb_out, SW);
                tile_regs_release();
                cb_push_back(cb_out, SW);
            } else {
                cb_reserve_back(cb_part, SW);
                tile_regs_wait();
                if (blk == 0) {
                    pack_reconfig_l1_acc(0);
                } else if (blk == 1) {
                    pack_reconfig_l1_acc(1);
                }
                pack_block(0, cb_part, SW);
                tile_regs_release();
                cb_push_back(cb_part, SW);
            }
        }
        // The partials CB holds exactly NC tiles, so draining it puts the next block's packs on the same addresses;
        // the second-to-last block's stay for the reload.
        if (blk + 2 < NBLK) {
            for (uint32_t s = 0; s < NSB; ++s) {
                cb_wait_front(cb_part, SW);
                cb_pop_front(cb_part, SW);
            }
        }
        if (blk + 2 == NBLK) {
            reload = true;
        }
        cb_pop_front(cb_w, IB * NC);
    }
    cb_pop_front(cb_x, Kt);
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// float32 x -> bf16 limbs (hi, lo), hi = bf16(x), lo = bf16(x - hi), in one pass per tile:
//   x (cb 0, float32, unpacked to DEST as float32) -> packed to hi (cb 16) and to a scratch copy (cb 24);
//   the scratch hi is widened back to float32 in DEST next to a second copy of x, subtracted with the
//   SFPU float32 subtract (round to nearest even), and packed to lo (cb 17).
// The packer's float32 -> bf16 narrowing and the SFPU subtract are the ones ttnn.typecast and
// ttnn.subtract use, so the limbs equal the typecast / subtract / typecast chain bit for bit.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_x = 0;
    constexpr uint32_t cb_hi = 16;
    constexpr uint32_t cb_lo = 17;
    constexpr uint32_t cb_tmp = 24;
    CircularBuffer x(cb_x);
    CircularBuffer hi(cb_hi);
    CircularBuffer lo(cb_lo);
    CircularBuffer tmp(cb_tmp);

    compute_kernel_hw_startup(cb_x, cb_hi);

    for (uint32_t t = 0; t < num_tiles; ++t) {
        x.wait_front(1);

        // hi = bf16(x), to the writer and to the scratch buffer
        hi.reserve_back(1);
        tmp.reserve_back(1);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_x);
        copy_init(cb_x);
        copy_tile(cb_x, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_hi);
        pack_tile(0, cb_tmp);
        tile_regs_release();
        hi.push_back(1);
        tmp.push_back(1);

        // lo = bf16(x - float32(hi))
        tmp.wait_front(1);
        lo.reserve_back(1);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_x);
        copy_init(cb_x);
        copy_tile(cb_x, 0, 0);
        reconfig_data_format_srca(cb_tmp);
        copy_init(cb_tmp);
        copy_tile(cb_tmp, 0, 1);
        sub_binary_tile_init();
        sub_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_lo);
        tile_regs_release();
        lo.push_back(1);
        tmp.pop_front(1);
        x.pop_front(1);
    }
}

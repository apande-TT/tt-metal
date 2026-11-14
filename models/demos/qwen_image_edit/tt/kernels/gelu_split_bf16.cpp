// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// float32 x -> bf16 limbs (hi, lo) of g = gelu_tanh(x) in one pass per tile (the feed-forward's GELU fused
// into the down projection's limb split, so g is never written to or read from DRAM):
//   x (cb 0, float32 -> DEST as float32), gelu_tanh_tile (the SFPU GELU ttnn.gelu(variant=Tanh) runs),
//   g packed to hi (cb 16, bf16), to a bf16 scratch (cb 24) and to a float32 scratch (cb 25);
//   then g (float32) and hi (widened) are reloaded, subtracted with the SFPU float32 subtract and packed
//   to lo (cb 17). Same limbs as kernels/split_bf16.cpp applied to ttnn.gelu's output.
// hi is rounded in DEST (round_bf16_packer.h: as the packer narrows, on a DEST copy of the float32 value) and lo is
// the SFPU float32 subtract of it, both packed from DEST: no bf16 / float32 scratch CB is packed and unpacked back
// (that srcA reload next to an unpack-to-DEST in one acquire deadlocked the unpacker / math handshake on rare tiles).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "round_bf16_packer.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_x = 0;
    constexpr uint32_t cb_hi = 16;
    constexpr uint32_t cb_lo = 17;
    constexpr uint32_t cb_tmp = 24;
    constexpr uint32_t cb_g = 25;
    CircularBuffer x(cb_x);
    CircularBuffer hi(cb_hi);
    CircularBuffer lo(cb_lo);
    CircularBuffer tmp(cb_tmp);
    CircularBuffer g(cb_g);

    compute_kernel_hw_startup(cb_x, cb_hi);

    for (uint32_t t = 0; t < num_tiles; ++t) {
        x.wait_front(1);

        // g = gelu(x) -> hi (bf16), scratch hi (bf16), scratch g (float32)
        hi.reserve_back(1);
        lo.reserve_back(1);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_x);
        copy_init(cb_x);
        copy_tile(cb_x, 0, 0);
        gelu_tanh_tile_init();
        gelu_tanh_tile(0);
        copy_dest_values_init();
        copy_dest_values<DataFormat::Float32>(0, 1);
        round_bf16_packer_tile_init();
        round_bf16_packer_tile(1);
        sub_binary_tile_init();
        sub_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_hi);
        pack_tile(1, cb_hi);
        pack_reconfig_data_format(cb_lo);
        pack_tile(0, cb_lo);
        tile_regs_release();
        hi.push_back(1);
        lo.push_back(1);
        x.pop_front(1);
    }
}

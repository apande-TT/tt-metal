// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// float32 x -> bf16 limbs (hi, lo), hi = bf16(x), lo = bf16(x - hi), in one pass per tile, all in DEST:
//   x (cb 0, float32, unpacked to DEST as float32) is copied to DEST tiles 0 and 1; tile 1 is rounded to bf16 in
//   place by the SFPU as the packer's narrowing rounds (round_bf16_packer.h; kept as float32 in DEST), subtracted from
//   tile 0 with the SFPU float32 subtract (exact: x - hi), and the two tiles are packed to hi (cb 16, exact: tile 1
//   holds bf16 values) and lo (cb 17, the packer's narrowing).
// Every unpack is the same float32 unpack-to-DEST of cb 0: the earlier variant packed hi to a bf16 scratch CB and
// unpacked it back through srcA within the same DEST acquire as an unpack-to-DEST of x, and that mix of the two
// unpack paths deadlocked the unpacker / math handshake on rare tiles (one core of one chip, long inputs).
// hi, lo are bit-identical to that variant's (hi packed by the packer, lo = packer(x - float32(hi))).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "round_bf16_packer.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_x = 0;
    constexpr uint32_t cb_hi = 16;
    constexpr uint32_t cb_lo = 17;
    CircularBuffer x(cb_x);
    CircularBuffer hi(cb_hi);
    CircularBuffer lo(cb_lo);

    compute_kernel_hw_startup(cb_x, cb_hi);

    for (uint32_t t = 0; t < num_tiles; ++t) {
        x.wait_front(1);
        hi.reserve_back(1);
        lo.reserve_back(1);
        tile_regs_acquire();
        copy_init(cb_x);
        copy_tile(cb_x, 0, 0);
        copy_tile(cb_x, 0, 1);
        round_bf16_packer_tile_init();
        round_bf16_packer_tile(1);
        sub_binary_tile_init();
        sub_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(1, cb_hi);
        pack_tile(0, cb_lo);
        tile_regs_release();
        hi.push_back(1);
        lo.push_back(1);
        x.pop_front(1);
    }
}

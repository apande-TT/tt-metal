// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Local sum of the exact all_to_all reduce: out tile = ((p0 + p1) + p2) + ... + p_{n-1}, the source
// tiles arriving in order on cb 0. float32 throughout: the tiles are unpacked to DEST as float32
// (UnpackToDestFp32) and added with the SFPU float32 add (round to nearest even), the same add and
// order as the ttnn.add chain it replaces.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t n_blocks = get_compile_time_arg_val(0);

    constexpr uint32_t cb_in_id = 0;
    constexpr uint32_t cb_out_id = 16;
    CircularBuffer cb_in(cb_in_id);
    CircularBuffer cb_out(cb_out_id);

    compute_kernel_hw_startup(cb_in_id, cb_out_id);
    copy_init(cb_in_id);
    add_binary_tile_init();

    for (uint32_t t = 0; t < num_tiles; ++t) {
        cb_out.reserve_back(1);
        tile_regs_acquire();
        cb_in.wait_front(1);
        copy_init(cb_in_id);
        copy_tile(cb_in_id, 0, 0);
        cb_in.pop_front(1);
        for (uint32_t i = 1; i < n_blocks; ++i) {
            cb_in.wait_front(1);
            copy_init(cb_in_id);
            copy_tile(cb_in_id, 0, 1);
            cb_in.pop_front(1);
            add_binary_tile_init();
            add_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out_id);
        tile_regs_release();
        cb_out.push_back(1);
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The compute kernel of the C++ stride-2 transposed convolution (see tt/cpp_upsample2.py): each output tile is
// its gathered "now" tile plus its gathered "delayed" tile, in float32 -- the same SFPU add
// (add_binary_tile<NearestEven>, operands unpacked straight to DEST) the stock ttnn.add(now, delayed) runs.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"

void kernel_main() {
    const uint32_t nu = get_arg_val<uint32_t>(0);

    constexpr auto cb_now = tt::CBIndex::c_0;
    constexpr auto cb_delayed = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup(cb_now, cb_delayed, cb_out);
    for (uint32_t u = 0; u < nu; ++u) {
        tile_regs_acquire();
        cb_wait_front(cb_now, 1);
        copy_init(cb_now);
        copy_tile(cb_now, 0, 0);
        cb_pop_front(cb_now, 1);
        cb_wait_front(cb_delayed, 1);
        copy_init(cb_delayed);
        copy_tile(cb_delayed, 0, 1);
        cb_pop_front(cb_delayed, 1);
        add_binary_tile_init();
        add_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
        tile_regs_commit();
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
}

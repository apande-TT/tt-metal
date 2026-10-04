// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The compute kernel of the C++ decode context merge (see tt/cpp_ctx_merge.py): each merged tile is its gathered
// P@V rows divided by their column-filled row sums in float32 -- the same SFPU divide (div_binary_tile, operands
// unpacked straight to DEST) the stock ttnn.divide(pv, sum) runs.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"

void kernel_main() {
    const uint32_t nu = get_arg_val<uint32_t>(0);

    constexpr auto cb_num = tt::CBIndex::c_0;
    constexpr auto cb_den = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup(cb_num, cb_den, cb_out);
    for (uint32_t u = 0; u < nu; ++u) {
        cb_wait_front(cb_num, 1);
        cb_wait_front(cb_den, 1);
        cb_reserve_back(cb_out, 1);
        tile_regs_acquire();
        copy_init(cb_num);
        copy_tile(cb_num, 0, 0);
        copy_init(cb_den);
        copy_tile(cb_den, 0, 1);
        div_binary_tile_init();
        div_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_num, 1);
        cb_pop_front(cb_den, 1);
    }
}

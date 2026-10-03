// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The compute kernel of the C++ tail-row regroup (see tt/cpp_tail_rows.py): each gathered bf16 tile is
// copied to DEST and packed in the cache dtype (bf8_b) -- the packer conversion the stock
// RM -> TILE tilize ends with.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"

void kernel_main() {
    const uint32_t nu = get_arg_val<uint32_t>(0);
    constexpr uint32_t DT = get_compile_time_arg_val(0);

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup(cb_in, cb_out);
    copy_init(cb_in);
    for (uint32_t u = 0; u < nu; ++u) {
        cb_wait_front(cb_in, DT);
        cb_reserve_back(cb_out, DT);
        tile_regs_acquire();
        for (uint32_t c = 0; c < DT; ++c) {
            copy_tile(cb_in, c, c);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t c = 0; c < DT; ++c) {
            pack_tile(c, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, DT);
        cb_pop_front(cb_in, DT);
    }
}

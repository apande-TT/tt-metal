// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// cb 16 tile i = cb 0 tile i + cb 1 tile i, in fp32 (inputs unpacked straight to DST).
// runtime args: n
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_a = 0, cb_b = 1, cb_out = 16;
    const uint32_t n = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_a, cb_out);
    for (uint32_t i = 0; i < n; ++i) {
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);
        cb_reserve_back(cb_out, 1);
        tile_regs_acquire();
        copy_init(cb_a);
        copy_tile(cb_a, 0, 0);
        copy_init(cb_b);
        copy_tile(cb_b, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);
    }
}

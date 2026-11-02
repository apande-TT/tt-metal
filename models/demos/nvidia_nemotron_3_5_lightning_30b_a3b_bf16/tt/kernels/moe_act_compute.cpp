// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// MoE decode between the up and down matmuls: cb 0 = n tiles of relu(x W_up) (any float format), cb 1 (with SCALE) =
// the fp32 tile of each row's routing weight for the expert those tiles belong to
//   ->  cb 16 = relu(.)^2 (times the weight), computed in fp32 and packed to cb 16's format.
// compile-time args: SCALE; runtime args: n
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t SCALE = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in = 0, cb_r = 1, cb_out = 16;
    const uint32_t n = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_in, cb_out);
    if constexpr (SCALE) {
        cb_wait_front(cb_r, 1);
    }
    for (uint32_t i = 0; i < n; ++i) {
        cb_wait_front(cb_in, 1);
        cb_reserve_back(cb_out, 1);
        tile_regs_acquire();
        copy_init(cb_in);
        copy_tile(cb_in, 0, 0);
        mul_binary_tile_init();
        mul_binary_tile(0, 0, 0);
        if constexpr (SCALE) {
            reconfig_data_format_srca(cb_in, cb_r);
            copy_init(cb_r);
            copy_tile(cb_r, 0, 1);
            reconfig_data_format_srca(cb_r, cb_in);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in, 1);
    }
    if constexpr (SCALE) {
        cb_pop_front(cb_r, 1);
    }
}

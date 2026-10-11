// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode down projection over the ROUTED experts only, phase 2 (compute). Per output tile: the routed experts'
// fp32 partials summed in expert order in fp32 dest (unpacked straight to dest), plus the shared expert's partial,
// packed bf16 once.
#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t Nt = get_compile_time_arg_val(0);
    constexpr uint32_t NC = get_compile_time_arg_val(1);
    constexpr uint32_t B = get_compile_time_arg_val(2);
    const uint32_t core_id = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_p = 0, cb_s = 1, cb_count = 4, cb_out = 16;

    compute_kernel_hw_startup(cb_p, cb_out);

    cb_wait_front(cb_count, 1);
    const uint32_t R = read_tile_value(cb_count, 0, 0);
    cb_pop_front(cb_count, 1);

    for (uint32_t n = core_id; n < Nt; n += NC) {
        tile_regs_acquire();
        // dest 0 = shared partial, then + each routed partial in expert order.
        reconfig_data_format_srca(cb_s);
        copy_init(cb_s);
        cb_wait_front(cb_s, 1);
        copy_tile(cb_s, 0, 0);
        cb_pop_front(cb_s, 1);
        reconfig_data_format_srca(cb_p);
        for (uint32_t r0 = 0; r0 < R; r0 += B) {
            const uint32_t b = R - r0 < B ? R - r0 : B;
            cb_wait_front(cb_p, b);
            for (uint32_t i = 0; i < b; ++i) {
                copy_init(cb_p);
                copy_tile(cb_p, i, 1);
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }
            cb_pop_front(cb_p, b);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
    }
}

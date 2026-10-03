// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The compute kernel of the C++ convolution shift-add (see tt/cpp_shift_add.py): the K shifted tap tiles
// of a unit summed in tap order in float32 -- acc = t0; acc = acc + t1; ... -- the same SFPU add
// (add_binary_tile<NearestEven>, operands unpacked straight to DEST) the stock per-tap ttnn.add chain runs.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"

void kernel_main() {
    const uint32_t nu = get_arg_val<uint32_t>(0);
    constexpr uint32_t K = get_compile_time_arg_val(0);

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup(cb_in, cb_out);
    for (uint32_t u = 0; u < nu; ++u) {
        tile_regs_acquire();
        cb_wait_front(cb_in, 1);
        copy_init(cb_in);
        copy_tile(cb_in, 0, 0);
        cb_pop_front(cb_in, 1);
        for (uint32_t k = 1; k < K; ++k) {
            cb_wait_front(cb_in, 1);
            copy_init(cb_in);
            copy_tile(cb_in, 0, 1);
            cb_pop_front(cb_in, 1);
            add_binary_tile_init();
            add_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
}

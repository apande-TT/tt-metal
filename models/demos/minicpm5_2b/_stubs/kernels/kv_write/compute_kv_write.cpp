// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode KV-cache write, compute: d = new - old (packed to bf16), out = old + d, bf16 DEST.
// The same FPU calls ttnn.subtract / ttnn.add run for bf16 operands in binary_ng's default
// (fast_and_approximate_mode) path -- binary_tiles_init<true, ELWSUB/ELWADD> + sub_tiles / add_tiles with
// fp32_dest_acc_en = false and the intermediate rounded to bf16 by the packer -- so the slot value written
// is bit-identical to the multi-op path it replaces.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/compute_kernel_hw_startup.h"

using std::uint32_t;

void kernel_main() {
    constexpr uint32_t Dt = get_compile_time_arg_val(0);
    constexpr uint32_t cb_old = get_compile_time_arg_val(1);
    constexpr uint32_t cb_new = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t cb_diff = get_compile_time_arg_val(4);

    const uint32_t num_tiles = get_arg_val<uint32_t>(0) * Dt;

    compute_kernel_hw_startup(cb_new, cb_old, cb_out);

    for (uint32_t t = 0; t < num_tiles; ++t) {
        cb_wait_front(cb_new, 1);
        cb_wait_front(cb_old, 1);

        // d = new - old
        cb_reserve_back(cb_diff, 1);
        binary_tiles_init<true, EltwiseBinaryType::ELWSUB>(cb_new, cb_old);
        tile_regs_acquire();
        sub_tiles(cb_new, cb_old, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_diff);
        tile_regs_release();
        cb_push_back(cb_diff, 1);
        cb_pop_front(cb_new, 1);

        // out = old + d
        cb_wait_front(cb_diff, 1);
        cb_reserve_back(cb_out, 1);
        binary_tiles_init<true, EltwiseBinaryType::ELWADD>(cb_old, cb_diff);
        tile_regs_acquire();
        add_tiles(cb_old, cb_diff, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_diff, 1);
        cb_pop_front(cb_old, 1);
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode attention bmm compute: each output tile is one uninterrupted fp32 DEST accumulation over
// k = 0..Kt-1 in order (DST += in0[k] @ in1[k], in1 transposed by the unpacker for scores), packed to bf16 --
// the arithmetic of the reuse-matmul it replaces, whose in0 block spans all of K (no partial reload).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/compute_kernel_hw_startup.h"

using std::uint32_t;

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t transpose_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_in1 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);

    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);
    matmul_init(cb_in0, cb_in1, transpose_b);

    for (uint32_t t = 0; t < num_tiles; ++t) {
        cb_wait_front(cb_in0, Kt);
        cb_wait_front(cb_in1, Kt);
        tile_regs_acquire();
        for (uint32_t k = 0; k < Kt; ++k) {
            matmul_tiles(cb_in0, cb_in1, k, k, 0);
        }
        tile_regs_commit();
        cb_pop_front(cb_in0, Kt);
        cb_pop_front(cb_in1, Kt);
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
}

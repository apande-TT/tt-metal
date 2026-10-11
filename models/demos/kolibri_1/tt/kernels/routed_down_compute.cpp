// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode down projection over the ROUTED experts only, phase 1 (compute). Per item: the expert's [32, inter] act
// row times its [inter, NB * 32] weight block, each output tile accumulated over the expert's CPE K tiles in fp32
// dest (NB tiles: full-sync dest), packed fp32 as the expert's partial.
#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t CPE = get_compile_time_arg_val(0);
    constexpr uint32_t NB = get_compile_time_arg_val(1);
    constexpr uint32_t cb_act = 0, cb_w = 1, cb_count = 4, cb_out = 16;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_act, cb_w, cb_out);
    matmul_init(cb_act, cb_w);

    cb_wait_front(cb_count, 1);
    const uint32_t n = read_tile_value(cb_count, 0, 0);
    cb_pop_front(cb_count, 1);

    for (uint32_t s = 0; s < n; ++s) {
        cb_wait_front(cb_act, CPE);
        cb_wait_front(cb_w, CPE * NB);
        tile_regs_acquire();
        for (uint32_t k = 0; k < CPE; ++k) {
            for (uint32_t j = 0; j < NB; ++j) {
                matmul_tiles(cb_act, cb_w, k, k * NB + j, j);
            }
        }
        tile_regs_commit();
        cb_pop_front(cb_act, CPE);
        cb_pop_front(cb_w, CPE * NB);
        tile_regs_wait();
        cb_reserve_back(cb_out, NB);
        for (uint32_t j = 0; j < NB; ++j) {
            pack_tile(j, cb_out);
        }
        cb_push_back(cb_out, NB);
        tile_regs_release();
    }
}

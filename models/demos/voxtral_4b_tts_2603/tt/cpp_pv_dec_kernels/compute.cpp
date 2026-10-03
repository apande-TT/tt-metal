// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The compute kernel of the C++ decode attention P@V (see tt/cpp_pv_dec.py): per unit, context tile
// d = sum over the span tiles j of e[j] @ v[j, d], all DT context tiles held in fp32 DEST across the
// whole span and summed in j order -- the stock reuse bmm's one-K-block accumulation.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"

void kernel_main() {
    const uint32_t nu = get_arg_val<uint32_t>(0);

    constexpr uint32_t DT = get_compile_time_arg_val(0);
    constexpr uint32_t ST = get_compile_time_arg_val(1);
    constexpr uint32_t PAD = get_compile_time_arg_val(2);  // pad span tiles the reader pushes after the ST

    constexpr auto cb_e = tt::CBIndex::c_0;
    constexpr auto cb_v = tt::CBIndex::c_1;
    constexpr auto cb_y = tt::CBIndex::c_16;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_e, cb_v, cb_y);
    matmul_init(cb_e, cb_v);

    for (uint32_t u = 0; u < nu; ++u) {
        tile_regs_acquire();
        for (uint32_t j = 0; j < ST; ++j) {
            cb_wait_front(cb_e, 1);
            cb_wait_front(cb_v, DT);
            for (uint32_t d = 0; d < DT; ++d) {
                matmul_tiles(cb_e, cb_v, 0, d, d);
            }
            cb_pop_front(cb_e, 1);
            cb_pop_front(cb_v, DT);
        }
        tile_regs_commit();
        if constexpr (PAD > 0) {
            cb_wait_front(cb_e, PAD);
            cb_pop_front(cb_e, PAD);
            cb_wait_front(cb_v, PAD * DT);
            cb_pop_front(cb_v, PAD * DT);
        }
        cb_reserve_back(cb_y, DT);
        tile_regs_wait();
        for (uint32_t d = 0; d < DT; ++d) {
            pack_tile(d, cb_y);
        }
        tile_regs_release();
        cb_push_back(cb_y, DT);
    }
}

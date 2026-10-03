// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The compute kernel of the C++ decode attention scores (see tt/cpp_scores_dec.py): per unit,
// score tile j = sum over the DT head_dim tiles of q[d] @ k[j, d]^T, the key tile transposed by the
// unpacker (matmul_init's transpose flag), summed in fp32 DEST in d order -- the stock reuse bmm's
// one-K-block accumulation.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"

void kernel_main() {
    const uint32_t nu = get_arg_val<uint32_t>(0);

    constexpr uint32_t DT = get_compile_time_arg_val(0);
    constexpr uint32_t ST = get_compile_time_arg_val(1);
    constexpr uint32_t PAD = get_compile_time_arg_val(2);  // pad key rows the reader pushes after the ST

    constexpr auto cb_q = tt::CBIndex::c_0;
    constexpr auto cb_k = tt::CBIndex::c_1;
    constexpr auto cb_y = tt::CBIndex::c_16;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q, cb_k, cb_y);
    matmul_init(cb_q, cb_k, 1);

    for (uint32_t u = 0; u < nu; ++u) {
        cb_wait_front(cb_q, DT);
        for (uint32_t j = 0; j < ST; ++j) {
            cb_wait_front(cb_k, DT);
            tile_regs_acquire();
            for (uint32_t d = 0; d < DT; ++d) {
                matmul_tiles(cb_q, cb_k, d, d, 0);
            }
            tile_regs_commit();
            cb_pop_front(cb_k, DT);
            cb_reserve_back(cb_y, 1);
            tile_regs_wait();
            pack_tile(0, cb_y);
            tile_regs_release();
            cb_push_back(cb_y, 1);
        }
        if constexpr (PAD > 0) {
            cb_wait_front(cb_k, PAD * DT);
            cb_pop_front(cb_k, PAD * DT);
        }
        cb_pop_front(cb_q, DT);
    }
}

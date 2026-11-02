// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Mamba2 decode, before the recurrence (fp32 throughout, inputs unpacked straight to DST):
//   conv cores, one tile of the conv channels each: cb 0 = x_t, x_{t-1}, x_{t-2}, x_{t-3}; cb 1 = the four taps
//     and the bias, rows replicated ->  silu(sum_s x_{t-s} tap_s + bias)
//   the dt core: cb 0 = the dt pre-activation tile; cb 1 = dt_bias, A, rows replicated
//     ->  dt = softplus(dt + dt_bias) = log1p(exp(.)), then exp(dt A)
// runtime arg: 0 conv, 1 dt; K (conv taps) compile time.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/log1p.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"

namespace {
constexpr uint32_t cb_x = 0, cb_k = 1, cb_out = 16, cb_out2 = 17;

ALWI void load(uint32_t cb, uint32_t tile, uint32_t dst) {
    copy_init(cb);
    copy_tile(cb, tile, dst);
}
ALWI void mul(uint32_t a, uint32_t b, uint32_t out) {
    mul_binary_tile_init();
    mul_binary_tile(a, b, out);
}
ALWI void add(uint32_t a, uint32_t b, uint32_t out) {
    add_binary_tile_init();
    add_binary_tile(a, b, out);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    const uint32_t role = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_x, cb_out);
    if (role == 0) {
        cb_wait_front(cb_x, K);
        cb_wait_front(cb_k, K + 1);
        cb_reserve_back(cb_out, 1);
        tile_regs_acquire();
        load(cb_x, 0, 0);
        load(cb_k, 0, 1);
        mul(0, 1, 0);
        for (uint32_t s = 1; s < K; ++s) {
            load(cb_x, s, 1);
            load(cb_k, s, 2);
            mul(1, 2, 1);
            add(0, 1, 0);
        }
        load(cb_k, K, 1);
        add(0, 1, 0);
        silu_tile_init();
        silu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_x, K);
        cb_pop_front(cb_k, K + 1);
    } else {
        cb_wait_front(cb_x, 1);
        cb_wait_front(cb_k, 2);
        cb_reserve_back(cb_out, 1);
        cb_reserve_back(cb_out2, 1);
        tile_regs_acquire();
        load(cb_x, 0, 0);
        load(cb_k, 0, 1);
        add(0, 1, 0);
        exp_tile_init<false>();
        exp_tile<false>(0);
        log1p_tile_init<false>();
        log1p_tile<false>(0);
        load(cb_k, 1, 1);
        mul(0, 1, 1);
        exp_tile_init<false>();
        exp_tile<false>(1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        pack_tile(1, cb_out2);
        tile_regs_release();
        cb_push_back(cb_out, 1);
        cb_push_back(cb_out2, 1);
        cb_pop_front(cb_x, 1);
        cb_pop_front(cb_k, 2);
    }
}

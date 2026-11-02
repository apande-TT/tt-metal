// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Mamba2 decode step, math (layout and inputs in ssm_step_reader.cpp). Per state tile (n, c), with dt, decay, B and
// C per user (tile row), each held replicated along its tile row:
//     S <- S * decay + (x * dt) * B_n          (written back by ssm_step_writer.cpp)
//     y_c += S * C_n                           (summed over this core's n by the packer's L1 accumulation; the
//                                               row ranges' partials summed by mamba_post_compute.cpp)
// The update is one SFPU pass over the tile (ssm_update). Everything stays fp32: every input is unpacked straight
// to DST (UnpackToDestFp32) and the products run on the SFPU, so the recurrent state is never rounded below fp32.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary_sfpu_macros.h"
#include "sfpi.h"

namespace ckernel::sfpu {
// DST tiles: 0 S, 1 x dt, 2 decay (left faces), 3 B_n (left faces) | C_n (right faces). In place: tile 0 <- the new
// S, tile 1 <- S C_n. The right face of a row band holds the same rows as its left face, so one set of per-row
// values serves both.
inline void ssm_update(const std::uint32_t, const std::uint32_t, const std::uint32_t) {
    constexpr std::uint32_t T = 32;  // sfpi rows per DST tile; a face is 8
#pragma GCC unroll 0
    for (int h = 0; h < 2; ++h) {
#pragma GCC unroll 8
        for (int j = 0; j < 8; ++j) {
            const sfpi::vFloat a = sfpi::dst_reg[2 * T];
            const sfpi::vFloat b = sfpi::dst_reg[3 * T];
            const sfpi::vFloat c = sfpi::dst_reg[3 * T + 8];
            const sfpi::vFloat s0 = sfpi::dst_reg[0] * a + sfpi::dst_reg[T] * b;
            sfpi::dst_reg[0] = s0;
            sfpi::dst_reg[T] = s0 * c;
            const sfpi::vFloat s1 = sfpi::dst_reg[8] * a + sfpi::dst_reg[T + 8] * b;
            sfpi::dst_reg[8] = s1;
            sfpi::dst_reg[T + 8] = s1 * c;
            sfpi::dst_reg++;
        }
#pragma GCC unroll 8
        for (int j = 0; j < 8; ++j) {
            sfpi::dst_reg++;  // past the right faces, to the next row band pair
        }
    }
}
}  // namespace ckernel::sfpu
#endif

namespace {
constexpr uint32_t cb_s = 0, cb_dec = 1, cb_x = 2, cb_dt = 3, cb_bc = 4, cb_xdt = 6;
constexpr uint32_t cb_so = 16, cb_y = 17;
}  // namespace

void kernel_main() {
    constexpr uint32_t NC = get_compile_time_arg_val(0);
    const uint32_t nn = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_s, cb_so);

    // x * dt for this core's column tiles, once
    cb_wait_front(cb_x, NC);
    cb_wait_front(cb_dt, NC);
    cb_reserve_back(cb_xdt, NC);
    for (uint32_t i = 0; i < NC; ++i) {
        tile_regs_acquire();
        copy_init(cb_x);
        copy_tile(cb_x, i, 0);
        copy_tile(cb_dt, i, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_xdt);
        tile_regs_release();
    }
    cb_push_back(cb_xdt, NC);
    cb_pop_front(cb_x, NC);
    cb_pop_front(cb_dt, NC);
    cb_wait_front(cb_xdt, NC);
    cb_wait_front(cb_dec, NC);

    cb_reserve_back(cb_y, NC);
    for (uint32_t k = 0; k < nn; ++k) {
        cb_wait_front(cb_bc, 1);
        cb_wait_front(cb_s, NC);
        cb_reserve_back(cb_so, NC);
        for (uint32_t i = 0; i < NC; ++i) {
            tile_regs_acquire();
            copy_init(cb_s);
            copy_tile(cb_s, i, 0);
            copy_tile(cb_xdt, i, 1);
            copy_tile(cb_dec, i, 2);
            copy_tile(cb_bc, 0, 3);
            MATH((::ckernel::llk_math_eltwise_binary_sfpu_init<::SfpuType::unused>()));
            MATH((_llk_math_eltwise_binary_sfpu_params_(::ckernel::sfpu::ssm_update, 0, 1, 0, VectorMode::None)));
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_so);
            pack_reconfig_l1_acc(k > 0 ? 1 : 0);
            pack_tile<true>(1, cb_y, i);
            pack_reconfig_l1_acc(0);
            tile_regs_release();
        }
        cb_push_back(cb_so, NC);
        cb_pop_front(cb_s, NC);
        cb_pop_front(cb_bc, 1);
    }
    cb_push_back(cb_y, NC);
    cb_pop_front(cb_xdt, NC);
    cb_pop_front(cb_dec, NC);
}

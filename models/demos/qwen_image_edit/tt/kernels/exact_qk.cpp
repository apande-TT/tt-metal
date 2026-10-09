// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Exact-lane a @ b^T of two-limb operands, all three lane patterns and their median, in one program:
//   est_p = sum over terms (i, j) in {(0, 0), (0, 1), (1, 0)}, lanes r = 0..7 of (a_i * mask_{p,r}) @ b_j^T
//   out   = median(est_0, est_1, est_2) = max(min(e0, e1), min(max(e0, e1), e2))
// The folded path (_precise.exact_matmul_bt with FOLD_LANES) builds the K-folded lane copies with a concat
// and one mask multiply per pattern and runs one matmul per pattern over 24 * K; here the lane copies are
// made per row block as exact matmuls with the diagonal 0/1 lane tiles (as lane_bmm.cpp), each pattern's
// 24 (term, lane) blocks accumulate in its own float32 DEST tile in the folded K order, and the median is
// taken in DEST with the SFPU float32 min / max. Replaces concat + 3 multiplies + 3 matmuls + 4 min / max.

#include <cstdint>

#include "api/compute/binary_max_min.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    const uint32_t start_unit = get_arg_val<uint32_t>(0);
    const uint32_t num_units = get_arg_val<uint32_t>(1);

    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Nt = get_compile_time_arg_val(2);
    constexpr uint32_t chunks = get_compile_time_arg_val(3);
    constexpr uint32_t csize = get_compile_time_arg_val(4);
    constexpr uint32_t lanes = 8;
    constexpr uint32_t patterns = 3;
    // lane copy (pattern p, limb i, lane r, K tile kt) -> tile ((p * 2 + i) * 8 + r) * Kt + kt of cb_m
    constexpr uint32_t copies = patterns * 2 * lanes * Kt;

    constexpr uint32_t cb_a = 0;
    constexpr uint32_t cb_b = 1;
    constexpr uint32_t cb_d = 2;
    constexpr uint32_t cb_m = 3;
    constexpr uint32_t cb_out = 16;
    CircularBuffer a(cb_a);
    CircularBuffer b(cb_b);
    CircularBuffer d(cb_d);
    CircularBuffer m(cb_m);
    CircularBuffer out(cb_out);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_d, cb_m);
    d.wait_front(patterns * lanes * Kt);

    for (uint32_t u = start_unit; u < start_unit + num_units; ++u) {
        const uint32_t c = u % chunks;
        const uint32_t nt0 = c * csize;
        const uint32_t nt1 = nt0 + csize < Nt ? nt0 + csize : Nt;

        a.wait_front(2 * Kt);
        pack_reconfig_data_format(cb_m);
        matmul_init(cb_a, cb_d, 0);
        for (uint32_t p = 0; p < patterns; ++p) {
            for (uint32_t i = 0; i < 2; ++i) {
                for (uint32_t r = 0; r < lanes; ++r) {
                    for (uint32_t kt = 0; kt < Kt; ++kt) {
                        m.reserve_back(1);
                        tile_regs_acquire();
                        matmul_tiles(cb_a, cb_d, i * Kt + kt, (p * lanes + r) * Kt + kt, 0);
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_tile(0, cb_m);
                        tile_regs_release();
                        m.push_back(1);
                    }
                }
            }
        }
        a.pop_front(2 * Kt);
        m.wait_front(copies);

        pack_reconfig_data_format(cb_out);
        for (uint32_t nt = nt0; nt < nt1; ++nt) {
            b.wait_front(2 * Kt);
            out.reserve_back(1);
            tile_regs_acquire();
            matmul_init(cb_m, cb_b, 1);
            for (uint32_t p = 0; p < patterns; ++p) {
                // the folded K order: (hi, hi) lanes 0..7, (hi, lo) lanes 0..7, (lo, hi) lanes 0..7
                for (uint32_t i = 0; i < 2; ++i) {
                    for (uint32_t j = 0; i + j < 2; ++j) {
                        for (uint32_t r = 0; r < lanes; ++r) {
                            for (uint32_t kt = 0; kt < Kt; ++kt) {
                                matmul_tiles(cb_m, cb_b, ((p * 2 + i) * lanes + r) * Kt + kt, j * Kt + kt, p);
                            }
                        }
                    }
                }
            }
            binary_max_tile_init();
            binary_max_tile(0, 1, 3);
            binary_min_tile_init();
            binary_min_tile(0, 1, 0);
            binary_min_tile(3, 2, 3);
            binary_max_tile_init();
            binary_max_tile(0, 3, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
            out.push_back(1);
            b.pop_front(2 * Kt);
        }
        m.pop_front(copies);
    }
}

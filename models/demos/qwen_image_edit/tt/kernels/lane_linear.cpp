// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Exact-lane linear for one lane pattern, in one program (split_linear's exact branch / _exact_sum):
//   out = sum over limbs i, lanes r = 0..7 of (a_i * mask_r) @ w
// in _exact_sum's order (limb-major, lanes inner, a float32 add after each full-K lane product, starting from
// the first product). As lane_bmm.cpp: the lane copy a_i * mask_r is the matmul a_i @ E_q with the diagonal
// 0/1 tile of the lane's columns (each output one product by 1 or 0, exact; q = (r - shift * kt) mod 8 for
// the pattern's per-32-group lane rotation), packed to bf16 (exact). Each lane product accumulates all its K
// tiles in the float32 DEST in K order, which is what the stock linear computes (one output tile per pass:
// matmul_block with ct_dim > 1 here broke the result, e2e PCC 0.29 / -0.14); the lane products are added with
// the SFPU float32 add (round to nearest even) as the ttnn.add chain does. The running sums of a unit's output columns live in
// cb_acc between lanes (float32 pack / UnpackToDestFp32 reload: exact) and the last lane's sums go to the
// output. The unit's weight column block is resident in cb_b, row-major (kt * csize + column).
// With patterns = 3 the unit's weight block and limbs are loaded once for all three lane patterns: pattern p's
// sums go to cb_est + p, and the output is their median, max(min(e0, e1), min(max(e0, e1), e2)), taken in DEST
// with the SFPU float32 min / max (exact, the same as _median3's ttnn ops). Every float32 reload reads its CB's
// front tile (index 0): an UnpackToDestFp32 copy_tile at a higher CB index read the wrong tile here (e2e PCC
// 0.51 / -0.04, and the 0.29 / -0.14 of the multi-column running-sum reloads). With a bias (has_bias, patterns = 3)
// the median takes one more SFPU float32 add (round to nearest even) of the column's bias tile (cb_bias, the bias
// in every row), the add a separate ttnn.add of the bias would do after the vote.

#include <cstdint>

#include "api/compute/binary_max_min.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    const uint32_t start_unit = get_arg_val<uint32_t>(0);
    const uint32_t num_units = get_arg_val<uint32_t>(1);

    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);
    constexpr uint32_t chunks = get_compile_time_arg_val(2);
    constexpr uint32_t csize = get_compile_time_arg_val(3);
    constexpr uint32_t limbs = get_compile_time_arg_val(4);
    // lane of K entry k under pattern p: (k % 8 + shifts[p] * (k / 32)) % 8
    constexpr uint32_t shifts[3] = {get_compile_time_arg_val(5), get_compile_time_arg_val(6), get_compile_time_arg_val(7)};
    constexpr uint32_t patterns = get_compile_time_arg_val(8);  // 1, or 3 with their median
    constexpr bool has_bias = get_compile_time_arg_val(9) != 0;
    constexpr uint32_t lanes = 8;
    constexpr uint32_t nb = 1;          // output columns per DEST pass (a running sum + a lane product)
    constexpr uint32_t copy_batch = 4;  // lane-copy tiles per DEST pass

    constexpr uint32_t cb_a = 0;
    constexpr uint32_t cb_b = 1;
    constexpr uint32_t cb_d = 2;
    constexpr uint32_t cb_m = 3;
    constexpr uint32_t cb_acc = 4;
    constexpr uint32_t cb_est = 5;  // 5, 6, 7: patterns 0, 1, 2
    constexpr uint32_t cb_bias = 8;
    constexpr uint32_t cb_out = 16;
    CircularBuffer a(cb_a);
    CircularBuffer b(cb_b);
    CircularBuffer d(cb_d);
    CircularBuffer m(cb_m);
    CircularBuffer acc(cb_acc);
    CircularBuffer out(cb_out);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_d, cb_m);
    d.wait_front(lanes);

    for (uint32_t u = start_unit; u < start_unit + num_units; ++u) {
        const uint32_t nt0 = (u % chunks) * csize;
        const uint32_t nt1 = nt0 + csize < Nt ? nt0 + csize : Nt;
        b.wait_front(csize * Kt);
        a.wait_front(limbs * Kt);
        for (uint32_t p = 0; p < patterns; ++p) {
            const uint32_t shift = shifts[p];
            for (uint32_t i = 0; i < limbs; ++i) {
                for (uint32_t r = 0; r < lanes; ++r) {
                    // lane copy of limb i: m[kt] = a_i[kt] @ E_q(r, kt), copy_batch tiles per DEST pass
                    pack_reconfig_data_format(cb_m);
                    matmul_init(cb_a, cb_d, 0);
                    for (uint32_t k0 = 0; k0 < Kt; k0 += copy_batch) {
                        const uint32_t kn = k0 + copy_batch < Kt ? copy_batch : Kt - k0;
                        tile_regs_acquire();
                        for (uint32_t j = 0; j < kn; ++j) {
                            const uint32_t kt = k0 + j;
                            matmul_tiles(cb_a, cb_d, i * Kt + kt, (r + lanes - (shift * kt) % lanes) % lanes, j);
                        }
                        tile_regs_commit();
                        m.reserve_back(kn);
                        tile_regs_wait();
                        for (uint32_t j = 0; j < kn; ++j) {
                            pack_tile(j, cb_m);
                        }
                        tile_regs_release();
                        m.push_back(kn);
                    }
                    m.wait_front(Kt);

                    // per nb output columns: sums (+)= the lane products over the full K
                    const bool first = i == 0 && r == 0;
                    const bool last = i == limbs - 1 && r == lanes - 1;
                    const uint32_t cb_dst = last ? (patterns == 1 ? cb_out : cb_est + p) : cb_acc;
                    CircularBuffer dst_cb(cb_dst);
                    pack_reconfig_data_format(cb_dst);
                    for (uint32_t c0 = nt0; c0 < nt1; c0 += nb) {
                        const uint32_t w = c0 + nb < nt1 ? nb : nt1 - c0;
                        const uint32_t prod = first ? 0 : nb;  // DEST tile of the lane products
                        tile_regs_acquire();
                        if (!first) {
                            acc.wait_front(w);
                            reconfig_data_format_srca(cb_b, cb_acc);
                            copy_init(cb_acc);
                            for (uint32_t j = 0; j < w; ++j) {
                                copy_tile(cb_acc, j, j);
                            }
                            reconfig_data_format_srca(cb_acc, cb_b);
                            fill_tile_init();
                            for (uint32_t j = 0; j < w; ++j) {
                                fill_tile(prod + j, 0.0f);
                            }
                        }
                        matmul_block_init(cb_m, cb_b, 0, w, 1, Kt);
                        for (uint32_t kt = 0; kt < Kt; ++kt) {
                            matmul_block(cb_m, cb_b, kt, kt * csize + (c0 - nt0), prod, 0, w, 1, Kt);
                        }
                        if (!first) {
                            add_binary_tile_init();
                            for (uint32_t j = 0; j < w; ++j) {
                                add_binary_tile<ckernel::DstRoundingMode::NearestEven>(j, prod + j, j);
                            }
                        }
                        tile_regs_commit();
                        if (!first) {
                            acc.pop_front(w);
                        }
                        dst_cb.reserve_back(w);
                        tile_regs_wait();
                        for (uint32_t j = 0; j < w; ++j) {
                            pack_tile(j, cb_dst);
                        }
                        tile_regs_release();
                        dst_cb.push_back(w);
                    }
                    m.pop_front(Kt);
                }
            }
        }
        a.pop_front(limbs * Kt);
        b.pop_front(csize * Kt);
        if constexpr (patterns == 3) {
            // per output column: the median of the three patterns' sums
            const uint32_t ncols = nt1 - nt0;
            CircularBuffer bias(cb_bias);
            if constexpr (has_bias) {
                bias.wait_front(csize);
            }
            pack_reconfig_data_format(cb_out);
            for (uint32_t c = 0; c < ncols; ++c) {
                tile_regs_acquire();
                // each pattern's front tile unpacked (UnpackToDestFp32) into DEST tile 0, then moved to its slot
                reconfig_data_format_srca(cb_b, cb_est);
                for (uint32_t q = 3; q-- > 0;) {
                    CircularBuffer e(cb_est + q);
                    e.wait_front(1);
                    copy_init(cb_est + q);
                    copy_tile(cb_est + q, 0, 0);
                    e.pop_front(1);
                    if (q > 0) {
                        copy_dest_values_init();
                        copy_dest_values<DataFormat::Float32>(0, q);
                    }
                }
                reconfig_data_format_srca(cb_est, cb_b);
                binary_max_tile_init();
                binary_max_tile(0, 1, 3);
                binary_min_tile_init();
                binary_min_tile(0, 1, 0);
                binary_min_tile(3, 2, 3);
                binary_max_tile_init();
                binary_max_tile(0, 3, 0);
                if constexpr (has_bias) {
                    // median to DEST 1, the column's bias tile (front of cb_bias) into DEST 0, then median + bias
                    copy_dest_values_init();
                    copy_dest_values<DataFormat::Float32>(0, 1);
                    reconfig_data_format_srca(cb_b, cb_bias);
                    copy_init(cb_bias);
                    copy_tile(cb_bias, 0, 0);
                    bias.pop_front(1);
                    reconfig_data_format_srca(cb_bias, cb_b);
                    add_binary_tile_init();
                    add_binary_tile<ckernel::DstRoundingMode::NearestEven>(1, 0, 0);
                }
                tile_regs_commit();
                out.reserve_back(1);
                tile_regs_wait();
                pack_tile(0, cb_out);
                tile_regs_release();
                out.push_back(1);
            }
            if (has_bias && ncols < csize) {
                bias.pop_front(csize - ncols);  // a short last chunk's unused tail
            }
        }
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Exact-lane batched product of two-limb operands for one lane pattern, in one program:
//   out = sum over terms (i, j) in {(0, 0), (0, 1), (1, 0)}, lanes r = 0..7 of (a_i * mask_r) @ b_j
// (the order of the stock path: terms outer, lanes inner, a float32 add after each lane product).
// The lane copy a_i * mask_r is the matmul a_i @ D_r with D_r the diagonal 0/1 tile of lane r (each
// output is one product by 1 or 0, exact), packed to bf16 (exact) once per row block and reused for its
// output columns. Each lane product accumulates its K tiles in the float32 DEST as the stock full-K
// matmul does; lane products are added in DEST with the SFPU float32 add (round to nearest even), the
// same add as the ttnn.add chain. Replaces 8 lanes x 3 terms of multiply + matmul + add ops (72 programs
// per pattern) by one.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/fill.h"
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
    constexpr uint32_t transpose_b = get_compile_time_arg_val(5);
    constexpr uint32_t lanes = 8;
    constexpr uint32_t products = 3 * lanes;
    // float32 DEST tiles beside the sum: 7 with the full-sync DEST (lane_program sets dst_full_sync_en)
    constexpr uint32_t group = 7;

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
    d.wait_front(lanes * Kt);

    for (uint32_t u = start_unit; u < start_unit + num_units; ++u) {
        const uint32_t c = u % chunks;
        const uint32_t nt0 = c * csize;
        const uint32_t nt1 = nt0 + csize < Nt ? nt0 + csize : Nt;

        // lane copies of both limbs of the row block: m[(i * 8 + r) * Kt + kt] = a_i[kt] @ D_r[kt]
        a.wait_front(2 * Kt);
        pack_reconfig_data_format(cb_m);
        matmul_init(cb_a, cb_d, 0);
        for (uint32_t i = 0; i < 2; ++i) {
            for (uint32_t r = 0; r < lanes; ++r) {
                for (uint32_t kt = 0; kt < Kt; ++kt) {
                    m.reserve_back(1);
                    tile_regs_acquire();
                    matmul_tiles(cb_a, cb_d, i * Kt + kt, r * Kt + kt, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, cb_m);
                    tile_regs_release();
                    m.push_back(1);
                }
            }
        }
        a.pop_front(2 * Kt);
        m.wait_front(2 * lanes * Kt);

        pack_reconfig_data_format(cb_out);
        for (uint32_t nt = nt0; nt < nt1; ++nt) {
            b.wait_front(2 * Kt);
            out.reserve_back(1);
            tile_regs_acquire();
            // the first lane product straight into the sum tile (DEST 0), then the rest in groups of up to
            // `group` products in DEST 1..group (zeroed, one matmul init per group) added to the sum in
            // order: the same products and the same float32 adds as one product + add at a time
            matmul_init(cb_m, cb_b, transpose_b);
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                matmul_tiles(cb_m, cb_b, kt, kt, 0);
            }
            for (uint32_t s0 = 1; s0 < products; s0 += group) {
                const uint32_t g = s0 + group < products ? group : products - s0;
                fill_tile_init();
                for (uint32_t q = 0; q < g; ++q) {
                    fill_tile(1 + q, 0.0f);
                }
                matmul_init(cb_m, cb_b, transpose_b);
                for (uint32_t q = 0; q < g; ++q) {
                    // product s = (term, lane): terms (0, 0), (0, 1), (1, 0), lanes 0..7 inner
                    const uint32_t s = s0 + q;
                    const uint32_t r = s % lanes;
                    const uint32_t i = s / lanes == 2 ? 1 : 0;
                    const uint32_t j = s / lanes == 1 ? 1 : 0;
                    for (uint32_t kt = 0; kt < Kt; ++kt) {
                        matmul_tiles(cb_m, cb_b, (i * lanes + r) * Kt + kt, j * Kt + kt, 1 + q);
                    }
                }
                add_binary_tile_init();
                for (uint32_t q = 0; q < g; ++q) {
                    add_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1 + q, 0);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
            out.push_back(1);
            b.pop_front(2 * Kt);
        }
        m.pop_front(2 * lanes * Kt);
    }
}

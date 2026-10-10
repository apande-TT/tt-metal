// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Two-limb linear y = hi @ w + lo @ w in one program: per work unit (mb row tiles held in cb 0 with both limbs'
// full K) and output block (mb x nb tiles), every K chunk of the multicast weight (cb 1) is multiplied by both
// limbs, each into its own float32 DEST tiles (hi: tile i, lo: tile mb * nb + i), so the weight is streamed once
// for both limbs and each product's whole K reduction stays in DEST in K order -- the fp32 sum the stock matmul
// forms (its partial reloads are UnpackToDestFp32, exact). The two products are then added with the SFPU
// float32 add (round to nearest even), the add of the separate-matmul path, so the result is the same.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);
    constexpr uint32_t mb = get_compile_time_arg_val(2);
    constexpr uint32_t nb = get_compile_time_arg_val(3);
    constexpr uint32_t kc = get_compile_time_arg_val(4);
    constexpr uint32_t units = get_compile_time_arg_val(5);

    constexpr uint32_t cb_a = 0;
    constexpr uint32_t cb_w = 1;
    constexpr uint32_t cb_out = 16;
    CircularBuffer a(cb_a);
    CircularBuffer w(cb_w);
    CircularBuffer out(cb_out);

    constexpr uint32_t lo_dst = mb * nb;
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_w, cb_out);
    for (uint32_t u = 0; u < units; ++u) {
        a.wait_front(2 * mb * Kt);
        for (uint32_t nblk = 0; nblk < Nt / nb; ++nblk) {
            matmul_init(cb_a, cb_w, 0);
            tile_regs_acquire();
            for (uint32_t kch = 0; kch < Kt / kc; ++kch) {
                w.wait_front(kc * nb);
                for (uint32_t kk = 0; kk < kc; ++kk) {
                    const uint32_t kt = kch * kc + kk;
                    for (uint32_t r = 0; r < mb; ++r) {
                        for (uint32_t c = 0; c < nb; ++c) {
                            matmul_tiles(cb_a, cb_w, r * Kt + kt, kk * nb + c, r * nb + c);
                            matmul_tiles(cb_a, cb_w, (mb + r) * Kt + kt, kk * nb + c, lo_dst + r * nb + c);
                        }
                    }
                }
                w.pop_front(kc * nb);
            }
            add_binary_tile_init();
            for (uint32_t i = 0; i < mb * nb; ++i) {
                add_binary_tile<ckernel::DstRoundingMode::NearestEven>(i, lo_dst + i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            out.reserve_back(mb * nb);
            for (uint32_t i = 0; i < mb * nb; ++i) {
                pack_tile(i, cb_out);
            }
            tile_regs_release();
            out.push_back(mb * nb);
        }
        a.pop_front(2 * mb * Kt);
    }
}

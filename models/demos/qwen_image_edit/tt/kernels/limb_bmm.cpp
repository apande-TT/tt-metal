// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The precise batched matmul of bf16 limbs (_precise.matmul): y = (ah @ bh + ah @ bl) + al @ bh, float32,
// in one program. Work unit = (batch, block of R output tile rows): the R rows of both a limbs are in cb 0
// for the whole unit (read once); per output column nt both b limbs' column nt stream through cb 1 and the
// unit's R output tiles of that column are formed: the three products accumulate side by side in DEST 0 / 1 / 2
// over the K tiles in K order (matmul_tiles, float32 DEST, HiFi4), then (hh + hl) + lh with the SFPU float32
// add (round to nearest even) as the two ttnn.add calls do (their float32 operands are exact copies of these
// DEST values). A full-K DEST accumulation is what the stock products compute (bmm.cpp directly; the 1-D
// multicast factory reloads its K-block partials through an UnpackToDestFp32 view, exact).
// reload_tf32 (compile time) instead mirrors MatmulMultiCoreReuseOptimized with in0_block_w = 1: after every
// K tile but the last the three partials are packed to cb 2 (float32) and reloaded through SrcA (that factory's
// intermed buffer is UnpackToSrc: a TF32 reload) before the next K tile accumulates.
// All packs go to float32 buffers and all matmul operands are bf16: no per-tile format reconfiguration except
// the reload_tf32 SrcA switch, made exactly as the stock reload makes it.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
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

    constexpr uint32_t Mt = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Nt = get_compile_time_arg_val(2);
    constexpr uint32_t blocks = get_compile_time_arg_val(3);  // row blocks per batch
    constexpr uint32_t R = get_compile_time_arg_val(4);       // output rows per block (the last may be short)
    constexpr bool reload_tf32 = get_compile_time_arg_val(5) != 0;

    constexpr uint32_t cb_a = 0;     // row j of the block: ah[kt] at 2 * j * Kt + kt, al[kt] at (2 * j + 1) * Kt + kt
    constexpr uint32_t cb_b = 1;     // column nt: bh[kt] at kt, bl[kt] at Kt + kt
    constexpr uint32_t cb_part = 2;  // reload_tf32: the three float32 partials (UnpackToSrc)
    constexpr uint32_t cb_out = 16;
    CircularBuffer a(cb_a);
    CircularBuffer b(cb_b);
    CircularBuffer part(cb_part);
    CircularBuffer out(cb_out);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_b, cb_out);

    for (uint32_t u = start_unit; u < start_unit + num_units; ++u) {
        const uint32_t r0 = (u % blocks) * R;
        const uint32_t nr = (r0 + R < Mt ? r0 + R : Mt) - r0;
        a.wait_front(2 * Kt * R);  // always R rows' worth (a short last block leaves the tail unused)
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            b.wait_front(2 * Kt);
            for (uint32_t j = 0; j < nr; ++j) {
                const uint32_t ah = 2 * j * Kt;
                const uint32_t al = ah + Kt;
                if constexpr (!reload_tf32) {
                    out.reserve_back(1);
                    tile_regs_acquire();
                    fill_tile_init();
                    fill_tile(0, 0.0f);
                    fill_tile(1, 0.0f);
                    fill_tile(2, 0.0f);
                    matmul_init(cb_a, cb_b, 0);
                    for (uint32_t kt = 0; kt < Kt; ++kt) {
                        matmul_tiles(cb_a, cb_b, ah + kt, kt, 0);       // ah @ bh
                        matmul_tiles(cb_a, cb_b, ah + kt, Kt + kt, 1);  // ah @ bl
                        matmul_tiles(cb_a, cb_b, al + kt, kt, 2);       // al @ bh
                    }
                } else {
                    for (uint32_t kt = 0; kt < Kt; ++kt) {
                        tile_regs_acquire();
                        if (kt == 0) {
                            fill_tile_init();
                            fill_tile(0, 0.0f);
                            fill_tile(1, 0.0f);
                            fill_tile(2, 0.0f);
                        } else {  // reload_from_dfb_to_dst of the reuse factory's compute kernel
                            reconfig_data_format_srca(cb_b, cb_part);
                            copy_init(cb_part);
                            part.wait_front(3);
                            copy_tile(cb_part, 0, 0);
                            copy_tile(cb_part, 1, 1);
                            copy_tile(cb_part, 2, 2);
                            part.pop_front(3);
                            reconfig_data_format_srca(cb_part, cb_b);
                        }
                        matmul_init(cb_a, cb_b, 0);
                        matmul_tiles(cb_a, cb_b, ah + kt, kt, 0);
                        matmul_tiles(cb_a, cb_b, ah + kt, Kt + kt, 1);
                        matmul_tiles(cb_a, cb_b, al + kt, kt, 2);
                        if (kt + 1 < Kt) {
                            tile_regs_commit();
                            part.reserve_back(3);
                            tile_regs_wait();
                            pack_tile(0, cb_part);
                            pack_tile(1, cb_part);
                            pack_tile(2, cb_part);
                            tile_regs_release();
                            part.push_back(3);
                        }
                    }
                    out.reserve_back(1);
                }
                // y = (hh + hl) + lh
                add_binary_tile_init();
                add_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
                add_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 2, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_out);
                tile_regs_release();
                out.push_back(1);
            }
            b.pop_front(2 * Kt);
        }
        a.pop_front(2 * Kt * R);
    }
}

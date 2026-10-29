// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute of tt/kv_append.py rope_append: rotary_embedding_llama_fused_qk's RoPE and the K / V append's bfloat8_b
// pack in ONE program. The RoPE is the stock kernel's arithmetic call for call
// (rotary_embedding_llama_fused_qk/device/kernels/compute/rotary_embedding_llama_sharded.cpp, same HiFi4 / fp32 DEST
// compute config, same bf16 intermediates): rotated = x @ trans_mat, sin_interm = rotated * sin (row broadcast),
// cos_interm = x * cos (row broadcast), out = cos_interm + sin_interm, each packed to bf16.
//   IS_Q: q's core -- q is rotated into its output shard, and the core's v row is packed to bfloat8_b for the cache.
//   else: k's core -- k is rotated into a plain bf16 CB and that is packed to bfloat8_b for the cache.
// The append pack is kernels/kv_append_compute.cpp's: the bf16 tiles copied into DEST and packed to the cache's
// bfloat8_b (DEST holds them exactly, so the packer sees the same values the 16-bit DEST append did).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

ALWI void ACQ() {
    tile_regs_acquire();
    tile_regs_wait();
}
ALWI void REL() {
    tile_regs_commit();
    tile_regs_release();
}

template <uint32_t Wt, uint32_t cb_app, uint32_t cb_bf8, uint32_t cb_back>
ALWI void append_pack() {
    // the append: Wt bf16 tiles -> DEST -> bfloat8_b; the packer goes back to bf16 (cb_back) afterwards
    cb_wait_front(cb_app, Wt);
    cb_reserve_back(cb_bf8, Wt);
    copy_tile_to_dst_init_short(cb_app);
    pack_reconfig_data_format(cb_bf8);
    tile_regs_acquire();
    for (uint32_t w = 0; w < Wt; ++w) {
        copy_tile(cb_app, w, w);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t w = 0; w < Wt; ++w) {
        pack_tile(w, cb_bf8);
    }
    tile_regs_release();
    cb_push_back(cb_bf8, Wt);
    cb_pop_front(cb_app, Wt);
    pack_reconfig_data_format(cb_back);
}

void kernel_main() {
    constexpr uint32_t IS_Q = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);  // tiles across a head row
    constexpr uint32_t in_cb = get_compile_time_arg_val(2);
    constexpr uint32_t out_cb = get_compile_time_arg_val(3);
    constexpr uint32_t cos_cb = get_compile_time_arg_val(4);
    constexpr uint32_t sin_cb = get_compile_time_arg_val(5);
    constexpr uint32_t trans_mat_cb = get_compile_time_arg_val(6);
    constexpr uint32_t rotated_in_interm_cb = get_compile_time_arg_val(7);
    constexpr uint32_t cos_interm_cb = get_compile_time_arg_val(8);
    constexpr uint32_t sin_interm_cb = get_compile_time_arg_val(9);
    constexpr uint32_t app_cb = get_compile_time_arg_val(10);  // IS_Q: v's shard; else out_cb (rotated k)
    constexpr uint32_t bf8_cb = get_compile_time_arg_val(11);

    compute_kernel_hw_startup<SrcOrder::Reverse>(in_cb, trans_mat_cb, out_cb);
    matmul_init(in_cb, trans_mat_cb);
    compute_kernel_hw_startup(rotated_in_interm_cb, sin_cb, sin_interm_cb);  // General Init for all binary ops

    if constexpr (IS_Q) {
        // v's append first: the data movement writes it into the cache while q rotates
        append_pack<Wt, app_cb, bf8_cb, out_cb>();
    }

    cb_reserve_back(rotated_in_interm_cb, Wt);
    cb_reserve_back(sin_interm_cb, Wt);
    cb_reserve_back(cos_interm_cb, Wt);
    cb_reserve_back(out_cb, Wt);

    // Get the input (q's shard is this kernel's own; k's is pushed by the data movement, kernels/kv_append_dm.cpp)
    if constexpr (IS_Q) {
        cb_reserve_back(in_cb, Wt);
        cb_push_back(in_cb, Wt);
    }
    cb_wait_front(in_cb, Wt);

    // rotated = x @ trans_mat
    matmul_init(in_cb, trans_mat_cb);
    ACQ();
    for (uint32_t j = 0; j < Wt; ++j) {
        matmul_tiles(in_cb, trans_mat_cb, j, 0, j);
        pack_tile(j, rotated_in_interm_cb, j);
    }
    REL();
    cb_push_back(rotated_in_interm_cb, Wt);
    cb_wait_front(rotated_in_interm_cb, Wt);

    mul_bcast_rows_init(rotated_in_interm_cb, sin_cb);
    ACQ();
    for (uint32_t j = 0; j < Wt; ++j) {
        // sin_interim = rotated * sin
        mul_tiles_bcast<BroadcastType::ROW>(rotated_in_interm_cb, sin_cb, j, j, j);
        pack_tile(j, sin_interm_cb, j);
    }
    REL();
    cb_push_back(sin_interm_cb, Wt);
    cb_pop_front(rotated_in_interm_cb, Wt);

    ACQ();
    for (uint32_t j = 0; j < Wt; ++j) {
        // cos_interim = x * cos
        mul_tiles_bcast<BroadcastType::ROW>(in_cb, cos_cb, j, j, j);
        pack_tile(j, cos_interm_cb, j);
    }
    REL();
    cb_push_back(cos_interm_cb, Wt);
    cb_pop_front(in_cb, Wt);  // Done with input

    cb_wait_front(sin_interm_cb, Wt);
    cb_wait_front(cos_interm_cb, Wt);
    add_init(cos_interm_cb, sin_interm_cb);
    ACQ();
    for (uint32_t j = 0; j < Wt; ++j) {
        // out = cos_interim + sin_interim
        add_tiles(cos_interm_cb, sin_interm_cb, j, j, j);
        pack_tile(j, out_cb, j);
    }
    REL();
    cb_push_back(out_cb, Wt);
    cb_pop_front(sin_interm_cb, Wt);
    cb_pop_front(cos_interm_cb, Wt);

    if constexpr (!IS_Q) {
        append_pack<Wt, app_cb, bf8_cb, out_cb>();  // the rotated k
    }
}

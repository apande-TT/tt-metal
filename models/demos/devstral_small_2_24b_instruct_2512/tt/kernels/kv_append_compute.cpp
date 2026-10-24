// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute of tt/kv_append.py: the core's input tile row (bf16, row h = kv head h) copied into DEST and packed to the
// cache's bfloat8_b. The packer forms a bfp8 face row (shared exponent + 16 mantissas) from those 16 values alone,
// so input row h's face rows are exactly the bytes paged_fused_update_cache's tilize (untilized rows -> DEST ->
// bfp8 pack, same 16-bit DEST) produces for the same values at the cache row; the data movement splices them in.

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t WT = get_compile_time_arg_val(0);  // tiles across a head row
    constexpr uint32_t cb_in = get_compile_time_arg_val(1);
    constexpr uint32_t cb_bf8 = get_compile_time_arg_val(2);

    compute_kernel_hw_startup(cb_in, cb_bf8);
    copy_tile_to_dst_init_short(cb_in);
    cb_wait_front(cb_in, WT);
    cb_reserve_back(cb_bf8, WT);
    tile_regs_acquire();
    for (uint32_t w = 0; w < WT; ++w) {
        copy_tile(cb_in, w, w);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t w = 0; w < WT; ++w) {
        pack_tile(w, cb_bf8);
    }
    tile_regs_release();
    cb_push_back(cb_bf8, WT);
    cb_pop_front(cb_in, WT);
}

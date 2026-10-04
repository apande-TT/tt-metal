// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The weight stream of tt/cpp_swiglu_mm.py over its BLOCKED re-laid weight (cpp_swiglu_mm.relayout, kb > 0): an
// interleaved tensor of NB tile rows, one per K block, whose tiles are permuted so that this core's KB x W block
// (K-row major) sits on DRAM bank BANK at consecutive bank-local slots SLOT0 .. SLOT0 + KB * W of its tile row.
// Each K block is ONE read from one bank. K blocks in this core's minimal_matmul order (REV: last block first).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t bank = get_arg_val<uint32_t>(1);
    const uint32_t slot0 = get_arg_val<uint32_t>(2);  // this core's first bank-local slot in a block row
    const uint32_t rev = get_arg_val<uint32_t>(3);    // K blocks last to first

    constexpr uint32_t KB = get_compile_time_arg_val(0);
    constexpr uint32_t NB = get_compile_time_arg_val(1);
    constexpr uint32_t W = get_compile_time_arg_val(2);
    constexpr uint32_t SLOTS = get_compile_time_arg_val(3);  // a block row's tiles on one bank

    constexpr uint32_t cb_w = 1;
    const uint32_t w_bytes = get_tile_size(cb_w);
    for (uint32_t i = 0; i < NB; ++i) {
        const uint32_t b = rev ? NB - 1 - i : i;
        cb_reserve_back(cb_w, KB * W);
        noc_async_read(
            get_noc_addr_from_bank_id<true>(bank, w_addr + (b * SLOTS + slot0) * w_bytes),
            get_write_ptr(cb_w),
            KB * W * w_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_w, KB * W);
    }
}

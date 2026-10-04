// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The weight stream of tt/cpp_swiglu_mm.py over its RE-LAID weight (cpp_swiglu_mm.relayout): the interleaved
// [K, 2N] tensor's tile columns are permuted so that this core's W tiles of a K row are pages
// k * NTF + (SLOT0 + i) * BANKS + BANK -- all on DRAM bank BANK, at consecutive bank-local addresses. Each K row
// is then ONE read of W tiles from one bank. BATCH K blocks are read per barrier, in this core's minimal_matmul
// K order (REV: last block first).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t bank = get_arg_val<uint32_t>(1);
    const uint32_t slot0 = get_arg_val<uint32_t>(2);  // first bank-local tile slot of a K row
    const uint32_t rev = get_arg_val<uint32_t>(3);    // K blocks last to first

    constexpr uint32_t KB = get_compile_time_arg_val(0);
    constexpr uint32_t NB = get_compile_time_arg_val(1);
    constexpr uint32_t W = get_compile_time_arg_val(2);
    constexpr uint32_t SLOTS = get_compile_time_arg_val(3);  // a K row's tiles on one bank (NTF / BANKS)
    constexpr uint32_t BATCH = get_compile_time_arg_val(4);  // K blocks a barrier (divides NB)

    constexpr uint32_t cb_w = 1;
    const uint32_t w_bytes = get_tile_size(cb_w);
    const uint32_t chunk = W * w_bytes;
    for (uint32_t i0 = 0; i0 < NB; i0 += BATCH) {
        cb_reserve_back(cb_w, BATCH * KB * W);
        uint32_t l1 = get_write_ptr(cb_w);
        for (uint32_t i = i0; i < i0 + BATCH; ++i) {
            const uint32_t b = rev ? NB - 1 - i : i;
            uint32_t src = w_addr + ((b * KB) * SLOTS + slot0) * w_bytes;
            for (uint32_t k = 0; k < KB; ++k) {
                noc_async_read(get_noc_addr_from_bank_id<true>(bank, src), l1, chunk);
                l1 += chunk;
                src += SLOTS * w_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_w, BATCH * KB * W);
    }
}

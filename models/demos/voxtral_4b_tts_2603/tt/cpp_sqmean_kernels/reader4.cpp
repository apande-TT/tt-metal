// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The input stream of the 4-way split square-mean (see tt/cpp_sqmean.py): this core folds ONE face of one
// tile row -- face f (0..3) of each of the row's WT float32 tiles, 1 KB each, packed four tiles to a CB slot.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t row = get_arg_val<uint32_t>(1);
    const uint32_t face = get_arg_val<uint32_t>(2);

    constexpr uint32_t WT = get_compile_time_arg_val(0);
    constexpr uint32_t BATCH = get_compile_time_arg_val(1);
    constexpr auto ax = TensorAccessorArgs<2>();
    const auto sx = TensorAccessor(ax, x_addr);

    constexpr uint32_t cb_x = 0;
    constexpr uint32_t face_bytes = 1024;
    const uint32_t off = face * face_bytes;

    // BATCH CB slots a barrier; slot j of a batch holds face `face` of the 4 tiles w + 4j .. w + 4j + 3, one
    // 1 KB face after another (the compute reads them as faces 0..3 of one tile).
    for (uint32_t w = 0; w < WT; w += 4 * BATCH) {
        cb_reserve_back(cb_x, BATCH);
        uint32_t l1 = get_write_ptr(cb_x);
        for (uint32_t i = 0; i < 4 * BATCH; ++i) {
            noc_async_read(sx.get_noc_addr(row * WT + w + i, off), l1, face_bytes);
            l1 += face_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_x, BATCH);
    }
}

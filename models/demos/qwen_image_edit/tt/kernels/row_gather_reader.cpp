// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// An exact row gather inside float32 tiles: output row j of every (entry, tile column) block is source row perm[j]
// of the same entry and column (rows j >= n_rows are left as they are: the caller drops them). Each row is two
// 64-byte face rows of its tile; they are read straight from the source tile's faces into the destination tile's
// faces in c_0 (one block of mt tiles per unit), which the writer stores whole.
// Compile-time: the source's accessor args, then mt (tile rows per entry), nt (tile columns), n_rows.
// Runtime: source addr, first unit, unit count, then perm[0 .. n_rows) (unit u: entry u / nt, column u % nt).

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t unit_start = get_arg_val<uint32_t>(1);
    const uint32_t unit_count = get_arg_val<uint32_t>(2);

    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr uint32_t base = src_args.next_compile_time_args_offset();
    constexpr uint32_t mt = get_compile_time_arg_val(base);
    constexpr uint32_t nt = get_compile_time_arg_val(base + 1);
    constexpr uint32_t n_rows = get_compile_time_arg_val(base + 2);
    constexpr uint32_t tile_bytes = 4096, face_bytes = 1024, face_row_bytes = 64;
    const auto src = TensorAccessor(src_args, src_addr);

    Noc noc;
    CircularBuffer cb(0);
    for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
        const uint32_t e = u / nt, c = u % nt;
        cb.reserve_back(mt);
        for (uint32_t j = 0; j < n_rows; ++j) {
            const uint32_t s = get_arg_val<uint32_t>(3 + j);
            const uint32_t src_page = (e * mt + s / 32) * nt + c;
            const uint32_t sr = s % 32, dr = j % 32;
            for (uint32_t h = 0; h < 2; ++h) {
                const uint32_t src_off = ((sr / 16) * 2 + h) * face_bytes + (sr % 16) * face_row_bytes;
                const uint32_t dst_off = (j / 32) * tile_bytes + ((dr / 16) * 2 + h) * face_bytes + (dr % 16) * face_row_bytes;
                noc.async_read(src, cb, face_row_bytes, {.page_id = src_page, .offset_bytes = src_off}, {.offset_bytes = dst_off});
            }
        }
        noc.async_read_barrier();
        cb.push_back(mt);
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode position step on one core: pos_f32 [B, 1] (fp32 tile, row b = batch b), pos_i32 [1, B]
// (int32 tile, column b = batch b) and pos_rm [B] (int32 row-major, one page) all += 1 in place, B = 32 logical
// entries each. Positions are small integers, so the fp32 increment is exact (the same value ttnn.add(pos, 1.0)
// stores).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_f = get_compile_time_arg_val(0);
    constexpr uint32_t cb_i = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr auto f_args = TensorAccessorArgs<3>();
    constexpr auto i_args = TensorAccessorArgs<f_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_r = get_compile_time_arg_val(i_args.next_compile_time_args_offset());
    constexpr uint32_t rm_bytes = get_compile_time_arg_val(i_args.next_compile_time_args_offset() + 1);
    constexpr auto r_args = TensorAccessorArgs<i_args.next_compile_time_args_offset() + 2>();

    const auto pos_f = TensorAccessor(f_args, get_common_arg_val<uint32_t>(0), tile_bytes);
    const auto pos_i = TensorAccessor(i_args, get_common_arg_val<uint32_t>(1), tile_bytes);
    const auto pos_r = TensorAccessor(r_args, get_common_arg_val<uint32_t>(2), rm_bytes);

    const uint32_t buf_f = get_write_ptr(cb_f);
    const uint32_t buf_i = get_write_ptr(cb_i);
    const uint32_t buf_r = get_write_ptr(cb_r);
    noc_async_read_page(0, pos_f, buf_f);
    noc_async_read_page(0, pos_i, buf_i);
    noc_async_read_page(0, pos_r, buf_r);
    noc_async_read_barrier();

    // Tile = four 16x16 faces (0: rows 0-15 cols 0-15, 1: rows 0-15 cols 16-31, 2: rows 16-31 cols 0-15, ...).
    volatile tt_l1_ptr float* f = reinterpret_cast<volatile tt_l1_ptr float*>(buf_f);
    volatile tt_l1_ptr int32_t* v = reinterpret_cast<volatile tt_l1_ptr int32_t*>(buf_i);
    volatile tt_l1_ptr int32_t* rm = reinterpret_cast<volatile tt_l1_ptr int32_t*>(buf_r);
    for (uint32_t b = 0; b < 32; ++b) {
        // pos_f32: row b, column 0.
        const uint32_t ef = (b >= 16 ? 2 * 256 : 0) + (b & 15) * 16;
        f[ef] = f[ef] + 1.0f;
        // pos_i32: row 0, column b.
        const uint32_t ei = (b >= 16 ? 256 : 0) + (b & 15);
        v[ei] = v[ei] + 1;
        // pos_rm: element b.
        rm[b] = rm[b] + 1;
    }

    noc_async_write_page(0, pos_f, buf_f);
    noc_async_write_page(0, pos_i, buf_i);
    noc_async_write_page(0, pos_r, buf_r);
    noc_async_write_barrier();
}

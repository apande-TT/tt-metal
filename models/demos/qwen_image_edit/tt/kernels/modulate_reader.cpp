// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The modulated LayerNorm tail's reader: for output tile i of a [B, S, D] float32 tensor (Ct = D / 32 tiles per tile
// row, St = S / 32 tile rows per sample), xc's page i into c_0, the row factor's page i / Ct (each row's 1/std across a
// tile) into c_1, and the sample's scale / shift pages (i / Ct / St) * Ct + i % Ct (each column's value down a tile)
// into c_2 / c_3. Compile-time: the four accessor args, then Ct, St. Runtime: the four addresses, num_tiles, start_id.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t addr_x = get_arg_val<uint32_t>(0);
    const uint32_t addr_r = get_arg_val<uint32_t>(1);
    const uint32_t addr_s = get_arg_val<uint32_t>(2);
    const uint32_t addr_h = get_arg_val<uint32_t>(3);
    const uint32_t num_tiles = get_arg_val<uint32_t>(4);
    const uint32_t start_id = get_arg_val<uint32_t>(5);

    constexpr auto args_x = TensorAccessorArgs<0>();
    constexpr auto args_r = TensorAccessorArgs<args_x.next_compile_time_args_offset()>();
    constexpr auto args_s = TensorAccessorArgs<args_r.next_compile_time_args_offset()>();
    constexpr auto args_h = TensorAccessorArgs<args_s.next_compile_time_args_offset()>();
    constexpr uint32_t ct = get_compile_time_arg_val(args_h.next_compile_time_args_offset());
    constexpr uint32_t st = get_compile_time_arg_val(args_h.next_compile_time_args_offset() + 1);
    const auto src_x = TensorAccessor(args_x, addr_x);
    const auto src_r = TensorAccessor(args_r, addr_r);
    const auto src_s = TensorAccessor(args_s, addr_s);
    const auto src_h = TensorAccessor(args_h, addr_h);
    const uint32_t bytes = get_local_cb_interface(0).fifo_page_size;

    Noc noc;
    CircularBuffer cb_x(0), cb_r(1), cb_s(2), cb_h(3);
    for (uint32_t i = start_id; i < start_id + num_tiles; ++i) {
        const uint32_t row = i / ct;
        const uint32_t vec = (row / st) * ct + i % ct;
        cb_x.reserve_back(1);
        noc.async_read(src_x, cb_x, bytes, {.page_id = i}, {.offset_bytes = 0});
        cb_r.reserve_back(1);
        noc.async_read(src_r, cb_r, bytes, {.page_id = row}, {.offset_bytes = 0});
        cb_s.reserve_back(1);
        noc.async_read(src_s, cb_s, bytes, {.page_id = vec}, {.offset_bytes = 0});
        cb_h.reserve_back(1);
        noc.async_read(src_h, cb_h, bytes, {.page_id = vec}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_x.push_back(1);
        cb_r.push_back(1);
        cb_s.push_back(1);
        cb_h.push_back(1);
    }
}

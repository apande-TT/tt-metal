// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// nlp_create_qkv_heads (num_kv_heads = num_heads, no k transpose) for a float32 tile tensor, as tile copies:
// qkv [N, 1, S, 3 * H * D] -> q, k, v [N, H, S, D]. Output page i of q / k / v (tile (d, s, e = n * H + h)) is
// tile column (w * H + h) * D/32 + d of qkv's tile row n * S/32 + s, for w = 0 / 1 / 2; the three are read into
// c_16 / c_17 / c_18 for the three-output writer, which stores page i of each. parts = 1: one projection
// [N, 1, S, H * D] -> [N, H, S, D] (the reshape + permute of a head split), its tiles into c_16 alone. merge = 1
// (parts = 1): the inverse, [N, H, S, D] -> [N, 1, S, H * D] (a head merge): output page i = tile (c = h * dt + d,
// s, n) is source tile (d, s, n * H + h). A ragged S runs over its padded tiles (same tile structure both sides).
// Compile-time: the source's accessor args, then H (heads), dt (D / 32), st (padded S / 32), parts (3 or 1), merge.
// Runtime: qkv addr, first output page, page count.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_id = get_arg_val<uint32_t>(1);
    const uint32_t num_pages = get_arg_val<uint32_t>(2);

    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr uint32_t base = src_args.next_compile_time_args_offset();
    constexpr uint32_t heads = get_compile_time_arg_val(base);
    constexpr uint32_t dt = get_compile_time_arg_val(base + 1);
    constexpr uint32_t st = get_compile_time_arg_val(base + 2);
    constexpr uint32_t parts = get_compile_time_arg_val(base + 3);
    constexpr uint32_t merge = get_compile_time_arg_val(base + 4);
    constexpr uint32_t ct = parts * heads * dt;  // source tile columns
    const auto src = TensorAccessor(src_args, src_addr);
    const uint32_t bytes = get_local_cb_interface(16).fifo_page_size;

    Noc noc;
    CircularBuffer cb_q(16), cb_k(17), cb_v(18);
    for (uint32_t i = start_id; i < start_id + num_pages; ++i) {
        uint32_t row;  // the (first) source page
        if constexpr (merge) {
            const uint32_t c = i % ct, r = i / ct;  // output tile (c, row r = n * st + s)
            const uint32_t s = r % st, n = r / st;
            row = ((n * heads + c / dt) * st + s) * dt + c % dt;
        } else {
            const uint32_t d = i % dt;
            const uint32_t r = i / dt;  // output tile row: e * st + s
            const uint32_t s = r % st;
            const uint32_t e = r / st;
            const uint32_t n = e / heads, h = e % heads;
            row = (n * st + s) * ct + h * dt + d;  // q's source page; k and v are heads * dt further
        }
        cb_q.reserve_back(1);
        noc.async_read(src, cb_q, bytes, {.page_id = row}, {.offset_bytes = 0});
        if constexpr (parts == 3) {
            cb_k.reserve_back(1);
            cb_v.reserve_back(1);
            noc.async_read(src, cb_k, bytes, {.page_id = row + heads * dt}, {.offset_bytes = 0});
            noc.async_read(src, cb_v, bytes, {.page_id = row + 2 * heads * dt}, {.offset_bytes = 0});
        }
        noc.async_read_barrier();
        cb_q.push_back(1);
        if constexpr (parts == 3) {
            cb_k.push_back(1);
            cb_v.push_back(1);
        }
    }
}

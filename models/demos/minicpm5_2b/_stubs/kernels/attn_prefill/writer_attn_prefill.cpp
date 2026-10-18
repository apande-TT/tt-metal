// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B fused prefill attention writer: the head merge (nlp_concat_heads) done by addressing. Unit u =
// (bg = b * n_kv + g, t = j * St + s) holds the Kt output tiles of query head h = g * G + j at sequence tile s,
// which land at tile row b * St + s, tile column h * Kt + dt of the merged [1, B * S, n_heads * D] output.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t St = get_compile_time_arg_val(2);
    constexpr uint32_t Rt = get_compile_time_arg_val(3);
    constexpr uint32_t n_kv = get_compile_time_arg_val(4);
    constexpr uint32_t G = get_compile_time_arg_val(5);
    constexpr auto out_args = TensorAccessorArgs<6>();
    constexpr uint32_t row_tiles = n_kv * G * Kt;

    const uint32_t u_start = get_arg_val<uint32_t>(0);
    const uint32_t u_count = get_arg_val<uint32_t>(1);
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(0));

    for (uint32_t u = u_start; u < u_start + u_count; ++u) {
        const uint32_t bg = u / Rt;
        const uint32_t t = u - bg * Rt;
        const uint32_t b = bg / n_kv;
        const uint32_t g = bg - b * n_kv;
        const uint32_t j = t / St;
        const uint32_t s = t - j * St;
        const uint32_t base = (b * St + s) * row_tiles + (g * G + j) * Kt;
        for (uint32_t dt = 0; dt < Kt; ++dt) {
            cb_wait_front(cb_out, 1);
            noc_async_write_page(base + dt, out, get_read_ptr(cb_out));
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
    }
}

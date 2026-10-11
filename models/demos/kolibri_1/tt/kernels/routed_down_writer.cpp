// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode down projection over the ROUTED experts only, phase 1 (writer): item (routed r, chunk c)'s NB fp32
// partial tiles to plane r of the [1, E, 32, hidden] partial buffer, columns c * NB ...
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Nt = get_compile_time_arg_val(0);
    constexpr uint32_t NB = get_compile_time_arg_val(1);
    constexpr auto p_args = TensorAccessorArgs<2>();
    const auto planes = TensorAccessor(p_args, get_arg_val<uint32_t>(0));

    constexpr uint32_t cb_list = 5, cb_out = 16;
    constexpr uint32_t CHUNKS = Nt / NB;
    constexpr uint32_t tile_bytes = 4096;

    cb_wait_front(cb_list, 1);
    volatile tt_l1_ptr uint32_t* list = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_list));
    const uint32_t n = list[0];
    for (uint32_t s = 0; s < n; ++s) {
        const uint32_t r = list[1 + s] / CHUNKS, c0 = (list[1 + s] % CHUNKS) * NB;
        cb_wait_front(cb_out, NB);
        const uint32_t l1 = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < NB; ++j) {
            noc_async_write_page(r * Nt + c0 + j, planes, l1 + j * tile_bytes);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_out, NB);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_list, 1);
}

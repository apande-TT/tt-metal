// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The weight-side exact lanes' reader: the 8 lane-mask tiles (pages 0..7 of the mask tensor) into c_1 once, held
// for the whole run, then this core's range of weight tiles into c_0, one at a time.
// Compile-time: the weight's accessor args, then the masks'. Runtime: weight addr, mask addr, num_tiles, start_id.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t m_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);
    const uint32_t start_id = get_arg_val<uint32_t>(3);

    constexpr auto w_args = TensorAccessorArgs<0>();
    constexpr auto m_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto w = TensorAccessor(w_args, w_addr);
    const auto m = TensorAccessor(m_args, m_addr);
    const uint32_t bytes = get_local_cb_interface(0).fifo_page_size;
    constexpr uint32_t lanes = 8;

    Noc noc;
    CircularBuffer cb_w(0), cb_m(1);
    cb_m.reserve_back(lanes);
    for (uint32_t r = 0; r < lanes; ++r) {
        noc.async_read(m, cb_m, bytes, {.page_id = r}, {.offset_bytes = r * bytes});
    }
    noc.async_read_barrier();
    cb_m.push_back(lanes);

    for (uint32_t i = start_id; i < start_id + num_tiles; ++i) {
        cb_w.reserve_back(1);
        noc.async_read(w, cb_w, bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_w.push_back(1);
    }
}

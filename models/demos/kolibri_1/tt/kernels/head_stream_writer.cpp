// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// LM head, one tile row of tokens (writer): first takes the activation the sender core multicasts (tells the sender
// this core's x CB is ready, waits for the data, hands it to compute), then writes the core's NC finished fp32 logits
// tiles.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t NC = get_compile_time_arg_val(0);
    constexpr uint32_t SW = get_compile_time_arg_val(1);
    constexpr uint32_t out_tile = get_compile_time_arg_val(2);
    constexpr uint32_t Kt = get_compile_time_arg_val(3);
    constexpr uint32_t sem_ready = get_compile_time_arg_val(4);
    constexpr uint32_t sem_valid = get_compile_time_arg_val(5);
    constexpr auto o_args = TensorAccessorArgs<6>();

    const auto out = TensorAccessor(o_args, get_arg_val<uint32_t>(0));
    const uint32_t core = get_arg_val<uint32_t>(1);
    const uint32_t sx = get_arg_val<uint32_t>(2), sy = get_arg_val<uint32_t>(3);  // the sender, NOC coordinates
    constexpr uint32_t cb_x = 0, cb_out = 16;

    cb_reserve_back(cb_x, Kt);
    noc_semaphore_inc(get_noc_addr(sx, sy, get_semaphore(sem_ready)), 1);
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_valid)), 1);
    cb_push_back(cb_x, Kt);

    for (uint32_t c = 0; c < NC; c += SW) {
        cb_wait_front(cb_out, SW);
        const uint32_t l1 = get_read_ptr(cb_out);
        for (uint32_t t = 0; t < SW; ++t) {
            noc_async_write_page(core * NC + c + t, out, l1 + t * out_tile);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_out, SW);
    }
    noc_async_write_barrier();
}

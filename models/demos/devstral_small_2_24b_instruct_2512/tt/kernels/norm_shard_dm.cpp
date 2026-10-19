// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Dataflow of the norm cores of tt/gate_up_decode.py's NormGateUpDecode (RISCV_0, NOC 0). Norm core `idx` reads
// width shard `idx` of the decode residual (WT fp32 tiles, from whichever core holds it) for compute (twice: the
// square and the final multiply each consume a copy) and writes compute's row sums (cb_part) into slot `idx` of the
// gather core's (norm core 0's) cb_ext, counting it there. The gather core hands the NP partials to its compute, which
// returns the per-row factor rsqrt(mean + eps) (cb_r); it multicasts that tile into every other norm core's cb_r and
// raises their factor semaphore (the same pattern as rms_norm's sharded kernels: unicast partials in, one multicast
// out). Each norm core then writes compute's normalized bf16 shard (cb_xn) into K slot `idx` of the activation
// sender's x CB (the same L1 address on every core: cb_x is allocated on the norm cores too) and counts it on the
// sender's gather semaphore (gate_up_sender.cpp, X_GATHER = NP). Counts follow the write barrier, so the data they
// announce has landed; all atomics are unicast to known cores.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t WT = get_compile_time_arg_val(0);  // tiles of a shard
    constexpr uint32_t NP = get_compile_time_arg_val(1);  // norm cores
    constexpr uint32_t x_tile = get_compile_time_arg_val(2);
    constexpr uint32_t part_bytes = get_compile_time_arg_val(3);  // an fp32 tile
    constexpr uint32_t sem_part = get_compile_time_arg_val(4);
    constexpr uint32_t sem_gather = get_compile_time_arg_val(5);
    constexpr uint32_t sem_r = get_compile_time_arg_val(6);
    constexpr auto h_args = TensorAccessorArgs<7>();
    constexpr uint32_t cb_x = 0, cb_ext = 3, cb_h = 4, cb_h2 = 5, cb_scaler = 7, cb_part = 8, cb_r = 9, cb_xn = 10;

    uint32_t a = 0;
    const uint32_t idx = get_arg_val<uint32_t>(a++);  // this core's shard (= its K slot); 0 = the gather core
    const auto h = TensorAccessor(h_args, get_arg_val<uint32_t>(a++));
    const uint32_t gx = get_arg_val<uint32_t>(a++), gy = get_arg_val<uint32_t>(a++);  // the activation sender
    const uint32_t cx = get_arg_val<uint32_t>(a++), cy = get_arg_val<uint32_t>(a++);  // the gather core
    const uint32_t rects = get_arg_val<uint32_t>(a++);  // the other norm cores as rectangles: x0 y0 x1 y1 cores
    const uint32_t rect0 = a;
    a += 5 * rects;

    cb_reserve_back(cb_h, WT);
    cb_reserve_back(cb_h2, WT);
    const uint32_t l1_h = get_write_ptr(cb_h), l1_h2 = get_write_ptr(cb_h2);
    for (uint32_t t = 0; t < WT; ++t) {
        noc_async_read_page(idx * WT + t, h, l1_h + t * part_bytes);
        noc_async_read_page(idx * WT + t, h, l1_h2 + t * part_bytes);
    }
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
    noc_async_read_barrier();
    cb_push_back(cb_h, WT);
    cb_push_back(cb_h2, WT);

    // row sums -> the gather core
    cb_wait_front(cb_part, 1);
    noc_async_write(get_read_ptr(cb_part), get_noc_addr(cx, cy, get_write_ptr(cb_ext) + idx * part_bytes), part_bytes);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(cx, cy, get_semaphore(sem_part)), 1);
    cb_pop_front(cb_part, 1);

    const uint32_t r_sem = get_semaphore(sem_r);
    if (idx == 0) {
        cb_reserve_back(cb_ext, NP);
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_part)), NP);
        cb_push_back(cb_ext, NP);
        cb_wait_front(cb_r, 1);  // the factor (compute pops it after its multiply)
        const uint32_t r_l1 = get_read_ptr(cb_r);
        for (uint32_t r = 0; r < rects; ++r) {
            const uint32_t x0 = get_arg_val<uint32_t>(rect0 + 5 * r), y0 = get_arg_val<uint32_t>(rect0 + 5 * r + 1);
            const uint32_t x1 = get_arg_val<uint32_t>(rect0 + 5 * r + 2), y1 = get_arg_val<uint32_t>(rect0 + 5 * r + 3);
            const uint32_t cores = get_arg_val<uint32_t>(rect0 + 5 * r + 4);
            noc_async_write_multicast(r_l1, get_noc_multicast_addr(x0, y0, x1, y1, r_l1), part_bytes, cores);
        }
        noc_async_write_barrier();
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r_sem) = 1;
        for (uint32_t r = 0; r < rects; ++r) {
            const uint32_t x0 = get_arg_val<uint32_t>(rect0 + 5 * r), y0 = get_arg_val<uint32_t>(rect0 + 5 * r + 1);
            const uint32_t x1 = get_arg_val<uint32_t>(rect0 + 5 * r + 2), y1 = get_arg_val<uint32_t>(rect0 + 5 * r + 3);
            const uint32_t cores = get_arg_val<uint32_t>(rect0 + 5 * r + 4);
            noc_semaphore_set_multicast(r_sem, get_noc_multicast_addr(x0, y0, x1, y1, r_sem), cores);
        }
    } else {
        cb_reserve_back(cb_r, 1);  // written by the gather core at this same address
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(r_sem), 1);
        cb_push_back(cb_r, 1);
    }

    cb_wait_front(cb_xn, WT);
    noc_async_write(get_read_ptr(cb_xn), get_noc_addr(gx, gy, get_write_ptr(cb_x) + idx * WT * x_tile), WT * x_tile);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(gx, gy, get_semaphore(sem_gather)), 1);
    noc_async_atomic_barrier();
    cb_pop_front(cb_xn, WT);
}

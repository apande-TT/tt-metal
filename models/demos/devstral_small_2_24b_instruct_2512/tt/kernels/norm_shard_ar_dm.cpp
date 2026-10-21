// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Dataflow of the norm cores of tt/gate_up_decode.py's NormGateUpDecode with the o_proj all_reduce folded in
// (ar_norm_call; RISCV_0, NOC 0). Norm core `idx` pairs with residual core `idx` (the core holding residual width shard
// idx, its all_reduce scratch and the new residual's shard): it waits until all TP devices' link workers (ar_send.cpp)
// have written their o_proj partial slot of the shard into that scratch -- TP counts on its own global semaphore --
// clears the count, reads the TP slots (bf16) and the residual shard x (fp32) for compute, and writes compute's
// h = x + sum (fp32) into the new residual's shard. Compute's own copies of h feed the norm, which then runs as in
// norm_shard_dm.cpp (row sums -> gather core -> rsqrt factor multicast -> normalized slice into the sender's x CB).
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
    constexpr uint32_t TP = get_compile_time_arg_val(7);
    constexpr uint32_t ar_tile = get_compile_time_arg_val(8);  // a bf16 partial tile
    constexpr uint32_t cb_x = 0, cb_ext = 3, cb_scaler = 7, cb_part = 8, cb_r = 9, cb_xn = 10;
    constexpr uint32_t cb_x0 = 11, cb_ar = 12, cb_hout = 14;

    uint32_t a = 0;
    const uint32_t idx = get_arg_val<uint32_t>(a++);  // this core's shard (= its K slot); 0 = the gather core
    const uint32_t res_x = get_arg_val<uint32_t>(a++), res_y = get_arg_val<uint32_t>(a++);  // its residual core
    const uint32_t x_addr = get_arg_val<uint32_t>(a++);        // residual x (fp32 shard)
    const uint32_t scratch_addr = get_arg_val<uint32_t>(a++);  // the TP partial slots
    const uint32_t hout_addr = get_arg_val<uint32_t>(a++);     // new residual h (fp32 shard)
    const uint32_t ar_sem_addr = get_arg_val<uint32_t>(a++);   // global semaphore: TP arrivals
    const uint32_t gx = get_arg_val<uint32_t>(a++), gy = get_arg_val<uint32_t>(a++);  // the activation sender
    const uint32_t cx = get_arg_val<uint32_t>(a++), cy = get_arg_val<uint32_t>(a++);  // the gather core
    const uint32_t rects = get_arg_val<uint32_t>(a++);  // the other norm cores as rectangles: x0 y0 x1 y1 cores
    const uint32_t rect0 = a;
    a += 5 * rects;

    cb_reserve_back(cb_x0, WT);
    noc_async_read(get_noc_addr(res_x, res_y, x_addr), get_write_ptr(cb_x0), WT * part_bytes);
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
    volatile tt_l1_ptr uint32_t* ar_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ar_sem_addr);
    noc_semaphore_wait(ar_sem, TP);
    noc_semaphore_set(ar_sem, 0);
    cb_reserve_back(cb_ar, TP * WT);
    noc_async_read(get_noc_addr(res_x, res_y, scratch_addr), get_write_ptr(cb_ar), TP * WT * ar_tile);
    noc_async_read_barrier();
    cb_push_back(cb_x0, WT);
    cb_push_back(cb_ar, TP * WT);

    // the new residual's shard
    cb_wait_front(cb_hout, WT);
    noc_async_write(get_read_ptr(cb_hout), get_noc_addr(res_x, res_y, hout_addr), WT * part_bytes);

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
    cb_pop_front(cb_hout, WT);
}

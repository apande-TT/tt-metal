// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Dataflow of tt/norm_decode.py (RISCV_0, NOC 0) on the residual's 8 x 5 shard cores (core (x, y), row worker (0, y),
// root (0, 0)): generates the reduce scalers and the eps tile (as layernorm's sharded writer does), hands the compute
// its own input shard (cb_x IS the shard), sends its partial row sums into slot x of its row worker's cb_ext and counts
// it there. A row worker (y > 0) waits for its 8, then sends its reduced tile into slot 8 + y - 1 of the root's cb_ext;
// the root waits for 12, and once the compute has the factor it sends it to the row workers, which pass it on to their
// row (unicast writes into each core's cb_exg, each followed by a semaphore increment). All transfers are unicast to
// NOC coordinates passed in, and every count follows its write barrier. MCAST_FACTOR: the root instead multicasts the
// factor tile and then the factor semaphore to every other core (rectangles in the runtime args, as layernorm's own
// sender does), and nothing is forwarded. SLOT_SEMS: each cb_ext slot has its own arrival semaphore (sem_gather + slot),
// so a worker hands its compute slot k as soon as slots 0..k have landed -- the reduce (one tile at a time, in slot
// order) overlaps the arrivals instead of starting after the last one.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp"

void kernel_main() {
    constexpr uint32_t WT = get_compile_time_arg_val(0);
    constexpr uint32_t ROLE = get_compile_time_arg_val(1);  // 0 plain core, 1 row worker, 2 root
    constexpr uint32_t sem_gather = get_compile_time_arg_val(2);
    constexpr uint32_t sem_factor = get_compile_time_arg_val(3);
    constexpr uint32_t ROW_LEN = get_compile_time_arg_val(4);  // cores in a row (8)
    constexpr uint32_t ROWS = get_compile_time_arg_val(5);     // rows (5)
    constexpr uint32_t TILE_BYTES = 4096;                      // fp32 tile

    constexpr uint32_t cb_x = 0, cb_scaler = 2, cb_part = 3, cb_ext = 4, cb_scg = 5;
    constexpr uint32_t cb_ex2 = 6, cb_eps = 7, cb_ex2pe = 8, cb_exg = 9, cb_out = 16;

    uint32_t a = 0;
    const uint32_t slot = get_arg_val<uint32_t>(a++);  // this core's x (its slot at the row worker)
    const uint32_t wx = get_arg_val<uint32_t>(a++), wy = get_arg_val<uint32_t>(a++);  // its row worker
    const uint32_t rx = get_arg_val<uint32_t>(a++), ry = get_arg_val<uint32_t>(a++);  // the root
    const uint32_t scalar_w = get_arg_val<uint32_t>(a++);
    const uint32_t scalar_c = get_arg_val<uint32_t>(a++);
    const uint32_t eps = get_arg_val<uint32_t>(a++);
    const uint32_t row = get_arg_val<uint32_t>(a++);   // this core's y (a row worker's slot at the root: 8 + y - 1)
    const uint32_t nfw = get_arg_val<uint32_t>(a++);   // factor targets that follow: (x, y) pairs
    const uint32_t fw0 = a;

    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
        __builtin_bit_cast(float, scalar_w));
    generate_bcast_col_scalar(CircularBuffer(cb_eps), eps);
    if constexpr (ROLE != 0) {
        dataflow_kernel_lib::prepare_reduce_scaler<cb_scg, ckernel::PoolType::AVG, ckernel::ReduceDim::REDUCE_ROW>(
            __builtin_bit_cast(float, scalar_c));
    }
    cb_reserve_back(cb_x, WT);
    cb_push_back(cb_x, WT);

    const uint32_t ext_base = get_write_ptr(cb_ext);  // the same address on every core
    const uint32_t gsem = get_semaphore(sem_gather);
    const uint32_t fsem = get_semaphore(sem_factor);

    // partial row sums -> slot `slot` of the row worker
    cb_wait_front(cb_part, 1);
    noc_async_write(get_read_ptr(cb_part), get_noc_addr(wx, wy, ext_base + slot * TILE_BYTES), TILE_BYTES);
    noc_async_write_barrier();
#ifdef SLOT_SEMS
    noc_semaphore_inc(get_noc_addr(wx, wy, get_semaphore(sem_gather + slot)), 1);
#else
    noc_semaphore_inc(get_noc_addr(wx, wy, gsem), 1);
#endif
    noc_async_atomic_barrier();
    cb_pop_front(cb_part, 1);

    if constexpr (ROLE == 1) {
#ifdef SLOT_SEMS
        for (uint32_t k = 0; k < ROW_LEN; ++k) {
            cb_reserve_back(cb_ext, 1);
            noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_gather + k)), 1);
            cb_push_back(cb_ext, 1);
        }
#else
        cb_reserve_back(cb_ext, ROW_LEN);
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gsem), ROW_LEN);
        cb_push_back(cb_ext, ROW_LEN);
#endif
        cb_wait_front(cb_ex2, 1);
        noc_async_write(
            get_read_ptr(cb_ex2), get_noc_addr(rx, ry, ext_base + (ROW_LEN + row - 1) * TILE_BYTES), TILE_BYTES);
        noc_async_write_barrier();
#ifdef SLOT_SEMS
        noc_semaphore_inc(get_noc_addr(rx, ry, get_semaphore(sem_gather + ROW_LEN + row - 1)), 1);
#else
        noc_semaphore_inc(get_noc_addr(rx, ry, gsem), 1);
#endif
        noc_async_atomic_barrier();
        cb_pop_front(cb_ex2, 1);
    } else if constexpr (ROLE == 2) {
#ifdef SLOT_SEMS
        for (uint32_t k = 0; k < ROW_LEN + ROWS - 1; ++k) {
            cb_reserve_back(cb_ext, 1);
            noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_gather + k)), 1);
            cb_push_back(cb_ext, 1);
        }
#else
        cb_reserve_back(cb_ext, ROW_LEN + ROWS - 1);
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gsem), ROW_LEN + ROWS - 1);
        cb_push_back(cb_ext, ROW_LEN + ROWS - 1);
#endif
    }

    // the factor: the root gets it from its compute, everyone else from the core that forwards it
    cb_reserve_back(cb_exg, 1);
    const uint32_t exg = get_write_ptr(cb_exg);  // the same address on every core
    if constexpr (ROLE == 2) {
        cb_wait_front(cb_ex2pe, 1);
        noc_async_write(get_read_ptr(cb_ex2pe), get_noc_addr(exg), TILE_BYTES);  // own copy
        noc_async_write_barrier();
        cb_pop_front(cb_ex2pe, 1);
    } else {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fsem), 1);
    }
#ifdef MCAST_FACTOR
    if constexpr (ROLE == 2) {
        // nfw = rectangles: x0 y0 x1 y1 cores each (NOC coordinates), together every core but this one
        for (uint32_t i = 0; i < nfw; ++i) {
            const uint32_t b = fw0 + 5 * i;
            noc_async_write_multicast(
                exg,
                get_noc_multicast_addr(
                    get_arg_val<uint32_t>(b), get_arg_val<uint32_t>(b + 1), get_arg_val<uint32_t>(b + 2),
                    get_arg_val<uint32_t>(b + 3), exg),
                TILE_BYTES,
                get_arg_val<uint32_t>(b + 4));
        }
        noc_async_write_barrier();
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fsem) = 1;
        for (uint32_t i = 0; i < nfw; ++i) {
            const uint32_t b = fw0 + 5 * i;
            noc_semaphore_set_multicast(
                fsem,
                get_noc_multicast_addr(
                    get_arg_val<uint32_t>(b), get_arg_val<uint32_t>(b + 1), get_arg_val<uint32_t>(b + 2),
                    get_arg_val<uint32_t>(b + 3), fsem),
                get_arg_val<uint32_t>(b + 4));
        }
        noc_async_write_barrier();
    }
#else
    for (uint32_t i = 0; i < nfw; ++i) {
        const uint32_t tx = get_arg_val<uint32_t>(fw0 + 2 * i), ty = get_arg_val<uint32_t>(fw0 + 2 * i + 1);
        noc_async_write(exg, get_noc_addr(tx, ty, exg), TILE_BYTES);
    }
    noc_async_write_barrier();
    for (uint32_t i = 0; i < nfw; ++i) {
        const uint32_t tx = get_arg_val<uint32_t>(fw0 + 2 * i), ty = get_arg_val<uint32_t>(fw0 + 2 * i + 1);
        noc_semaphore_inc(get_noc_addr(tx, ty, fsem), 1);
    }
    noc_async_atomic_barrier();
#endif
    cb_push_back(cb_exg, 1);

    cb_wait_front(cb_out, WT);  // cb_out IS this core's output shard
    cb_pop_front(cb_out, WT);
}

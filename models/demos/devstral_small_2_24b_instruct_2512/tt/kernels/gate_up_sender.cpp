// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Activation sender of tt/gate_up_decode.py (a spare core that computes nothing, RISCV_0 on NOC 0): reads the
// [32, K] decode activation once and multicasts it into every compute core's x CB (the same L1 address on every
// core), then raises their valid semaphore. It first waits for every compute core to report its x CB reserved
// (ready semaphore). Both semaphores start from 0 on every launch of the program.
// X_GATHER (NormGateUpDecode): the activation is not read from a tensor; X_GATHER norm cores write its slices straight
// into this core's x CB, each counting itself on the gather semaphore (compile-time arg 5).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);  // K tiles
    constexpr uint32_t x_tile = get_compile_time_arg_val(1);
    constexpr uint32_t n_cores = get_compile_time_arg_val(2);  // compute cores
    constexpr uint32_t sem_ready = get_compile_time_arg_val(3);
    constexpr uint32_t sem_valid = get_compile_time_arg_val(4);
    constexpr uint32_t cb_x = 0;

    cb_reserve_back(cb_x, Kt);
    const uint32_t x_l1 = get_write_ptr(cb_x);
#ifdef X_GATHER
    constexpr uint32_t sem_gather = get_compile_time_arg_val(5);
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_gather)), X_GATHER);
#elif defined(X_SLICE)
    // One of the two RISCs of the sender core (tt/gate_up_decode.py dual_noc_sender_kernels): this one reads and
    // multicasts only K tiles [k0, k1) of the activation, over its own NOC, and then raises the compute cores' valid
    // semaphore by one (they wait for a count from each). A whole row multicast from one RISC ran at ~10-15 GB/s, and
    // concurrent multicasts on one NOC serialize, so the two halves go out on the two NOCs at once. Both RISCs wait on
    // the core's own ready count. Runtime args: x, k0, k1, rects.
    {
        constexpr auto x_args = TensorAccessorArgs<5>();
        const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(0));
        const uint32_t k0 = get_arg_val<uint32_t>(1), k1 = get_arg_val<uint32_t>(2);
        for (uint32_t k = k0; k < k1; ++k) {
            noc_async_read_page(k, x, x_l1 + k * x_tile);
        }
        noc_async_read_barrier();
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_ready)), n_cores);
        const uint32_t valid_l1 = get_semaphore(sem_valid);
        constexpr uint32_t a0 = 3;
        const uint32_t rects = get_arg_val<uint32_t>(a0);
        const uint32_t off = k0 * x_tile, bytes = (k1 - k0) * x_tile;
        // NOC 1 multicasts from the far corner
        auto mcast_addr = [&](uint32_t r, uint32_t l1) {
            const uint32_t x0 = get_arg_val<uint32_t>(a0 + 1 + 5 * r), y0 = get_arg_val<uint32_t>(a0 + 2 + 5 * r);
            const uint32_t x1 = get_arg_val<uint32_t>(a0 + 3 + 5 * r), y1 = get_arg_val<uint32_t>(a0 + 4 + 5 * r);
            return noc_index == 0 ? get_noc_multicast_addr(x0, y0, x1, y1, l1) : get_noc_multicast_addr(x1, y1, x0, y0, l1);
        };
        for (uint32_t r = 0; r < rects; ++r) {
            const uint32_t cores = get_arg_val<uint32_t>(a0 + 5 + 5 * r);
            if (cores == 1) {
                const uint32_t cx = get_arg_val<uint32_t>(a0 + 1 + 5 * r), cy = get_arg_val<uint32_t>(a0 + 2 + 5 * r);
                noc_async_write(x_l1 + off, get_noc_addr(cx, cy, x_l1 + off), bytes);
            } else {
                noc_async_write_multicast(x_l1 + off, mcast_addr(r, x_l1 + off), bytes, cores);
            }
        }
        noc_async_write_barrier();  // the slice has landed everywhere before it is counted
        for (uint32_t r = 0; r < rects; ++r) {
            const uint32_t cores = get_arg_val<uint32_t>(a0 + 5 + 5 * r);
            if (cores == 1) {
                const uint32_t cx = get_arg_val<uint32_t>(a0 + 1 + 5 * r), cy = get_arg_val<uint32_t>(a0 + 2 + 5 * r);
                noc_semaphore_inc(get_noc_addr(cx, cy, valid_l1), 1);
            } else {
                noc_semaphore_inc_multicast(mcast_addr(r, valid_l1), 1, cores);
            }
        }
        noc_async_atomic_barrier();
        return;
    }
#else
    constexpr auto x_args = TensorAccessorArgs<5>();
    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(0));
    for (uint32_t k = 0; k < Kt; ++k) {
        noc_async_read_page(k, x, x_l1 + k * x_tile);
    }
    noc_async_read_barrier();
#endif
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_ready)), n_cores);
    const uint32_t valid_l1 = get_semaphore(sem_valid);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_l1) = 1;
    // The compute cores as rectangles (NOC 0 order: start <= end, NOC coordinates): count, then x0 y0 x1 y1 cores.
    const uint32_t rects = get_arg_val<uint32_t>(1);
    for (uint32_t r = 0; r < rects; ++r) {
        const uint32_t x0 = get_arg_val<uint32_t>(2 + 5 * r), y0 = get_arg_val<uint32_t>(3 + 5 * r);
        const uint32_t x1 = get_arg_val<uint32_t>(4 + 5 * r), y1 = get_arg_val<uint32_t>(5 + 5 * r);
        const uint32_t cores = get_arg_val<uint32_t>(6 + 5 * r);
        if (cores == 1) {
            noc_async_write(x_l1, get_noc_addr(x0, y0, x_l1), Kt * x_tile);
            noc_semaphore_set_remote(valid_l1, get_noc_addr(x0, y0, valid_l1));
        } else {
            noc_async_write_multicast(x_l1, get_noc_multicast_addr(x0, y0, x1, y1, x_l1), Kt * x_tile, cores);
            noc_semaphore_set_multicast(valid_l1, get_noc_multicast_addr(x0, y0, x1, y1, valid_l1), cores);
        }
    }
    noc_async_write_barrier();
}

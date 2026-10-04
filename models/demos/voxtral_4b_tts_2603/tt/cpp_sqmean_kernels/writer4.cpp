// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The writer of the 4-way split square-mean (see tt/cpp_sqmean.py).
//   role 1 (odd face): sends its folded face (1 KB) into face 1 of its partner's assembly CB, then bumps the
//                      partner's semaphore.
//   role 0 (even face): copies its own folded face into face 0 of the assembly CB, waits for the partner's
//                      face, hands the assembled tile to the compute, and writes the reduced half (2 KB:
//                      the half's 16 row means in column 0) into its half of the output tile.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t role = get_arg_val<uint32_t>(0);
    const uint32_t y_addr = get_arg_val<uint32_t>(1);
    const uint32_t row = get_arg_val<uint32_t>(2);
    const uint32_t half = get_arg_val<uint32_t>(3);
    const uint32_t partner_x = get_arg_val<uint32_t>(4);
    const uint32_t partner_y = get_arg_val<uint32_t>(5);

    constexpr auto ay = TensorAccessorArgs<0>();
    const auto sy = TensorAccessor(ay, y_addr);

    constexpr uint32_t cb_part = 1;
    constexpr uint32_t cb_asm = 2;
    constexpr uint32_t cb_y = 16;
    constexpr uint32_t face_bytes = 1024;
    constexpr uint32_t half_bytes = 2048;
    const uint32_t sem_addr = get_semaphore(0);

    cb_wait_front(cb_part, 1);
    if (role == 1) {
        // The partner's assembly CB is a one-slot CB at the same L1 address on every core.
        const uint64_t dst = get_noc_addr(partner_x, partner_y, get_write_ptr(cb_asm) + face_bytes);
        noc_async_write(get_read_ptr(cb_part), dst, face_bytes);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(partner_x, partner_y, sem_addr), 1);
        noc_async_atomic_barrier();
        cb_pop_front(cb_part, 1);
        return;
    }
    cb_reserve_back(cb_asm, 1);
    {
        auto* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_part));
        auto* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_asm));
        for (uint32_t i = 0; i < face_bytes / 4; ++i) {
            dst[i] = src[i];
        }
    }
    cb_pop_front(cb_part, 1);
    auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    noc_semaphore_wait(sem, 1);
    noc_semaphore_set(sem, 0);
    cb_push_back(cb_asm, 1);

    cb_wait_front(cb_y, 1);
    noc_async_write(get_read_ptr(cb_y), sy.get_noc_addr(row, half * half_bytes), half_bytes);
    noc_async_write_barrier();
    cb_pop_front(cb_y, 1);
}

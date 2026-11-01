// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Receiver reader of the two-limb linear (limb_linear.cpp). Per work unit (mb row tiles of the flattened
// activation, unit u = first_unit + i * unit_stride) it reads both limbs' full-K rows into cb 0 (limb i, row r,
// K tile kt -> (i * mb + r) * Kt + kt), then takes the weight in chunks of kc K tiles x nb columns that the
// sender core multicasts into cb 1: reserve a slot, tell the sender it is free, wait until the data is in.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
    const uint32_t hi_addr = get_arg_val<uint32_t>(0);
    const uint32_t lo_addr = get_arg_val<uint32_t>(1);
    const uint32_t first_unit = get_arg_val<uint32_t>(2);
    const uint32_t sender_x = get_arg_val<uint32_t>(3);
    const uint32_t sender_y = get_arg_val<uint32_t>(4);
    const uint32_t lo_off = get_arg_val<uint32_t>(5);  // lo's page offset (both limbs in one stacked tensor; else 0)
    const uint32_t lo_byte = get_arg_val<uint32_t>(6);  // lo's byte offset in hi's page (limbs paired per page; else 0)

    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);
    constexpr uint32_t mb = get_compile_time_arg_val(2);
    constexpr uint32_t nb = get_compile_time_arg_val(3);
    constexpr uint32_t kc = get_compile_time_arg_val(4);
    constexpr uint32_t units = get_compile_time_arg_val(5);
    constexpr uint32_t unit_stride = get_compile_time_arg_val(6);
    constexpr auto hi_args = TensorAccessorArgs<7>();
    constexpr auto lo_args = TensorAccessorArgs<hi_args.next_compile_time_args_offset()>();
    constexpr uint32_t sender_sem_id = get_compile_time_arg_val(lo_args.next_compile_time_args_offset());
    constexpr uint32_t receiver_sem_id = get_compile_time_arg_val(lo_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t slot1_sem_id = get_compile_time_arg_val(lo_args.next_compile_time_args_offset() + 2);
    constexpr uint32_t slot2_sem_id = get_compile_time_arg_val(lo_args.next_compile_time_args_offset() + 3);

    constexpr uint32_t cb_a = 0;
    constexpr uint32_t cb_w = 1;
    const uint32_t page = get_local_cb_interface(cb_a).fifo_page_size;
    const auto hi = TensorAccessor(hi_args, hi_addr);
    const auto lo = TensorAccessor(lo_args, lo_addr);

    Noc noc;
    DataflowBuffer da(cb_a);
    DataflowBuffer dw(cb_w);
    Semaphore<> receiver_sem(receiver_sem_id);
    // one credit counter per cb 1 slot on the sender (slot 0: sender_sem_id), so a receiver ahead of the others
    // can never stand in for one that has not freed its slot yet
    const uint32_t credit_ids[3] = {sender_sem_id, slot1_sem_id, slot2_sem_id};

    // weight chunks by credit: this core grants the sender one credit per free cb 1 slot (up to `slots` chunks
    // ahead, on that slot's own counter) and waits for the sender's count of delivered chunks, so the sender can
    // multicast the next chunks without a fresh round trip per chunk
    constexpr uint32_t slots = 3;  // cb 1 holds three chunks (limb_linear.py)
    constexpr uint32_t chunk = kc * nb;
    constexpr uint32_t per_unit = (Nt / nb) * (Kt / kc);
    constexpr uint32_t total = units * per_unit;
    uint32_t granted = 0;  // credits granted (chunks whose slot is reserved)
    uint32_t got = 0;      // chunks received
    auto grant = [&]() {
        while (granted < total && granted < got + slots) {
            dw.reserve_back((granted - got + 1) * chunk);  // slots for chunks got .. granted are free
            Semaphore<>(credit_ids[granted % slots]).up(noc, sender_x, sender_y, 1);
            ++granted;
        }
    };
    grant();
    for (uint32_t i = 0; i < units; ++i) {
        const uint32_t row0 = (first_unit + i * unit_stride) * mb;
        da.reserve_back(2 * mb * Kt);
        for (uint32_t r = 0; r < mb; ++r) {
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                const uint32_t id = (row0 + r) * Kt + kt;
                noc.async_read(hi, da, page, {.page_id = id}, {.offset_bytes = (r * Kt + kt) * page});
                noc.async_read(
                    lo, da, page, {.page_id = id + lo_off, .offset_bytes = lo_byte}, {.offset_bytes = ((mb + r) * Kt + kt) * page});
            }
        }
        noc.async_read_barrier();
        da.push_back(2 * mb * Kt);

        for (uint32_t c = 0; c < per_unit; ++c) {
            receiver_sem.wait_min(got + 1);  // the sender has delivered chunk `got`
            dw.push_back(chunk);
            ++got;
            grant();
        }
    }
    // the sender's last count has landed: leave the semaphore at 0 for the next program on this core
    receiver_sem.set(0);
}

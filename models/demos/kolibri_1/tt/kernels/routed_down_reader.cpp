// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode down projection over the ROUTED experts only, phase 1 (reader). The [1, E] mask says which local experts
// any of the step's tokens routed to; routed expert r (in expert order) contributes act[:, e rows] @ Wd[e rows, :].
// Work items are (routed expert, block of NB output columns), dealt round-robin over the cores. Per item: the
// expert's CPE act tiles, then for each of its CPE K rows the NB weight tiles of the item's columns.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t E = get_compile_time_arg_val(0);          // local experts
    constexpr uint32_t CPE = get_compile_time_arg_val(1);        // K tiles per expert (inter / 32)
    constexpr uint32_t Nt = get_compile_time_arg_val(2);         // output column tiles (hidden / 32)
    constexpr uint32_t NB = get_compile_time_arg_val(3);         // output column tiles per item
    constexpr uint32_t NC = get_compile_time_arg_val(4);         // cores
    constexpr uint32_t MAX_ITEMS = get_compile_time_arg_val(5);  // max items one core can own
    constexpr uint32_t a_tile_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t w_tile_bytes = get_compile_time_arg_val(7);
    constexpr auto a_args = TensorAccessorArgs<8>();
    constexpr auto w_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto m_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();

    const auto act = TensorAccessor(a_args, get_arg_val<uint32_t>(0));
    const auto wd = TensorAccessor(w_args, get_arg_val<uint32_t>(1));
    const auto mask_acc = TensorAccessor(m_args, get_arg_val<uint32_t>(2));
    const uint32_t core_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_act = 0, cb_w = 1, cb_mask = 3, cb_count = 4, cb_list = 5;
    constexpr uint32_t CHUNKS = Nt / NB;

    cb_reserve_back(cb_mask, 1);
    const uint32_t mask_l1 = get_write_ptr(cb_mask);
    noc_async_read_page(0, mask_acc, mask_l1);
    noc_async_read_barrier();

    // Routed experts in expert order (+-0.0 both count as unrouted); item i = (routed r, chunk c), i = r * CHUNKS + c.
    volatile tt_l1_ptr uint32_t* mask = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mask_l1);
    uint32_t experts[E];
    uint32_t R = 0;
    for (uint32_t e = 0; e < E; ++e) {
        if ((mask[e] & 0x7fffffffu) != 0) {
            experts[R++] = e;
        }
    }
    uint32_t items[MAX_ITEMS];
    uint32_t n = 0;
    for (uint32_t i = core_id; i < R * CHUNKS; i += NC) {
        items[n++] = i;
    }

    cb_reserve_back(cb_list, 1);
    volatile tt_l1_ptr uint32_t* list = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_list));
    list[0] = n;
    for (uint32_t s = 0; s < n; ++s) {
        list[1 + s] = items[s];
    }
    cb_push_back(cb_list, 1);
    cb_reserve_back(cb_count, 1);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_count)) = n;
    cb_push_back(cb_count, 1);

    for (uint32_t s = 0; s < n; ++s) {
        const uint32_t e = experts[items[s] / CHUNKS];
        const uint32_t c0 = (items[s] % CHUNKS) * NB;
        // The whole item in flight at once (one barrier): weight tile (k, j) at k * NB + j.
        cb_reserve_back(cb_act, CPE);
        cb_reserve_back(cb_w, CPE * NB);
        const uint32_t a_l1 = get_write_ptr(cb_act);
        const uint32_t w_l1 = get_write_ptr(cb_w);
        for (uint32_t k = 0; k < CPE; ++k) {
            noc_async_read_page(e * CPE + k, act, a_l1 + k * a_tile_bytes);
            const uint32_t row = (e * CPE + k) * Nt + c0;
            for (uint32_t j = 0; j < NB; ++j) {
                noc_async_read_page(row + j, wd, w_l1 + (k * NB + j) * w_tile_bytes);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_act, CPE);
        cb_push_back(cb_w, CPE * NB);
    }
}

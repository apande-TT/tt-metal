// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The input stream of the C++ decode attention P@V (see tt/cpp_pv_dec.py). For each of this core's
// (user, kv head) units it reads, per span tile j, the exp-weight tile e[j] and the values' tile row
// v[j] (DT tiles of head_dim).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t e_addr = get_arg_val<uint32_t>(0);
    const uint32_t v_addr = get_arg_val<uint32_t>(1);
    const uint32_t u0 = get_arg_val<uint32_t>(2);
    const uint32_t nu = get_arg_val<uint32_t>(3);

    constexpr uint32_t DT = get_compile_time_arg_val(0);
    constexpr uint32_t ST = get_compile_time_arg_val(1);
    constexpr uint32_t SS = get_compile_time_arg_val(2);  // the unit's cache tile rows (read in place)
    constexpr uint32_t RB = get_compile_time_arg_val(3);  // span tiles a read barrier (padded to RB)
    constexpr auto ae = TensorAccessorArgs<4>();
    constexpr auto av = TensorAccessorArgs<ae.next_compile_time_args_offset()>();
    const auto se = TensorAccessor(ae, e_addr);
    const auto sv = TensorAccessor(av, v_addr);

    constexpr uint32_t cb_e = 0;
    constexpr uint32_t cb_v = 1;
    const uint32_t e_bytes = get_tile_size(cb_e);
    const uint32_t v_bytes = get_tile_size(cb_v);

    for (uint32_t u = u0; u < u0 + nu; ++u) {
        for (uint32_t j0 = 0; j0 < ST; j0 += RB) {
            cb_reserve_back(cb_e, RB);
            cb_reserve_back(cb_v, RB * DT);
            uint32_t le = get_write_ptr(cb_e);
            uint32_t l1 = get_write_ptr(cb_v);
            for (uint32_t j = j0; j < j0 + RB && j < ST; ++j) {
                noc_async_read_page(u * ST + j, se, le);
                le += e_bytes;
                const uint32_t t = (u * SS + j) * DT;
                for (uint32_t d = 0; d < DT; ++d) {
                    noc_async_read_page(t + d, sv, l1);
                    l1 += v_bytes;
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_e, RB);
            cb_push_back(cb_v, RB * DT);
        }
    }
}

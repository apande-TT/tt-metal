// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The input stream of the C++ decode attention scores (see tt/cpp_scores_dec.py). For each of this
// core's (user, kv head) units it reads the query's one tile row (DT tiles of head_dim), then the
// unit's keys one tile row (DT tiles) at a time.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t u0 = get_arg_val<uint32_t>(2);
    const uint32_t nu = get_arg_val<uint32_t>(3);

    constexpr uint32_t DT = get_compile_time_arg_val(0);
    constexpr uint32_t ST = get_compile_time_arg_val(1);
    constexpr auto aq = TensorAccessorArgs<2>();
    constexpr auto ak = TensorAccessorArgs<aq.next_compile_time_args_offset()>();
    const auto sq = TensorAccessor(aq, q_addr);
    const auto sk = TensorAccessor(ak, k_addr);

    constexpr uint32_t cb_q = 0;
    constexpr uint32_t cb_k = 1;
    const uint32_t q_bytes = get_tile_size(cb_q);
    const uint32_t k_bytes = get_tile_size(cb_k);

    for (uint32_t u = u0; u < u0 + nu; ++u) {
        cb_reserve_back(cb_q, DT);
        uint32_t l1 = get_write_ptr(cb_q);
        for (uint32_t d = 0; d < DT; ++d) {
            noc_async_read_page(u * DT + d, sq, l1);
            l1 += q_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_q, DT);
        for (uint32_t j = 0; j < ST; ++j) {
            cb_reserve_back(cb_k, DT);
            l1 = get_write_ptr(cb_k);
            const uint32_t t = (u * ST + j) * DT;
            for (uint32_t d = 0; d < DT; ++d) {
                noc_async_read_page(t + d, sk, l1);
                l1 += k_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_k, DT);
        }
    }
}

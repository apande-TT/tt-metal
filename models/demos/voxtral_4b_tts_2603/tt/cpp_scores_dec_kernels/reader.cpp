// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The input stream of the C++ decode attention scores (see tt/cpp_scores_dec.py). With G > 0 the query is
// the RoPE output `[B, 1, n_heads, head_dim]` itself: unit (b, h)'s G query rows (heads h*G..) are gathered
// into rows 0..G-1 of a zeroed tile, two 64-byte face segments a row -- the grouped layout, built in place.
// For each of this
// core's (user, kv head) units it reads the query's one tile row (DT tiles of head_dim), then the
// unit's keys one tile row (DT tiles) at a time -- the first ST of the unit's SS cache tile rows, so the
// keys are read in place from the whole cache. With PT > 0 the first PT key tile rows -- the prompt prefix every
// user shares -- come from ONE shared copy `[1, NKV, PT * 32, head_dim]` (the bytes the cache would hold there).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t u0 = get_arg_val<uint32_t>(2);
    const uint32_t nu = get_arg_val<uint32_t>(3);
    const uint32_t p_addr = get_arg_val<uint32_t>(4);

    constexpr uint32_t DT = get_compile_time_arg_val(0);
    constexpr uint32_t ST = get_compile_time_arg_val(1);
    constexpr uint32_t SS = get_compile_time_arg_val(2);
    constexpr uint32_t RB = get_compile_time_arg_val(3);  // key rows a read barrier (rows padded to RB)
    constexpr uint32_t G = get_compile_time_arg_val(4);   // query rows a kv head (0: q already grouped)
    constexpr uint32_t NKV = get_compile_time_arg_val(5);
    constexpr uint32_t PT = get_compile_time_arg_val(6);  // shared prefix key tile rows (0: none)
    constexpr auto aq = TensorAccessorArgs<7>();
    constexpr auto ak = TensorAccessorArgs<aq.next_compile_time_args_offset()>();
    constexpr auto ap = TensorAccessorArgs<ak.next_compile_time_args_offset()>();
    const auto sq = TensorAccessor(aq, q_addr);
    const auto sk = TensorAccessor(ak, k_addr);
    const auto sp = TensorAccessor(ap, p_addr);

    constexpr uint32_t cb_q = 0;
    constexpr uint32_t cb_k = 1;
    const uint32_t q_bytes = get_tile_size(cb_q);
    const uint32_t k_bytes = get_tile_size(cb_k);

    if constexpr (G > 0) {
        // Both q slots start zeroed; only rows 0..G-1 are ever written, so the pad rows stay zero.
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        const uint32_t base = get_write_ptr(cb_q);
        for (uint32_t off = 0; off < 2 * DT * q_bytes; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, base + off, MEM_ZEROS_SIZE);
        }
        noc_async_read_barrier();
    }
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        cb_reserve_back(cb_q, DT);
        uint32_t l1 = get_write_ptr(cb_q);
        if constexpr (G > 0) {
            const uint32_t b = u / NKV;
            const uint32_t h = u % NKV;
            for (uint32_t d = 0; d < DT; ++d) {
                for (uint32_t r = 0; r < G; ++r) {
                    const uint32_t src = h * G + r;
                    for (uint32_t s = 0; s < 2; ++s) {
                        const uint32_t so = (((src >> 4) * 2 + s) * 1024) + (src & 15) * 64;
                        const uint32_t dof = (((r >> 4) * 2 + s) * 1024) + (r & 15) * 64;
                        noc_async_read(sq.get_noc_addr(b * DT + d, so), l1 + d * q_bytes + dof, 64);
                    }
                }
            }
        } else {
            for (uint32_t d = 0; d < DT; ++d) {
                noc_async_read_page(u * DT + d, sq, l1);
                l1 += q_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_q, DT);
        for (uint32_t j0 = 0; j0 < ST; j0 += RB) {
            cb_reserve_back(cb_k, RB * DT);
            l1 = get_write_ptr(cb_k);
            for (uint32_t j = j0; j < j0 + RB && j < ST; ++j) {
                if (j < PT) {
                    const uint32_t t = ((u % NKV) * PT + j) * DT;
                    for (uint32_t d = 0; d < DT; ++d) {
                        noc_async_read_page(t + d, sp, l1);
                        l1 += k_bytes;
                    }
                } else {
                    const uint32_t t = (u * SS + j) * DT;
                    for (uint32_t d = 0; d < DT; ++d) {
                        noc_async_read_page(t + d, sk, l1);
                        l1 += k_bytes;
                    }
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_k, RB * DT);
        }
    }
}

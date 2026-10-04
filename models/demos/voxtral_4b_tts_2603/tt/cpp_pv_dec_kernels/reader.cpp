// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The input stream of the C++ decode attention P@V (see tt/cpp_pv_dec.py). For each of this core's
// units -- a (user, kv head)'s query tile row -- it reads, per span tile j, the exp-weight tile e[j] and
// the values' tile row v[j] (DT tiles of head_dim).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t e_addr = get_arg_val<uint32_t>(0);
    const uint32_t v_addr = get_arg_val<uint32_t>(1);
    const uint32_t u0 = get_arg_val<uint32_t>(2);
    const uint32_t nu = get_arg_val<uint32_t>(3);
    const uint32_t p_addr = get_arg_val<uint32_t>(4);  // the shared prefix's values (PT > 0)

    constexpr uint32_t DT = get_compile_time_arg_val(0);
    constexpr uint32_t ST = get_compile_time_arg_val(1);
    constexpr uint32_t SS = get_compile_time_arg_val(2);  // the unit's cache tile rows (read in place)
    constexpr uint32_t RB = get_compile_time_arg_val(3);  // span tiles a read barrier (padded to RB)
    constexpr uint32_t MT = get_compile_time_arg_val(4);  // query tile rows a (user, kv head): MT units share its V
    // PACKED: e is the packed [B, 1, 32, span] (kv head h's G rows at rows h * G ..): unit (b, h) gathers its G
    // rows into rows 0..G-1 of a zeroed tile -- the grouped tile it had before, its pad rows zero.
    constexpr uint32_t PACKED = get_compile_time_arg_val(5);
    constexpr uint32_t G = get_compile_time_arg_val(6);
    constexpr uint32_t NKV = get_compile_time_arg_val(7);
    // PT > 0: the first PT value tile rows (the prompt prefix every user shares) come from ONE shared copy
    // `[1, NKV, PT * 32, head_dim]` -- the bytes each user's cache would hold there.
    constexpr uint32_t PT = get_compile_time_arg_val(8);
    constexpr auto ae = TensorAccessorArgs<9>();
    constexpr auto av = TensorAccessorArgs<ae.next_compile_time_args_offset()>();
    constexpr auto ap = TensorAccessorArgs<av.next_compile_time_args_offset()>();
    const auto se = TensorAccessor(ae, e_addr);
    const auto sv = TensorAccessor(av, v_addr);
    const auto sp = TensorAccessor(ap, p_addr);

    constexpr uint32_t cb_e = 0;
    constexpr uint32_t cb_v = 1;
    const uint32_t e_bytes = get_tile_size(cb_e);
    const uint32_t v_bytes = get_tile_size(cb_v);

    constexpr uint32_t face = 1024;
    constexpr uint32_t seg = 64;
    if constexpr (PACKED) {
        // Every e slot starts zeroed; only rows 0..G-1 are ever written, so the pad rows stay zero.
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        const uint32_t base = get_write_ptr(cb_e);
        for (uint32_t off = 0; off < 2 * RB * e_bytes; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, base + off, MEM_ZEROS_SIZE);
        }
        noc_async_read_barrier();
    }
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t pb = u / NKV;
        const uint32_t r0 = (u % NKV) * G;
        const uint32_t sof = ((r0 >> 4) * 2) * face + (r0 & 15) * seg;
        for (uint32_t j0 = 0; j0 < ST; j0 += RB) {
            cb_reserve_back(cb_e, RB);
            cb_reserve_back(cb_v, RB * DT);
            uint32_t le = get_write_ptr(cb_e);
            uint32_t l1 = get_write_ptr(cb_v);
            for (uint32_t j = j0; j < j0 + RB && j < ST; ++j) {
                if constexpr (PACKED) {
                    for (uint32_t f = 0; f < 2; ++f) {
                        noc_async_read(se.get_noc_addr(pb * ST + j, sof + f * face), le + f * face, G * seg);
                    }
                } else {
                    noc_async_read_page(u * ST + j, se, le);
                }
                le += e_bytes;
                if (j < PT) {
                    const uint32_t t = (((u / MT) % NKV) * PT + j) * DT;
                    for (uint32_t d = 0; d < DT; ++d) {
                        noc_async_read_page(t + d, sp, l1);
                        l1 += v_bytes;
                    }
                } else {
                    const uint32_t t = ((u / MT) * SS + j) * DT;
                    for (uint32_t d = 0; d < DT; ++d) {
                        noc_async_read_page(t + d, sv, l1);
                        l1 += v_bytes;
                    }
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_e, RB);
            cb_push_back(cb_v, RB * DT);
        }
    }
}

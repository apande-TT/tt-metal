// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Prefill routed experts, token-gathered (reader). The work is cut into UNITS: 64 consecutive tokens of one
// expert's routed-token list (expert e has ceil(n_e / 64) of them), enumerated in expert order and dealt
// round-robin over the cores, so a heavily routed expert is spread over many cores instead of serialising one.
// Per unit: the unit's token list (to the writer), the two routing-weight tiles (every element of row j = the
// fp32 weight of the j-th token), the tokens' x rows (row-major, one page per token) for compute to tilize,
// and expert e's gate/up and down weight tiles, streamed once per unit.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// Non-negative fp32 bit pattern -> integer part (the counts are exact small integers).
inline uint32_t f32_to_u32(uint32_t b) {
    const int32_t ex = static_cast<int32_t>((b >> 23) & 0xffu) - 127;
    if (ex < 0) {
        return 0;
    }
    const uint32_t mant = (b & 0x7fffffu) | 0x800000u;
    return ex >= 23 ? mant << (ex - 23) : mant >> (23 - ex);
}

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);     // hidden tiles: K of gate/up, N of down
    constexpr uint32_t NT_GU = get_compile_time_arg_val(1);  // tiles in one row of the interleaved gate/up weight
    constexpr uint32_t CPE = get_compile_time_arg_val(2);    // intermediate tiles per expert
    constexpr uint32_t M = get_compile_time_arg_val(3);      // tokens
    constexpr uint32_t KB = get_compile_time_arg_val(4);     // K tiles per gate/up weight push
    constexpr uint32_t E = get_compile_time_arg_val(5);      // local experts
    constexpr uint32_t w_tile_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(7);  // one row-major x row
    constexpr uint32_t NC = get_compile_time_arg_val(8);         // cores
    constexpr uint32_t MAX_UNITS = get_compile_time_arg_val(9);  // most units one core can be dealt
    constexpr auto x_args = TensorAccessorArgs<10>();
    constexpr auto gu_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto wd_args = TensorAccessorArgs<gu_args.next_compile_time_args_offset()>();
    constexpr auto wt_args = TensorAccessorArgs<wd_args.next_compile_time_args_offset()>();
    constexpr auto n_args = TensorAccessorArgs<wt_args.next_compile_time_args_offset()>();

    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(0));
    const auto gu = TensorAccessor(gu_args, get_arg_val<uint32_t>(1));
    const auto wd = TensorAccessor(wd_args, get_arg_val<uint32_t>(2));
    const auto wt = TensorAccessor(wt_args, get_arg_val<uint32_t>(3));
    const auto counts = TensorAccessor(n_args, get_arg_val<uint32_t>(4));
    const uint32_t core = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_rm = 0, cb_g = 1, cb_u = 2, cb_scale = 3, cb_count = 4, cb_list = 5, cb_wt = 6, cb_wd = 7;
    constexpr uint32_t U = 64;  // tokens per unit (two 32-row blocks)

    // This core's units, from the per-expert routed-token counts ([1, E] fp32).
    cb_reserve_back(cb_wt, 1);
    const uint32_t wt_l1 = get_write_ptr(cb_wt);
    noc_async_read_page(0, counts, wt_l1);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wt_l1);
    uint16_t unit_e[MAX_UNITS], unit_g[MAX_UNITS];
    uint32_t mine = 0, u = 0;
    for (uint32_t e = 0; e < E; ++e) {
        const uint32_t groups = (f32_to_u32(w[e]) + U - 1) / U;
        for (uint32_t g = 0; g < groups; ++g, ++u) {
            if (u % NC == core && mine < MAX_UNITS) {
                unit_e[mine] = e;
                unit_g[mine++] = g;
            }
        }
    }
    cb_reserve_back(cb_count, 1);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_count)) = mine;
    cb_push_back(cb_count, 1);
    cb_reserve_back(cb_list, 1);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_list)) = mine;
    cb_push_back(cb_list, 1);

    uint32_t loaded = E;
    for (uint32_t i = 0; i < mine; ++i) {
        const uint32_t e = unit_e[i], first = unit_g[i] * U;
        if (e != loaded) {  // row e of the transposed [E, M] fp32 weights
            noc_async_read_page(e, wt, wt_l1);
            noc_async_read_barrier();
            loaded = e;
        }
        // [e, n, t_0 .. t_{n-1}]: tokens first .. first + U - 1 of e's routed list (weight non-zero; +-0 are not).
        cb_reserve_back(cb_list, 1);
        volatile tt_l1_ptr uint32_t* list = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_list));
        uint32_t n = 0, seen = 0;
        for (uint32_t t = 0; t < M && n < U; ++t) {
            if (w[t] & 0x7fffffffu) {
                if (seen++ >= first) {
                    list[2 + n++] = t;
                }
            }
        }
        list[0] = e;
        list[1] = n;
        if (n == 0) {  // only on inconsistent counts; keep the pad-row reads in bounds
            list[2] = 0;
        }

        // Routing-weight tiles (fp32): all 32 elements of row j of tile rr = weight of token rr * 32 + j, i.e. its
        // two 16-wide face rows; pad rows are 0, so their activations are exactly 0.
        cb_reserve_back(cb_scale, 2);
        volatile tt_l1_ptr uint32_t* s32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_scale));
        for (uint32_t j = 0; j < 2 * 32; ++j) {
            const uint32_t f = j < n ? w[list[2 + j]] : 0;
            volatile tt_l1_ptr uint32_t* row = s32 + (j / 32) * 1024 + ((j % 32) / 16) * 512 + (j % 16) * 16;
            for (uint32_t q = 0; q < 16; ++q) {
                row[q] = f;
                row[256 + q] = f;
            }
        }
        cb_push_back(cb_scale, 2);

        // The two row blocks' x rows, 32 row-major rows each (pad rows repeat the unit's last token).
        for (uint32_t rr = 0; rr < 2; ++rr) {
            cb_reserve_back(cb_rm, Kt);
            const uint32_t l1 = get_write_ptr(cb_rm);
            for (uint32_t j = 0; j < 32; ++j) {
                const uint32_t idx = rr * 32 + j;
                noc_async_read_page(list[2 + (idx < n ? idx : (n ? n - 1 : 0))], x, l1 + j * row_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb_rm, Kt);
        }
        cb_push_back(cb_list, 1);

        // Gate/up column pairs of e (gate tile at 2c, up at 2c + 1 of the interleaved weight), K in blocks.
        for (uint32_t c = 0; c < CPE; ++c) {
            const uint32_t col = (e * CPE + c) * 2;
            for (uint32_t kb = 0; kb < Kt; kb += KB) {
                cb_reserve_back(cb_g, KB);
                cb_reserve_back(cb_u, KB);
                const uint32_t g_l1 = get_write_ptr(cb_g);
                const uint32_t u_l1 = get_write_ptr(cb_u);
                for (uint32_t k = 0; k < KB; ++k) {
                    const uint32_t page = (kb + k) * NT_GU + col;
                    noc_async_read_page(page, gu, g_l1 + k * w_tile_bytes);
                    noc_async_read_page(page + 1, gu, u_l1 + k * w_tile_bytes);
                }
                noc_async_read_barrier();
                cb_push_back(cb_g, KB);
                cb_push_back(cb_u, KB);
            }
        }

        // Down rows of e ([CPE x Kt] tiles), one output column at a time.
        for (uint32_t nn = 0; nn < Kt; ++nn) {
            cb_reserve_back(cb_wd, CPE);
            const uint32_t l1 = get_write_ptr(cb_wd);
            for (uint32_t k = 0; k < CPE; ++k) {
                noc_async_read_page((e * CPE + k) * Kt + nn, wd, l1 + k * w_tile_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb_wd, CPE);
        }
    }
}

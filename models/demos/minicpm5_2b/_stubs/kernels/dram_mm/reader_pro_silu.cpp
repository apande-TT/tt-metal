// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode down projection, helper-core reader (BRISC/NCRISC pair of a dram_mm program whose helpers
// run a prologue and an epilogue):
//   prologue: gate and up tiles [p_start, p_start + p_tiles) of the MLP for h = silu(gate) * up;
//   epilogue: residual tile and the projection's tile [e_start, e_start + e_tiles), the latter once the
//             projection writers have stored it and bumped this core's semaphore (once per tile).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_gate = get_compile_time_arg_val(0);
    constexpr uint32_t cb_up = get_compile_time_arg_val(1);
    constexpr uint32_t cb_res = get_compile_time_arg_val(2);
    constexpr uint32_t cb_mm = get_compile_time_arg_val(3);
    constexpr uint32_t ep_sem = get_compile_time_arg_val(4);
    constexpr auto g_args = TensorAccessorArgs<5>();
    constexpr auto u_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto r_args = TensorAccessorArgs<u_args.next_compile_time_args_offset()>();
    constexpr auto m_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = 32 * 32 * 2;

    const uint32_t p_start = get_arg_val<uint32_t>(0);
    const uint32_t p_tiles = get_arg_val<uint32_t>(1);
    const uint32_t e_start = get_arg_val<uint32_t>(2);
    const uint32_t e_tiles = get_arg_val<uint32_t>(3);
    const auto gate = TensorAccessor(g_args, get_common_arg_val<uint32_t>(0));
    const auto up = TensorAccessor(u_args, get_common_arg_val<uint32_t>(1));
    const auto res = TensorAccessor(r_args, get_common_arg_val<uint32_t>(2));
    const auto mm = TensorAccessor(m_args, get_common_arg_val<uint32_t>(3));

    auto read = [&](uint32_t cb, const auto& acc, uint32_t start, uint32_t n) {
        cb_reserve_back(cb, n);
        uint32_t l1 = get_write_ptr(cb);
        for (uint32_t t = start; t < start + n; ++t) {
            noc_async_read_page(t, acc, l1);
            l1 += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb, n);
    };
    read(cb_gate, gate, p_start, p_tiles);
    read(cb_up, up, p_start, p_tiles);
    read(cb_res, res, e_start, e_tiles);

    auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ep_sem));
    noc_semaphore_wait(sem, e_tiles);
    noc_semaphore_set(sem, 0);
    read(cb_mm, mm, e_start, e_tiles);
}

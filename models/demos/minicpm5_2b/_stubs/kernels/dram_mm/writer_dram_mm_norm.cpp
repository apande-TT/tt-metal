// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode projection with the RMSNorm of its input fused in front (BRISC): the dataflow of
// kernels/rms_norm_shared over this program's cores, then an all-gather of the normalized row into every core's
// cb_in0, then the output writes of writer_dram_mm.cpp. The weight reader (NCRISC) streams from the start, so the
// weight CB fills while the norm runs.
//   1. Read this core's tpc input tiles [t_start, t_start + tpc) and gamma tiles (reducer: the constant tiles).
//   2. Its fp32 x^2 tiles -> slot t_start of the reducer's cb_x2all, bump the reducer's sem_gather.
//   3. Reducer: publish cb_x2all once all cores are in; copy the rstd tile into every core's cb_ex2pe, bump sem_bcast.
//   4. Publish the rstd tile; copy this core's normalized tiles into slot t_start of every core's cb_in0, bump
//      sem_all everywhere; publish cb_in0 once all slices are in.
//   5. Output tiles [n_start, n_start + num_groups * group_w) (columns from split_t on into the second output).
// CBs sit at the same L1 address on every core and cb_x2all / cb_ex2pe / cb_in0 are untouched before these writes,
// so their base write pointers are the remote destinations.
// Only cores 0..n_norm-1 (runtime arg does_norm) take part in steps 1-4's norm; every core receives the row.
// Common args: [w, x, out, out_hi, gamma, consts, reducer_x, reducer_y, x_0, y_0, ..., x_{n-1}, y_{n-1}].

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t group_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t split_t = get_compile_time_arg_val(4);
    constexpr uint32_t tpc = get_compile_time_arg_val(5);
    constexpr uint32_t cb_xin = get_compile_time_arg_val(6);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(7);
    constexpr uint32_t cb_eps = get_compile_time_arg_val(8);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(9);
    constexpr uint32_t cb_x2 = get_compile_time_arg_val(10);
    constexpr uint32_t cb_x2all = get_compile_time_arg_val(11);
    constexpr uint32_t cb_ex2pe = get_compile_time_arg_val(12);
    constexpr uint32_t cb_ex2pe_send = get_compile_time_arg_val(13);
    constexpr uint32_t cb_normout = get_compile_time_arg_val(14);
    constexpr uint32_t sem_gather = get_compile_time_arg_val(15);
    constexpr uint32_t sem_bcast = get_compile_time_arg_val(16);
    constexpr uint32_t sem_all = get_compile_time_arg_val(17);
    constexpr uint32_t n_cores = get_compile_time_arg_val(18);
    constexpr uint32_t n_norm = get_compile_time_arg_val(19);  // cores 0..n_norm-1 do the norm, all receive its row
    constexpr uint32_t sem_start = get_compile_time_arg_val(20);  // releases this core's weight stream
    // Epilogue (residual add on helper cores): signal each stored output tile's helper (common args from ep_off).
    constexpr uint32_t ep_on = get_compile_time_arg_val(21);
    constexpr uint32_t ep_sem = get_compile_time_arg_val(22);
    constexpr uint32_t ep_tpc = get_compile_time_arg_val(23);  // output tiles per helper core
    constexpr uint32_t ep_off = get_compile_time_arg_val(24);
    constexpr auto x_args = TensorAccessorArgs<25>();
    constexpr auto out_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = 32 * 32 * 2;
    constexpr uint32_t f32_tile_bytes = 32 * 32 * 4;

    const uint32_t n_start = get_arg_val<uint32_t>(2);
    const uint32_t num_groups = get_arg_val<uint32_t>(4);
    const uint32_t t_start = get_arg_val<uint32_t>(6);
    const uint32_t is_reducer = get_arg_val<uint32_t>(7);
    const uint32_t does_norm = get_arg_val<uint32_t>(8);
    const auto x = TensorAccessor(x_args, get_common_arg_val<uint32_t>(1));
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(2));
    const auto out_hi = TensorAccessor(out_args, get_common_arg_val<uint32_t>(3));  // same spec, other buffer
    const auto gamma = TensorAccessor(g_args, get_common_arg_val<uint32_t>(4));
    const auto consts = TensorAccessor(c_args, get_common_arg_val<uint32_t>(5));
    const uint32_t red_x = get_common_arg_val<uint32_t>(6);
    const uint32_t red_y = get_common_arg_val<uint32_t>(7);

    cb_reserve_back(cb_in0, Kt);
    if (!does_norm) {
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_start)) = 1;
    }
    if (does_norm) {
        // 1. Input slice, gamma slice, constants.
        cb_reserve_back(cb_xin, tpc);
        cb_reserve_back(cb_gamma, tpc);
        const uint32_t in_l1 = get_write_ptr(cb_xin);
        const uint32_t g_l1 = get_write_ptr(cb_gamma);
        for (uint32_t i = 0; i < tpc; ++i) {
            noc_async_read_page(t_start + i, x, in_l1 + i * tile_bytes);
            noc_async_read_page(t_start + i, gamma, g_l1 + i * tile_bytes);
        }
        if (is_reducer) {
            cb_reserve_back(cb_scaler, 1);
            cb_reserve_back(cb_eps, 1);
            noc_async_read_page(0, consts, get_write_ptr(cb_scaler));
            noc_async_read_page(1, consts, get_write_ptr(cb_eps));
        }
        noc_async_read_barrier();
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_start)) = 1;  // DRAM reads done
        cb_push_back(cb_xin, tpc);
        cb_push_back(cb_gamma, tpc);
        if (is_reducer) {
            cb_push_back(cb_scaler, 1);
            cb_push_back(cb_eps, 1);
        }

        // 2. x^2 slice -> reducer.
        cb_wait_front(cb_x2, tpc);
        noc_async_write(
            get_read_ptr(cb_x2),
            get_noc_addr(red_x, red_y, get_write_ptr(cb_x2all) + t_start * f32_tile_bytes),
            tpc * f32_tile_bytes);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(red_x, red_y, get_semaphore(sem_gather)), 1);
        noc_async_atomic_barrier();
        cb_pop_front(cb_x2, tpc);

        // 3. Reducer: the whole x^2 row to its compute, then its rstd tile -> every core.
        if (is_reducer) {
            cb_reserve_back(cb_x2all, Kt);
            auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_gather));
            noc_semaphore_wait(sem, n_norm);
            noc_semaphore_set(sem, 0);
            cb_push_back(cb_x2all, Kt);

            cb_wait_front(cb_ex2pe_send, 1);
            const uint32_t src = get_read_ptr(cb_ex2pe_send);
            const uint32_t dst = get_write_ptr(cb_ex2pe);
            for (uint32_t c = 0; c < n_norm; ++c) {
                const uint32_t cx = get_common_arg_val<uint32_t>(8 + 2 * c);
                const uint32_t cy = get_common_arg_val<uint32_t>(9 + 2 * c);
                noc_async_write(src, get_noc_addr(cx, cy, dst), f32_tile_bytes);
            }
            noc_async_write_barrier();
            const uint32_t sem_l1 = get_semaphore(sem_bcast);
            for (uint32_t c = 0; c < n_norm; ++c) {
                const uint32_t cx = get_common_arg_val<uint32_t>(8 + 2 * c);
                const uint32_t cy = get_common_arg_val<uint32_t>(9 + 2 * c);
                noc_semaphore_inc(get_noc_addr(cx, cy, sem_l1), 1);
            }
            noc_async_atomic_barrier();
            cb_pop_front(cb_ex2pe_send, 1);
        }

        // 4. rstd to compute; normalized slice -> every core's cb_in0; the gathered row to compute.
        cb_reserve_back(cb_ex2pe, 1);
        {
            auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_bcast));
            noc_semaphore_wait(sem, 1);
            noc_semaphore_set(sem, 0);
        }
        cb_push_back(cb_ex2pe, 1);

        const uint32_t in0_dst = get_write_ptr(cb_in0) + t_start * tile_bytes;
        cb_wait_front(cb_normout, tpc);
        {
            const uint32_t src = get_read_ptr(cb_normout);
            for (uint32_t c = 0; c < n_cores; ++c) {
                const uint32_t cx = get_common_arg_val<uint32_t>(8 + 2 * c);
                const uint32_t cy = get_common_arg_val<uint32_t>(9 + 2 * c);
                noc_async_write(src, get_noc_addr(cx, cy, in0_dst), tpc * tile_bytes);
            }
            noc_async_write_barrier();
            const uint32_t sem_l1 = get_semaphore(sem_all);
            for (uint32_t c = 0; c < n_cores; ++c) {
                const uint32_t cx = get_common_arg_val<uint32_t>(8 + 2 * c);
                const uint32_t cy = get_common_arg_val<uint32_t>(9 + 2 * c);
                noc_semaphore_inc(get_noc_addr(cx, cy, sem_l1), 1);
            }
            noc_async_atomic_barrier();
        }
        cb_pop_front(cb_normout, tpc);
    }
    {
        auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_all));
        noc_semaphore_wait(sem, n_norm);
        noc_semaphore_set(sem, 0);
    }
    cb_push_back(cb_in0, Kt);

    // 5. Output.
    uint32_t n = n_start;
    for (uint32_t g = 0; g < num_groups; ++g) {
        cb_wait_front(cb_out, group_w);
        uint32_t src = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < group_w; ++j) {
            if (n + j < split_t) {
                noc_async_write_page(n + j, out, src);
            } else {
                noc_async_write_page(n + j - split_t, out_hi, src);
            }
            src += tile_bytes;
        }
        noc_async_writes_flushed();
        if constexpr (ep_on) {
            // The tiles are stored: hand each to its epilogue core.
            noc_async_write_barrier();
            const uint32_t sem_l1 = get_semaphore(ep_sem);
            for (uint32_t j = 0; j < group_w; ++j) {
                const uint32_t h = (n + j) / ep_tpc;
                const uint32_t hx = get_common_arg_val<uint32_t>(ep_off + 2 * h);
                const uint32_t hy = get_common_arg_val<uint32_t>(ep_off + 2 * h + 1);
                noc_semaphore_inc(get_noc_addr(hx, hy, sem_l1), 1);
            }
        }
        cb_pop_front(cb_out, group_w);
        n += group_w;
    }
    noc_async_write_barrier();
    if constexpr (ep_on) {
        noc_async_atomic_barrier();
    }
}

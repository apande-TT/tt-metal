// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of the exact-lane linear (lane_linear.cpp). Streams, for this core's work units:
//   cb 2: the 8 diagonal 0/1 lane tiles E_q = diag(col % 8 == q), once;
//   per unit (row block rb, output columns [nt0, nt1)):
//     cb 1: the weight's column block, row-major kt * csize + (nt - nt0), once per unit (csize * Kt entries);
//     per limb i: cb 0: the limb's row block (K tile kt -> kt);
//     with a bias (patterns = 3): cb 8: the bias tiles of the unit's output columns (float32, the bias in every
//     row), for the median epilogue.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t a0_addr = get_arg_val<uint32_t>(0);
    const uint32_t a1_addr = get_arg_val<uint32_t>(1);
    const uint32_t a2_addr = get_arg_val<uint32_t>(2);
    const uint32_t w_addr = get_arg_val<uint32_t>(3);
    const uint32_t d_addr = get_arg_val<uint32_t>(4);
    const uint32_t start_unit = get_arg_val<uint32_t>(5);
    const uint32_t num_units = get_arg_val<uint32_t>(6);
    const uint32_t bias_addr = get_arg_val<uint32_t>(7);

    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);
    constexpr uint32_t chunks = get_compile_time_arg_val(2);
    constexpr uint32_t csize = get_compile_time_arg_val(3);
    constexpr uint32_t limbs = get_compile_time_arg_val(4);
    constexpr bool has_bias = get_compile_time_arg_val(5) != 0;
    constexpr auto a0_args = TensorAccessorArgs<6>();
    constexpr auto a1_args = TensorAccessorArgs<a0_args.next_compile_time_args_offset()>();
    constexpr auto a2_args = TensorAccessorArgs<a1_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<a2_args.next_compile_time_args_offset()>();
    constexpr auto d_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto bias_args = TensorAccessorArgs<d_args.next_compile_time_args_offset()>();
    constexpr uint32_t lanes = 8;

    constexpr uint32_t cb_a = 0;
    constexpr uint32_t cb_b = 1;
    constexpr uint32_t cb_d = 2;
    const uint32_t page = get_local_cb_interface(cb_a).fifo_page_size;
    const auto a0 = TensorAccessor(a0_args, a0_addr);
    const auto a1 = TensorAccessor(a1_args, a1_addr);
    const auto a2 = TensorAccessor(a2_args, a2_addr);
    const auto w = TensorAccessor(w_args, w_addr);
    const auto dg = TensorAccessor(d_args, d_addr);

    Noc noc;
    DataflowBuffer da(cb_a);
    DataflowBuffer db(cb_b);
    DataflowBuffer dd(cb_d);

    dd.reserve_back(lanes);
    for (uint32_t t = 0; t < lanes; ++t) {
        noc.async_read(dg, dd, page, {.page_id = t}, {.offset_bytes = t * page});
    }
    noc.async_read_barrier();
    dd.push_back(lanes);

    for (uint32_t u = start_unit; u < start_unit + num_units; ++u) {
        const uint32_t rb = u / chunks;
        const uint32_t nt0 = (u % chunks) * csize;
        const uint32_t nt1 = nt0 + csize < Nt ? nt0 + csize : Nt;

        // always a full csize * Kt block (a short last chunk leaves its tail columns unused), so blocks never wrap
        db.reserve_back(csize * Kt);
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            for (uint32_t nt = nt0; nt < nt1; ++nt) {
                noc.async_read(w, db, page, {.page_id = kt * Nt + nt}, {.offset_bytes = (kt * csize + nt - nt0) * page});
            }
        }
        noc.async_read_barrier();
        db.push_back(csize * Kt);
        for (uint32_t i = 0; i < limbs; ++i) {
            da.reserve_back(Kt);
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                const uint32_t id = rb * Kt + kt;
                if (i == 0) {
                    noc.async_read(a0, da, page, {.page_id = id}, {.offset_bytes = kt * page});
                } else if (i == 1) {
                    noc.async_read(a1, da, page, {.page_id = id}, {.offset_bytes = kt * page});
                } else {
                    noc.async_read(a2, da, page, {.page_id = id}, {.offset_bytes = kt * page});
                }
            }
            noc.async_read_barrier();
            da.push_back(Kt);
        }
        if constexpr (has_bias) {
            constexpr uint32_t cb_bias = 8;
            const auto bs = TensorAccessor(bias_args, bias_addr);
            const uint32_t bpage = get_local_cb_interface(cb_bias).fifo_page_size;
            DataflowBuffer dbias(cb_bias);
            dbias.reserve_back(csize);
            for (uint32_t nt = nt0; nt < nt1; ++nt) {
                noc.async_read(bs, dbias, bpage, {.page_id = nt}, {.offset_bytes = (nt - nt0) * bpage});
            }
            noc.async_read_barrier();
            dbias.push_back(csize);
        }
    }
}

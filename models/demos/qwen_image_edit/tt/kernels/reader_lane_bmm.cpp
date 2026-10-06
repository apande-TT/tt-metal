// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of the exact-lane batched product (lane_bmm.cpp). Streams, for this core's work units:
//   cb 2: the 8 * Kt diagonal 0/1 lane tiles (lane r, K tile kt -> tile r * Kt + kt), once;
//   per unit (batch b, row tile mt, output columns [nt0, nt1)):
//     cb 0: a's row block of both limbs (limb i, K tile kt -> i * Kt + kt);
//     cb 1: per output column nt, b's column block of both limbs (limb j, K tile kt -> j * Kt + kt),
//           read as b[nt, kt] tiles when transpose_b (the matmul transposes within the tile).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t a0_addr = get_arg_val<uint32_t>(0);
    const uint32_t a1_addr = get_arg_val<uint32_t>(1);
    const uint32_t b0_addr = get_arg_val<uint32_t>(2);
    const uint32_t b1_addr = get_arg_val<uint32_t>(3);
    const uint32_t d_addr = get_arg_val<uint32_t>(4);
    const uint32_t start_unit = get_arg_val<uint32_t>(5);
    const uint32_t num_units = get_arg_val<uint32_t>(6);

    constexpr uint32_t Mt = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Nt = get_compile_time_arg_val(2);
    constexpr uint32_t chunks = get_compile_time_arg_val(3);
    constexpr uint32_t csize = get_compile_time_arg_val(4);
    constexpr uint32_t transpose_b = get_compile_time_arg_val(5);
    constexpr auto a0_args = TensorAccessorArgs<6>();
    constexpr auto a1_args = TensorAccessorArgs<a0_args.next_compile_time_args_offset()>();
    constexpr auto b0_args = TensorAccessorArgs<a1_args.next_compile_time_args_offset()>();
    constexpr auto b1_args = TensorAccessorArgs<b0_args.next_compile_time_args_offset()>();
    constexpr auto d_args = TensorAccessorArgs<b1_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_a = 0;
    constexpr uint32_t cb_b = 1;
    constexpr uint32_t cb_d = 2;
    const uint32_t page = get_local_cb_interface(cb_a).fifo_page_size;
    const auto a0 = TensorAccessor(a0_args, a0_addr);
    const auto a1 = TensorAccessor(a1_args, a1_addr);
    const auto b0 = TensorAccessor(b0_args, b0_addr);
    const auto b1 = TensorAccessor(b1_args, b1_addr);
    const auto dg = TensorAccessor(d_args, d_addr);

    Noc noc;
    DataflowBuffer da(cb_a);
    DataflowBuffer db(cb_b);
    DataflowBuffer dd(cb_d);

    dd.reserve_back(8 * Kt);
    for (uint32_t t = 0; t < 8 * Kt; ++t) {
        noc.async_read(dg, dd, page, {.page_id = t}, {.offset_bytes = t * page});
    }
    noc.async_read_barrier();
    dd.push_back(8 * Kt);

    for (uint32_t u = start_unit; u < start_unit + num_units; ++u) {
        const uint32_t rb = u / chunks;
        const uint32_t c = u % chunks;
        const uint32_t bt = rb / Mt;
        const uint32_t mt = rb % Mt;
        const uint32_t nt0 = c * csize;
        const uint32_t nt1 = nt0 + csize < Nt ? nt0 + csize : Nt;

        da.reserve_back(2 * Kt);
        const uint32_t a_base = (bt * Mt + mt) * Kt;
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            noc.async_read(a0, da, page, {.page_id = a_base + kt}, {.offset_bytes = kt * page});
            noc.async_read(a1, da, page, {.page_id = a_base + kt}, {.offset_bytes = (Kt + kt) * page});
        }
        noc.async_read_barrier();
        da.push_back(2 * Kt);

        for (uint32_t nt = nt0; nt < nt1; ++nt) {
            db.reserve_back(2 * Kt);
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                const uint32_t id = transpose_b ? (bt * Nt + nt) * Kt + kt : (bt * Kt + kt) * Nt + nt;
                noc.async_read(b0, db, page, {.page_id = id}, {.offset_bytes = kt * page});
                noc.async_read(b1, db, page, {.page_id = id}, {.offset_bytes = (Kt + kt) * page});
            }
            noc.async_read_barrier();
            db.push_back(2 * Kt);
        }
    }
}

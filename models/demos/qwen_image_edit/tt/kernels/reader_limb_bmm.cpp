// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of the precise limb batched matmul (limb_bmm.cpp). Work unit u = (batch bt = u / blocks, rows
// [r0, r0 + nr) with r0 = (u % blocks) * R):
//   cb 0, once per unit (R rows' worth of entries): row r0 + j of both a limbs (ah[kt] at 2 * j * Kt + kt,
//         al[kt] at (2 * j + 1) * Kt + kt), one barrier;
//   cb 1, per output column nt: column nt of both b limbs (bh[kt] at kt, bl[kt] at Kt + kt), one barrier.
// Runtime arg 6 (k_tiles): when nonzero, args 7 + i * R + j hold the K tiles row j of this core's i-th unit uses
// (a's K tiles past them are zero: limb_bmm.cpp skips them): only those of a's row, and of b the most any row of
// the unit uses, are read (the rest of the cb entries are left unread and unused).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t ah_addr = get_arg_val<uint32_t>(0);
    const uint32_t al_addr = get_arg_val<uint32_t>(1);
    const uint32_t bh_addr = get_arg_val<uint32_t>(2);
    const uint32_t bl_addr = get_arg_val<uint32_t>(3);
    const uint32_t start_unit = get_arg_val<uint32_t>(4);
    const uint32_t num_units = get_arg_val<uint32_t>(5);

    constexpr uint32_t Mt = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Nt = get_compile_time_arg_val(2);
    constexpr uint32_t blocks = get_compile_time_arg_val(3);
    constexpr uint32_t R = get_compile_time_arg_val(4);
    constexpr auto ah_args = TensorAccessorArgs<6>();
    constexpr auto al_args = TensorAccessorArgs<ah_args.next_compile_time_args_offset()>();
    constexpr auto bh_args = TensorAccessorArgs<al_args.next_compile_time_args_offset()>();
    constexpr auto bl_args = TensorAccessorArgs<bh_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_a = 0;
    constexpr uint32_t cb_b = 1;
    const uint32_t page = get_local_cb_interface(cb_a).fifo_page_size;
    const auto ah = TensorAccessor(ah_args, ah_addr);
    const auto al = TensorAccessor(al_args, al_addr);
    const auto bh = TensorAccessor(bh_args, bh_addr);
    const auto bl = TensorAccessor(bl_args, bl_addr);

    const bool k_tiles = get_arg_val<uint32_t>(6) != 0;

    Noc noc;
    DataflowBuffer da(cb_a);
    DataflowBuffer db(cb_b);
    for (uint32_t u = start_unit; u < start_unit + num_units; ++u) {
        const uint32_t bt = u / blocks;
        const uint32_t r0 = (u % blocks) * R;
        const uint32_t nr = (r0 + R < Mt ? r0 + R : Mt) - r0;

        uint32_t kb = 0;  // the most K tiles any row of the unit uses
        da.reserve_back(2 * Kt * R);
        for (uint32_t j = 0; j < nr; ++j) {
            const uint32_t base = (bt * Mt + r0 + j) * Kt;
            const uint32_t kj = k_tiles ? get_arg_val<uint32_t>(7 + (u - start_unit) * R + j) : Kt;
            kb = kj > kb ? kj : kb;
            for (uint32_t kt = 0; kt < kj; ++kt) {
                noc.async_read(ah, da, page, {.page_id = base + kt}, {.offset_bytes = (2 * j * Kt + kt) * page});
                noc.async_read(al, da, page, {.page_id = base + kt}, {.offset_bytes = ((2 * j + 1) * Kt + kt) * page});
            }
        }
        noc.async_read_barrier();
        da.push_back(2 * Kt * R);

        for (uint32_t nt = 0; nt < Nt; ++nt) {
            db.reserve_back(2 * Kt);
            for (uint32_t kt = 0; kt < kb; ++kt) {
                const uint32_t id = (bt * Kt + kt) * Nt + nt;
                noc.async_read(bh, db, page, {.page_id = id}, {.offset_bytes = kt * page});
                noc.async_read(bl, db, page, {.page_id = id}, {.offset_bytes = (Kt + kt) * page});
            }
            noc.async_read_barrier();
            db.push_back(2 * Kt);
        }
    }
}

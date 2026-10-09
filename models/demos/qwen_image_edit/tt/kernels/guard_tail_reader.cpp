// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Streams the same tile range of up to six float32 operands into CBs c_0..c_5 (the guarded tail's ex, dn, nn,
// tol, lo; the VAE vote tail's acc, bn, f0, e_neg, tol; the TP all-reduce's gathered partials). Operand k reads
// page (i % mod_k when mod_k is nonzero, else i) + off_k: mod broadcasts a (32, C) row block over every row of
// tiles (C / 32 tiles per row, so tile i's column is i % mod); off picks one slab of a stacked tensor.
// Compile-time: the six operands' accessor args (unused ones repeat operand 0's), then the operand count.
// Runtime: addr0..4, num_tiles, start_id, mod0..4, off0..4, then addr5, mod5, off5.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t addr0 = get_arg_val<uint32_t>(0);
    const uint32_t addr1 = get_arg_val<uint32_t>(1);
    const uint32_t addr2 = get_arg_val<uint32_t>(2);
    const uint32_t addr3 = get_arg_val<uint32_t>(3);
    const uint32_t addr4 = get_arg_val<uint32_t>(4);
    const uint32_t num_tiles = get_arg_val<uint32_t>(5);
    const uint32_t start_id = get_arg_val<uint32_t>(6);
    const uint32_t mod0 = get_arg_val<uint32_t>(7);
    const uint32_t mod1 = get_arg_val<uint32_t>(8);
    const uint32_t mod2 = get_arg_val<uint32_t>(9);
    const uint32_t mod3 = get_arg_val<uint32_t>(10);
    const uint32_t mod4 = get_arg_val<uint32_t>(11);
    const uint32_t off0 = get_arg_val<uint32_t>(12);
    const uint32_t off1 = get_arg_val<uint32_t>(13);
    const uint32_t off2 = get_arg_val<uint32_t>(14);
    const uint32_t off3 = get_arg_val<uint32_t>(15);
    const uint32_t off4 = get_arg_val<uint32_t>(16);
    const uint32_t addr5 = get_arg_val<uint32_t>(17);
    const uint32_t mod5 = get_arg_val<uint32_t>(18);
    const uint32_t off5 = get_arg_val<uint32_t>(19);

    constexpr auto args0 = TensorAccessorArgs<0>();
    constexpr auto args1 = TensorAccessorArgs<args0.next_compile_time_args_offset()>();
    constexpr auto args2 = TensorAccessorArgs<args1.next_compile_time_args_offset()>();
    constexpr auto args3 = TensorAccessorArgs<args2.next_compile_time_args_offset()>();
    constexpr auto args4 = TensorAccessorArgs<args3.next_compile_time_args_offset()>();
    constexpr auto args5 = TensorAccessorArgs<args4.next_compile_time_args_offset()>();
    constexpr uint32_t n_in = get_compile_time_arg_val(args5.next_compile_time_args_offset());
    const auto src0 = TensorAccessor(args0, addr0);
    const auto src1 = TensorAccessor(args1, addr1);
    const auto src2 = TensorAccessor(args2, addr2);
    const auto src3 = TensorAccessor(args3, addr3);
    const auto src4 = TensorAccessor(args4, addr4);
    const auto src5 = TensorAccessor(args5, addr5);
    const uint32_t bytes = get_local_cb_interface(0).fifo_page_size;

    Noc noc;
    CircularBuffer cb0(0), cb1(1), cb2(2), cb3(3), cb4(4), cb5(5);
    for (uint32_t i = start_id; i < start_id + num_tiles; ++i) {
        cb0.reserve_back(1);
        noc.async_read(src0, cb0, bytes, {.page_id = (mod0 ? i % mod0 : i) + off0}, {.offset_bytes = 0});
        if constexpr (n_in > 1) {
            cb1.reserve_back(1);
            noc.async_read(src1, cb1, bytes, {.page_id = (mod1 ? i % mod1 : i) + off1}, {.offset_bytes = 0});
        }
        if constexpr (n_in > 2) {
            cb2.reserve_back(1);
            noc.async_read(src2, cb2, bytes, {.page_id = (mod2 ? i % mod2 : i) + off2}, {.offset_bytes = 0});
        }
        if constexpr (n_in > 3) {
            cb3.reserve_back(1);
            noc.async_read(src3, cb3, bytes, {.page_id = (mod3 ? i % mod3 : i) + off3}, {.offset_bytes = 0});
        }
        if constexpr (n_in > 4) {
            cb4.reserve_back(1);
            noc.async_read(src4, cb4, bytes, {.page_id = (mod4 ? i % mod4 : i) + off4}, {.offset_bytes = 0});
        }
        if constexpr (n_in > 5) {
            cb5.reserve_back(1);
            noc.async_read(src5, cb5, bytes, {.page_id = (mod5 ? i % mod5 : i) + off5}, {.offset_bytes = 0});
        }
        noc.async_read_barrier();
        cb0.push_back(1);
        if constexpr (n_in > 1) {
            cb1.push_back(1);
        }
        if constexpr (n_in > 2) {
            cb2.push_back(1);
        }
        if constexpr (n_in > 3) {
            cb3.push_back(1);
        }
        if constexpr (n_in > 4) {
            cb4.push_back(1);
        }
        if constexpr (n_in > 5) {
            cb5.push_back(1);
        }
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MiniCPM5-2B decode: the fused attention's per-step L1 constants (bf16 mask table, fp32 sum-scaler replicas) copied
// DRAM -> L1 by one program: this core copies pages [a0, a0 + na) of tensor A and [b0, b0 + nb) of tensor B, each
// page staged through a local buffer (verbatim bytes).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_buf = get_compile_time_arg_val(0);
    constexpr uint32_t a_page = get_compile_time_arg_val(1);
    constexpr uint32_t b_page = get_compile_time_arg_val(2);
    constexpr auto a_src_args = TensorAccessorArgs<3>();
    constexpr auto a_dst_args = TensorAccessorArgs<a_src_args.next_compile_time_args_offset()>();
    constexpr auto b_src_args = TensorAccessorArgs<a_dst_args.next_compile_time_args_offset()>();
    constexpr auto b_dst_args = TensorAccessorArgs<b_src_args.next_compile_time_args_offset()>();

    const uint32_t a0 = get_arg_val<uint32_t>(0);
    const uint32_t na = get_arg_val<uint32_t>(1);
    const uint32_t b0 = get_arg_val<uint32_t>(2);
    const uint32_t nb = get_arg_val<uint32_t>(3);
    const auto a_src = TensorAccessor(a_src_args, get_common_arg_val<uint32_t>(0), a_page);
    const auto a_dst = TensorAccessor(a_dst_args, get_common_arg_val<uint32_t>(1), a_page);
    const auto b_src = TensorAccessor(b_src_args, get_common_arg_val<uint32_t>(2), b_page);
    const auto b_dst = TensorAccessor(b_dst_args, get_common_arg_val<uint32_t>(3), b_page);

    const uint32_t buf = get_write_ptr(cb_buf);
    uint32_t l1 = buf;
    for (uint32_t i = 0; i < na; ++i) {
        noc_async_read_page(a0 + i, a_src, l1);
        l1 += a_page;
    }
    for (uint32_t i = 0; i < nb; ++i) {
        noc_async_read_page(b0 + i, b_src, l1);
        l1 += b_page;
    }
    noc_async_read_barrier();
    l1 = buf;
    for (uint32_t i = 0; i < na; ++i) {
        noc_async_write_page(a0 + i, a_dst, l1);
        l1 += a_page;
    }
    for (uint32_t i = 0; i < nb; ++i) {
        noc_async_write_page(b0 + i, b_dst, l1);
        l1 += b_page;
    }
    noc_async_write_barrier();
}

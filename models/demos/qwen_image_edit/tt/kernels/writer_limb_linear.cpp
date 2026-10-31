// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer of the two-limb linear (limb_linear.cpp): each mb x nb float32 output block of cb 16 (tile r * nb + c)
// to y[row0 + r, n0 + nblk * nb + c], per work unit in the compute kernel's order. With a bias (has_bias), it
// first reads the block's nb float32 bias tiles (bias rows [32, N], tile n0 + nblk * nb + c) into cb 2. With
// `heads`, y's [q | k | v] column blocks go straight to head-major q / k / v tensors instead of y.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t first_unit = get_arg_val<uint32_t>(1);
    const uint32_t n0 = get_arg_val<uint32_t>(2);  // first output column tile of this core's column group

    constexpr uint32_t Nt = get_compile_time_arg_val(0);
    constexpr uint32_t Ntg = get_compile_time_arg_val(1);  // column tiles of this column group
    constexpr uint32_t mb = get_compile_time_arg_val(2);
    constexpr uint32_t nb = get_compile_time_arg_val(3);
    constexpr uint32_t units = get_compile_time_arg_val(4);
    constexpr uint32_t unit_stride = get_compile_time_arg_val(5);
    constexpr auto y_args = TensorAccessorArgs<6>();
    constexpr bool has_bias = get_compile_time_arg_val(y_args.next_compile_time_args_offset()) != 0;
    constexpr auto b_args = TensorAccessorArgs<y_args.next_compile_time_args_offset() + 1>();
    const uint32_t b_addr = get_arg_val<uint32_t>(3);
    // heads: output column tile c is (q / k / v = c / (H * Dt), head (c / Dt) % H, d = c % Dt) of three head-major
    // tensors [B, H, Jt tile rows, Dt]; output tile row R = (b = R / St, s = R % St) lands on joint row row_off + s
    constexpr bool heads = get_compile_time_arg_val(b_args.next_compile_time_args_offset()) != 0;
    constexpr auto q_args = TensorAccessorArgs<b_args.next_compile_time_args_offset() + 1>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_out = 16;
    const uint32_t page = get_local_cb_interface(cb_out).fifo_page_size;
    const auto y = TensorAccessor(y_args, y_addr);
    const auto b = TensorAccessor(b_args, b_addr);
    const auto qo = TensorAccessor(q_args, get_arg_val<uint32_t>(4));
    const auto ko = TensorAccessor(k_args, get_arg_val<uint32_t>(5));
    const auto vo = TensorAccessor(v_args, get_arg_val<uint32_t>(6));
    const uint32_t St = get_arg_val<uint32_t>(7);       // this projection's tile rows per batch entry
    const uint32_t Jt = get_arg_val<uint32_t>(8);       // joint tile rows per batch entry of the head tensors
    const uint32_t row_off = get_arg_val<uint32_t>(9);  // this projection's first joint tile row
    const uint32_t H = get_arg_val<uint32_t>(10);
    const uint32_t Dt = get_arg_val<uint32_t>(11);

    Noc noc;
    DataflowBuffer dout(cb_out);
    DataflowBuffer dbias(2);
    for (uint32_t i = 0; i < units; ++i) {
        const uint32_t row0 = (first_unit + i * unit_stride) * mb;
        for (uint32_t nblk = 0; nblk < Ntg / nb; ++nblk) {
            if constexpr (has_bias) {
                dbias.reserve_back(nb);
                for (uint32_t c = 0; c < nb; ++c) {
                    noc.async_read(b, dbias, page, {.page_id = n0 + nblk * nb + c}, {.offset_bytes = c * page});
                }
                noc.async_read_barrier();
                dbias.push_back(nb);
            }
            dout.wait_front(mb * nb);
            for (uint32_t r = 0; r < mb; ++r) {
                for (uint32_t c = 0; c < nb; ++c) {
                    const uint32_t col = n0 + nblk * nb + c;
                    if constexpr (heads) {
                        const uint32_t row = row0 + r;
                        const uint32_t which = col / (H * Dt);
                        const uint32_t h = (col / Dt) % H;
                        const uint32_t id = (((row / St) * H + h) * Jt + row_off + row % St) * Dt + col % Dt;
                        const uint32_t src = (r * nb + c) * page;
                        if (which == 0) {
                            noc.async_write(dout, qo, page, {.offset_bytes = src}, {.page_id = id});
                        } else if (which == 1) {
                            noc.async_write(dout, ko, page, {.offset_bytes = src}, {.page_id = id});
                        } else {
                            noc.async_write(dout, vo, page, {.offset_bytes = src}, {.page_id = id});
                        }
                    } else {
                        const uint32_t id = (row0 + r) * Nt + col;
                        noc.async_write(dout, y, page, {.offset_bytes = (r * nb + c) * page}, {.page_id = id});
                    }
                }
            }
            noc.async_write_barrier();
            dout.pop_front(mb * nb);
        }
    }
}

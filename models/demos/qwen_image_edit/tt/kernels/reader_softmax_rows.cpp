// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of the fused attention softmax (softmax_rows.cpp): for each tile row r of this core it streams the
// row's Wt logit tiles r * Wt + wt (one barrier per `batch` tiles; batch divides Wt, so a batch never wraps
// the 2 * batch tile buffer), and, when r starts a new mask batch, first the Wt tiles of that batch's
// [1, 1, 1, Sk] mask row (tile key * Wt + wt, row 0 valid; the compute kernel broadcasts it over the rows).
// Row r is (n, h, ht) with r = (n * H + h) * Ht + ht; the mask batch is n, or 0 for a mask shared by all.
// With implicit tile padding it first writes fill_pad's right / bottom / corner padding mask tiles (1.0 in the
// padding) once, with fill_pad's own generate_mask_tile.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/kernels/dataflow/fill_pad_dataflow_common.hpp"

void kernel_main() {
    const uint32_t s_addr = get_arg_val<uint32_t>(0);
    const uint32_t mask_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_rows = get_arg_val<uint32_t>(2);
    const uint32_t start_row = get_arg_val<uint32_t>(3);

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr uint32_t mask_batches = get_compile_time_arg_val(3);
    constexpr bool has_mask = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t w_mod = get_compile_time_arg_val(5);  // Sk % 32
    constexpr uint32_t h_mod = get_compile_time_arg_val(6);  // Sq % 32
    constexpr uint32_t batch = get_compile_time_arg_val(7);
    constexpr auto s_args = TensorAccessorArgs<8>();
    constexpr auto mask_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_s = 0;
    constexpr uint32_t cb_mask = 1;
    constexpr uint32_t cb_pad = 5;
    const uint32_t page = get_local_cb_interface(cb_s).fifo_page_size;
    const auto s = TensorAccessor(s_args, s_addr);
    const auto mask = TensorAccessor(mask_args, mask_addr);

    Noc noc;
    DataflowBuffer in(cb_s);
    if constexpr (w_mod != 0 || h_mod != 0) {  // right, bottom, corner (those present), as fill_pad's masks
        constexpr uint32_t one = 0x3F800000u;  // fill_pad's MASK_VALUE for float32
        constexpr uint32_t num_pad = (w_mod != 0) + (h_mod != 0) + (w_mod != 0 && h_mod != 0);
        DataflowBuffer pad(cb_pad);
        pad.reserve_back(num_pad);
        auto* p = reinterpret_cast<uint32_t*>(pad.get_write_ptr());
        constexpr uint32_t tile = 1024;
        if constexpr (w_mod != 0) {
            generate_mask_tile<uint32_t, w_mod, 32, 32>(p, one);
            p += tile;
        }
        if constexpr (h_mod != 0) {
            generate_mask_tile<uint32_t, 32, h_mod, 32>(p, one);
            p += tile;
        }
        if constexpr (w_mod != 0 && h_mod != 0) {
            generate_mask_tile<uint32_t, w_mod, h_mod, 32>(p, one);
        }
        pad.push_back(num_pad);
    }
    uint32_t mask_key = 0xFFFFFFFF;
    for (uint32_t r = start_row; r < start_row + num_rows; ++r) {
        if constexpr (has_mask) {
            const uint32_t key = mask_batches == 1 ? 0 : r / Ht / H;
            if (key != mask_key) {
                DataflowBuffer m(cb_mask);
                m.reserve_back(Wt);
                for (uint32_t wt = 0; wt < Wt; ++wt) {
                    noc.async_read(mask, m, page, {.page_id = key * Wt + wt}, {.offset_bytes = wt * page});
                }
                noc.async_read_barrier();
                m.push_back(Wt);
                mask_key = key;
            }
        }
        for (uint32_t wt = 0; wt < Wt; wt += batch) {
            in.reserve_back(batch);
            for (uint32_t k = 0; k < batch; ++k) {
                noc.async_read(s, in, page, {.page_id = r * Wt + wt + k}, {.offset_bytes = k * page});
            }
            noc.async_read_barrier();
            in.push_back(batch);
        }
    }
}

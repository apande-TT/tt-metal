// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The latents' 2 x 2 pack / unpack (QwenImageEditPipeline._pack_latents / _unpack_latents) on ROW_MAJOR 32-bit
// tensors, one (b, i) pair of image rows per unit, rearranged word by word in L1 (pure data movement):
//   pack   (mode 0): x [B, C, H, W] -> y [B, (H / 2)(W / 2), 4C], y[b, i * Wh + j, c * 4 + p * 2 + q] = x[b, c, 2i + p, 2j + q]
//   unpack (mode 1): the inverse.
// A unit reads the 2C image rows (c, 2i + p) (W words each) and the Wh packed rows (i, j) (4C words each) on the
// other side, instead of a 6-D ROW_MAJOR permute whose innermost dim (q, 2 words) moves 8 bytes at a time.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_units = get_arg_val<uint32_t>(2);
    const uint32_t start = get_arg_val<uint32_t>(3);

    constexpr uint32_t mode = get_compile_time_arg_val(0);
    constexpr uint32_t C = get_compile_time_arg_val(1);
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr uint32_t W = get_compile_time_arg_val(3);
    constexpr uint32_t Hh = H / 2;
    constexpr uint32_t Wh = W / 2;
    constexpr uint32_t img_row = W * 4;      // bytes of an image row
    constexpr uint32_t pack_row = 4 * C * 4;  // bytes of a packed row
    constexpr auto src_args = TensorAccessorArgs<4>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const auto src = TensorAccessor(src_args, src_addr);
    const auto dst = TensorAccessor(dst_args, dst_addr);

    Noc noc;
    DataflowBuffer in(0);   // one unit's source rows (2C image rows, or Wh packed rows: 2C * W words either way)
    DataflowBuffer out(1);  // its destination rows
    in.reserve_back(1);
    out.reserve_back(1);
    auto* a = reinterpret_cast<volatile uint32_t*>(in.get_write_ptr());
    auto* y = reinterpret_cast<volatile uint32_t*>(out.get_write_ptr());

    for (uint32_t u = start; u < start + num_units; ++u) {
        const uint32_t b = u / Hh;
        const uint32_t i = u % Hh;
        if constexpr (mode == 0) {
            for (uint32_t c = 0; c < C; ++c) {
                for (uint32_t p = 0; p < 2; ++p) {
                    noc.async_read(src, in, img_row, {.page_id = (b * C + c) * H + 2 * i + p},
                                   {.offset_bytes = (c * 2 + p) * img_row});
                }
            }
            noc.async_read_barrier();
            for (uint32_t j = 0; j < Wh; ++j) {
                for (uint32_t c = 0; c < C; ++c) {
                    for (uint32_t p = 0; p < 2; ++p) {
                        for (uint32_t q = 0; q < 2; ++q) {
                            y[j * 4 * C + c * 4 + p * 2 + q] = a[(c * 2 + p) * W + 2 * j + q];
                        }
                    }
                }
            }
            for (uint32_t j = 0; j < Wh; ++j) {
                noc.async_write(out, dst, pack_row, {.offset_bytes = j * pack_row}, {.page_id = b * Hh * Wh + i * Wh + j});
            }
        } else {
            for (uint32_t j = 0; j < Wh; ++j) {
                noc.async_read(src, in, pack_row, {.page_id = b * Hh * Wh + i * Wh + j}, {.offset_bytes = j * pack_row});
            }
            noc.async_read_barrier();
            for (uint32_t j = 0; j < Wh; ++j) {
                for (uint32_t c = 0; c < C; ++c) {
                    for (uint32_t p = 0; p < 2; ++p) {
                        for (uint32_t q = 0; q < 2; ++q) {
                            y[(c * 2 + p) * W + 2 * j + q] = a[j * 4 * C + c * 4 + p * 2 + q];
                        }
                    }
                }
            }
            for (uint32_t c = 0; c < C; ++c) {
                for (uint32_t p = 0; p < 2; ++p) {
                    noc.async_write(out, dst, img_row, {.offset_bytes = (c * 2 + p) * img_row},
                                    {.page_id = (b * C + c) * H + 2 * i + p});
                }
            }
        }
        noc.async_write_barrier();
    }
}

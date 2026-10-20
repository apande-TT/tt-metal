// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader / assembler of tt/patch_unfold.py: the 2 x 2 spatial unfold of the vision features straight in tile layout.
// A unit is (image, output tile row R, channel tile c): its 32 merged tokens come from the 128 source tokens of source
// tile rows 4R .. 4R + 3, so the unit reads those 4 tiles of channel tile c and assembles the 4 output tiles of
// sub-position j = a * 2 + b (output column tile j * DT + c): output row t is source row (2r + a) * W + 2s + b, with
// (r, s) = divmod(32 R + t, W / 2). Rows move as whole 16-element face rows (32 bytes) by local NOC reads (L1 to L1;
// RISC word copies: 90 us, local NOC reads: 40 us for 8 images), so the bytes are copied, not recomputed.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
// byte offset of the left half (columns 0..15) of tile row r in a 32 x 32 bf16 tile of 16 x 16 faces
constexpr uint32_t row_offset(uint32_t r) { return (r < 16 ? 0 : 1024) + (r & 15) * 32; }
}  // namespace

void kernel_main() {
    constexpr uint32_t DT = get_compile_time_arg_val(0);  // channel tiles (d / 32)
    constexpr uint32_t RT = get_compile_time_arg_val(1);  // output tile rows an image (h * w / 4 / 32)
    constexpr uint32_t W = get_compile_time_arg_val(2);   // patches a source row (w)
    constexpr uint32_t ST = get_compile_time_arg_val(3);  // source tile rows an image (h * w / 32)
    constexpr uint32_t TILE = 2048;                       // bf16 tile bytes
    constexpr uint32_t cb_src = 0, cb_out = 16;
    constexpr auto x_args = TensorAccessorArgs<4>();
    static_assert(32 % (W / 2) == 0, "an output tile row must hold whole merged rows");

    const auto x = TensorAccessor(x_args, get_arg_val<uint32_t>(0));
    const uint32_t u0 = get_arg_val<uint32_t>(1);
    const uint32_t nu = get_arg_val<uint32_t>(2);

    cb_reserve_back(cb_src, 4);  // scratch for a unit's 4 source tiles (never pushed)
    const uint32_t src = get_write_ptr(cb_src);
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t c = u % DT;
        const uint32_t R = (u / DT) % RT;
        const uint32_t img = u / (DT * RT);
        for (uint32_t i = 0; i < 4; ++i) {
            noc_async_read_page((img * ST + 4 * R + i) * DT + c, x, src + i * TILE);
        }
        noc_async_read_barrier();

        cb_reserve_back(cb_out, 4);
        const uint32_t out = get_write_ptr(cb_out);
        for (uint32_t j = 0; j < 4; ++j) {
            const uint32_t a = j >> 1, b = j & 1;
            const uint32_t dst = out + j * TILE;
            for (uint32_t t = 0; t < 32; ++t) {
                const uint32_t m = 32 * R + t;
                const uint32_t r = m / (W / 2), s = m % (W / 2);
                const uint32_t tok = (2 * r + a) * W + 2 * s + b - 128 * R;  // token within the unit's 4 source tiles
                const uint32_t from = src + (tok >> 5) * TILE + row_offset(tok & 31);
                const uint32_t to = dst + row_offset(t);
                noc_async_read(get_noc_addr(from), to, 32);              // columns 0..15
                noc_async_read(get_noc_addr(from + 512), to + 512, 32);  // columns 16..31
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_out, 4);
    }
}

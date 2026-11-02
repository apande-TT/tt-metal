// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Reads runs of whole tiles from other cores' L1 (a width-sharded tensor's shards) into circular buffers, in list
// order per buffer. With SCALE it first builds cb 1 = the column e of the fp32 tile W spread along every row (each
// row's own W[r, e] in all 32 columns), W read from DRAM through cb 2.
// compile-time args: SCALE (+ W's accessor args when SCALE)
// runtime args: W address, e, n, then n x (cb, noc x, noc y, L1 address, tiles)
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
template <bool SCALE, uint32_t W_ARGS>
void spread_weight() {
    constexpr uint32_t cb_r = 1, cb_w = 2;
    if constexpr (SCALE) {
        constexpr auto w_args = TensorAccessorArgs<W_ARGS>();
        const auto w = TensorAccessor(w_args, get_arg_val<uint32_t>(0));
        const uint32_t e = get_arg_val<uint32_t>(1);
        cb_reserve_back(cb_w, 1);
        cb_reserve_back(cb_r, 1);
        const uint32_t w_l1 = (get_write_ptr(cb_w) + 63) & ~63u;  // DRAM reads need 64 B-aligned landing addresses
        noc_async_read(w.get_noc_addr(0), w_l1, 4096);
        noc_async_read_barrier();
        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w_l1);
        volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_r));
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t row = (r / 16) * 512 + (r % 16) * 16;  // 32-bit words: faces of 256, face rows of 16
            const uint32_t v = src[row + (e / 16) * 256 + e % 16];
            for (uint32_t c = 0; c < 16; ++c) {
                dst[row + c] = v;
                dst[row + 256 + c] = v;
            }
        }
        cb_push_back(cb_r, 1);
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t MAX_CB = 8;
    spread_weight<get_compile_time_arg_val(0) != 0, 1>();

    const uint32_t n = get_arg_val<uint32_t>(2);
    uint32_t count[MAX_CB] = {0};
    for (uint32_t i = 0; i < n; ++i) {
        count[get_arg_val<uint32_t>(3 + 5 * i)] += get_arg_val<uint32_t>(3 + 5 * i + 4);
    }
    uint32_t next[MAX_CB];
    for (uint32_t c = 0; c < MAX_CB; ++c) {
        if (count[c]) {
            cb_reserve_back(c, count[c]);
            next[c] = get_write_ptr(c);
        }
    }
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t* a = reinterpret_cast<const uint32_t*>(get_arg_addr(3 + 5 * i));
        const uint32_t bytes = a[4] * get_tile_size(a[0]);
        noc_async_read(get_noc_addr(a[1], a[2], a[3]), next[a[0]], bytes);
        next[a[0]] += bytes;
    }
    noc_async_read_barrier();
    for (uint32_t c = 0; c < MAX_CB; ++c) {
        if (count[c]) {
            cb_push_back(c, count[c]);
        }
    }
}

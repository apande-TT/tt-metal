// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Sampler top-k, final merge, one core per user row b. The first tree stage (ttnn.topk over G groups of the
// vocabulary) leaves each group's top K sorted largest first, so the row's top K is a G-way merge of those lists:
// K steps, each taking the largest list head. fp32 values are compared through an order-preserving integer key;
// ties go to the earlier group, i.e. the smaller token ids. Reads the row's G * K values and labels as two 64 B
// face rows per tile; writes the K winners, largest first, to the row's face rows of the [B, K] outputs.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// fp32 bits -> unsigned key with the same order (negatives reversed below the positives).
inline uint32_t order_key(uint32_t bits) { return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u); }

void kernel_main() {
    constexpr uint32_t G = get_compile_time_arg_val(0);  // sorted lists in the row
    constexpr uint32_t K = get_compile_time_arg_val(1);  // entries per list = entries kept (a multiple of 32)
    constexpr auto v_args = TensorAccessorArgs<2>();
    constexpr auto l_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto ov_args = TensorAccessorArgs<l_args.next_compile_time_args_offset()>();
    constexpr auto ol_args = TensorAccessorArgs<ov_args.next_compile_time_args_offset()>();
    const auto vin = TensorAccessor(v_args, get_arg_val<uint32_t>(0));
    const auto lin = TensorAccessor(l_args, get_arg_val<uint32_t>(1));
    const auto vout = TensorAccessor(ov_args, get_arg_val<uint32_t>(2));
    const auto lout = TensorAccessor(ol_args, get_arg_val<uint32_t>(3));
    const uint32_t b = get_arg_val<uint32_t>(4);  // the user row this core owns

    constexpr uint32_t N = G * K, NT = N / 32, KT = K / 32;
    constexpr uint32_t face = 1024, frow = 64;  // 32-bit tile geometry
    constexpr uint32_t cb_scratch = 0;

    cb_reserve_back(cb_scratch, 1);
    const uint32_t v_l1 = (get_write_ptr(cb_scratch) + 63) & ~63u;  // DRAM reads need 64 B-aligned landing addresses
    const uint32_t l_l1 = v_l1 + N * 4;
    const uint32_t ov_l1 = l_l1 + N * 4;
    const uint32_t ol_l1 = ov_l1 + K * 4;
    const uint32_t head_l1 = ol_l1 + K * 4;
    const uint32_t hkey_l1 = head_l1 + ((G * 4 + 63) & ~63u);

    // Row r of tile row b / 32: two 64 B face rows per tile; column c of the row lands at index c.
    const uint32_t tile_row = b / 32, r = b % 32;
    const uint32_t off = (r / 16) * 2 * face + (r % 16) * frow;
    for (uint32_t t = 0; t < NT; ++t) {
        noc_async_read(vin.get_noc_addr(tile_row * NT + t, off), v_l1 + t * 2 * frow, frow);
        noc_async_read(vin.get_noc_addr(tile_row * NT + t, off + face), v_l1 + t * 2 * frow + frow, frow);
        noc_async_read(lin.get_noc_addr(tile_row * NT + t, off), l_l1 + t * 2 * frow, frow);
        noc_async_read(lin.get_noc_addr(tile_row * NT + t, off + face), l_l1 + t * 2 * frow + frow, frow);
    }
    noc_async_read_barrier();

    volatile tt_l1_ptr uint32_t* v = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v_l1);
    volatile tt_l1_ptr uint32_t* l = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_l1);
    volatile tt_l1_ptr uint32_t* ov = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ov_l1);
    volatile tt_l1_ptr uint32_t* ol = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ol_l1);
    volatile tt_l1_ptr uint32_t* head = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(head_l1);
    volatile tt_l1_ptr uint32_t* hkey = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hkey_l1);
    for (uint32_t g = 0; g < G; ++g) {
        head[g] = 0;
        hkey[g] = order_key(v[g * K]);
    }
    for (uint32_t i = 0; i < K; ++i) {
        uint32_t best = 0, best_key = hkey[0];
        for (uint32_t g = 1; g < G; ++g) {
            const uint32_t k = hkey[g];
            if (k > best_key) {  // strict: ties keep the earlier list
                best_key = k;
                best = g;
            }
        }
        const uint32_t h = head[best];
        ov[i] = v[best * K + h];
        ol[i] = l[best * K + h];
        head[best] = h + 1;
        hkey[best] = h + 1 < K ? order_key(v[best * K + h + 1]) : 0u;  // an exhausted list never wins again
    }

    // Winner i is column i: tile i / 32, face half (i % 32) / 16, so each 64 B run of the outputs is one face row.
    for (uint32_t t = 0; t < KT; ++t) {
        for (uint32_t hf = 0; hf < 2; ++hf) {
            const uint32_t src = (t * 2 + hf) * frow;
            noc_async_write(ov_l1 + src, vout.get_noc_addr(tile_row * KT + t, off + hf * face), frow);
            noc_async_write(ol_l1 + src, lout.get_noc_addr(tile_row * KT + t, off + hf * face), frow);
        }
    }
    noc_async_write_barrier();
}

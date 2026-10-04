// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The input stream of the C++ banded codec attention (see tt/cpp_band_attn.py). Per unit -- a (batch,
// head)'s query tile row r -- it reads the DT query tiles, the key and value tile rows of the band
// (slot 0: row r - 1, slot 1: row r), and the two additive-mask tiles of the band, read in place from
// the prebuilt [1, H, MM, MM] mask. Row 0 has no row before it: its slot 0 re-reads row 0 and takes
// the mask tile (r, 1) -- fully blocked -- so that slot's weights come out exactly zero.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t m_addr = get_arg_val<uint32_t>(3);
    const uint32_t u0 = get_arg_val<uint32_t>(4);
    const uint32_t nu = get_arg_val<uint32_t>(5);
    const uint32_t scale_bits = get_arg_val<uint32_t>(6);

    constexpr uint32_t DT = get_compile_time_arg_val(0);   // head_dim tiles
    constexpr uint32_t RT = get_compile_time_arg_val(1);   // sequence tile rows a (batch, head)
    constexpr uint32_t H = get_compile_time_arg_val(2);    // heads
    constexpr uint32_t MMT = get_compile_time_arg_val(3);  // mask row tiles
    constexpr uint32_t MST = get_compile_time_arg_val(4);  // mask column tiles
    // 0: q / k / v are [B, H, T, D]; 1: they are the MERGED [B, 1, T, H * D] (head h in column tiles
    // h * DT ..), read in place -- no concat + head split before the kernel, no head merge after it.
    constexpr uint32_t MERGED = get_compile_time_arg_val(5);
    // q / k tile bytes: 4096 (float32) or 2048 (bf16, the fused codec qk-norm's output).
    constexpr uint32_t QK_BYTES = get_compile_time_arg_val(6);
    constexpr auto aq = TensorAccessorArgs<7>();
    constexpr auto ak = TensorAccessorArgs<aq.next_compile_time_args_offset()>();
    constexpr auto av = TensorAccessorArgs<ak.next_compile_time_args_offset()>();
    constexpr auto am = TensorAccessorArgs<av.next_compile_time_args_offset()>();
    const auto sq = TensorAccessor(aq, q_addr);
    const auto sk = TensorAccessor(ak, k_addr);
    const auto sv = TensorAccessor(av, v_addr);
    const auto sm = TensorAccessor(am, m_addr);

    constexpr uint32_t cb_q = 0;
    constexpr uint32_t cb_k = 1;
    constexpr uint32_t cb_v = 2;
    constexpr uint32_t cb_scale = 4;
    constexpr uint32_t cb_mask = 5;
    constexpr uint32_t tile_bytes = 4096;

    // The scale as a whole float32 tile (binary_ng's scalar operand tile).
    cb_reserve_back(cb_scale, 1);
    {
        auto* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_scale));
        for (uint32_t i = 0; i < 1024; ++i) {
            p[i] = scale_bits;
        }
    }
    cb_push_back(cb_scale, 1);

    // The page of head bh's tile row t, head_dim tile d.
    auto page = [](uint32_t bh, uint32_t t, uint32_t d) -> uint32_t {
        return MERGED ? ((bh / H) * RT + t) * (H * DT) + (bh % H) * DT + d : (bh * RT + t) * DT + d;
    };

    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t r = u % RT;
        const uint32_t bh = u / RT;
        const uint32_t h = bh % H;
        const uint32_t prev = r > 0 ? r - 1 : 0;
        const uint32_t mrow = (h * MMT + r) * MST;

        cb_reserve_back(cb_q, DT);
        cb_reserve_back(cb_k, 2 * DT);
        cb_reserve_back(cb_mask, 2);
        uint32_t lq = get_write_ptr(cb_q);
        uint32_t lk = get_write_ptr(cb_k);
        const uint32_t lm = get_write_ptr(cb_mask);
        for (uint32_t d = 0; d < DT; ++d) {
            noc_async_read_page(page(bh, r, d), sq, lq);
            lq += QK_BYTES;
        }
        for (uint32_t d = 0; d < DT; ++d) {
            noc_async_read_page(page(bh, prev, d), sk, lk);
            lk += QK_BYTES;
        }
        for (uint32_t d = 0; d < DT; ++d) {
            noc_async_read_page(page(bh, r, d), sk, lk);
            lk += QK_BYTES;
        }
        noc_async_read_page(mrow + (r > 0 ? r - 1 : 1), sm, lm);
        noc_async_read_page(mrow + r, sm, lm + tile_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_q, DT);
        cb_push_back(cb_k, 2 * DT);
        cb_push_back(cb_mask, 2);

        cb_reserve_back(cb_v, 2 * DT);
        uint32_t lv = get_write_ptr(cb_v);
        for (uint32_t d = 0; d < DT; ++d) {
            noc_async_read_page(page(bh, prev, d), sv, lv);
            lv += tile_bytes;
        }
        for (uint32_t d = 0; d < DT; ++d) {
            noc_async_read_page(page(bh, r, d), sv, lv);
            lv += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_v, 2 * DT);
    }
}

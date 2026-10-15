// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Prefill last-row gather of tt/last_row.py: row R of every user's (S, D) slice of the tiled hidden state (B, S, D),
// B = 32, emitted as the tiled (32, D) LM-head input. Output tile j holds column tile j of all 32 users: user b's
// row is row b of the tile. Inside a 32 x 32 tile a row is two 16-element face rows (faces 2 * (row / 16) and + 1,
// 32 bytes each), so each RISC fetches, per user and face, the 64-byte DRAM-aligned row pair holding row R into
// scratch, copies the 32 bytes of row R into row b of its output tile and writes the tile. One output tile per pass;
// a RISC takes tiles FIRST, FIRST + STRIDE, ...

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t hid_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t first = get_arg_val<uint32_t>(2);  // this RISC's first output tile
    const uint32_t row_page0 = get_arg_val<uint32_t>(3);  // page of user 0's tile row holding R, column tile 0
    const uint32_t r = get_arg_val<uint32_t>(4);  // R % 32
    const uint32_t slot = get_arg_val<uint32_t>(5);  // this RISC's scratch slot on the core (0 or 1)

    constexpr uint32_t DT = get_compile_time_arg_val(0);  // column tiles (D / 32)
    constexpr uint32_t USER_PAGES = get_compile_time_arg_val(1);  // pages between users' rows (S / 32 * DT)
    constexpr uint32_t STRIDE = get_compile_time_arg_val(2);  // RISCs over all cores
    constexpr auto a_hid = TensorAccessorArgs<3>();
    constexpr auto a_out = TensorAccessorArgs<a_hid.next_compile_time_args_offset()>();
    const auto s_hid = TensorAccessor(a_hid, hid_addr);
    const auto s_out = TensorAccessor(a_out, out_addr);

    constexpr uint32_t USERS = 32, FACE = 512, PAIR = 64;  // bytes: a 16 x 16 bf16 face, two 16-element face rows
    constexpr uint32_t cb = 0;  // per slot: USERS x 2 row pairs of scratch, then the output tile
    constexpr uint32_t SLOT_BYTES = USERS * 2 * PAIR + 2048;
    // 64-byte aligned like the DRAM row pairs it receives (the CB carries 64 spare bytes)
    const uint32_t scratch = ((get_write_ptr(cb) + PAIR - 1) & ~(PAIR - 1)) + slot * SLOT_BYTES;
    const uint32_t tile = scratch + USERS * 2 * PAIR;
    const uint32_t src_off = (r >> 4) * 2 * FACE + ((r & 15) & ~1u) * 32;  // row pair in face 2 * (R / 16)
    const uint32_t half = (r & 1) * 8;  // row R's 32 bytes in its pair, in words

    for (uint32_t j = first; j < DT; j += STRIDE) {
        for (uint32_t b = 0; b < USERS; ++b) {
            const uint32_t page = row_page0 + b * USER_PAGES + j;
            const uint32_t dst = scratch + b * 2 * PAIR;
            noc_async_read(s_hid.get_noc_addr(page, src_off), dst, PAIR);
            noc_async_read(s_hid.get_noc_addr(page, src_off + FACE), dst + PAIR, PAIR);
        }
        noc_async_read_barrier();
        for (uint32_t b = 0; b < USERS; ++b) {
            for (uint32_t k = 0; k < 2; ++k) {
                const volatile tt_l1_ptr uint32_t* from =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + (b * 2 + k) * PAIR) + half;
                volatile tt_l1_ptr uint32_t* to =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile + ((b >> 4) * 2 + k) * FACE + (b & 15) * 32);
                for (uint32_t w = 0; w < 8; ++w) {
                    to[w] = from[w];
                }
            }
        }
        noc_async_write(tile, s_out.get_noc_addr(j), 2048);
        noc_async_write_barrier();
    }
}

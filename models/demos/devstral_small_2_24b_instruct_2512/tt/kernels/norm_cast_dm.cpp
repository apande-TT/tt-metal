// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Data movement of tt/norm_cast.py (one kernel, two roles). ROLE 0 (reader, RISCV_0): the constant tiles first --
// eps, the reduce scaler, the unit gamma row (the stock reader's tiles, see FILL_* below) -- then this core's x tile
// rows [row0, row0 + nrows), each row's Wt tiles in blocks of BLOCK pages. ROLE 1 (writer, RISCV_1): the bf16 output
// tiles of the same rows, BLOCK pages at a time.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// Fill a tile (4 faces of 16 x 16 datums, DATUM bytes each): ROW0 -- the tile zeroed, row 0 of every face = value
// (the stock reduce scaler); COL0 -- the tile zeroed, column 0 of every face = value (the stock eps tile holds it in
// column 0 of faces 0 / 2; columns 16 + only feed rsqrt lanes nothing reads); TOP_ROW -- row 0 of faces 0 / 1 only, the
// rest left as it is (the only datums a row-broadcast operand reads: the stock gamma tile).
enum Fill : uint32_t { ROW0 = 0, COL0 = 1, TOP_ROW = 2 };

template <uint32_t DATUM>
inline void fill_tile(uint32_t l1, uint32_t value, Fill how) {
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    const uint32_t word = DATUM == 4 ? value : (value & 0xFFFF) | (value << 16);
    constexpr uint32_t FACE_WORDS = 256 * DATUM / 4, ROW_WORDS = 16 * DATUM / 4;
    if (how != TOP_ROW) {
        for (uint32_t i = 0; i < 4 * FACE_WORDS; ++i) {
            w[i] = 0;
        }
    }
    for (uint32_t face = 0; face < 4; ++face) {
        if (how == COL0) {
            for (uint32_t r = 0; r < 16; ++r) {
                if constexpr (DATUM == 4) {
                    w[face * FACE_WORDS + r * ROW_WORDS] = value;
                } else {
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1)[(face * 16 + r) * 16] = value & 0xFFFF;
                }
            }
        } else if (how == ROW0 || face < 2) {
            for (uint32_t i = 0; i < ROW_WORDS; ++i) {
                w[face * FACE_WORDS + i] = word;
            }
        }
    }
}

void kernel_main() {
    constexpr uint32_t ROLE = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t BLOCK = get_compile_time_arg_val(2);
    constexpr uint32_t CB = get_compile_time_arg_val(3);  // cb_in (reader) / cb_out (writer)
    constexpr uint32_t PAGE = get_compile_time_arg_val(4);
    constexpr auto t_args = TensorAccessorArgs<5>();
    const auto t = TensorAccessor(t_args, get_arg_val<uint32_t>(0));
    const uint32_t row0 = get_arg_val<uint32_t>(1);
    const uint32_t nrows = get_arg_val<uint32_t>(2);

    if constexpr (ROLE == 0) {
        // constant tiles: (cb, datum bytes, value bits, fill) for eps, scaler, gamma
        constexpr uint32_t CB_EPS = get_compile_time_arg_val(t_args.next_compile_time_args_offset() + 0);
        constexpr uint32_t EPS_DATUM = get_compile_time_arg_val(t_args.next_compile_time_args_offset() + 1);
        constexpr uint32_t EPS_FILL = get_compile_time_arg_val(t_args.next_compile_time_args_offset() + 2);
        constexpr uint32_t CB_SCALER = get_compile_time_arg_val(t_args.next_compile_time_args_offset() + 3);
        constexpr uint32_t SC_DATUM = get_compile_time_arg_val(t_args.next_compile_time_args_offset() + 4);
        constexpr uint32_t SC_FILL = get_compile_time_arg_val(t_args.next_compile_time_args_offset() + 5);
        constexpr uint32_t CB_GAMMA = get_compile_time_arg_val(t_args.next_compile_time_args_offset() + 6);
        constexpr uint32_t G_DATUM = get_compile_time_arg_val(t_args.next_compile_time_args_offset() + 7);
        constexpr uint32_t G_FILL = get_compile_time_arg_val(t_args.next_compile_time_args_offset() + 8);
        const uint32_t eps_bits = get_arg_val<uint32_t>(3);
        const uint32_t sc_bits = get_arg_val<uint32_t>(4);
        const uint32_t g_bits = get_arg_val<uint32_t>(5);

        cb_reserve_back(CB_EPS, 1);
        if constexpr (EPS_DATUM == 4) {
            fill_tile<4>(get_write_ptr(CB_EPS), eps_bits, static_cast<Fill>(EPS_FILL));
        } else {
            fill_tile<2>(get_write_ptr(CB_EPS), eps_bits, static_cast<Fill>(EPS_FILL));
        }
        cb_push_back(CB_EPS, 1);
        cb_reserve_back(CB_SCALER, 1);
        if constexpr (SC_DATUM == 4) {
            fill_tile<4>(get_write_ptr(CB_SCALER), sc_bits, static_cast<Fill>(SC_FILL));
        } else {
            fill_tile<2>(get_write_ptr(CB_SCALER), sc_bits, static_cast<Fill>(SC_FILL));
        }
        cb_push_back(CB_SCALER, 1);
        cb_reserve_back(CB_GAMMA, Wt);
        for (uint32_t i = 0; i < Wt; ++i) {
            const uint32_t l1 = get_write_ptr(CB_GAMMA) + i * 1024 * G_DATUM;
            if constexpr (G_DATUM == 4) {
                fill_tile<4>(l1, g_bits, static_cast<Fill>(G_FILL));
            } else {
                fill_tile<2>(l1, g_bits, static_cast<Fill>(G_FILL));
            }
        }
        cb_push_back(CB_GAMMA, Wt);

        for (uint32_t r = row0; r < row0 + nrows; ++r) {
            for (uint32_t k = 0; k < Wt; k += BLOCK) {
                const uint32_t n = Wt - k < BLOCK ? Wt - k : BLOCK;
                cb_reserve_back(CB, BLOCK);
                const uint32_t wp = get_write_ptr(CB);
                for (uint32_t j = 0; j < n; ++j) {
                    noc_async_read_page(r * Wt + k + j, t, wp + j * PAGE);
                }
                noc_async_read_barrier();
                cb_push_back(CB, BLOCK);
            }
        }
    } else {
        for (uint32_t r = row0; r < row0 + nrows; ++r) {
            for (uint32_t k = 0; k < Wt; k += BLOCK) {
                const uint32_t n = Wt - k < BLOCK ? Wt - k : BLOCK;
                cb_wait_front(CB, BLOCK);
                const uint32_t rp = get_read_ptr(CB);
                for (uint32_t j = 0; j < n; ++j) {
                    noc_async_write_page(r * Wt + k + j, t, rp + j * PAGE);
                }
                noc_async_writes_flushed();
                cb_pop_front(CB, BLOCK);
            }
        }
        noc_async_write_barrier();
    }
}

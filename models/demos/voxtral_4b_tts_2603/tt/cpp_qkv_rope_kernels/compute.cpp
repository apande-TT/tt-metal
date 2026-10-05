// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The compute kernel of tt/cpp_qkv_rope.py: the stock RoPE kernel's arithmetic (rotary_embedding.cpp, the
// multi-tile prefill path) op for op -- per column tile j of a head tile row:
//   j <  WT/2:  rot_i = rot[j] * (-1)       (FPU mul, scalar broadcast)    sin_i = rot_i * sin[j]
//   j >= WT/2:                                                             sin_i = rot[j] * sin[j]
//   cos_i = in[j] * cos[j];  out = cos_i + sin_i                           (FPU mul / add)
// at HiFi4 with a 16-bit DEST, every intermediate packed bf16 as the stock CBs hold it -- but each op runs over
// the row's WT tiles under ONE init instead of re-initialising per tile.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/pack.h"

constexpr uint32_t WT = get_compile_time_arg_val(0);
constexpr uint32_t HALF = WT / 2;
constexpr uint32_t HEADS = get_compile_time_arg_val(1);  // q + k heads a tile row (units of one row)

constexpr auto cb_in = tt::CBIndex::c_0;
constexpr auto cb_rot = tt::CBIndex::c_1;
constexpr auto cb_cos = tt::CBIndex::c_2;
constexpr auto cb_sin = tt::CBIndex::c_3;
constexpr auto cb_scalar = tt::CBIndex::c_4;
constexpr auto cb_out = tt::CBIndex::c_16;
constexpr auto cb_roti = tt::CBIndex::c_24;
constexpr auto cb_cosi = tt::CBIndex::c_25;
constexpr auto cb_sini = tt::CBIndex::c_26;

template <uint32_t out_cb>
inline void pack_one() {
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out_cb);
    tile_regs_release();
}

void kernel_main() {
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t units = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_rot, cb_scalar, cb_roti);
    cb_wait_front(cb_scalar, 1);
    uint32_t row = 0xFFFFFFFF;
    for (uint32_t u = u0; u < u0 + units; ++u) {
        if (u / HEADS != row) {
            // the reader sends cos / sin once per tile row
            if (row != 0xFFFFFFFF) {
                cb_pop_front(cb_sin, WT);
                cb_pop_front(cb_cos, WT);
            }
            row = u / HEADS;
            cb_wait_front(cb_sin, WT);
            cb_wait_front(cb_cos, WT);
        }
        cb_wait_front(cb_rot, WT);
        cb_wait_front(cb_in, WT);

        // rot_i = rot * -1 on the first half
        cb_reserve_back(cb_roti, HALF);
        mul_bcast_scalar_init(cb_rot, cb_scalar);
        for (uint32_t j = 0; j < HALF; ++j) {
            tile_regs_acquire();
            mul_tiles_bcast_scalar(cb_rot, cb_scalar, j, 0, 0);
            pack_one<cb_roti>();
        }
        cb_push_back(cb_roti, HALF);

        // sin_i = rot_i * sin (first half), rot * sin (second half)
        cb_wait_front(cb_roti, HALF);
        cb_reserve_back(cb_sini, WT);
        mul_init(cb_roti, cb_sin, false);
        for (uint32_t j = 0; j < HALF; ++j) {
            tile_regs_acquire();
            mul_tiles(cb_roti, cb_sin, j, j, 0);
            pack_one<cb_sini>();
        }
        mul_init(cb_rot, cb_sin, false);
        for (uint32_t j = HALF; j < WT; ++j) {
            tile_regs_acquire();
            mul_tiles(cb_rot, cb_sin, j, j, 0);
            pack_one<cb_sini>();
        }
        cb_push_back(cb_sini, WT);
        cb_pop_front(cb_roti, HALF);

        // cos_i = in * cos
        cb_reserve_back(cb_cosi, WT);
        mul_init(cb_in, cb_cos, false);
        for (uint32_t j = 0; j < WT; ++j) {
            tile_regs_acquire();
            mul_tiles(cb_in, cb_cos, j, j, 0);
            pack_one<cb_cosi>();
        }
        cb_push_back(cb_cosi, WT);

        // out = cos_i + sin_i
        cb_wait_front(cb_cosi, WT);
        cb_wait_front(cb_sini, WT);
        cb_reserve_back(cb_out, WT);
        add_init(cb_cosi, cb_sini, false);
        for (uint32_t j = 0; j < WT; ++j) {
            tile_regs_acquire();
            add_tiles(cb_cosi, cb_sini, j, j, 0);
            pack_one<cb_out>();
        }
        cb_push_back(cb_out, WT);
        cb_pop_front(cb_cosi, WT);
        cb_pop_front(cb_sini, WT);

        cb_pop_front(cb_rot, WT);
        cb_pop_front(cb_in, WT);
    }
    if (row != 0xFFFFFFFF) {
        cb_pop_front(cb_sin, WT);
        cb_pop_front(cb_cos, WT);
    }
}

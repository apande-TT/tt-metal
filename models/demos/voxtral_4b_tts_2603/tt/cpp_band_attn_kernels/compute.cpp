// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The compute kernel of the C++ banded codec attention (see tt/cpp_band_attn.py). Per unit (a query
// tile row and its two-tile key band) it replays the stock chain's own primitives in the stock order:
//   S = q @ k^T              reuse bmm, one K block   matmul_tiles (k transposed by the unpacker), HiFi2
//   S' = S * scale           binary_ng MUL (scalar)   mul_binary_tile against a scale-filled tile
//   s = S' + mask            tt/cpp_softmax           add_binary_tile<NearestEven>
//   m, e = exp(s - m), z, e / z                       binary_max fold + sfpu_reduce, exp_tile, add fold + sfpu_reduce, div
//   out = P @ v              reuse bmm, one K block   matmul_tiles, the DT context tiles held in DEST
// Every tile outside the band is fully masked in the stock chain: its weights are exact zeros, so
// it adds nothing to the max, the sums or the context -- skipping it changes no value.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"

constexpr uint32_t DT = get_compile_time_arg_val(0);
constexpr uint32_t NB = 2;  // key tiles in the band
constexpr uint32_t OUT_NARROW = get_compile_time_arg_val(1);  // 1: the context CB is narrower than float32

constexpr auto cb_q = tt::CBIndex::c_0;
constexpr auto cb_k = tt::CBIndex::c_1;
constexpr auto cb_v = tt::CBIndex::c_2;
constexpr auto cb_raw = tt::CBIndex::c_3;
constexpr auto cb_scale = tt::CBIndex::c_4;
constexpr auto cb_mask = tt::CBIndex::c_5;
constexpr auto cb_s = tt::CBIndex::c_6;
constexpr auto cb_max = tt::CBIndex::c_7;
constexpr auto cb_maxf = tt::CBIndex::c_8;
constexpr auto cb_e = tt::CBIndex::c_9;
constexpr auto cb_sum = tt::CBIndex::c_10;
constexpr auto cb_sumf = tt::CBIndex::c_11;
constexpr auto cb_p = tt::CBIndex::c_12;
constexpr auto cb_out = tt::CBIndex::c_16;

template <bool IS_MAX>
inline void reduce_row(uint32_t cb_in, uint32_t cb_out_) {
    cb_reserve_back(cb_out_, 1);
    tile_regs_acquire();
    copy_init(cb_in);
    if constexpr (IS_MAX) {
        binary_max_tile_init();
    } else {
        add_binary_tile_init();
    }
    copy_tile(cb_in, 0, 0);
    for (uint32_t w = 1; w < NB; ++w) {
        copy_tile(cb_in, w, 1);
        if constexpr (IS_MAX) {
            binary_max_tile(0, 1, 0);
        } else {
            add_binary_tile(0, 1, 0);
        }
    }
    if constexpr (IS_MAX) {
        sfpu_reduce_init<ckernel::PoolType::MAX, DataFormat::Float32>();
        sfpu_reduce<ckernel::PoolType::MAX, DataFormat::Float32, ckernel::ReduceDim::REDUCE_ROW>(0, 1, 1);
    } else {
        sfpu_reduce_init<ckernel::PoolType::SUM, DataFormat::Float32>();
        sfpu_reduce<ckernel::PoolType::SUM, DataFormat::Float32, ckernel::ReduceDim::REDUCE_ROW>(0, 1, 1);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out_);
    tile_regs_release();
    cb_push_back(cb_out_, 1);
}

void kernel_main() {
    const uint32_t nu = get_arg_val<uint32_t>(0);

    // The packer starts on a float32 CB (every intermediate is float32); a narrower context CB is switched to
    // around its own pack only.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q, cb_k, OUT_NARROW ? cb_raw : cb_out);
    cb_wait_front(cb_scale, 1);

    for (uint32_t u = 0; u < nu; ++u) {
        // S = q @ k^T over the band, one tile at a time, summed over head_dim in fp32 DEST.
        cb_wait_front(cb_q, DT);
        cb_wait_front(cb_k, NB * DT);
        matmul_init(cb_q, cb_k, 1);
        for (uint32_t j = 0; j < NB; ++j) {
            tile_regs_acquire();
            for (uint32_t d = 0; d < DT; ++d) {
                matmul_tiles(cb_q, cb_k, d, j * DT + d, 0);
            }
            tile_regs_commit();
            cb_reserve_back(cb_raw, 1);
            tile_regs_wait();
            pack_tile(0, cb_raw);
            tile_regs_release();
            cb_push_back(cb_raw, 1);
        }
        cb_pop_front(cb_q, DT);
        cb_pop_front(cb_k, NB * DT);

        // s = S * scale + mask
        cb_wait_front(cb_raw, NB);
        cb_wait_front(cb_mask, NB);
        for (uint32_t j = 0; j < NB; ++j) {
            cb_reserve_back(cb_s, 1);
            tile_regs_acquire();
            copy_init(cb_raw);
            copy_tile(cb_raw, j, 0);
            copy_init(cb_scale);
            copy_tile(cb_scale, 0, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            copy_init(cb_mask);
            copy_tile(cb_mask, j, 1);
            add_binary_tile_init();
            add_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_s);
            tile_regs_release();
            cb_push_back(cb_s, 1);
        }
        cb_pop_front(cb_raw, NB);
        cb_pop_front(cb_mask, NB);

        // m = row max
        cb_wait_front(cb_s, NB);
        reduce_row<true>(cb_s, cb_max);

        // e = exp(s - m)
        cb_wait_front(cb_maxf, 1);
        for (uint32_t j = 0; j < NB; ++j) {
            cb_reserve_back(cb_e, 1);
            tile_regs_acquire();
            copy_init(cb_s);
            copy_tile(cb_s, j, 0);
            copy_init(cb_maxf);
            copy_tile(cb_maxf, 0, 1);
            sub_binary_tile_init();
            sub_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
            exp_tile_init();
            exp_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_e);
            tile_regs_release();
            cb_push_back(cb_e, 1);
        }
        cb_pop_front(cb_s, NB);
        cb_pop_front(cb_maxf, 1);

        // z = row sum
        cb_wait_front(cb_e, NB);
        reduce_row<false>(cb_e, cb_sum);

        // P = e / z
        cb_wait_front(cb_sumf, 1);
        for (uint32_t j = 0; j < NB; ++j) {
            cb_reserve_back(cb_p, 1);
            tile_regs_acquire();
            copy_init(cb_e);
            copy_tile(cb_e, j, 0);
            copy_init(cb_sumf);
            copy_tile(cb_sumf, 0, 1);
            div_binary_tile_init();
            div_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_p);
            tile_regs_release();
            cb_push_back(cb_p, 1);
        }
        cb_pop_front(cb_e, NB);
        cb_pop_front(cb_sumf, 1);

        // out = P @ v: the DT context tiles in DEST, summed over the band in key order.
        cb_wait_front(cb_p, NB);
        cb_wait_front(cb_v, NB * DT);
        matmul_init(cb_p, cb_v, 0);
        tile_regs_acquire();
        for (uint32_t j = 0; j < NB; ++j) {
            for (uint32_t d = 0; d < DT; ++d) {
                matmul_tiles(cb_p, cb_v, j, j * DT + d, d);
            }
        }
        tile_regs_commit();
        cb_pop_front(cb_p, NB);
        cb_pop_front(cb_v, NB * DT);
        cb_reserve_back(cb_out, DT);
        tile_regs_wait();
        if constexpr (OUT_NARROW) {
            pack_reconfig_data_format(cb_raw, cb_out);
        }
        for (uint32_t d = 0; d < DT; ++d) {
            pack_tile(d, cb_out);
        }
        if constexpr (OUT_NARROW) {
            pack_reconfig_data_format(cb_out, cb_raw);
        }
        tile_regs_release();
        cb_push_back(cb_out, DT);
    }
    cb_pop_front(cb_scale, 1);
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sampler noise lookup of tt/pipeline.py NoiseLookup, one RISC per R rows of the (32, 32) candidate tile:
// out[b, j] = noise[step, b, ids[b, j] % NB] as fp32. Reads the decode step counter, this RISC's R rows of that
// step's row-major bf16 noise row (B * NB) and of the uint32 id tile (R rows of a 16-row face pair are two
// contiguous runs, one per column face), looks every candidate's bucket up with integer ops (bf16 -> fp32 is the
// bit pattern << 16, exact) and writes the same runs of the fp32 output tile.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t step_addr = get_arg_val<uint32_t>(0);
    const uint32_t noise_addr = get_arg_val<uint32_t>(1);
    const uint32_t ids_addr = get_arg_val<uint32_t>(2);
    const uint32_t out_addr = get_arg_val<uint32_t>(3);
    const uint32_t r0 = get_arg_val<uint32_t>(4);  // first row (R rows, inside one 16-row face pair)
    const uint32_t slot = get_arg_val<uint32_t>(5);  // this RISC's scratch slot on the core (0 or 1)

    constexpr uint32_t NB = get_compile_time_arg_val(0);  // noise buckets per row (power of two)
    constexpr uint32_t R = get_compile_time_arg_val(1);   // rows per RISC (divides 16)
    constexpr auto a_step = TensorAccessorArgs<2>();
    constexpr auto a_noise = TensorAccessorArgs<a_step.next_compile_time_args_offset()>();
    constexpr auto a_ids = TensorAccessorArgs<a_noise.next_compile_time_args_offset()>();
    constexpr auto a_out = TensorAccessorArgs<a_ids.next_compile_time_args_offset()>();
    const auto s_step = TensorAccessor(a_step, step_addr);
    const auto s_noise = TensorAccessor(a_noise, noise_addr);
    const auto s_ids = TensorAccessor(a_ids, ids_addr);
    const auto s_out = TensorAccessor(a_out, out_addr);

    constexpr uint32_t cb_step = 0;   // 64 B per slot
    constexpr uint32_t cb_noise = 1;  // R rows x NB bf16 per slot
    constexpr uint32_t cb_ids = 2;    // 2 runs of R x 16 uint32 per slot
    constexpr uint32_t cb_out = 3;    // 2 runs of R x 16 fp32 per slot
    constexpr uint32_t ROWS_BYTES = R * NB * 2;
    constexpr uint32_t RUN = R * 16;  // elements in one face's run
    constexpr uint32_t RUN_BYTES = RUN * 4;

    const uint32_t l_step = get_write_ptr(cb_step) + slot * 64;
    const uint32_t l_noise = get_write_ptr(cb_noise) + slot * ROWS_BYTES;
    const uint32_t l_ids = get_write_ptr(cb_ids) + slot * 2 * RUN_BYTES;
    const uint32_t l_out = get_write_ptr(cb_out) + slot * 2 * RUN_BYTES;
    // byte offset in the tile of face (2 * (r0 / 16) + f), row r0 % 16
    const uint32_t run0 = ((r0 >> 4) * 2 * 256 + (r0 & 15) * 16) * 4;
    const uint32_t run1 = run0 + 256 * 4;

    noc_async_read(s_step.get_noc_addr(0), l_step, 64);
    noc_async_read(s_ids.get_noc_addr(0, run0), l_ids, RUN_BYTES);
    noc_async_read(s_ids.get_noc_addr(0, run1), l_ids + RUN_BYTES, RUN_BYTES);
    noc_async_read_barrier();
    const uint32_t step = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_step);
    noc_async_read(s_noise.get_noc_addr(step, r0 * NB * 2), l_noise, ROWS_BYTES);
    noc_async_read_barrier();

    const volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_ids);
    const volatile tt_l1_ptr uint16_t* noise = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l_noise);
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_out);
    // element e of the two runs: run e / RUN (column face), local row (e % RUN) / 16
    for (uint32_t e = 0; e < 2 * RUN; ++e) {
        const uint32_t r = (e % RUN) >> 4;
        out[e] = static_cast<uint32_t>(noise[r * NB + (ids[e] & (NB - 1))]) << 16;
    }
    noc_async_write(l_out, s_out.get_noc_addr(0, run0), RUN_BYTES);
    noc_async_write(l_out + RUN_BYTES, s_out.get_noc_addr(0, run1), RUN_BYTES);
    noc_async_write_barrier();
}

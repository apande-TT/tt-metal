// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// MoE router after the sigmoid, one core per user row b (the HF NemotronH gate with n_group = topk_group = 1):
//   the K experts with the largest choice = sigmoid(logits) + e_score_correction_bias (ties: the lower expert id),
//   their weights scores[e], divided by (their sum + 1e-20) when NORM, times SCALE.
// Writes this chip's ELOC columns, experts [d ELOC, (d + 1) ELOC), d read from the chip-index tensor: the weight of
// a selected expert, 0 for the rest, to row b of the (1, 32, ELOC) fp32 output.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
// fp32 bits -> unsigned key in float order (-0 taken as +0).
inline uint32_t order_key(uint32_t bits) {
    bits = bits == 0x80000000u ? 0u : bits;
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

inline float as_float(uint32_t bits) {
    union {
        uint32_t u;
        float f;
    } v{bits};
    return v.f;
}

inline uint32_t as_bits(float f) {
    union {
        float f;
        uint32_t u;
    } v{f};
    return v.u;
}
}  // namespace

void kernel_main() {
    constexpr uint32_t E = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t ELOC = get_compile_time_arg_val(2);
    constexpr uint32_t NORM = get_compile_time_arg_val(3);
    constexpr uint32_t SCALE_BITS = get_compile_time_arg_val(4);
    constexpr auto c_args = TensorAccessorArgs<5>();
    constexpr auto s_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr auto d_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<d_args.next_compile_time_args_offset()>();
    const auto choice = TensorAccessor(c_args, get_arg_val<uint32_t>(0));
    const auto scores = TensorAccessor(s_args, get_arg_val<uint32_t>(1));
    const auto chip = TensorAccessor(d_args, get_arg_val<uint32_t>(2));
    const auto out = TensorAccessor(o_args, get_arg_val<uint32_t>(3));
    const uint32_t b = get_arg_val<uint32_t>(4);
    static_assert(E % 32 == 0 && ELOC == 32 && K <= 32, "one output tile column; whole input tiles");

    constexpr uint32_t ET = E / 32;
    constexpr uint32_t face = 1024, frow = 64;  // 32-bit tile geometry
    constexpr uint32_t cb_row = 0;
    cb_reserve_back(cb_row, 1);
    const uint32_t c_l1 = (get_write_ptr(cb_row) + 63) & ~63u;  // DRAM reads need 64 B-aligned landing addresses
    const uint32_t s_l1 = c_l1 + E * 4, d_l1 = s_l1 + E * 4, o_l1 = d_l1 + frow;

    const uint32_t off = (b / 16) * 2 * face + (b % 16) * frow;
    for (uint32_t t = 0; t < ET; ++t) {
        noc_async_read(choice.get_noc_addr(t, off), c_l1 + t * 2 * frow, frow);
        noc_async_read(choice.get_noc_addr(t, off + face), c_l1 + t * 2 * frow + frow, frow);
        noc_async_read(scores.get_noc_addr(t, off), s_l1 + t * 2 * frow, frow);
        noc_async_read(scores.get_noc_addr(t, off + face), s_l1 + t * 2 * frow + frow, frow);
    }
    noc_async_read(chip.get_noc_addr(0, 0), d_l1, frow);
    noc_async_read_barrier();

    volatile tt_l1_ptr uint32_t* c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(c_l1);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s_l1);
    const uint32_t lo = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(d_l1)[0] * ELOC;

    uint32_t key[E];
    for (uint32_t e = 0; e < E; ++e) {
        key[e] = order_key(c[e]);
    }
    uint32_t top[K];
    float w[K], sum = 0.0f;
    for (uint32_t k = 0; k < K; ++k) {
        uint32_t best = 0;
        for (uint32_t e = 1; e < E; ++e) {
            best = key[e] > key[best] ? e : best;
        }
        top[k] = best;
        key[best] = 0;  // below every real key: order_key sets the top bit of positives, and ~ of a negative's
        w[k] = as_float(s[best]);
        sum += w[k];
    }

    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(o_l1);
    for (uint32_t j = 0; j < ELOC; ++j) {
        o[j] = 0;
    }
    const float scale = as_float(SCALE_BITS);
    const float denom = sum + 1e-20f;
    for (uint32_t k = 0; k < K; ++k) {
        if (top[k] >= lo && top[k] < lo + ELOC) {
            o[top[k] - lo] = as_bits(NORM ? w[k] / denom * scale : w[k] * scale);
        }
    }
    noc_async_write(o_l1, out.get_noc_addr(0, off), frow);
    noc_async_write(o_l1 + frow, out.get_noc_addr(0, off + face), frow);
    noc_async_write_barrier();
}

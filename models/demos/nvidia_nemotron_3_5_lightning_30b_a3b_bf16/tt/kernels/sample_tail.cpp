// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Sampler after the top-k, one core per user row b (golden.host_sample's rule):
//   e[k] = exp(vals[k] - vals[0]) over the K candidates (descending; computed by the caller, fp32);
//   n_keep = #{k : sum_{j<k} e[j] < top_p * sum_j e[j]} (at least 1), kept = the first n_keep plus any candidate
//   equal to the n_keep-th value; the kept candidates in token-id order, and the token is the first whose inclusive
//   sum of e exceeds u * (their total), u = u_table[b, col[b]] (the step's uniform for this user's position).
// The e's are summed as exact integers (value * 2^56, 64-bit; the cores have no FPU, and integer sums carry no
// rounding at all), top_p and u as 32-bit binary fractions, so the only error left is the e's own (fp32 exp).
// Writes the token id (as an exact fp32) to column 0 of row b of the [B, 1] output.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// fp32 bits -> unsigned key in float order (-0 taken as +0).
inline uint32_t order_key(uint32_t bits) {
    bits = bits == 0x80000000u ? 0u : bits;
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Non-negative fp32 bits (value <= 1) -> floor(value * 2^56): exact for values >= 2^-33, the rest is < 2^-56.
inline uint64_t to_q56(uint32_t bits) {
    const uint32_t ex = (bits >> 23) & 0xffu;
    if (ex == 0) {
        return 0;
    }
    const uint64_t m = (bits & 0x7fffffu) | 0x800000u;  // value = m * 2^(ex - 150)
    if (ex >= 94) {
        return m << (ex - 94);
    }
    const uint32_t s = 94 - ex;
    return s >= 24 ? 0 : m >> s;
}

// fp32 bits of a value in [0, 1) -> floor(value * 2^32).
inline uint32_t to_q32(uint32_t bits) {
    const uint32_t ex = (bits >> 23) & 0xffu;
    if (ex == 0) {
        return 0;
    }
    const uint32_t m = (bits & 0x7fffffu) | 0x800000u;
    if (ex >= 118) {
        return m << (ex - 118);
    }
    const uint32_t s = 118 - ex;
    return s >= 24 ? 0 : m >> s;
}

// floor(z * f / 2^32) for z < 2^63, f < 2^32.
inline uint64_t mul_q32(uint64_t z, uint32_t f) {
    const uint32_t hi = static_cast<uint32_t>(z >> 32), lo = static_cast<uint32_t>(z);
    return static_cast<uint64_t>(hi) * f + static_cast<uint32_t>((static_cast<uint64_t>(lo) * f) >> 32);
}

// Non-negative fp32 bits of an exact integer -> the integer.
inline uint32_t f32_to_u32(uint32_t b) {
    const int32_t ex = static_cast<int32_t>((b >> 23) & 0xffu) - 127;
    if (ex < 0) {
        return 0;
    }
    const uint32_t mant = (b & 0x7fffffu) | 0x800000u;
    return ex >= 23 ? mant << (ex - 23) : mant >> (23 - ex);
}

// Integer < 2^24 -> its fp32 bits.
inline uint32_t u32_to_f32(uint32_t v) {
    if (v == 0) {
        return 0;
    }
    const uint32_t msb = 31 - __builtin_clz(v);
    return ((msb + 127) << 23) | ((v << (23 - msb)) & 0x7fffffu);
}

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);         // tiles across the K candidates
    constexpr uint32_t CT = get_compile_time_arg_val(1);         // tiles across a u_table row
    constexpr uint32_t TOP_P_Q32 = get_compile_time_arg_val(2);  // round(top_p * 2^32)
    constexpr auto e_args = TensorAccessorArgs<3>();
    constexpr auto v_args = TensorAccessorArgs<e_args.next_compile_time_args_offset()>();
    constexpr auto i_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto u_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<u_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    const auto ein = TensorAccessor(e_args, get_arg_val<uint32_t>(0));
    const auto vin = TensorAccessor(v_args, get_arg_val<uint32_t>(1));
    const auto iin = TensorAccessor(i_args, get_arg_val<uint32_t>(2));
    const auto utab = TensorAccessor(u_args, get_arg_val<uint32_t>(3));
    const auto cin = TensorAccessor(c_args, get_arg_val<uint32_t>(4));
    const auto out = TensorAccessor(o_args, get_arg_val<uint32_t>(5));
    const uint32_t b = get_arg_val<uint32_t>(6);  // the user row this core owns

    constexpr uint32_t K = KT * 32;
    constexpr uint32_t face = 1024, frow = 64;  // 32-bit tile geometry
    constexpr uint32_t cb_row = 0;

    cb_reserve_back(cb_row, 1);
    const uint32_t e_l1 = (get_write_ptr(cb_row) + 63) & ~63u;  // DRAM reads need 64 B-aligned landing addresses
    const uint32_t v_l1 = e_l1 + K * 4, i_l1 = v_l1 + K * 4, c_l1 = i_l1 + K * 4, u_l1 = c_l1 + frow;
    const uint32_t w_l1 = u_l1 + frow, kid_l1 = w_l1 + frow, eq_l1 = kid_l1 + K * 4, keq_l1 = eq_l1 + K * 8;

    const uint32_t tile_row = b / 32, r = b % 32;
    const uint32_t off = (r / 16) * 2 * face + (r % 16) * frow;
    for (uint32_t t = 0; t < KT; ++t) {
        const uint32_t page = tile_row * KT + t, dst = t * 2 * frow;
        noc_async_read(ein.get_noc_addr(page, off), e_l1 + dst, frow);
        noc_async_read(ein.get_noc_addr(page, off + face), e_l1 + dst + frow, frow);
        noc_async_read(vin.get_noc_addr(page, off), v_l1 + dst, frow);
        noc_async_read(vin.get_noc_addr(page, off + face), v_l1 + dst + frow, frow);
        noc_async_read(iin.get_noc_addr(page, off), i_l1 + dst, frow);
        noc_async_read(iin.get_noc_addr(page, off + face), i_l1 + dst + frow, frow);
    }
    noc_async_read(cin.get_noc_addr(tile_row, off), c_l1, frow);  // column 0 of row b
    noc_async_read_barrier();

    volatile tt_l1_ptr uint32_t* e = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(e_l1);
    volatile tt_l1_ptr uint32_t* v = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v_l1);
    volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(i_l1);
    // Scratch only this core touches: plain accesses.
    uint32_t* kid = reinterpret_cast<uint32_t*>(kid_l1);
    uint64_t* eq = reinterpret_cast<uint64_t*>(eq_l1);
    uint64_t* keq = reinterpret_cast<uint64_t*>(keq_l1);

    // The uniform: u_table[b, pos], pos = col[b] (an exact integer in fp32).
    const uint32_t pos = f32_to_u32(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(c_l1)[0]);
    const uint32_t pc = pos % 32;
    noc_async_read(utab.get_noc_addr(tile_row * CT + pos / 32, off + (pc / 16) * face), u_l1, frow);

    // top_p over the descending candidates, as exact integer sums.
    uint64_t z = 0;
    for (uint32_t k = 0; k < K; ++k) {
        eq[k] = to_q56(e[k]);
        z += eq[k];
    }
    const uint64_t lim = mul_q32(z, TOP_P_Q32);
    uint64_t excl = 0;
    uint32_t n_keep = 0;
    while (n_keep < K && excl < lim) {
        excl += eq[n_keep++];
    }
    if (n_keep == 0) {
        n_keep = 1;
    }
    const uint32_t cut = order_key(v[n_keep - 1]);
    uint32_t m = n_keep;
    while (m < K && order_key(v[m]) >= cut) {  // descending: >= is a tie with the cut
        ++m;
    }

    // The token: the smallest kept id t with sum_{kept id <= t} e > u * (kept total). The sums are exact integers, so
    // no order is needed to form them: quickselect on the id -- split around a pivot id, and the left part's sum says
    // which side holds t (expected O(m), where sorting the kept ids was O(m^2) for flat rows of ~128).
    uint64_t zk = 0;
    for (uint32_t k = 0; k < m; ++k) {
        kid[k] = ids[k];
        keq[k] = eq[k];
        zk += keq[k];
    }
    noc_async_read_barrier();
    uint64_t thr = mul_q32(zk, to_q32(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(u_l1)[pc % 16]));
    uint32_t lo = 0, hi = m, tok = 0;  // the candidates for t are kid[lo .. hi)
    bool found = false;
    while (lo < hi) {
        const uint32_t pivot = kid[lo + (hi - lo) / 2];
        uint64_t q_pivot = 0, s_left = 0;
        uint32_t i = lo;  // kid[lo .. i) < pivot after the pass; kid[i .. hi) the rest
        for (uint32_t j = lo; j < hi; ++j) {
            const uint32_t id = kid[j];
            if (id < pivot) {
                const uint64_t q = keq[j];
                s_left += q;
                kid[j] = kid[i];
                keq[j] = keq[i];
                kid[i] = id;
                keq[i] = q;
                ++i;
            } else if (id == pivot) {
                q_pivot = keq[j];
            }
        }
        if (s_left > thr) {
            hi = i;  // t < pivot
        } else if (s_left + q_pivot > thr) {
            tok = pivot;
            found = true;
            break;
        } else {
            thr -= s_left + q_pivot;  // t > pivot: drop the pivot from kid[i .. hi)
            uint32_t k = i;
            while (kid[k] != pivot) {
                ++k;
            }
            kid[k] = kid[i];
            keq[k] = keq[i];
            lo = i + 1;
        }
    }
    if (!found) {  // only if rounding left u * z at the total: the largest kept id
        tok = 0;
        for (uint32_t k = 0; k < m; ++k) {
            tok = ids[k] > tok ? ids[k] : tok;
        }
    }

    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w_l1);
    w[0] = u32_to_f32(tok);
    w[1] = 0;
    w[2] = 0;
    w[3] = 0;
    noc_async_write(w_l1, out.get_noc_addr(tile_row, off), 16);
    noc_async_write_barrier();
}

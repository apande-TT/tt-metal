// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Weight sender of the two-limb linear (limb_linear.cpp), on one core outside the compute rectangle. For every
// work unit round it streams its column group's slice of the weight (Ntg columns from n0) once, column block (nb tiles) outer and K chunk (kc tiles) inner, each
// chunk read from DRAM into one of the two cb 1 slots (tile kk * nb + c = w[k0 + kk, n0 + c]) and multicast to
// the same L1 address on every compute core once all of them have freed that slot. The sender's own cb 1 is
// only a staging ring (no consumer here): it alternates the two slots at the receivers' cb 1 addresses.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_start = get_arg_val<uint32_t>(1);
    const uint32_t y_start = get_arg_val<uint32_t>(2);
    const uint32_t x_end = get_arg_val<uint32_t>(3);
    const uint32_t y_end = get_arg_val<uint32_t>(4);
    const uint32_t n0 = get_arg_val<uint32_t>(5);  // first output column tile of this sender's column group

    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);
    constexpr uint32_t Ntg = get_compile_time_arg_val(2);  // column tiles of this column group
    constexpr uint32_t nb = get_compile_time_arg_val(3);
    constexpr uint32_t kc = get_compile_time_arg_val(4);
    constexpr uint32_t rounds = get_compile_time_arg_val(5);
    constexpr uint32_t num_dests = get_compile_time_arg_val(6);
    constexpr auto w_args = TensorAccessorArgs<7>();
    constexpr uint32_t sender_sem_id = get_compile_time_arg_val(w_args.next_compile_time_args_offset());
    constexpr uint32_t receiver_sem_id = get_compile_time_arg_val(w_args.next_compile_time_args_offset() + 1);

    constexpr uint32_t cb_w = 1;
    const uint32_t page = get_local_cb_interface(cb_w).fifo_page_size;
    constexpr uint32_t chunk_tiles = kc * nb;
    const uint32_t chunk_bytes = chunk_tiles * page;
    const auto w = TensorAccessor(w_args, w_addr);

    Noc noc;
    DataflowBuffer dw(cb_w);
    const uint32_t base = dw.get_write_ptr();
    Semaphore<> sender_sem(sender_sem_id);
    Semaphore<> receiver_sem(receiver_sem_id);
    receiver_sem.set(VALID);  // the local value multicast to the receivers once a chunk has landed

    uint32_t slot = 0;
    for (uint32_t round = 0; round < rounds; ++round) {
        for (uint32_t nblk = 0; nblk < Ntg / nb; ++nblk) {
            for (uint32_t kch = 0; kch < Kt / kc; ++kch) {
                noc.async_write_barrier();  // the multicast that last used this slot has left L1
                const uint32_t off = slot * chunk_bytes;
                for (uint32_t kk = 0; kk < kc; ++kk) {
                    for (uint32_t c = 0; c < nb; ++c) {
                        const uint32_t id = (kch * kc + kk) * Nt + n0 + nblk * nb + c;
                        noc.async_read(w, dw, page, {.page_id = id}, {.offset_bytes = off + (kk * nb + c) * page});
                    }
                }
                noc.async_read_barrier();

                sender_sem.wait(num_dests);  // every receiver has freed this slot
                sender_sem.set(0);
                const uint32_t addr = base + off;
                const MulticastEndpoint mcast_dst;
                noc.async_write_multicast(
                    CoreLocalMem<uint32_t>(addr),
                    mcast_dst,
                    chunk_bytes,
                    num_dests,
                    {},
                    {.noc_x_start = x_start, .noc_y_start = y_start, .noc_x_end = x_end, .noc_y_end = y_end, .addr = addr},
                    true);
#ifdef ARCH_BLACKHOLE
                noc.async_writes_flushed();
#endif
                receiver_sem.set_multicast(noc, x_start, y_start, x_end, y_end, num_dests);
                slot ^= 1;
            }
        }
    }
    noc.async_write_barrier();
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// One fabric-link worker of tt/gate_up_decode.py's fused o_proj all_reduce (RISCV_0, NOC 0): the same exchange
// ttnn.experimental.all_reduce_async runs (its worker_writer.cpp), inside the gate / up program so that it overlaps
// the weight streams. The worker owns N_RES residual shards (each SHARD_BYTES = 4 bf16 tiles of this device's o_proj
// partials, at P_ADDR on its residual core): it reads them all, then, per shard, writes it into slot MY_CHIP of that
// residual core's scratch (SCRATCH_ADDR) on this device and -- one fused write + atomic increment packet per direction,
// line-multicast over the ring -- on every other device, raising the paired norm core's global semaphore (SEM_ADDR)
// there; the local write is counted on the local norm core once it has landed. Each norm core waits for TP counts.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"

constexpr uint32_t MY_CHIP = get_compile_time_arg_val(0);
constexpr uint32_t CB_HDR = get_compile_time_arg_val(1);
constexpr uint32_t CB_BUF = get_compile_time_arg_val(2);
constexpr uint32_t SHARD_BYTES = get_compile_time_arg_val(3);
constexpr uint32_t N_RES = get_compile_time_arg_val(4);
constexpr ccl_routing_utils::line_multicast_route_info_t fwd_route =
    ccl_routing_utils::get_line_multicast_route_info_from_args<5>();
constexpr ccl_routing_utils::line_multicast_route_info_t bwd_route =
    ccl_routing_utils::get_line_multicast_route_info_from_args<5 + ccl_routing_utils::num_line_multicast_args>();

void kernel_main() {
    size_t a = 0;
    const uint32_t p_addr = get_arg_val<uint32_t>(a++);
    const uint32_t scratch_addr = get_arg_val<uint32_t>(a++);
    const uint32_t sem_addr = get_arg_val<uint32_t>(a++);
    const size_t xy0 = a;  // N_RES x (res x, res y, norm x, norm y), NOC 0 coordinates
    a += 4 * N_RES;
    auto fabric = FabricConnectionManager::build_from_args<FabricConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(a);

    cb_reserve_back(CB_HDR, 2);
    volatile PACKET_HEADER_TYPE* hdr_fwd = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(get_write_ptr(CB_HDR));
    volatile PACKET_HEADER_TYPE* hdr_bwd =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(get_write_ptr(CB_HDR) + sizeof(PACKET_HEADER_TYPE));
    ccl_routing_utils::fabric_set_line_multicast_route(hdr_fwd, fwd_route);
    ccl_routing_utils::fabric_set_line_multicast_route(hdr_bwd, bwd_route);

    // this device's partial shards (N_RES x SHARD_BYTES), read in batches of BATCH: the next batch's reads are in
    // flight while the current one is sent
    constexpr uint32_t BATCH = 4;
    cb_reserve_back(CB_BUF, N_RES);
    const uint32_t buf = get_write_ptr(CB_BUF);
    auto read_batch = [&](uint32_t b0) {
        for (uint32_t i = b0; i < b0 + BATCH && i < N_RES; ++i) {
            const uint32_t rx = get_arg_val<uint32_t>(xy0 + 4 * i), ry = get_arg_val<uint32_t>(xy0 + 4 * i + 1);
            noc_async_read(get_noc_addr(rx, ry, p_addr), buf + i * SHARD_BYTES, SHARD_BYTES);
        }
    };
    read_batch(0);
    noc_async_read_barrier();

    if (fabric.is_logically_connected()) {
        fabric.open_finish();
    }
    size_t l1 = buf;
    for (uint32_t b0 = 0; b0 < N_RES; b0 += BATCH) {
        if (b0 + BATCH < N_RES) {
            read_batch(b0 + BATCH);
        }
        for (uint32_t i = b0; i < b0 + BATCH && i < N_RES; ++i) {
            const uint32_t rx = get_arg_val<uint32_t>(xy0 + 4 * i), ry = get_arg_val<uint32_t>(xy0 + 4 * i + 1);
            const uint32_t nx = get_arg_val<uint32_t>(xy0 + 4 * i + 2), ny = get_arg_val<uint32_t>(xy0 + 4 * i + 3);
            const uint64_t dst = safe_get_noc_addr(rx, ry, scratch_addr + MY_CHIP * SHARD_BYTES);
            const uint64_t sem = safe_get_noc_addr(nx, ny, sem_addr);
            fused_write_atomic_and_advance_local_read_address_for_fabric_write(
                dst, hdr_fwd, hdr_bwd, fabric, l1, SHARD_BYTES, sem, 1, true);
            noc_async_writes_flushed();
        }
        noc_async_read_barrier();
    }
    noc_async_write_barrier();  // the local copies have landed: count them on the local norm cores
    for (uint32_t i = 0; i < N_RES; ++i) {
        const uint32_t nx = get_arg_val<uint32_t>(xy0 + 4 * i + 2), ny = get_arg_val<uint32_t>(xy0 + 4 * i + 3);
        noc_semaphore_inc(safe_get_noc_addr(nx, ny, sem_addr), 1);
    }
    if (fabric.is_logically_connected()) {
        fabric.close_start();
        fabric.close_finish();
    }
    noc_async_atomic_barrier();
    noc_async_write_barrier();
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// One fabric-link worker of tt/down_decode.py's all_reduce fold (StreamDownDecode.ar_call; RISCV_0, NOC 0): ar_send.cpp's
// exchange (all_reduce_async's worker_writer pattern), but sent in two waves while the down projection still streams.
// The worker owns N1 + N2 residual shards (SHARD_BYTES = WT bf16 tiles of this device's down partials each, at P_ADDR
// on its residual core). The down cores count every partial tile they write on this worker's wave semaphore (SEM_W1 for
// the first N1 shards, SEM_W2 for the rest); once all of a wave's tiles are counted, the worker reads its shards and,
// per shard, writes it into slot MY_CHIP of that residual core's scratch on this device and -- one fused write +
// atomic increment packet per direction, line-multicast over the ring -- on every other device, raising the residual
// core's global semaphore there; the local writes are counted once they have landed. Each residual core waits for TP.
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
constexpr uint32_t N1 = get_compile_time_arg_val(4);  // wave-1 shards
constexpr uint32_t N2 = get_compile_time_arg_val(5);  // wave-2 shards
constexpr uint32_t SEM_W1 = get_compile_time_arg_val(6);
constexpr uint32_t SEM_W2 = get_compile_time_arg_val(7);
constexpr uint32_t WT = get_compile_time_arg_val(8);  // tiles a shard
constexpr ccl_routing_utils::line_multicast_route_info_t fwd_route =
    ccl_routing_utils::get_line_multicast_route_info_from_args<9>();
constexpr ccl_routing_utils::line_multicast_route_info_t bwd_route =
    ccl_routing_utils::get_line_multicast_route_info_from_args<9 + ccl_routing_utils::num_line_multicast_args>();

void kernel_main() {
    size_t a = 0;
    const uint32_t p_addr = get_arg_val<uint32_t>(a++);
    const uint32_t scratch_addr = get_arg_val<uint32_t>(a++);
    const uint32_t sem_addr = get_arg_val<uint32_t>(a++);
    const size_t xy0 = a;  // (N1 + N2) x (res x, res y), NOC 0 coordinates
    a += 2 * (N1 + N2);
    auto fabric = FabricConnectionManager::build_from_args<FabricConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(a);

    cb_reserve_back(CB_HDR, 2);
    volatile PACKET_HEADER_TYPE* hdr_fwd = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(get_write_ptr(CB_HDR));
    volatile PACKET_HEADER_TYPE* hdr_bwd =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(get_write_ptr(CB_HDR) + sizeof(PACKET_HEADER_TYPE));
    ccl_routing_utils::fabric_set_line_multicast_route(hdr_fwd, fwd_route);
    ccl_routing_utils::fabric_set_line_multicast_route(hdr_bwd, bwd_route);
    if (fabric.is_logically_connected()) {
        fabric.open_finish();
    }

    cb_reserve_back(CB_BUF, N1 + N2);
    const uint32_t buf = get_write_ptr(CB_BUF);
    for (uint32_t wave = 0; wave < 2; ++wave) {
        const uint32_t s0 = wave == 0 ? 0 : N1, s1 = wave == 0 ? N1 : N1 + N2;
        if (s1 == s0) {
            continue;
        }
        noc_semaphore_wait(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(wave == 0 ? SEM_W1 : SEM_W2)), (s1 - s0) * WT);
        for (uint32_t i = s0; i < s1; ++i) {
            const uint32_t rx = get_arg_val<uint32_t>(xy0 + 2 * i), ry = get_arg_val<uint32_t>(xy0 + 2 * i + 1);
            noc_async_read(get_noc_addr(rx, ry, p_addr), buf + i * SHARD_BYTES, SHARD_BYTES);
        }
        noc_async_read_barrier();
        size_t l1 = buf + s0 * SHARD_BYTES;
        for (uint32_t i = s0; i < s1; ++i) {
            const uint32_t rx = get_arg_val<uint32_t>(xy0 + 2 * i), ry = get_arg_val<uint32_t>(xy0 + 2 * i + 1);
            const uint64_t dst = safe_get_noc_addr(rx, ry, scratch_addr + MY_CHIP * SHARD_BYTES);
            const uint64_t sem = safe_get_noc_addr(rx, ry, sem_addr);
            fused_write_atomic_and_advance_local_read_address_for_fabric_write(
                dst, hdr_fwd, hdr_bwd, fabric, l1, SHARD_BYTES, sem, 1, true);
            noc_async_writes_flushed();
        }
        noc_async_write_barrier();  // this wave's local copies have landed: count them on their residual cores
        for (uint32_t i = s0; i < s1; ++i) {
            const uint32_t rx = get_arg_val<uint32_t>(xy0 + 2 * i), ry = get_arg_val<uint32_t>(xy0 + 2 * i + 1);
            noc_semaphore_inc(safe_get_noc_addr(rx, ry, sem_addr), 1);
        }
    }
    if (fabric.is_logically_connected()) {
        fabric.close_start();
        fabric.close_finish();
    }
    noc_async_atomic_barrier();
    noc_async_write_barrier();
}

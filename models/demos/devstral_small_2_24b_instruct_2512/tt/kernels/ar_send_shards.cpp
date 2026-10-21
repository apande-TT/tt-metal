// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// One fabric-link worker of tt/down_decode.py's all_reduce fold (StreamDownDecode.ar_call; RISCV_0, NOC 0): ar_send.cpp's
// exchange (all_reduce_async's worker_writer pattern), one shard at a time AS SOON AS IT IS COMPLETE while the down
// projection still streams. The worker owns N residual shards (SHARD_BYTES = WT bf16 tiles of this device's down
// partials each). The down cores write every partial tile straight into this worker's send buffer (CB_BUF, the same L1
// address on every core: shard s at s * SHARD_BYTES) and count it in shard s's 3-bit field of semaphore SEM_A (shards
// 0..9) or SEM_B (10..19). The worker sends each shard whose field reached WT -- into slot MY_CHIP of its residual
// core's scratch on this device and, one fused write + atomic increment packet per direction, line-multicast over the
// ring, on every other device, raising the residual core's global semaphore there -- and counts the local copies once
// no shard is ready (after a write barrier). Each residual core waits for TP counts.
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
constexpr uint32_t N = get_compile_time_arg_val(4);  // shards of this worker (<= 20)
constexpr uint32_t SEM_A = get_compile_time_arg_val(5);
constexpr uint32_t SEM_B = get_compile_time_arg_val(6);
constexpr uint32_t WT = get_compile_time_arg_val(7);  // tiles a shard (< 8: a 3-bit field)
constexpr ccl_routing_utils::line_multicast_route_info_t fwd_route =
    ccl_routing_utils::get_line_multicast_route_info_from_args<8>();
constexpr ccl_routing_utils::line_multicast_route_info_t bwd_route =
    ccl_routing_utils::get_line_multicast_route_info_from_args<8 + ccl_routing_utils::num_line_multicast_args>();
static_assert(N <= 20 && WT < 8, "shard fields");

void kernel_main() {
    size_t a = 0;
    const uint32_t scratch_addr = get_arg_val<uint32_t>(a++);
    const uint32_t sem_addr = get_arg_val<uint32_t>(a++);
    const size_t xy0 = a;  // N x (res x, res y), NOC 0 coordinates
    a += 2 * N;
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

    const uint32_t buf = get_write_ptr(CB_BUF);
    volatile tt_l1_ptr uint32_t* sem_a = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_A));
    volatile tt_l1_ptr uint32_t* sem_b = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_B));
    constexpr uint32_t all = (1u << N) - 1;
    uint32_t sent = 0, counted = 0;
    while (counted != all) {
        invalidate_l1_cache();
        const uint32_t va = *sem_a, vb = *sem_b;
        bool any = false;
        for (uint32_t s = 0; s < N; ++s) {
            if ((sent >> s) & 1) {
                continue;
            }
            const uint32_t field = (s < 10 ? va >> (3 * s) : vb >> (3 * (s - 10))) & 7;
            if (field != WT) {
                continue;
            }
            const uint32_t rx = get_arg_val<uint32_t>(xy0 + 2 * s), ry = get_arg_val<uint32_t>(xy0 + 2 * s + 1);
            const uint64_t dst = safe_get_noc_addr(rx, ry, scratch_addr + MY_CHIP * SHARD_BYTES);
            const uint64_t sem = safe_get_noc_addr(rx, ry, sem_addr);
            size_t l1 = buf + s * SHARD_BYTES;
            fused_write_atomic_and_advance_local_read_address_for_fabric_write(
                dst, hdr_fwd, hdr_bwd, fabric, l1, SHARD_BYTES, sem, 1, true);
            noc_async_writes_flushed();
            sent |= 1u << s;
            any = true;
        }
        if (!any && sent != counted) {
            noc_async_write_barrier();  // the local copies sent so far have landed: count them on their residual cores
            for (uint32_t s = 0; s < N; ++s) {
                if (((sent & ~counted) >> s) & 1) {
                    const uint32_t rx = get_arg_val<uint32_t>(xy0 + 2 * s), ry = get_arg_val<uint32_t>(xy0 + 2 * s + 1);
                    noc_semaphore_inc(safe_get_noc_addr(rx, ry, sem_addr), 1);
                }
            }
            counted = sent;
        }
    }
    if (fabric.is_logically_connected()) {
        fabric.close_start();
        fabric.close_finish();
    }
    noc_async_atomic_barrier();
    noc_async_write_barrier();
}

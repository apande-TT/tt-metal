// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute of the norm cores in tt/gate_up_decode.py's spread layout (NormGateUpDecode.SPREAD): the o_proj all_reduce
// sum, residual add and norm of norm_shard_ar_compute.cpp, then this core's own gate / up columns exactly as
// gate_up_compute.cpp computes them (GU_AFTER_NORM: reconfigured, not started up, and at the matmul's own fidelity --
// the kernel is built at the norm's HiFi4; its args follow the norm's, GU_CTA comes in as a define).
#include <cstdint>

// every header the two parts include, first, so the renaming below reaches only their kernel_main
#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reduce.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

#define kernel_main norm_part
#include "norm_shard_ar_compute.cpp"
#undef kernel_main

#define kernel_main gate_up_part
#include "gate_up_compute.cpp"
#undef kernel_main

void kernel_main() {
    norm_part();
    gate_up_part();
}

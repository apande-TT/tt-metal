// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RISCV_0 of the norm cores in tt/gate_up_decode.py's spread layout (NormGateUpDecode.SPREAD): the o_proj all_reduce,
// residual add and norm dataflow of norm_shard_ar_dm.cpp, then -- its slice of the activation handed to the sender --
// the gate weight stream of this core's own gate / up columns (gate_up_stream.cpp, non-owner; its args follow the
// norm's: GU_CTA / GU_RTA come in as defines). The up stream runs on RISCV_1 from the program's start.
#include <cstdint>

// every header the two parts include, first, so the renaming below reaches only their kernel_main
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

#define kernel_main norm_part
#include "norm_shard_ar_dm.cpp"
#undef kernel_main

#define kernel_main stream_part
#include "gate_up_stream.cpp"
#undef kernel_main

void kernel_main() {
    norm_part();
    stream_part();
}

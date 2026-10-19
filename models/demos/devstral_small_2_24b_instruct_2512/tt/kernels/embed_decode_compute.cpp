// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel of tt/embed_decode.py: tilizes the reader's 32-row x WT-tile row-major bf16 block into WT fp32 tiles
// (bf16 -> fp32 is exact: the tilize unpacks the bf16 values and the packer widens them).

#include <cstdint>

#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t WT = get_compile_time_arg_val(0);
    constexpr uint32_t CB_RM = get_compile_time_arg_val(1);
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(2);
    compute_kernel_hw_startup(CB_RM, CB_OUT);
    compute_kernel_lib::tilize<WT, CB_RM, CB_OUT>(1);
}

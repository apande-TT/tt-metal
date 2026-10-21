# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.typecast of an L1-interleaved tile tensor into an L1-interleaved output, with every core casting the pages of
its OWN L1 bank (ttnn.generic_op: kernels/local_cast_dm.cpp, kernels/local_cast_compute.cpp).

Input and output have the same page count, so page p of both sits in bank p % banks at slot p // banks: each core
reads its bank's input slots and writes its bank's output slots over loopback NOC transfers, where ttnn.typecast
hands each core a contiguous page range and reads / writes every page across the chip, one page at a time. The
compute is ttnn.typecast's own (copy into DEST -- fp32 input unpacked straight to DEST --, typecast_tile's instructions
with the fp32 -> bf16 row loop unrolled, pack; same compute config), so the result is bit-identical (A/B on the vision
norm outputs: 0 of 33.5M values differ). Vision norm cast (8 x 256 x 1024 fp32 -> bf16): 10.3 -> 7.2 us."""

from __future__ import annotations

from pathlib import Path

import ttnn

_DIR = Path(__file__).resolve().parent / "kernels"
_DM = str(_DIR / "local_cast_dm.cpp")
_COMPUTE = str(_DIR / "local_cast_compute.cpp")
_DF = {ttnn.float32: 0, ttnn.bfloat16: 5}  # tt::DataFormat Float32 / Float16_b
_PAGE = {ttnn.float32: 4096, ttnn.bfloat16: 2048}


class LocalCast:
    CB_IN, CB_OUT = 0, 16
    CHUNK = 4  # pages a read / write batch (both CBs hold two batches)

    def __init__(self, device):
        self.device = device
        g = device.compute_with_storage_grid_size()
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
        self.core_xy = [(x, y) for y in range(g.y) for x in range(g.x)]
        self.n_cores = g.x * g.y
        try:  # every L1 bank must be one worker core's (the kernels cast only their own core's first bank)
            self.banks = int(ttnn.get_memory_view(device, ttnn.BufferType.L1).num_banks)
        except Exception:
            self.banks = 0

    def supports(self, x, dtype) -> bool:
        mem = x.memory_config()
        return (
            self.banks == self.n_cores
            and x.dtype in _DF
            and dtype in _DF
            and x.dtype != dtype
            and x.layout == ttnn.TILE_LAYOUT
            and mem.buffer_type == ttnn.BufferType.L1
            and not x.is_sharded()
        )

    def __call__(self, x, dtype):
        """x: L1-interleaved tiles (fp32 <-> bf16) -> the same shape in dtype, L1-interleaved."""
        out = ttnn.allocate_tensor_on_device(x.shape, dtype, ttnn.TILE_LAYOUT, self.device, ttnn.L1_MEMORY_CONFIG)
        shape = [int(d) for d in x.padded_shape]
        npages = 1
        for d in shape[:-2]:
            npages *= d
        npages *= (shape[-2] // ttnn.TILE_SIZE) * (shape[-1] // ttnn.TILE_SIZE)
        nmax = -(-npages // self.banks)
        chunk = min(self.CHUNK, nmax)
        pin, pout = _PAGE[x.dtype], _PAGE[dtype]

        def cb(index, dt, page):
            fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=dt, page_size=page)
            return ttnn.CBDescriptor(total_size=2 * chunk * page, core_ranges=self.cores, format_descriptors=[fmt])

        cfg = ttnn.ComputeConfigDescriptor()
        cfg.math_fidelity = ttnn.MathFidelity.HiFi4
        cfg.fp32_dest_acc_en = True
        cfg.math_approx_mode = False
        cfg.dst_full_sync_en = False
        if x.dtype == ttnn.float32:  # ttnn.typecast's preserve_fp32_precision: fp32 input unpacked straight to DEST
            modes = [ttnn.UnpackToDestMode.Default] * 64
            modes[self.CB_IN] = ttnn.UnpackToDestMode.UnpackToDestFp32
            cfg.unpack_to_dest_mode = modes
        dm = lambda role, t, risc, noc, cbi, page: ttnn.KernelDescriptor(
            kernel_source=_DM,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.cores,
            compile_time_args=[role, cbi, page, nmax, chunk] + list(ttnn.TensorAccessorArgs(t).get_compile_time_args()),
            runtime_args=[(xy, [t.buffer_address()]) for xy in self.core_xy],
            config=ttnn.DataMovementConfigDescriptor(processor=risc, noc=noc),
        )
        kernels = [
            dm(0, x, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0, self.CB_IN, pin),
            dm(1, out, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1, self.CB_OUT, pout),
            ttnn.KernelDescriptor(
                kernel_source=_COMPUTE,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.cores,
                compile_time_args=[nmax, chunk, _DF[x.dtype], _DF[dtype]],
                runtime_args=[],
                config=cfg,
            ),
        ]
        cbs = [cb(self.CB_IN, x.dtype, pin), cb(self.CB_OUT, dtype, pout)]
        ttnn.generic_op([x, out], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
        return out

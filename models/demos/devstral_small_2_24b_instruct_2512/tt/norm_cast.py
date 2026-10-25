# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.rms_norm (unit weight, interleaved) of an fp32 tile tensor followed by ttnn.typecast to bf16, as ONE program
(ttnn.generic_op: kernels/norm_cast_dm.cpp, kernels/norm_cast_compute.cpp).

The compute is the stock interleaved layernorm kernel's RMSNORM + FUSE_GAMMA path (compute/layernorm.cpp) on the same
circular-buffer formats, the same compute_kernel_lib calls in the same order, with the gamma product rounded to bf16
by typecast's own instructions before it is packed, so the output is the bits ttnn.typecast(ttnn.rms_norm(x)) writes,
without the fp32 norm output or the cast program. Work split: whole tile rows over cores, as the stock factory."""

from __future__ import annotations

import struct
from pathlib import Path

import ttnn

_KDIR = Path(__file__).resolve().parent / "kernels"
_DM = str(_KDIR / "norm_cast_dm.cpp")
_COMPUTE = str(_KDIR / "norm_cast_compute.cpp")
CB_IN, CB_SCALER, CB_EPS, CB_GAMMA, CB_OUT, CB_EX2, CB_XMM2, CB_EX2PE, CB_FUSION = 0, 2, 3, 5, 16, 19, 20, 21, 22
_PAGE = {ttnn.float32: 4096, ttnn.bfloat16: 2048}

# The stock factory's choices for this call (LayerNormMultiCoreProgramFactory, fp32 input, HiFi4 + fp32 DEST, the
# canonical unit bf16 weight): 4-tile blocks with fp32 DEST, float32_reduction (unused by the row reduce).
BLOCK = 4
FLOAT32_REDUCTION = 1
# intermediate CB formats (x^2, E[x^2], E[x^2] + eps -> rsqrt, x * r): fp32, re-entering SrcA / SrcB as TF32
INTERM_DTYPE = ttnn.float32
# (cb dtype, value, fill) of the reader-generated tiles as the stock reader writes them; fill (kernels/norm_cast_dm.cpp):
# 0 row 0 of each face, 1 column 0, 2 row 0 of faces 0 / 1 only. eps: bf16 TRUNCATED (generate_bcast_scalar: bits >>
# 16) in column 0 (only column 0 of the rsqrt tile is read, by the column-broadcast multiply); scaler: bf16 1.0 on row 0
# of every face (SUM reduce, 1/N applied after by an SFPU multiply); gamma: the unit weight (only row 0 of faces 0 / 1
# is broadcast).
EPS_TILE = (ttnn.bfloat16, "eps", 1)
SCALER_TILE = (ttnn.bfloat16, 1.0, 0)
GAMMA_TILE = (ttnn.bfloat16, 1.0, 2)


def _bits(dtype, v: float) -> int:
    f = struct.unpack("<I", struct.pack("<f", float(v)))[0]
    return f if dtype == ttnn.float32 else f >> 16  # bf16 by truncation (exact for 1.0)


class NormCast:
    def __init__(self, device):
        self.device = device

    @staticmethod
    def supports(x) -> bool:
        mem = x.memory_config()
        return (
            x.dtype == ttnn.float32
            and x.layout == ttnn.TILE_LAYOUT
            and not x.is_sharded()
            and mem.buffer_type == ttnn.BufferType.L1
            and int(x.padded_shape[-1]) % ttnn.TILE_SIZE == 0
            and int(x.padded_shape[-2]) % ttnn.TILE_SIZE == 0
        )

    def __call__(self, x, eps: float):
        """x: fp32 L1-interleaved tiles (..., S, W) -> bf16(rms_norm(x)) in L1, interleaved."""
        device = self.device
        shape = [int(d) for d in x.padded_shape]
        W = shape[-1]
        Wt = W // ttnn.TILE_SIZE
        rows = shape[-2] // ttnn.TILE_SIZE
        for d in shape[:-2]:
            rows *= d
        out = ttnn.allocate_tensor_on_device(x.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG)
        g = device.compute_with_storage_grid_size()
        ncores = min(rows, g.x * g.y)
        base, extra = divmod(rows, ncores)
        cores = ttnn.num_cores_to_corerangeset(ncores, g, row_wise=True)
        blk = min(BLOCK, Wt)
        f32, bf = _PAGE[ttnn.float32], _PAGE[ttnn.bfloat16]

        def cb(index, tiles, dtype):
            page = _PAGE[dtype]
            fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)
            return ttnn.CBDescriptor(total_size=tiles * page, core_ranges=cores, format_descriptors=[fmt])

        wt_full = -(-Wt // blk) * blk
        cbs = [
            cb(CB_IN, wt_full, ttnn.float32),
            cb(CB_SCALER, 1, SCALER_TILE[0]),
            cb(CB_EPS, 1, EPS_TILE[0]),
            cb(CB_GAMMA, wt_full, GAMMA_TILE[0]),
            cb(CB_OUT, 2 * blk, ttnn.bfloat16),
            cb(CB_EX2, 1, INTERM_DTYPE),
            cb(CB_XMM2, wt_full, INTERM_DTYPE),  # the whole row's x^2 is packed before the reduce reads it
            cb(CB_EX2PE, 1, INTERM_DTYPE),
            cb(CB_FUSION, 2 * blk, INTERM_DTYPE),
        ]
        ccfg = ttnn.ComputeConfigDescriptor()
        ccfg.math_fidelity = ttnn.MathFidelity.HiFi4
        ccfg.fp32_dest_acc_en = True
        ccfg.math_approx_mode = False
        ccfg.dst_full_sync_en = False

        val = lambda spec: _bits(spec[0], eps if spec[1] == "eps" else spec[1])
        datum = lambda dt: 4 if dt == ttnn.float32 else 2
        rr, rw, rc = [], [], []
        r0 = 0
        for i in range(ncores):
            n = base + (1 if i < extra else 0)
            xy = (i % g.x, i // g.x)
            rr.append((xy, [x.buffer_address(), r0, n, val(EPS_TILE), val(SCALER_TILE), val(GAMMA_TILE)]))
            rw.append((xy, [out.buffer_address(), r0, n]))
            rc.append((xy, [n]))
            r0 += n
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        consts = [CB_EPS, datum(EPS_TILE[0]), EPS_TILE[2], CB_SCALER, datum(SCALER_TILE[0]), SCALER_TILE[2]]
        consts += [CB_GAMMA, datum(GAMMA_TILE[0]), GAMMA_TILE[2]]
        dm = lambda role, risc, noc, cbi, page, t, rt, extra_ct: ttnn.KernelDescriptor(
            kernel_source=_DM,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[role, Wt, blk, cbi, page] + acc(t) + extra_ct,
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(processor=risc, noc=noc),
        )
        kernels = [
            dm(0, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0, CB_IN, f32, x, rr, consts),
            dm(1, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1, CB_OUT, bf, out, rw, consts),
            ttnn.KernelDescriptor(
                kernel_source=_COMPUTE,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=cores,
                compile_time_args=[Wt, blk, W, FLOAT32_REDUCTION],
                runtime_args=rc,
                config=ccfg,
            ),
        ]
        ttnn.generic_op([x, out], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
        return out

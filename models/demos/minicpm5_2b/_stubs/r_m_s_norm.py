# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn port of `LlamaRMSNorm` (model.layers.0.input_layernorm) for `openbmb/MiniCPM5-2B`.

HF semantics: x * rsqrt(mean(x^2) + eps) * weight over the 2048-wide hidden dim.
"""
from __future__ import annotations

import math
import os
import struct

import torch

import ttnn

# Decode (one tile row) RMSNorm as a multi-core generic_op: every core recomputes ttnn.rms_norm's row statistic
# with its exact arithmetic and normalizes its own slice of the row (the stock op runs one tile row on one core).
_USE_KERNEL = True
_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
_TILES_PER_CORE = 4
# Row statistic once on a reducer core and broadcast (kernels/rms_norm_shared) instead of recomputed on every core.
_SHARED_STAT = True
_SHARED_COLS = 8  # the cores form a _SHARED_COLS-wide rectangle


class TtRMSNorm:
    def __init__(self, device, torch_module):
        self.device = device
        self.eps = getattr(torch_module, "variance_epsilon", None) or getattr(torch_module, "eps", 1e-6)
        self.compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        w = torch_module.weight.detach().to(torch.float32).reshape(1, 1, -1)
        self.weight = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.consts = None
        if _USE_KERNEL:
            # The decode kernel's constant tiles [scaler | eps], built here once instead of by the reader (whose
            # element-wise L1 stores were most of its time): ttnn layernorm's reduce scaler (1.0 in rows 0 and 16)
            # and eps truncated (not rounded) to bf16 in column 0.
            eps_bf16 = torch.tensor([float(self.eps)], dtype=torch.float32).view(torch.int32)
            eps_bf16 = (eps_bf16 & ~0xFFFF).view(torch.float32)
            c = torch.zeros(32, 64)
            c[[0, 16], 0:32] = 1.0
            c[:, 32] = eps_bf16
            self.consts = ttnn.from_torch(c, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def _kernel_ok(self, x):
        ps = list(x.padded_shape)
        return (
            _USE_KERNEL
            and x.dtype == ttnn.bfloat16
            and x.layout == ttnn.TILE_LAYOUT
            and not x.memory_config().is_sharded()
            and math.prod(ps[:-1]) == 32
            and ps[-1] == x.shape[-1]
            and (ps[-1] // 32) % _TILES_PER_CORE == 0
        )

    def _kernel_shared(self, x, memory_config):
        """The row statistic computed once instead of on every core: each core squares its own slice, the reducer
        (core 0) row-reduces all Wt x^2 tiles in order and broadcasts the rstd tile, each core normalizes its
        slice. Same per-tile arithmetic and reduction order as the per-core-statistic kernel."""
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(x.shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device, memory_config
        )
        n_cores = x.padded_shape[-1] // 32 // _TILES_PER_CORE
        cols = _SHARED_COLS if n_cores % _SHARED_COLS == 0 else n_cores
        core_list = [ttnn.CoreCoord(i % cols, i // cols) for i in range(n_cores)]
        kernels, cbs, semaphores = self.shared_parts(x, out, core_list)
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)
        return ttnn.generic_op([x, self.weight, self.consts, out], program)

    def shared_parts(self, x, out, core_list, pro=None):
        """Kernels, CBs and semaphores of the shared-statistic norm of x into out on core_list (logical cores, the
        first the reducer; Wt / _TILES_PER_CORE of them). pro=(sem_id, noc_xy): as the prologue of another
        program, each core then bumps semaphore sem_id on every core of noc_xy once its output slice is stored."""
        w_t = x.padded_shape[-1] // 32
        n_cores = w_t // _TILES_PER_CORE
        assert len(core_list) == n_cores
        from models.demos.minicpm5_2b._stubs.dram_mm import _core_rects

        cores = _core_rects(core_list)
        rt = ttnn.RuntimeArgs()
        noc_xy = []
        for i, c in enumerate(core_list):
            rt[c.x][c.y] = [i * _TILES_PER_CORE, _TILES_PER_CORE, 1 if i == 0 else 0]
            q = self.device.worker_core_from_logical_core(c)
            noc_xy += [q.x, q.y]
        bf16_tile, f32_tile = 2 * 32 * 32, 4 * 32 * 32
        cb_in, cb_scaler, cb_eps, cb_gamma, cb_out = 0, 1, 2, 3, 16
        cb_x2, cb_ex2, cb_ex2pe, cb_fusion, cb_x2all, cb_ex2pe_send = 24, 25, 26, 27, 28, 29
        sem_gather, sem_bcast = 0, 1

        def cb(idx, n, fmt, size):
            return ttnn.CBDescriptor(
                total_size=n * size,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(idx, fmt, size)],
            )

        bf, f32 = ttnn.bfloat16, ttnn.float32
        cbs = [
            cb(cb_in, _TILES_PER_CORE, bf, bf16_tile),
            cb(cb_scaler, 1, bf, bf16_tile),
            cb(cb_eps, 1, bf, bf16_tile),
            cb(cb_gamma, _TILES_PER_CORE, bf, bf16_tile),
            cb(cb_out, _TILES_PER_CORE, bf, bf16_tile),
            cb(cb_x2, _TILES_PER_CORE, f32, f32_tile),
            cb(cb_ex2, 1, f32, f32_tile),
            cb(cb_ex2pe, 1, f32, f32_tile),
            cb(cb_fusion, _TILES_PER_CORE, f32, f32_tile),
            cb(cb_x2all, w_t, f32, f32_tile),
            cb(cb_ex2pe_send, 1, f32, f32_tile),
        ]
        inv_w_bits = struct.unpack("<I", struct.pack("<f", 1.0 / x.shape[-1]))[0]
        d = os.path.join(_KERNEL_DIR, "rms_norm_shared")
        reader = ttnn.KernelDescriptor(
            kernel_source=os.path.join(d, "reader_rms_norm_shared.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[w_t, cb_in, cb_scaler, cb_eps, cb_gamma, cb_x2all, cb_ex2pe, sem_gather, sem_bcast]
            + [n_cores]
            + list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(self.weight).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(self.consts).get_compile_time_args()),
            runtime_args=rt,
            common_runtime_args=[x.buffer_address(), self.weight.buffer_address(), self.consts.buffer_address()],
            config=ttnn.ReaderConfigDescriptor(),
        )
        pro_sem, pro_xy = pro if pro is not None else (0xFFFFFFFF, [])
        writer = ttnn.KernelDescriptor(
            kernel_source=os.path.join(d, "writer_rms_norm_shared.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[cb_x2, cb_x2all, cb_ex2pe, cb_ex2pe_send, cb_out, sem_gather, sem_bcast, n_cores]
            + [pro_sem]
            + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
            runtime_args=rt,
            common_runtime_args=[out.buffer_address(), noc_xy[0], noc_xy[1]]
            + noc_xy
            + ([len(pro_xy) // 2] + list(pro_xy) if pro is not None else []),
            config=ttnn.WriterConfigDescriptor(),
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=os.path.join(d, "compute_rms_norm_shared.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[
                w_t, inv_w_bits, cb_in, cb_scaler, cb_eps, cb_gamma, cb_x2, cb_ex2, cb_ex2pe, cb_fusion, cb_out,
                cb_x2all, cb_ex2pe_send,
            ],
            runtime_args=rt,
            config=ttnn.ComputeConfigDescriptor(  # as self.compute_cfg
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                dst_full_sync_en=False,
            ),
        )
        semaphores = [
            ttnn.SemaphoreDescriptor(id=sem_gather, core_ranges=cores, initial_value=0),
            ttnn.SemaphoreDescriptor(id=sem_bcast, core_ranges=cores, initial_value=0),
        ]
        return [reader, writer, compute], cbs, semaphores

    def _kernel(self, x, memory_config):
        if _SHARED_STAT:
            return self._kernel_shared(x, memory_config)
        w_t = x.padded_shape[-1] // 32
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(x.shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device, memory_config
        )
        grid = self.device.compute_with_storage_grid_size()
        n_cores = w_t // _TILES_PER_CORE
        full_rows, rem = divmod(n_cores, grid.x)
        ranges = []
        if full_rows:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, full_rows - 1)))
        if rem:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
        cores = ttnn.CoreRangeSet(ranges)
        rt = ttnn.RuntimeArgs()
        for i in range(n_cores):
            rt[i % grid.x][i // grid.x] = [i * _TILES_PER_CORE, _TILES_PER_CORE]
        bf16_tile, f32_tile = 2 * 32 * 32, 4 * 32 * 32
        cb_in, cb_scaler, cb_eps, cb_gamma, cb_out = 0, 1, 2, 3, 16
        cb_x2, cb_ex2, cb_ex2pe, cb_fusion = 24, 25, 26, 27

        def cb(idx, n, fmt, size):
            return ttnn.CBDescriptor(
                total_size=n * size,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(idx, fmt, size)],
            )

        bf, f32 = ttnn.bfloat16, ttnn.float32
        cbs = [
            cb(cb_in, w_t, bf, bf16_tile),
            cb(cb_scaler, 1, bf, bf16_tile),
            cb(cb_eps, 1, bf, bf16_tile),
            cb(cb_gamma, _TILES_PER_CORE, bf, bf16_tile),
            cb(cb_out, _TILES_PER_CORE, bf, bf16_tile),
            cb(cb_x2, w_t, f32, f32_tile),
            cb(cb_ex2, 1, f32, f32_tile),
            cb(cb_ex2pe, 1, f32, f32_tile),
            cb(cb_fusion, _TILES_PER_CORE, f32, f32_tile),
        ]
        inv_w_bits = struct.unpack("<I", struct.pack("<f", 1.0 / x.shape[-1]))[0]
        d = os.path.join(_KERNEL_DIR, "rms_norm")
        reader = ttnn.KernelDescriptor(
            kernel_source=os.path.join(d, "reader_rms_norm.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[w_t, cb_in, cb_scaler, cb_eps, cb_gamma]
            + list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(self.weight).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(self.consts).get_compile_time_args()),
            runtime_args=rt,
            common_runtime_args=[x.buffer_address(), self.weight.buffer_address(), self.consts.buffer_address()],
            config=ttnn.ReaderConfigDescriptor(),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KERNEL_DIR, "rope", "writer_rope.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[cb_out] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
            runtime_args=rt,
            common_runtime_args=[out.buffer_address()],
            config=ttnn.WriterConfigDescriptor(),
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=os.path.join(d, "compute_rms_norm.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[
                w_t, inv_w_bits, cb_in, cb_scaler, cb_eps, cb_gamma, cb_x2, cb_ex2, cb_ex2pe, cb_fusion, cb_out
            ],
            runtime_args=rt,
            config=ttnn.ComputeConfigDescriptor(  # as self.compute_cfg
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                dst_full_sync_en=False,
            ),
        )
        program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
        return ttnn.generic_op([x, self.weight, self.consts, out], program)

    def __call__(self, x, *args, memory_config=None, fused=None, **kwargs):
        """fused: a program computing this norm together with the op that consumes it (e.g. a dram_mm.DramMatmul
        bound to this norm); its result is returned instead of the normalized x."""
        if fused is not None:
            return fused(x, memory_config or ttnn.DRAM_MEMORY_CONFIG)
        if not isinstance(x, ttnn.Tensor):
            x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        if self._kernel_ok(x):
            return self._kernel(x, memory_config or ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.rms_norm(
            x,
            epsilon=self.eps,
            weight=self.weight,
            memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_cfg,
        )


def build(device, torch_module=None):
    return TtRMSNorm(device, torch_module)


def r_m_s_norm(device, torch_module=None):
    return TtRMSNorm(device, torch_module)

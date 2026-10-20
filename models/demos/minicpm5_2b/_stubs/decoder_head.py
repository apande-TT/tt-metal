# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn port of `lm_head` (decoder_head) for `openbmb/MiniCPM5-2B`.

HF semantics: logits = x @ W^T, Linear(2048 -> 130560, bias=False).
"""
from __future__ import annotations

import math
import os

import torch

import ttnn
from models.demos.minicpm5_2b._stubs.dram_mm import DramMatmul

_VOCAB_CHUNK = 16320  # 510 tiles; 130560 = 8 chunks
_TILE = 32

# Decode (M = one tile row): custom C++ skinny-matmul kernel via ttnn.generic_op; anything else uses ttnn.linear.
_USE_CPP_KERNEL = True
_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "lm_head")
_MAX_GROUP_W = 4  # output tiles held in DEST per accumulation (fp32 dest, half-sync => 4)
_BLOCK_TILES = 32  # target weight tiles per reader block (k_block * group_w)
_IN1_BUFFERS = 3  # weight CB depth in blocks
_CB_IN0, _CB_IN1, _CB_OUT = 0, 1, 16
# Decode LM head on the DRAM-bank-local generic_op (_stubs/dram_mm.py), same arithmetic as the kernel below.
_USE_DRAM_MM = True
_DRAM_MM_WORKERS_PER_BANK = 6  # HiFi4 bf8: compute-bound on fewer cores
_DRAM_MM_BLOCK_TILES = 8  # bf8 weight read in 8-tile blocks (one read each)
# The final RMSNorm as a prologue of the decode LM head program (bind_prenorm).
_USE_NORM_PROLOGUE = True


def _mesh_kw(device, **kw):
    kw["device"] = device
    try:
        if isinstance(device, ttnn.MeshDevice):
            kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
    except AttributeError:
        pass
    return kw


class DecoderHead:
    def __init__(self, device, torch_module):
        self.device = device
        self.compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # Split the 130560-wide vocab into tile-aligned column chunks (as tt_transformers' LMHead does)
        # so no single matmul spans the full vocab.
        w = torch_module.weight.detach().to(torch.float32).t().contiguous()
        self.weights = [
            ttnn.from_torch(
                chunk.contiguous(),
                # bfloat8_b halves the ~535 MB vocab-projection read that bounds the decode LM head.
                **_mesh_kw(device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            )
            for chunk in torch.split(w, _VOCAB_CHUNK, dim=-1)
        ]
        self.k_dim = w.shape[0]
        self.vocab = w.shape[1]
        self._kernel = self._build_kernel() if _USE_CPP_KERNEL else None
        self.dmm = None
        if _USE_DRAM_MM:
            self.dmm = DramMatmul(
                device,
                w,
                ttnn.MathFidelity.HiFi2,
                dtype=ttnn.bfloat8_b,
                workers_per_bank=_DRAM_MM_WORKERS_PER_BANK,
                block_tiles=_DRAM_MM_BLOCK_TILES,
            )

    def _build_kernel(self):
        """Static part of the generic_op program: one program covers all vocab chunks."""
        if self.k_dim % _TILE or any(t.shape[-1] != _VOCAB_CHUNK for t in self.weights):
            return None
        Kt = self.k_dim // _TILE
        Nt_chunk = _VOCAB_CHUNK // _TILE
        Nt = Nt_chunk * len(self.weights)
        grid = self.device.compute_with_storage_grid_size()
        n_avail = grid.x * grid.y

        # Group width g (tiles accumulated together in DEST): minimise the per-core tile load, prefer larger g.
        best = None
        for g in range(_MAX_GROUP_W, 0, -1):
            if Nt % g:
                continue
            load = math.ceil(Nt // g / n_avail) * g
            if best is None or load < best[0]:
                best = (load, g)
        group_w = best[1]
        num_groups = Nt // group_w
        n_cores = min(n_avail, num_groups)
        base, extra = divmod(num_groups, n_cores)
        k_block = max(d for d in range(1, Kt + 1) if Kt % d == 0 and d * group_w <= max(_BLOCK_TILES, group_w))

        # Cores row-wise over the logical grid.
        full_rows, rem = divmod(n_cores, grid.x)
        ranges = []
        if full_rows:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, full_rows - 1)))
        if rem:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
        cores = ttnn.CoreRangeSet(ranges)

        dm_rt, cmp_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        n = 0
        for i in range(n_cores):
            x, y = i % grid.x, i // grid.x
            groups = base + (1 if i < extra else 0)
            dm_rt[x][y] = [n * group_w, groups]
            cmp_rt[x][y] = [groups]
            n += groups
        assert n == num_groups

        in0_tile = 2 * _TILE * _TILE  # bf16
        in1_tile = self.weights[0].tile.get_tile_size(ttnn.bfloat8_b)  # 1088
        out_tile = in0_tile
        cbs = [
            ttnn.CBDescriptor(
                total_size=Kt * in0_tile,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(_CB_IN0, ttnn.bfloat16, in0_tile)],
            ),
            ttnn.CBDescriptor(
                total_size=_IN1_BUFFERS * k_block * group_w * in1_tile,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(_CB_IN1, ttnn.bfloat8_b, in1_tile)],
            ),
            ttnn.CBDescriptor(
                total_size=2 * group_w * out_tile,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(_CB_OUT, ttnn.bfloat16, out_tile)],
            ),
        ]
        compute = ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KERNEL_DIR, "compute_lm_head.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[Kt, group_w, k_block, _CB_IN0, _CB_IN1, _CB_OUT],
            runtime_args=cmp_rt,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                dst_full_sync_en=False,
            ),
        )
        return dict(
            Kt=Kt,
            cores=cores,
            cbs=cbs,
            compute=compute,
            dm_rt=dm_rt,
            reader_ct=[Kt, Nt_chunk, group_w, k_block, _CB_IN0, _CB_IN1, in0_tile, in1_tile],
            w_ct=list(ttnn.TensorAccessorArgs(self.weights[0]).get_compile_time_args()),
            writer_ct=[group_w, _CB_OUT, out_tile],
            w_addrs=[t.buffer_address() for t in self.weights],
        )

    def _kernel_ok(self, x):
        k = self._kernel
        if k is None or x.dtype != ttnn.bfloat16 or x.memory_config().is_sharded():
            return False
        ps = list(x.padded_shape)
        return ps[-1] == self.k_dim and math.prod(ps[:-1]) == _TILE  # exactly one tile row

    def _run_kernel(self, x, memory_config=None, segments=None):
        k = self._kernel
        # With `segments` S the logits are allocated as [1, S, 32, V / S]: one tile row per segment, so its tile
        # pages are exactly those of [.., 32, V] in the same order and the kernel writes them unchanged.
        shape = list(x.shape)[:-1] + [self.vocab] if segments is None else [1, segments, 32, self.vocab // segments]
        if self.dmm is not None and self.dmm.accepts(x):
            return self.dmm(x, memory_config or ttnn.DRAM_MEMORY_CONFIG, out_shape=shape)
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape(shape),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.device,
            memory_config or ttnn.DRAM_MEMORY_CONFIG,
        )
        # Buffer addresses go in common runtime args (rebuilt every call; values are not part of the program hash).
        reader = ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KERNEL_DIR, "reader_lm_head.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=k["cores"],
            compile_time_args=k["reader_ct"] + list(ttnn.TensorAccessorArgs(x).get_compile_time_args()) + k["w_ct"],
            runtime_args=k["dm_rt"],
            common_runtime_args=[x.buffer_address()] + k["w_addrs"],
            config=ttnn.ReaderConfigDescriptor(),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KERNEL_DIR, "writer_lm_head.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=k["cores"],
            compile_time_args=k["writer_ct"] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
            runtime_args=k["dm_rt"],
            common_runtime_args=[out.buffer_address()],
            config=ttnn.WriterConfigDescriptor(),
        )
        program = ttnn.ProgramDescriptor(kernels=[reader, writer, k["compute"]], semaphores=[], cbs=k["cbs"])
        return ttnn.generic_op([x, *self.weights, out], program)

    def bind_prenorm(self, norm):
        """Run `norm` (the final RMSNorm) as a prologue of the decode LM head program (DramMatmul norm_prologue)."""
        self.dmm_norm = None
        if _USE_NORM_PROLOGUE and self.dmm is not None and self.dmm.ok and hasattr(norm, "consts"):
            dmm = DramMatmul(
                self.device,
                None,
                ttnn.MathFidelity.HiFi2,
                dtype=ttnn.bfloat8_b,
                workers_per_bank=_DRAM_MM_WORKERS_PER_BANK,
                block_tiles=_DRAM_MM_BLOCK_TILES,
                prebuilt=self.dmm,
                norm_prologue=norm,
            )
            self.dmm_norm = dmm if dmm.ok else None
        return self.dmm_norm is not None

    def __call__(self, x, *args, memory_config=None, segments=None, prenorm=None, **kwargs):
        """prenorm: the RMSNorm still to be applied to x (inside the decode LM head program when bound)."""
        dmm = getattr(self, "dmm_norm", None)
        if prenorm is not None:
            if dmm is not None and dmm.norm_pro is prenorm and self._kernel_ok(x) and prenorm._kernel_ok(x):
                if segments is not None and (self.vocab // 32) % segments:
                    segments = None
                shape = (
                    list(x.shape)[:-1] + [self.vocab] if segments is None else [1, segments, 32, self.vocab // segments]
                )
                mem = memory_config or ttnn.DRAM_MEMORY_CONFIG

                def fused(xx, _mem):
                    return dmm(xx, mem, out_shape=shape)

                return prenorm(x, memory_config=mem, fused=fused)  # final RMSNorm + LM head in one program
            x = prenorm(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        if not isinstance(x, ttnn.Tensor):
            x = ttnn.from_torch(x, **_mesh_kw(self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT))
        elif x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        if self._kernel_ok(x):
            if segments is not None and (self.vocab // 32) % segments:
                segments = None
            return self._run_kernel(x, memory_config, segments)
        outs = [
            ttnn.linear(x, w, compute_kernel_config=self.compute_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for w in self.weights
        ]
        out = ttnn.concat(outs, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for t in outs:
            ttnn.deallocate(t)
        return out


def build(device, torch_module=None):
    return DecoderHead(device, torch_module)


def decoder_head(device, torch_module=None):
    return DecoderHead(device, torch_module)

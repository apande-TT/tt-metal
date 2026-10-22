# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn port of `LlamaRotaryEmbedding` (model.rotary_emb) for `openbmb/MiniCPM5-2B`.

HF semantics: freqs = position_ids[..., None] * inv_freq; emb = cat(freqs, freqs);
returns (cos(emb) * scale, sin(emb) * scale), each [B, S, head_dim=128].
Angles are formed in fp32 on device so large positions keep their phase.
"""
from __future__ import annotations

import os

import torch

import ttnn

_ROPE_TABLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "rope_table")
_GATHER_CORES = 4  # batch rows split over this many cores (fewer cores: less per-program launch cost in trace)


class TtRotaryEmbedding:
    def __init__(self, device, torch_module):
        self.device = device
        self.scaling = float(getattr(torch_module, "attention_scaling", 1.0) or 1.0)
        inv_freq = torch_module.inv_freq.detach().to(torch.float32)
        inv_freq = inv_freq.repeat(2).reshape(1, 1, 1, -1)  # emb = (freqs, freqs) along head_dim
        self.head_dim = inv_freq.shape[-1]
        self.inv_freq = ttnn.from_torch(inv_freq, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        self.rows_table = None

    def build_rows_table(self, n_pos, rot_sign):
        """Decode rows tables (once, at init): cos and rot_sign-signed sin of every position < n_pos as [P, D] bf16
        tiles, made by the very ops of the per-step rows path (fp32 position * inv_freq, cos / sin, bf16 typecast,
        then * rot_sign), so each table row is bit-identical to what that path computes for the position."""
        assert n_pos % 32 == 0
        pos = ttnn.from_torch(
            torch.arange(n_pos, dtype=torch.float32).reshape(1, 1, n_pos, 1),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        cos, sin = self._rows_math(pos)
        sin_signed = ttnn.multiply(sin, rot_sign)
        self.rows_table = (cos, sin_signed)

    def _rows_math(self, pos):
        emb = ttnn.multiply(pos, self.inv_freq)
        cos, sin = ttnn.cos(emb), ttnn.sin(emb)
        ttnn.deallocate(emb)
        if self.scaling != 1.0:
            cos = ttnn.multiply(cos, self.scaling)
            sin = ttnn.multiply(sin, self.scaling)
        return ttnn.typecast(cos, ttnn.bfloat16), ttnn.typecast(sin, ttnn.bfloat16)

    def _gather_rows(self, pos_i32, memory_config):
        """(cos, signed sin) rows [1, 1, B, D] for the int32 [1, B] positions, gathered from the tables by one
        data-movement generic_op (kernels/rope_table)."""
        cos_t, sin_t = self.rows_table
        b = int(pos_i32.shape[-1])
        assert b == 32 and b % _GATHER_CORES == 0
        d_t = self.head_dim // 32
        outs = [
            ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, b, self.head_dim]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device, memory_config
            )
            for _ in range(2)
        ]
        users = b // _GATHER_CORES
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(_GATHER_CORES - 1, 0))])
        rt = ttnn.RuntimeArgs()
        for i in range(_GATHER_CORES):
            rt[i][0] = [i * users]
        reader = ttnn.KernelDescriptor(
            kernel_source=os.path.join(_ROPE_TABLE_DIR, "gather_rows.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[0, 1, d_t, users]
            + list(ttnn.TensorAccessorArgs(pos_i32).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(cos_t).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(outs[0]).get_compile_time_args()),
            runtime_args=rt,
            common_runtime_args=[pos_i32.buffer_address(), cos_t.buffer_address(), sin_t.buffer_address()]
            + [t.buffer_address() for t in outs],
            config=ttnn.ReaderConfigDescriptor(),
        )
        cbs = [
            ttnn.CBDescriptor(
                total_size=4 * 32 * 32,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.int32, 4 * 32 * 32)],
            ),
            ttnn.CBDescriptor(  # one 64 B (DRAM-aligned) span per gathered face row, plus alignment slack
                total_size=users * 2 * d_t * 2 * 64 + 64,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(1, ttnn.bfloat16, users * 2 * d_t * 2 * 64 + 64)],
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=[reader], semaphores=[], cbs=cbs)
        ttnn.generic_op([pos_i32, cos_t, sin_t] + outs, program)
        return outs[0], outs[1]

    def __call__(self, x, position_ids=None, *args, rows=False, table_pos=None, memory_config=None, **kwargs):
        """rows: one position per batch row (s = 1), returned as [1, 1, B, head_dim] (row b = batch b, one tile
        row) instead of [B, 1, head_dim]; the same elementwise arithmetic on the same values.
        table_pos (rows only, with build_rows_table done): the int32 [1, B] positions; returns (cos, rot_sign-signed
        sin) gathered from the tables in `memory_config` -- the values the rows path and its * rot_sign produce."""
        if rows and table_pos is not None and self.rows_table is not None:
            return self._gather_rows(table_pos, memory_config or ttnn.DRAM_MEMORY_CONFIG)
        if position_ids is None:
            raise ValueError("TtRotaryEmbedding requires position_ids")
        if not isinstance(position_ids, ttnn.Tensor):
            raise TypeError("TtRotaryEmbedding reads position_ids from device; upload them before the call")
        pos = position_ids if position_ids.dtype == ttnn.float32 else ttnn.typecast(position_ids, ttnn.float32)
        b, s = pos.shape[0], pos.shape[-1]
        if rows:
            assert s == 1
            pos = ttnn.to_layout(ttnn.reshape(pos, (1, 1, b, 1)), ttnn.TILE_LAYOUT)
            return self._rows_math(pos)
        pos = ttnn.reshape(pos, (1, b, s, 1))
        pos = ttnn.to_layout(pos, ttnn.TILE_LAYOUT)
        emb = ttnn.multiply(pos, self.inv_freq)
        cos = ttnn.cos(emb)
        sin = ttnn.sin(emb)
        ttnn.deallocate(emb)
        if self.scaling != 1.0:
            cos = ttnn.multiply(cos, self.scaling)
            sin = ttnn.multiply(sin, self.scaling)
        cos = ttnn.reshape(ttnn.typecast(cos, ttnn.bfloat16), (b, s, self.head_dim))
        sin = ttnn.reshape(ttnn.typecast(sin, ttnn.bfloat16), (b, s, self.head_dim))
        return cos, sin


def build(device, torch_module=None):
    return TtRotaryEmbedding(device, torch_module)


def rotary_embedding(device, torch_module=None):
    return TtRotaryEmbedding(device, torch_module)

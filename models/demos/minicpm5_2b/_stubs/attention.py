# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn port of `LlamaAttention` for `openbmb/MiniCPM5-2B`.

GQA attention: 16 query heads, 2 KV heads, head_dim 128, no projection biases.
HF semantics reproduced:
  q/k/v = x @ W^T, split into heads
  q, k  = q*cos + rotate_half(q)*sin   (cos/sin are the model's `position_embeddings`)
  [k, v] = cat(past, new) when a cache is passed (DynamicCache.update appends)
  out   = softmax(q k^T * head_dim^-0.5 + causal) v, merged heads, @ Wo^T
rotate_half is a fixed [head_dim, head_dim] matmul so the whole forward stays on device.

Batch is read from the activation ([B, S, hidden]); cos/sin may be torch (the HF capture) or ttnn
([B, S, head_dim], from the rotary stub). Passing `kv_cache` (a `KVCache`) together with `attn_ctx`
(an `AttnContext`) selects the resident-cache path the e2e pipeline uses: prefill writes the cache
and attends causally over its own keys, decode writes one position per batch row and attends over the
whole fixed-capacity cache. Every mask and position lives in a persistent device tensor, so neither
path creates a tensor from host and both are trace-capturable.
"""
from __future__ import annotations

import math
import os

import torch

import ttnn
from models.demos.minicpm5_2b._stubs.decode_mm import decode_linear_config

# Decode KV-cache write as one C++ kernel via ttnn.generic_op (else: slot-read matmuls + sub/add + paged_update_cache).
_USE_KV_WRITE_KERNEL = True
_KV_WRITE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "kv_write")
# Decode head split/merge as one data-movement kernel each (else: slice + tiled reshape on both sides).
_USE_HEAD_KERNELS = True
_HEADS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "heads")
# Decode rope as one kernel (x * cos + rotate_half(x) * sin) when the context carries full-tile cos/signed sin.
_USE_ROPE_KERNEL = True
_ROPE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "rope")
# Decode score scale + key mask as one kernel when the context carries the row-broadcast mask.
_USE_SCALE_MASK_KERNEL = True
_SCALE_MASK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "scale_mask")


def _rotate_half_matrix(head_dim):
    # x @ R == cat(-x[..., d/2:], x[..., :d/2])
    half = head_dim // 2
    rot = torch.zeros(head_dim, head_dim)
    rot[half:, :half] = -torch.eye(half)
    rot[:half, half:] = torch.eye(half)
    return rot


class KVCache:
    """One layer's resident cache: k and v, each [B, n_kv_heads, capacity, head_dim] on device."""

    def __init__(self, device, batch, n_kv_heads, capacity, head_dim):
        zeros = torch.zeros(batch, n_kv_heads, capacity, head_dim)
        self.k = ttnn.from_torch(zeros, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.v = ttnn.from_torch(zeros, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.capacity = capacity


class AttnContext:
    """Per-forward device tensors shared by every layer.

    mode "prefill": `mask` is the additive causal mask [1, 1, S, S].
    mode "decode":  `mask` is the additive key mask [B, 1, 1, capacity] (keys past each row's position
                    are -inf), `write` the one-hot [B, n_kv, 1, capacity] of each row's write slot and
                    `cur_pos` that slot as int32 [B], which paged_update_cache writes in place.
    """

    def __init__(self, mode, mask, write=None, cur_pos=None, rope=None, mask_full=None):
        self.mode, self.mask, self.write, self.cur_pos = mode, mask, write, cur_pos
        # decode: `mask` broadcast over the tile rows, [B, 1, 32, capacity], for the scale + mask kernel.
        self.mask_full = mask_full
        # decode: (cos, signed sin) [B, 1, 32, D] broadcast over the tile rows, sin negated on the first
        # half of D (rotate_half's sign), for the one-kernel rope.
        self.rope = rope


class TtAttention:
    def __init__(self, device, torch_module) -> None:
        cfg = torch_module.config
        self.device = device
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.kv_groups = self.n_heads // self.n_kv_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.n_heads
        self.scale = self.head_dim**-0.5
        self.layer_idx = getattr(torch_module, "layer_idx", 0)

        self.compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            # packer_l1_acc with fp32 dest accumulation intermittently stalled the final readback here.
            packer_l1_acc=False,
        )

        def _w(linear):
            return ttnn.from_torch(
                linear.weight.detach().to(torch.float32).t().contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.wq = _w(torch_module.q_proj)
        self.wk = _w(torch_module.k_proj)
        self.wv = _w(torch_module.v_proj)
        # [Wq | Wk | Wv] for the cached prefill: one projection whose output nlp_create_qkv_heads splits.
        projs = (torch_module.q_proj, torch_module.k_proj, torch_module.v_proj)
        self.wqkv = ttnn.from_torch(
            torch.cat([m.weight.detach().to(torch.float32) for m in projs]).t().contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Decode copy with the columns grouped per KV head, [q heads of group g | k head g] for each g, then V:
        # q and k then split as one [B, kv, groups + 1, D] tensor and are roped together.
        d, g = self.head_dim, self.kv_groups
        wq, wk, wv = (m.weight.detach().to(torch.float32) for m in projs)
        grouped = []
        for h in range(self.n_kv_heads):
            grouped += [wq[h * g * d : (h + 1) * g * d], wk[h * d : (h + 1) * d]]
        self.wqkv_dec = ttnn.from_torch(
            torch.cat(grouped + [wv]).t().contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.wo = _w(torch_module.o_proj)
        # Decode (one tile row) projections: 1-D multicast with a wide K block (bit-identical to the default).
        self.cfg_qkv = decode_linear_config(device, self.wqkv.shape[-2], self.wqkv.shape[-1])
        self.cfg_o = decode_linear_config(device, self.wo.shape[-2], self.wo.shape[-1])
        self.rot = ttnn.from_torch(
            _rotate_half_matrix(self.head_dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        # The score scale as ttnn.multiply(scores, scale) sees it: a bf16 tile, made by that same scalar multiply.
        ones = ttnn.from_torch(torch.ones(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.scale_tile = ttnn.multiply(ones, self.scale)

    def _to_tt(self, t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

    def _heads(self, x, n_heads, seq):
        # [B, S, n*D] -> [B, n, S, D]
        x = ttnn.reshape(x, (x.shape[0], seq, n_heads, self.head_dim))
        return ttnn.permute(x, (0, 2, 1, 3))

    def _cos_sin(self, position_embeddings, batch, seq):
        cos, sin = position_embeddings
        if isinstance(cos, ttnn.Tensor):
            return ttnn.reshape(cos, (batch, 1, seq, self.head_dim)), ttnn.reshape(sin, (batch, 1, seq, self.head_dim))
        cb = cos.shape[0] if cos.dim() == 3 else 1
        cos = ttnn.reshape(self._to_tt(cos), (cb, 1, seq, self.head_dim))
        sin = ttnn.reshape(self._to_tt(sin), (cb, 1, seq, self.head_dim))
        return cos, sin

    def _cached(self, x, cos, sin, kv_cache, ctx, b, seq):
        """Resident-cache attention: prefill on [B, S, hidden], one decode step on [1, B, hidden]."""
        if ctx.mode == "prefill":
            # Batch folded into the rows: one M = B*S fused-QKV matmul reading its weight once, then the
            # head split [B, 1, S, (n + 2 kv) * D] -> q [B, n, S, D], k/v [B, kv, S, D] in one op.
            qkv = self._prefill_mm(ttnn.reshape(x, (1, b * seq, x.shape[-1])), self.wqkv)
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                ttnn.reshape(qkv, (b, 1, seq, qkv.shape[-1])),
                num_heads=self.n_heads,
                num_kv_heads=self.n_kv_heads,
                transpose_k_heads=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            # The rope products ([B, heads, S, D], 16 MB for q) feed only the add: L1. Its output feeds the score
            # bmm (large static CBs), so it stays in DRAM.
            dram, l1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
            q = self._rope(q, cos, sin, self._rope_cfg(q), memory_config=dram, products_memory_config=l1)
            k = self._rope(k, cos, sin, self._rope_cfg(k), memory_config=dram, products_memory_config=l1)
            pad = kv_cache.capacity - seq
            ttnn.copy(ttnn.pad(k, ((0, 0), (0, 0), (0, pad), (0, 0)), 0.0) if pad else k, kv_cache.k)
            ttnn.copy(ttnn.pad(v, ((0, 0), (0, 0), (0, pad), (0, 0)), 0.0) if pad else v, kv_cache.v)
            # GQA without repeating K/V: head h = g * groups + j, so [B, n, S, D] viewed as [B, kv, groups * S, D]
            # puts each KV head's query heads in its rows; every output tile is the same dot product as before.
            s_t, d_t = seq // 32, self.head_dim // 32
            # The scale and mask outputs ([B, n, S, S], 16 MB each at B=32, S=128) live in L1 between the two
            # bmms; the bmms' outputs stay in DRAM, as their per-(batch, KV head) blocks fill most of L1 with CBs.
            l1 = ttnn.L1_MEMORY_CONFIG
            scores = ttnn.matmul(
                ttnn.reshape(q, (b, self.n_kv_heads, self.kv_groups * seq, self.head_dim)),
                k,
                transpose_b=True,
                program_config=self._bmm_cfg(self.kv_groups * s_t, d_t, s_t),
                compute_kernel_config=self.compute_cfg,
            )
            scaled = ttnn.multiply(ttnn.reshape(scores, (b, self.n_heads, seq, seq)), self.scale, memory_config=l1)
            ttnn.deallocate(scores)
            masked = ttnn.add(scaled, ctx.mask, memory_config=l1)
            ttnn.deallocate(scaled)
            probs = ttnn.softmax(
                masked, dim=-1, compute_kernel_config=self.compute_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            ttnn.deallocate(masked)
            attn = ttnn.matmul(
                ttnn.reshape(probs, (b, self.n_kv_heads, self.kv_groups * seq, seq)),
                v,
                program_config=self._bmm_cfg(self.kv_groups * s_t, s_t, d_t),
                compute_kernel_config=self.compute_cfg,
            )
            attn = ttnn.reshape(attn, (b, self.n_heads, seq, self.head_dim))
            attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [B, 1, S, n*D]
            attn = ttnn.reshape(attn, (1, b * seq, self.n_heads * self.head_dim))
            out = self._prefill_mm(attn, self.wo)
            return ttnn.reshape(out, (b, seq, out.shape[-1]))
        else:
            # One fused-QKV projection for the B decode rows, sliced into q / k / v. Every decode intermediate
            # is a few hundred tiles, so it lives in L1 instead of DRAM.
            l1 = ttnn.L1_MEMORY_CONFIG
            qkv = ttnn.linear(
                x, self.wqkv_dec, compute_kernel_config=self.compute_cfg, program_config=self.cfg_qkv, memory_config=l1
            )
            nq, nkv = self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim
            g, d = self.kv_groups, self.head_dim
            # One token per row: the grouped columns split straight into [B, kv, groups + 1, D] (rows 0..groups-1
            # the query heads h = kv * groups + j, row `groups` that KV head's key), so q and k share one rope and
            # the cache is attended per KV head with no repeat. The key row's scores are computed and discarded.
            if _USE_HEAD_KERNELS and b == 32:
                qk = self._heads_op("head_split", qkv, (b, self.n_kv_heads, g + 1, d), [g + 1], l1)
            else:
                qk = ttnn.slice(qkv, (0, 0, 0), (1, b, nq + nkv), memory_config=l1)
                qk = ttnn.reshape(qk, (b, self.n_kv_heads, g + 1, d))
            if _USE_ROPE_KERNEL and ctx.rope is not None:
                qk = self._rope_kernel(qk, *ctx.rope, l1)
            else:
                qk = self._rope(qk, cos, sin, self._rope_cfg(qk), memory_config=l1)
            d_t, c_t = self.head_dim // 32, kv_cache.capacity // 32
            if _USE_KV_WRITE_KERNEL:
                # One kernel writes slot pos[b] of both caches with old + (new - old), V read straight from qkv.
                self._kv_write(qk, qkv, kv_cache, ctx.cur_pos, b, k_row=g)
            else:
                k = ttnn.slice(qk, (0, 0, g, 0), (b, self.n_kv_heads, g + 1, d))
                v = ttnn.slice(qkv, (0, 0, nq + nkv), (1, b, nq + 2 * nkv))
                v = ttnn.reshape(v, (b, self.n_kv_heads, 1, self.head_dim))
                # Write only row b's slot pos[b] in place with old + (new - old): the value the one-hot blend
                # cache + onehot * (new - cache) leaves there, without touching the rest of the cache.
                shard = self._decode_kv_shard(b)
                for new_kv, cache in ((k, kv_cache.k), (v, kv_cache.v)):
                    # onehot [B, kv, 1, C] @ cache: the slot's current value, exactly (one non-zero term).
                    old = ttnn.matmul(
                        ctx.write,
                        cache,
                        program_config=self._bmm_cfg(1, c_t, d_t),
                        compute_kernel_config=self.compute_cfg,
                    )
                    new_kv = ttnn.add(old, ttnn.subtract(new_kv, old))
                    new_kv = ttnn.to_memory_config(
                        ttnn.reshape(new_kv, (1, b, self.n_kv_heads, self.head_dim)), shard
                    )
                    ttnn.experimental.paged_update_cache(cache, new_kv, update_idxs_tensor=ctx.cur_pos)
            scores = ttnn.matmul(
                qk,
                kv_cache.k,
                transpose_b=True,
                program_config=self._bmm_cfg(1, d_t, c_t),
                compute_kernel_config=self.compute_cfg,
                memory_config=l1,
            )
            if _USE_SCALE_MASK_KERNEL and ctx.mask_full is not None:
                scores = self._scale_mask_kernel(scores, ctx.mask_full, l1)
            else:
                scores = ttnn.add(ttnn.multiply(scores, self.scale, memory_config=l1), ctx.mask, memory_config=l1)
            probs = ttnn.softmax(scores, dim=-1, compute_kernel_config=self.compute_cfg, memory_config=l1)
            attn = ttnn.matmul(
                probs,
                kv_cache.v,
                program_config=self._bmm_cfg(1, c_t, d_t),
                compute_kernel_config=self.compute_cfg,
                memory_config=l1,
            )
            if _USE_HEAD_KERNELS and b == 32:
                attn = self._heads_op("head_merge", attn, (1, b, self.n_heads * d), [g, d // 32, b], l1)
            else:
                attn = ttnn.slice(attn, (0, 0, 0, 0), (b, self.n_kv_heads, g, d), memory_config=l1)
                attn = ttnn.reshape(attn, (1, b, self.n_heads * self.head_dim))
        return ttnn.linear(
            attn,
            self.wo,
            compute_kernel_config=self.compute_cfg,
            memory_config=l1,
            program_config=self.cfg_o,
        )

    def _rope_kernel(self, x, cos_full, sin_signed, memory_config):
        """Decode rope on x [B, n_kv, rows, D] as one generic_op (kernels/rope), bit-identical to _rope."""
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(x.shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device, memory_config
        )
        n_tiles = math.prod(x.padded_shape) // (32 * 32)
        cores, rt = self._spread(n_tiles)
        tile = 2 * 32 * 32
        cb_x, cb_cos, cb_xr, cb_sin, cb_t1, cb_t2, cb_out = 0, 1, 2, 3, 4, 5, 16
        cbs = [
            ttnn.CBDescriptor(
                total_size=2 * tile,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(idx, ttnn.bfloat16, tile)],
            )
            for idx in (cb_x, cb_cos, cb_xr, cb_sin, cb_t1, cb_t2, cb_out)
        ]
        compute_cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False
        )
        # As binary_ng sets for its SFPU multiply operands; the FPU add operands keep the default.
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for idx in (cb_x, cb_cos, cb_xr, cb_sin):
            modes[idx] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_cfg.unpack_to_dest_mode = modes
        d_t = self.head_dim // 32
        reader = ttnn.KernelDescriptor(
            kernel_source=os.path.join(_ROPE_DIR, "reader_rope.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[self.n_kv_heads, d_t, cb_x, cb_cos, cb_xr, cb_sin]
            + list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(cos_full).get_compile_time_args()),
            runtime_args=rt["dm"],
            common_runtime_args=[x.buffer_address(), cos_full.buffer_address(), sin_signed.buffer_address()],
            config=ttnn.ReaderConfigDescriptor(),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=os.path.join(_ROPE_DIR, "writer_rope.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[cb_out] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
            runtime_args=rt["dm"],
            common_runtime_args=[out.buffer_address()],
            config=ttnn.WriterConfigDescriptor(),
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=os.path.join(_ROPE_DIR, "compute_rope.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[cb_x, cb_cos, cb_xr, cb_sin, cb_t1, cb_t2, cb_out],
            runtime_args=rt["cmp"],
            config=compute_cfg,
        )
        program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
        return ttnn.generic_op([x, cos_full, sin_signed, out], program)

    def _scale_mask_kernel(self, scores, mask_full, memory_config):
        """Decode scores * scale + mask as one generic_op (kernels/scale_mask), bit-identical to the two ops."""
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(scores.shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device, memory_config
        )
        n_tiles = math.prod(out.padded_shape) // (32 * 32)
        cores, rt = self._spread(n_tiles)
        tile = 2 * 32 * 32
        cb_s, cb_scale, cb_mask, cb_t, cb_out = 0, 1, 2, 3, 16
        cbs = [
            ttnn.CBDescriptor(
                total_size=2 * tile,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(idx, ttnn.bfloat16, tile)],
            )
            for idx in (cb_s, cb_scale, cb_mask, cb_t, cb_out)
        ]
        compute_cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False
        )
        # As binary_ng sets for its SFPU multiply operands; the FPU add operands keep the default.
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for idx in (cb_s, cb_scale):
            modes[idx] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_cfg.unpack_to_dest_mode = modes
        c_t = scores.padded_shape[-1] // 32
        reader = ttnn.KernelDescriptor(
            kernel_source=os.path.join(_SCALE_MASK_DIR, "reader_scale_mask.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[self.n_kv_heads, c_t, cb_s, cb_scale, cb_mask]
            + list(ttnn.TensorAccessorArgs(scores).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(self.scale_tile).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(mask_full).get_compile_time_args()),
            runtime_args=rt["dm"],
            common_runtime_args=[scores.buffer_address(), self.scale_tile.buffer_address(), mask_full.buffer_address()],
            config=ttnn.ReaderConfigDescriptor(),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=os.path.join(_ROPE_DIR, "writer_rope.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[cb_out] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
            runtime_args=rt["dm"],
            common_runtime_args=[out.buffer_address()],
            config=ttnn.WriterConfigDescriptor(),
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=os.path.join(_SCALE_MASK_DIR, "compute_scale_mask.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[cb_s, cb_scale, cb_mask, cb_t, cb_out],
            runtime_args=rt["cmp"],
            config=compute_cfg,
        )
        program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
        return ttnn.generic_op([scores, self.scale_tile, mask_full, out], program)

    def _spread(self, n_tiles):
        """Row-major core set over the grid for n_tiles work items, with [start, count] / [count] runtime args."""
        grid = self.device.compute_with_storage_grid_size()
        n_cores = min(n_tiles, grid.x * grid.y)
        base, extra = divmod(n_tiles, n_cores)
        full_rows, rem = divmod(n_cores, grid.x)
        ranges = []
        if full_rows:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, full_rows - 1)))
        if rem:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
        rt = {"dm": ttnn.RuntimeArgs(), "cmp": ttnn.RuntimeArgs()}
        start = 0
        for i in range(n_cores):
            count = base + (1 if i < extra else 0)
            rt["dm"][i % grid.x][i // grid.x] = [start, count]
            rt["cmp"][i % grid.x][i // grid.x] = [count]
            start += count
        return ttnn.CoreRangeSet(ranges), rt

    def _heads_op(self, kernel, src, out_shape, args, memory_config):
        """Decode head split/merge (kernels/heads): a bf16 row-half gather into `out_shape` tiles, spread over
        the grid. head_split args: [rows]; head_merge args: [groups, Dt, B]."""
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(out_shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device, memory_config
        )
        n_tiles = math.prod(out.padded_shape) // (32 * 32)
        grid = self.device.compute_with_storage_grid_size()
        n_cores = min(n_tiles, grid.x * grid.y)
        base, extra = divmod(n_tiles, n_cores)
        full_rows, rem = divmod(n_cores, grid.x)
        ranges = []
        if full_rows:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, full_rows - 1)))
        if rem:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
        cores = ttnn.CoreRangeSet(ranges)
        rt, start = ttnn.RuntimeArgs(), 0
        for i in range(n_cores):
            count = base + (1 if i < extra else 0)
            rt[i % grid.x][i // grid.x] = [start, count]
            start += count
        tile, cb_tile = 2 * 32 * 32, 0
        if kernel == "head_split":
            ct = [self.n_kv_heads, args[0], self.head_dim // 32, cb_tile]
        else:
            ct = [self.n_kv_heads, *args, cb_tile]
        reader = ttnn.KernelDescriptor(
            kernel_source=os.path.join(_HEADS_DIR, f"{kernel}.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ct
            + list(ttnn.TensorAccessorArgs(src).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
            runtime_args=rt,
            common_runtime_args=[src.buffer_address(), out.buffer_address()],
            config=ttnn.ReaderConfigDescriptor(),
        )
        cbs = [
            ttnn.CBDescriptor(
                total_size=tile,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(cb_tile, ttnn.bfloat16, tile)],
            )
        ]
        program = ttnn.ProgramDescriptor(kernels=[reader], semaphores=[], cbs=cbs)
        return ttnn.generic_op([src, out], program)

    def _prefill_mm(self, x, w):
        """Prefill projection on minimal_matmul's small streaming CBs, which span the full grid under fp32
        accumulation. K_block is all of K, so each tile is one uninterrupted K-order accumulation as in ttnn.linear."""
        grid = self.device.compute_with_storage_grid_size()
        return ttnn.experimental.minimal_matmul(
            x,
            w,
            config=ttnn.MinimalMatmulConfig(
                M_block_size=2,
                K_block_size=w.shape[-2] // 32,
                N_block_size=2,
                subblock_h=2,
                subblock_w=2,
                compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_cfg,
        )

    def _kv_write(self, k, qkv, kv_cache, cur_pos, batch, k_row=0):
        """Decode KV-cache write as one generic_op: per (cache, b, kv head) unit, the slot row pos[b] of
        cache[b, h] becomes old + (new - old) (binary_ng's default bf16 FPU sub/add), every other row untouched."""
        grid = self.device.compute_with_storage_grid_size()
        n_kv, d_t = self.n_kv_heads, self.head_dim // 32
        units = 2 * batch * n_kv
        n_cores = min(units, grid.x * grid.y)
        base, extra = divmod(units, n_cores)
        full_rows, rem = divmod(n_cores, grid.x)
        ranges = []
        if full_rows:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, full_rows - 1)))
        if rem:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
        cores = ttnn.CoreRangeSet(ranges)

        dm_rt, cmp_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        start = 0
        for i in range(n_cores):
            count = base + (1 if i < extra else 0)
            dm_rt[i % grid.x][i // grid.x] = [start, count]
            cmp_rt[i % grid.x][i // grid.x] = [count]
            start += count

        tile = 2 * 32 * 32
        cb_old, cb_new, cb_scr, cb_pos, cb_diff, cb_out, cb_wscr, cb_wpos = 0, 1, 2, 3, 4, 16, 17, 18

        def cb(idx, n_tiles):
            return ttnn.CBDescriptor(
                total_size=n_tiles * tile,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(idx, ttnn.bfloat16, tile)],
            )

        # A unit's d_t tiles move as one batch: old/new double-buffered per unit, d_t-tile scratch on both movers.
        cbs = [cb(cb_old, 2 * d_t), cb(cb_new, 2 * d_t), cb(cb_scr, d_t), cb(cb_pos, 1), cb(cb_diff, 1)]
        cbs += [cb(cb_out, 2), cb(cb_wscr, d_t), cb(cb_wpos, 1)]
        shape = (batch, n_kv, kv_cache.capacity // 32, d_t, qkv.padded_shape[-1] // 32)
        cache_ct = list(ttnn.TensorAccessorArgs(kv_cache.k).get_compile_time_args())
        pos_ct = list(ttnn.TensorAccessorArgs(cur_pos).get_compile_time_args())
        addrs = [kv_cache.k, kv_cache.v, k, qkv, cur_pos]
        common = [t.buffer_address() for t in addrs]
        v_col0 = (self.n_heads + n_kv) * self.head_dim // 32
        reader = ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KV_WRITE_DIR, "reader_kv_write.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[*shape, v_col0, cb_old, cb_new, cb_scr, cb_pos, k_row]
            + cache_ct
            + list(ttnn.TensorAccessorArgs(k).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(qkv).get_compile_time_args())
            + pos_ct,
            runtime_args=dm_rt,
            common_runtime_args=common,
            config=ttnn.ReaderConfigDescriptor(),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KV_WRITE_DIR, "writer_kv_write.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[*shape[:4], cb_out, cb_wscr, cb_wpos] + cache_ct + pos_ct,
            runtime_args=dm_rt,
            common_runtime_args=common,
            config=ttnn.WriterConfigDescriptor(),
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KV_WRITE_DIR, "compute_kv_write.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[d_t, cb_old, cb_new, cb_out, cb_diff],
            runtime_args=cmp_rt,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
            ),
        )
        program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
        # The caches are updated in place; generic_op's last tensor is its (here: in-place) output.
        ttnn.generic_op([k, qkv, cur_pos, kv_cache.k, kv_cache.v], program)

    def _decode_kv_shard(self, batch):
        """Height-shard spec for one token's K/V [1, B, n_kv, D]: one batch row (a padded tile row) per core."""
        grid = self.device.compute_with_storage_grid_size()
        return ttnn.create_sharded_memory_config(
            shape=(32, self.head_dim),
            core_grid=ttnn.num_cores_to_corerangeset(batch, grid, True),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _rope_cfg(self, x):
        """Rotate-half x @ rot with x's batch/head dims folded into M, spread over the grid.

        in0_block_w spans all of K, so each tile is one uninterrupted accumulation."""
        grid = self.device.compute_with_storage_grid_size()
        cores, d_t = grid.x * grid.y, self.head_dim // 32
        m_t = math.prod(x.padded_shape) // (self.head_dim * 32)
        per_core_m = next(d for d in range(1, m_t + 1) if m_t % d == 0 and m_t // d <= cores)
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=d_t,
            out_subblock_h=1,
            out_subblock_w=d_t,
            per_core_M=per_core_m,
            per_core_N=d_t,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )

    def _bmm_cfg(self, m_t, k_t, n_t):
        """Attention bmm with a batched second operand: one [m_t, n_t]-tile output block per (batch, head),
        blocks spread over the grid, in0_block_w spanning all of K so each tile is one uninterrupted accumulation."""
        grid = self.device.compute_with_storage_grid_size()
        return ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=k_t,
            out_subblock_h=1,
            out_subblock_w=max(w for w in range(1, 5) if n_t % w == 0),
            per_core_M=m_t,
            per_core_N=n_t,
        )

    def _rope(self, x, cos, sin, program_config=None, memory_config=None, products_memory_config=None):
        """x * cos + (x @ rot) * sin. The two products (consumed only by the add) may go to their own memory."""
        x_rot = ttnn.matmul(
            x,
            self.rot,
            program_config=program_config,
            compute_kernel_config=self.compute_cfg,
            memory_config=memory_config,
        )
        pmem = products_memory_config or memory_config
        x_cos = ttnn.multiply(x, cos, memory_config=pmem)
        rot_sin = ttnn.multiply(x_rot, sin, memory_config=pmem)
        ttnn.deallocate(x_rot)
        out = ttnn.add(x_cos, rot_sin, memory_config=memory_config)
        ttnn.deallocate(x_cos)
        ttnn.deallocate(rot_sin)
        return out

    def _past_kv(self, past_key_values):
        if past_key_values is None:
            return None
        layers = getattr(past_key_values, "layers", None)
        if layers is None or self.layer_idx >= len(layers):
            return None
        layer = layers[self.layer_idx]
        keys, values = getattr(layer, "keys", None), getattr(layer, "values", None)
        if keys is None or values is None or keys.numel() == 0:
            return None
        return self._to_tt(keys), self._to_tt(values), keys.shape[2]

    def __call__(
        self, hidden_states, position_embeddings=None, past_key_values=None, kv_cache=None, attn_ctx=None, **kwargs
    ):
        x = hidden_states
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (x.shape[0] * x.shape[1], x.shape[-2], x.shape[-1]))
        b, seq = x.shape[0], x.shape[-2]
        if kv_cache is not None:
            if attn_ctx.mode == "decode":
                # Decode rows arrive as [1, B, hidden] so every linear sees B real rows, not B padded tiles.
                b, seq = x.shape[-2], 1
            cos, sin = self._cos_sin(position_embeddings, b, seq)
            return self._cached(x, cos, sin, kv_cache, attn_ctx, b, seq), None

        q = ttnn.linear(x, self.wq, compute_kernel_config=self.compute_cfg)
        k = ttnn.linear(x, self.wk, compute_kernel_config=self.compute_cfg)
        v = ttnn.linear(x, self.wv, compute_kernel_config=self.compute_cfg)
        q = self._heads(q, self.n_heads, seq)
        k = self._heads(k, self.n_kv_heads, seq)
        v = self._heads(v, self.n_kv_heads, seq)

        cos, sin = self._cos_sin(position_embeddings, b, seq)
        q = self._rope(q, cos, sin)
        k = self._rope(k, cos, sin)

        # Explicit GQA attention for both prefill and decode (SDPA's default chunking hung at S=64).
        past = self._past_kv(past_key_values)
        past_len = 0
        if past is not None:
            past_k, past_v, past_len = past
            k = ttnn.concat([past_k, k], dim=2)
            v = ttnn.concat([past_v, v], dim=2)
        k = ttnn.repeat_interleave(k, self.kv_groups, dim=1)
        v = ttnn.repeat_interleave(v, self.kv_groups, dim=1)
        scores = ttnn.matmul(q, k, transpose_b=True, compute_kernel_config=self.compute_cfg)
        scores = ttnn.multiply(scores, self.scale)
        if seq > 1:
            # Query i sits at absolute position past_len + i and sees keys 0..past_len + i.
            total = past_len + seq
            mask = ttnn.full((1, 1, seq, total), -1e9, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
            mask = ttnn.triu(mask, diagonal=past_len + 1)
            scores = ttnn.add(scores, mask)
            ttnn.deallocate(mask)
        probs = ttnn.softmax(scores, dim=-1, compute_kernel_config=self.compute_cfg)
        attn = ttnn.matmul(probs, v, compute_kernel_config=self.compute_cfg)
        for t in (q, k, v):
            ttnn.deallocate(t)

        attn = ttnn.permute(attn, (0, 2, 1, 3))
        attn = ttnn.reshape(attn, (b, seq, self.n_heads * self.head_dim))
        out = ttnn.linear(attn, self.wo, compute_kernel_config=self.compute_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn)
        return out, None

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)


# Module-level `build` — primary test entry point.
def build(device, torch_module=None):
    return TtAttention.build(device, torch_module)


# Module-level shim with the component's lowercase slug name.
def attention(device, torch_module=None):
    return TtAttention.build(device, torch_module)

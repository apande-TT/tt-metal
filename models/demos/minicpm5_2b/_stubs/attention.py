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
from models.demos.minicpm5_2b._stubs.dram_mm import DramMatmul, _core_rects

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
# Decode q.K^T and probs.V against the caches as a generic_op spreading output tiles over the whole grid
# (the reuse bmm gives each (batch, KV head) row to one core: 64 of the cores).
_USE_ATTN_BMM_KERNEL = True
# Decode: the score kernel applies the scale + key mask itself (no separate scale_mask op).
_FUSE_SCALE_MASK = True
_ATTN_BMM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "attn_bmm")
# Decode q.K^T (+ scale + mask), softmax and probs.V (+ head merge) as ONE generic_op, one (batch, KV head) row
# block per core, bit-identical to the three ops (kernels/attn_fused).
_USE_FUSED_DECODE_ATTN = True
# Decode QKV / o_proj on the DRAM-bank-local generic_op (_stubs/dram_mm.py).
_USE_DRAM_MM = True
# Decode rope on the QKV projection's own layout (72 real tiles, no head split): the fused attention gathers its
# head rows from it and the KV write takes its key rows from it.
_USE_QKV_ROPE = True
# ... with the layer's input RMSNorm fused into the QKV program (bind_prenorm) as a prologue on helper cores
# (DramMatmul norm_prologue). On the QKV cores with an all-gather it did not pay (16 cores: 39.7 us; 40 cores:
# 38.3 us, the all-gather to 40 cores alone ~18 us; vs 11.7 + 25.7 us unfused).
_USE_DRAM_MM_NORM = True
# QKV is compute-bound (HiFi4) once its norm prologue is done: 3 workers per bank (24 cores) beat 2 (+ the norm
# prologue gives the stream a head start), 4 and 5 lose to bank contention.
_QKV_DMM = dict(workers_per_bank=3)
# The decode KV-cache write done by each fused-attention core for its own (batch, KV head) instead of 46 extra
# cores behind semaphores (the kv_write arithmetic under fp32 DEST: e2e bit-identical).
_INLINE_KV_WRITE = True
# Decode o_proj with the layer's residual add on helper cores of the same program (DramMatmul residual_add).
_USE_DRAM_MM_RES = True
_ATTN_FUSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "attn_fused")
# Prefill score bmm + scale/mask + softmax + probs.V + head merge as ONE generic_op (kernels/attn_prefill): the
# attn_fused arithmetic per query tile row, the (batch, KV head) block's K / V resident in L1 across its rows.
_USE_FUSED_PREFILL_ATTN = True
# Prefill Q and K|V as two full-grid 2-D multicast matmuls (Q on the o_proj config) instead of one 100-core fused QKV.
_SPLIT_PREFILL_QKV = True
_ATTN_PREFILL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "attn_prefill")


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

    def __init__(self, mode, mask, write=None, cur_pos=None, rope=None, mask_full=None, rope_rows=None):
        self.mode, self.mask, self.write, self.cur_pos = mode, mask, write, cur_pos
        # decode: (cos, signed sin) [1, 1, B, D] (row b = batch b's position), for the rope on the QKV layout.
        self.rope_rows = rope_rows
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
        # [Wk | Wv] for the split prefill projection (q on wq, k / v on this; nlp_create_qkv_heads' input_kv).
        self.wkv = ttnn.from_torch(
            torch.cat([m.weight.detach().to(torch.float32) for m in projs[1:]]).t().contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Decode (one tile row) projections: 1-D multicast with a wide K block (bit-identical to the default).
        self.cfg_qkv = decode_linear_config(device, self.wqkv.shape[-2], self.wqkv.shape[-1], per_core_n=2)  # 80 tiles on 40 cores
        self.cfg_o = decode_linear_config(device, self.wo.shape[-2], self.wo.shape[-1], per_core_n=2)  # 64 tiles on 32 cores
        # The same decode projections on the DRAM-bank-local generic_op (same arithmetic as the 1-D multicast).
        self.dmm_qkv = self.dmm_o = self.dmm_o_res = None
        if _USE_DRAM_MM:
            fid = ttnn.MathFidelity.HiFi4  # as self.compute_cfg
            self.dmm_qkv = DramMatmul(device, torch.cat(grouped + [wv]).t().contiguous(), fid, **_QKV_DMM)
            wo_t = torch_module.o_proj.weight.detach().to(torch.float32).t().contiguous()
            self.dmm_o = DramMatmul(device, wo_t, fid)
            # ... and o_proj + the layer's residual add in one program (ttnn.add's arithmetic on helper cores).
            if _USE_DRAM_MM_RES and self.dmm_o.ok:
                dmm = DramMatmul(device, None, fid, prebuilt=self.dmm_o, residual_add=True)
                self.dmm_o_res = dmm if dmm.ok else None
        self.dmm_qkv_norm = None  # bind_prenorm: the input norm fused into the QKV program
        self.rot = ttnn.from_torch(
            _rotate_half_matrix(self.head_dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        # The score scale as ttnn.multiply(scores, scale) sees it: a bf16 tile, made by that same scalar multiply.
        ones = ttnn.from_torch(torch.ones(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.scale_tile = ttnn.multiply(ones, self.scale)
        # Stand-in for cache tiles past a row's position tile in the decode attention bmms (see _attn_bmm).
        self.zero_tile = ttnn.from_torch(
            torch.zeros(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        # ttnn softmax's reduce scalers as its reader builds them: 1.0 in row 0 of every face (tile rows 0 and 16),
        # bf16 for the max (the input's format) and fp32 for the sum (the fp32 intermediates' format).
        scaler = torch.zeros(1, 1, 32, 32)
        scaler[..., [0, 16], :] = 1.0
        self.max_scaler = ttnn.from_torch(scaler, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.sum_scaler = ttnn.from_torch(scaler, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

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
            dram, l1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
            rope_kernel = _USE_ROPE_KERNEL and ctx.rope is not None
            # With the rope kernel the fused-QKV output only feeds the head split: L1.
            grid = self.device.compute_with_storage_grid_size()
            x2 = ttnn.reshape(x, (1, b * seq, x.shape[-1]))
            if _SPLIT_PREFILL_QKV:
                # Q on o_proj's full-grid 2-D multicast config (same shape), K | V as a narrow second matmul: every
                # output tile is the same full-K accumulation as in the fused-QKV matmul, on all 110 cores.
                q_lin = ttnn.linear(
                    x2, self.wq, memory_config=l1 if rope_kernel else dram, compute_kernel_config=self.compute_cfg,
                    program_config=self._prefill_2d_cfg(x2, self.wq, 8, 3),
                )
                kv_lin = ttnn.linear(
                    x2, self.wkv, memory_config=l1 if rope_kernel else dram, compute_kernel_config=self.compute_cfg,
                    program_config=self._prefill_2d_cfg(x2, self.wkv, 8, 2),
                )
            else:
                qkv = ttnn.linear(
                    x2,
                    self.wqkv,
                    memory_config=l1 if rope_kernel else dram,
                    compute_kernel_config=self.compute_cfg,
                    core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x),
                )
            if x.memory_config().buffer_type == ttnn.BufferType.L1:
                ttnn.deallocate(x)  # an L1 norm output: free it before the score bmm's large static CBs
            # With the rope kernel (small CBs) the head split lands in L1: q / k are freed once roped and v moves
            # to DRAM (it would otherwise sit below the freed q / k, inside the score bmm's static CB region).
            if _SPLIT_PREFILL_QKV:
                q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                    ttnn.reshape(q_lin, (b, 1, seq, q_lin.shape[-1])),
                    input_kv=ttnn.reshape(kv_lin, (b, 1, seq, kv_lin.shape[-1])),
                    num_heads=self.n_heads,
                    num_kv_heads=self.n_kv_heads,
                    transpose_k_heads=False,
                    memory_config=l1 if rope_kernel else dram,
                )
                ttnn.deallocate(q_lin)
                ttnn.deallocate(kv_lin)
            else:
                q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                    ttnn.reshape(qkv, (b, 1, seq, qkv.shape[-1])),
                    num_heads=self.n_heads,
                    num_kv_heads=self.n_kv_heads,
                    transpose_k_heads=False,
                    memory_config=l1 if rope_kernel else dram,
                )
                ttnn.deallocate(qkv)
            fused = _USE_FUSED_PREFILL_ATTN and _USE_SCALE_MASK_KERNEL and seq % 32 == 0 and ctx.mask.shape[-1] == seq
            fused = fused and kv_cache.capacity % 32 == 0 and kv_cache.capacity >= seq
            # The roped q / k feed the score bmm (large static CBs), so they stay in DRAM.
            if rope_kernel:
                q_in, k_in = q, k
                q = self._rope_kernel(q_in, *ctx.rope, dram)
                k = self._rope_kernel(k_in, *ctx.rope, dram)
                ttnn.deallocate(q_in)
                ttnn.deallocate(k_in)
                if not fused:  # the fused attention (small CBs) reads v from L1
                    v_l1, v = v, ttnn.to_memory_config(v, dram)
                    ttnn.deallocate(v_l1)
            else:
                # The rope products ([B, heads, S, D], 16 MB for q) feed only the add: L1.
                q = self._rope(q, cos, sin, self._rope_cfg(q), memory_config=dram, products_memory_config=l1)
                k = self._rope(k, cos, sin, self._rope_cfg(k), memory_config=dram, products_memory_config=l1)
            if fused:
                # The fused attention also stores K / V (zero-padded to capacity) into the caches, and returns
                # [1, B * S, n * D] in L1, already head-merged: o_proj's input.
                attn = self._attn_prefill_fused(q, k, v, ctx.mask, b, seq, ttnn.L1_MEMORY_CONFIG, kv_cache=kv_cache)
                for t in (q, k, v):
                    ttnn.deallocate(t)
                return self._prefill_o_proj(attn, b, seq)
            pad = kv_cache.capacity - seq
            # The capacity-padded K / V feed only the copy into the cache: L1.
            for new, cache in ((k, kv_cache.k), (v, kv_cache.v)):
                if pad:
                    padded = ttnn.pad(new, ((0, 0), (0, 0), (0, pad), (0, 0)), 0.0, memory_config=l1)
                    ttnn.copy(padded, cache)
                    ttnn.deallocate(padded)
                else:
                    ttnn.copy(new, cache)
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
            if _USE_SCALE_MASK_KERNEL:
                masked = self._scale_mask_kernel(ttnn.reshape(scores, (b, self.n_heads, seq, seq)), ctx.mask, l1)
                ttnn.deallocate(scores)
            else:
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
            # [B, 1, S, n*D] in L1: o_proj's input and output (two 16 MB L1 tensors) still fit beside its static CBs.
            attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.L1_MEMORY_CONFIG)
            attn = ttnn.reshape(attn, (1, b * seq, self.n_heads * self.head_dim))
            return self._prefill_o_proj(attn, b, seq)
        else:
            # One fused-QKV projection for the B decode rows, sliced into q / k / v. Every decode intermediate
            # is a few hundred tiles, so it lives in L1 instead of DRAM.
            l1 = ttnn.L1_MEMORY_CONFIG
            if self._fused_prenorm:
                # Input RMSNorm + QKV in one program, run through the norm's own call.
                qkv = self._prenorm(x, memory_config=l1, fused=self.dmm_qkv_norm)
            elif self.dmm_qkv is not None and self.dmm_qkv.accepts(x):
                qkv = self.dmm_qkv(x, l1)
            else:
                qkv = ttnn.linear(
                    x, self.wqkv_dec, compute_kernel_config=self.compute_cfg, program_config=self.cfg_qkv, memory_config=l1
                )
            nq, nkv = self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim
            g, d = self.kv_groups, self.head_dim
            # One token per row: the grouped columns split straight into [B, kv, groups + 1, D] (rows 0..groups-1
            # the query heads h = kv * groups + j, row `groups` that KV head's key), so q and k share one rope and
            # the cache is attended per KV head with no repeat. The key row's scores are computed and discarded.
            fused_attn = _USE_FUSED_DECODE_ATTN and _USE_HEAD_KERNELS and b == 32 and ctx.mask_full is not None
            if _USE_QKV_ROPE and fused_attn and _USE_KV_WRITE_KERNEL and ctx.rope_rows is not None:
                # Rope on the QKV layout; attention and the KV write read head rows straight from it.
                qkr = self._rope_qkv(qkv, *ctx.rope_rows, (g + 1) * self.n_kv_heads * self.head_dim, l1)
                attn = self._attn_fused(
                    qkr, kv_cache, ctx, l1, kv_write=(qkv, g), q_rows=(b, g + 1), inline_kv=_INLINE_KV_WRITE
                )
                ttnn.deallocate(qkr)
                return self._decode_o_proj(attn, l1)
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
            fused_attn = _USE_FUSED_DECODE_ATTN and _USE_HEAD_KERNELS and b == 32 and ctx.mask_full is not None
            if fused_attn and _USE_KV_WRITE_KERNEL:
                # The KV write runs inside the fused attention program, on the cores attention leaves idle.
                attn = self._attn_fused(qk, kv_cache, ctx, l1, kv_write=(qkv, g))
                return self._decode_o_proj(attn, l1)
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
            if fused_attn:
                attn = self._attn_fused(qk, kv_cache, ctx, l1)
                return self._decode_o_proj(attn, l1)
            fused_mask = _USE_ATTN_BMM_KERNEL and _FUSE_SCALE_MASK and ctx.mask_full is not None
            if _USE_ATTN_BMM_KERNEL:
                scores = self._attn_bmm(
                    qk, kv_cache.k, True, l1, mask_full=ctx.mask_full if fused_mask else None, cur_pos=ctx.cur_pos
                )
            else:
                scores = ttnn.matmul(
                    qk,
                    kv_cache.k,
                    transpose_b=True,
                    program_config=self._bmm_cfg(1, d_t, c_t),
                    compute_kernel_config=self.compute_cfg,
                    memory_config=l1,
                )
            if fused_mask:
                pass  # scaled and masked by the score kernel
            elif _USE_SCALE_MASK_KERNEL and ctx.mask_full is not None:
                scores = self._scale_mask_kernel(scores, ctx.mask_full, l1)
            else:
                scores = ttnn.add(ttnn.multiply(scores, self.scale, memory_config=l1), ctx.mask, memory_config=l1)
            probs = ttnn.softmax(scores, dim=-1, compute_kernel_config=self.compute_cfg, memory_config=l1)
            if _USE_ATTN_BMM_KERNEL and _USE_HEAD_KERNELS and b == 32:
                # probs.V with the head merge in its writer: straight to the o_proj input [1, B, n * D].
                attn = self._attn_bmm(probs, kv_cache.v, False, l1, merge_groups=g, cur_pos=ctx.cur_pos)
            elif _USE_ATTN_BMM_KERNEL:
                attn = self._attn_bmm(probs, kv_cache.v, False, l1, cur_pos=ctx.cur_pos)
            else:
                attn = ttnn.matmul(
                    probs,
                    kv_cache.v,
                    program_config=self._bmm_cfg(1, c_t, d_t),
                    compute_kernel_config=self.compute_cfg,
                    memory_config=l1,
                )
            if _USE_ATTN_BMM_KERNEL and _USE_HEAD_KERNELS and b == 32:
                pass  # already merged by the probs.V writer
            elif _USE_HEAD_KERNELS and b == 32:
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

    def _rope_qkv(self, qkv, cos_rows, sin_rows, width, memory_config):
        """Rope on the first `width` columns (the grouped q / k heads) of the one-tile-row QKV output [1, B, N],
        as _rope_kernel's program with reader_rope_qkv.cpp: tile c (head c / Dt, column tile c % Dt) with the
        step's cos / signed sin tile c % Dt ([1, 1, B, D], row b = batch b) and its rotate-half partner tile."""
        b = qkv.shape[-2]
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, b, width]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device, memory_config)
        n_tiles = width // 32
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
        compute_cfg = ttnn.ComputeConfigDescriptor(  # as _rope_kernel
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False
        )
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for idx in (cb_x, cb_cos, cb_xr, cb_sin):
            modes[idx] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_cfg.unpack_to_dest_mode = modes
        d_t = self.head_dim // 32
        reader = ttnn.KernelDescriptor(
            kernel_source=os.path.join(_ROPE_DIR, "reader_rope_qkv.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[d_t, cb_x, cb_cos, cb_xr, cb_sin]
            + list(ttnn.TensorAccessorArgs(qkv).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(cos_rows).get_compile_time_args()),
            runtime_args=rt["dm"],
            common_runtime_args=[qkv.buffer_address(), cos_rows.buffer_address(), sin_rows.buffer_address()],
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
        return ttnn.generic_op([qkv, cos_rows, sin_rows, out], program)

    def _rope_kernel(self, x, cos_full, sin_signed, memory_config):
        """Rope on x [B, H, S, D] as one generic_op (kernels/rope), bit-identical to _rope. cos_full / sin_signed
        are [B, 1, S, D] (decode: S = one tile row broadcast from the step's position)."""
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
            compile_time_args=[x.shape[1], d_t, x.padded_shape[-2] // 32, cb_x, cb_cos, cb_xr, cb_sin]
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
        """scores [B, H, S, C] * scale + mask as one generic_op (kernels/scale_mask), bit-identical to the two ops.
        mask_full is [B, 1, S, C] (decode, row-broadcast key mask) or [1, 1, S, C] (prefill causal mask)."""
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
            compile_time_args=[
                scores.shape[1],
                c_t,
                scores.padded_shape[-2] // 32,
                int(mask_full.shape[0] > 1),
                cb_s,
                cb_scale,
                cb_mask,
            ]
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

    def _attn_bmm(self, x, cache, transpose_b, memory_config, merge_groups=None, mask_full=None, cur_pos=None):
        """Decode x [B, n_kv, rows, K] @ cache^T (transpose_b, cache [B, n_kv, C, D] -> [.., C]) or @ cache
        (-> [.., D]) as one generic_op (kernels/attn_bmm), bit-identical to the reuse bmm with a full-K block.
        With merge_groups (probs.V only) the writer scatters rows 0..groups-1 straight into the merged
        [1, B, n_kv * groups * D] (the head merge). With mask_full (scores only, [B, 1, 32, C]) the kernel also
        applies _scale_mask_kernel's scale + mask to each packed score tile on the same core.
        With cur_pos (int32 [B], one page) the reader stands a resident zero tile in for every cache tile past row b's
        position tile instead of reading it from DRAM: those keys are masked (score -> the mask value, prob 0), and
        unwritten slots are zero anyway, so the compute sees the inputs that leave its output unchanged."""
        fuse = mask_full is not None
        skip = cur_pos is not None
        k_t = x.padded_shape[-1] // 32
        n = cache.padded_shape[-2] if transpose_b else cache.padded_shape[-1]
        if merge_groups is None:
            out_shape = list(x.shape)[:-1] + [n]
        else:
            out_shape = [1, x.shape[0], x.shape[1] * merge_groups * n]
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape(out_shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device, memory_config
        )
        n_t = n // 32
        cores, rt = self._spread(math.prod(list(x.padded_shape)[:-1]) // 32 * n_t)
        tile = 2 * 32 * 32
        cb_in0, cb_in1, cb_scale, cb_mask, cb_t, cb_t2, cb_pos, cb_zero, cb_out = 0, 1, 2, 3, 4, 5, 6, 7, 16
        sizes = [(cb_in0, 2 * k_t), (cb_in1, 2 * k_t), (cb_out, 2)]
        if fuse:
            sizes += [(cb_scale, 1), (cb_mask, 2), (cb_t, 2), (cb_t2, 2)]
        if skip:
            sizes += [(cb_pos, 1), (cb_zero, 1)]
        cbs = [
            ttnn.CBDescriptor(
                total_size=size * tile,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(idx, ttnn.bfloat16, tile)],
            )
            for idx, size in sizes
        ]
        scale_t, mask_t = (self.scale_tile, mask_full) if fuse else (cache, cache)  # unused when not fused
        pos_t, zero_t = (cur_pos, self.zero_tile) if skip else (cache, cache)  # unused without cur_pos
        reader = ttnn.KernelDescriptor(
            kernel_source=os.path.join(_ATTN_BMM_DIR, "reader_attn_bmm.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[k_t, n_t, int(transpose_b), cb_in0, cb_in1, int(fuse), x.shape[1], cb_scale, cb_mask]
            + [int(skip), cb_pos, cb_zero]
            + list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(cache).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(scale_t).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(mask_t).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(pos_t).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(zero_t).get_compile_time_args()),
            runtime_args=rt["dm"],
            common_runtime_args=[
                x.buffer_address(),
                cache.buffer_address(),
                scale_t.buffer_address(),
                mask_t.buffer_address(),
                pos_t.buffer_address(),
                zero_t.buffer_address(),
            ],
            config=ttnn.ReaderConfigDescriptor(),
        )
        if merge_groups is None:
            writer_src, writer_ct = os.path.join(_ROPE_DIR, "writer_rope.cpp"), [cb_out]
        else:
            writer_src = os.path.join(_ATTN_BMM_DIR, "writer_attn_merge.cpp")
            writer_ct = [cb_out, x.shape[1], merge_groups, n_t]
        writer = ttnn.KernelDescriptor(
            kernel_source=writer_src,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=writer_ct + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
            runtime_args=rt["dm"],
            common_runtime_args=[out.buffer_address()],
            config=ttnn.WriterConfigDescriptor(),
        )
        compute_cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,  # as self.compute_cfg
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            dst_full_sync_en=False,
        )
        if fuse:
            # As _scale_mask_kernel sets for its SFPU multiply operands.
            modes = [ttnn.UnpackToDestMode.Default] * 64
            for idx in (cb_t, cb_scale):
                modes[idx] = ttnn.UnpackToDestMode.UnpackToDestFp32
            compute_cfg.unpack_to_dest_mode = modes
        compute = ttnn.KernelDescriptor(
            kernel_source=os.path.join(_ATTN_BMM_DIR, "compute_attn_bmm.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[k_t, int(transpose_b), cb_in0, cb_in1, cb_out, int(fuse), cb_scale, cb_mask, cb_t, cb_t2],
            runtime_args=rt["cmp"],
            config=compute_cfg,
        )
        program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
        io = [x, cache] + ([mask_full] if fuse else []) + ([cur_pos, self.zero_tile] if skip else [])
        return ttnn.generic_op(io + [out], program)

    def bind_prenorm(self, norm):
        """Fuse `norm` (the layer's input RMSNorm, HiFi4 like the QKV projection) into the decode QKV program."""
        if _USE_DRAM_MM_NORM and self.dmm_qkv is not None and self.dmm_qkv.ok and hasattr(norm, "consts"):
            dmm = DramMatmul(
                self.device, None, ttnn.MathFidelity.HiFi4, norm_prologue=norm, prebuilt=self.dmm_qkv, **_QKV_DMM
            )
            self.dmm_qkv_norm = dmm if dmm.ok else None
        return self.dmm_qkv_norm is not None

    def _decode_o_proj(self, attn, l1):
        r = self._residual
        if (
            r is not None
            and self.dmm_o_res is not None
            and self.dmm_o_res.accepts(attn)
            and list(r.padded_shape) == list(attn.padded_shape)
            and r.dtype == ttnn.bfloat16
            and not r.memory_config().is_sharded()
        ):
            self._residual_added = True
            return self.dmm_o_res(attn, l1, residual=r)
        if self.dmm_o is not None and self.dmm_o.accepts(attn):
            return self.dmm_o(attn, l1)
        return ttnn.linear(
            attn, self.wo, compute_kernel_config=self.compute_cfg, memory_config=l1, program_config=self.cfg_o
        )

    def _attn_fused(self, qk, kv_cache, ctx, memory_config, kv_write=None, q_rows=None, inline_kv=False):
        """Decode attention for qk [B, n_kv, rows, D] against the caches as one generic_op (kernels/attn_fused): per
        (batch, KV head) row block, the score kernel's q.K^T + scale + mask, ttnn.softmax's arithmetic and probs.V,
        the writer scattering the query rows straight into the merged [1, B, n_kv * groups * D] o_proj input.

        kv_write=(qkv, k_row): this step's KV-cache write (the _kv_write kernels, unchanged, bf16 DEST) joins the same
        program on the cores the row blocks leave idle; each unit bumps its row block's K or V semaphore once written,
        and the attention reader waits on it before streaming the slot's tile."""
        # q_rows=(B, G): qk is the roped QKV layout [1, B, n_kv * G * D] (the reader gathers the head rows).
        if q_rows is not None:
            b, n_kv, g = q_rows[0], self.n_kv_heads, self.kv_groups
        else:
            b, n_kv, g = qk.shape[0], qk.shape[1], self.kv_groups
        d_t, c_t = self.head_dim // 32, kv_cache.capacity // 32
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, b, n_kv * g * self.head_dim]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device, memory_config
        )
        grid = self.device.compute_with_storage_grid_size()
        n_rows = b * n_kv
        assert n_rows <= grid.x * grid.y
        full_rows, rem = divmod(n_rows, grid.x)
        ranges = []
        if full_rows:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, full_rows - 1)))
        if rem:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
        cores = ttnn.CoreRangeSet(ranges)
        sem_k, sem_v = 0, 1
        kv_kernels, kv_cbs, kv_tensors = [], [], []
        # inline_kv (with kv_write and q_rows): each row block's core does its own KV-cache write (reader + compute,
        # fp32 DEST, the kv_write arithmetic -- e2e bit-identical) instead of separate cores behind semaphores.
        inline_kv = inline_kv and kv_write is not None and q_rows is not None
        qkv_t = kv_write[0] if kv_write is not None else None
        if kv_write is not None and not inline_kv:
            qkv, k_row = kv_write
            idle = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(n_rows, grid.x * grid.y)]
            coords = []
            for i in range(n_rows):
                c = self.device.worker_core_from_logical_core(ttnn.CoreCoord(i % grid.x, i // grid.x))
                coords += [c.x, c.y]
            kv_kernels, kv_cbs, kv_tensors = self._kv_write_parts(
                qk, qkv, kv_cache, ctx.cur_pos, b, k_row, idle, signal=(sem_k, sem_v, coords),
                k_from_rows=q_rows is not None,
            )
        rd_rt, wr_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i in range(n_rows):
            rd_rt[i % grid.x][i // grid.x] = [i]
            wr_rt[i % grid.x][i // grid.x] = [i * d_t, d_t]
        bf, f32 = ttnn.bfloat16, ttnn.float32
        tile, f32_tile = 2 * 32 * 32, 4 * 32 * 32
        (cb_q, cb_k, cb_scale, cb_mask, cb_t, cb_t2, cb_pos, cb_zero, cb_sm_in, cb_mx, cb_sm, cb_v, cb_probs) = range(13)
        cb_cnt = 13  # the row's live key-tile count, reader -> compute
        cb_kv_old, cb_kv_new, cb_kv_scr, cb_kv_diff, cb_kv_out = 17, 18, 19, 20, 21  # inline_kv
        cb_wpos, kv_flag = 14, 0  # inline_kv: the writer's copy of the positions; staged tiles ready (reader -> writer)
        cb_out, cb_max, cb_exps, cb_recip = 16, 24, 25, 26
        spec = [
            (cb_q, d_t, bf),
            (cb_k, 2 * d_t, bf),
            (cb_scale, 1, bf),
            (cb_mask, 2, bf),
            (cb_t, 2, bf),
            (cb_t2, 2, bf),
            (cb_pos, 1, bf),
            (cb_zero, 1, bf),
            (cb_sm_in, c_t, bf),
            (cb_mx, 1, bf),
            (cb_sm, 1, f32),
            (cb_v, 2 * c_t, bf),
            (cb_probs, c_t, bf),
            (cb_out, 2, bf),
            (cb_max, 1, f32),
            (cb_exps, c_t, f32),
            (cb_recip, 1, f32),
        ]
        if inline_kv:
            spec += [(cb_kv_old, 2 * d_t, bf), (cb_kv_new, 2 * d_t, bf), (cb_kv_scr, d_t, bf)]
            spec += [(cb_kv_diff, 1, bf), (cb_kv_out, 2 * d_t, bf), (cb_wpos, 1, bf)]
        cbs = [
            ttnn.CBDescriptor(
                total_size=n * (f32_tile if fmt == f32 else tile),
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(idx, fmt, f32_tile if fmt == f32 else tile)],
            )
            for idx, n, fmt in spec
        ]
        cbs.append(
            ttnn.CBDescriptor(
                total_size=64, core_ranges=cores, format_descriptors=[ttnn.CBFormatDescriptor(cb_cnt, ttnn.uint32, 64)]
            )
        )
        tensors = [
            qk,
            kv_cache.k,
            kv_cache.v,
            self.scale_tile,
            ctx.mask_full,
            ctx.cur_pos,
            self.zero_tile,
            self.max_scaler,
            self.sum_scaler,
        ]
        accessor_ct = []
        for t in tensors:
            accessor_ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        reader = ttnn.KernelDescriptor(
            kernel_source=os.path.join(_ATTN_FUSED_DIR, "reader_attn_fused.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[d_t, c_t, n_kv, cb_q, cb_k, cb_scale, cb_mask, cb_pos, cb_zero, cb_mx, cb_sm, cb_v, cb_cnt]
            + accessor_ct
            + [int(kv_write is not None and not inline_kv), sem_k, sem_v, int(q_rows is not None), g + 1]
            + [int(inline_kv), cb_kv_old, cb_kv_new, cb_kv_scr, cb_kv_out]
            + [(self.n_heads + self.n_kv_heads) * d_t, (qkv_t.padded_shape[-1] // 32) if inline_kv else 0, kv_flag]
            + list(ttnn.TensorAccessorArgs(qkv_t if inline_kv else qk).get_compile_time_args()),
            runtime_args=rd_rt,
            common_runtime_args=[t.buffer_address() for t in tensors]
            + [(qkv_t if inline_kv else qk).buffer_address()],
            config=ttnn.ReaderConfigDescriptor(),
        )
        if inline_kv:
            # The head merge, plus storing this step's staged position tiles once the reader has spliced them.
            writer = ttnn.KernelDescriptor(
                kernel_source=os.path.join(_ATTN_FUSED_DIR, "writer_attn_fused.cpp"),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=cores,
                compile_time_args=[cb_out, n_kv, g, d_t, c_t, cb_kv_old, cb_wpos, kv_flag]
                + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
                + list(ttnn.TensorAccessorArgs(kv_cache.k).get_compile_time_args())
                + list(ttnn.TensorAccessorArgs(ctx.cur_pos).get_compile_time_args()),
                runtime_args=wr_rt,
                common_runtime_args=[
                    out.buffer_address(),
                    kv_cache.k.buffer_address(),
                    kv_cache.v.buffer_address(),
                    ctx.cur_pos.buffer_address(),
                ],
                config=ttnn.WriterConfigDescriptor(),
            )
        else:
            writer = ttnn.KernelDescriptor(
                kernel_source=os.path.join(_ATTN_BMM_DIR, "writer_attn_merge.cpp"),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=cores,
                compile_time_args=[cb_out, n_kv, g, d_t] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
                runtime_args=wr_rt,
                common_runtime_args=[out.buffer_address()],
                config=ttnn.WriterConfigDescriptor(),
            )
        compute_cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,  # as self.compute_cfg (scores, softmax and probs.V)
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            dst_full_sync_en=False,
        )
        # As _attn_bmm sets for the score scale's SFPU multiply operands; softmax's fp32 intermediates keep the
        # default (unpacked to SrcA/SrcB, as ttnn's softmax factory leaves them).
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for idx in (cb_t, cb_scale):
            modes[idx] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_cfg.unpack_to_dest_mode = modes
        compute = ttnn.KernelDescriptor(
            kernel_source=os.path.join(_ATTN_FUSED_DIR, "compute_attn_fused.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[
                d_t, c_t, cb_q, cb_k, cb_scale, cb_mask, cb_t, cb_t2, cb_sm_in, cb_mx, cb_sm, cb_max, cb_exps, cb_recip,
                cb_probs, cb_v, cb_out, cb_cnt, int(inline_kv), cb_kv_old, cb_kv_new, cb_kv_diff, cb_kv_out,
            ],
            runtime_args=ttnn.RuntimeArgs(),
            config=compute_cfg,
        )
        semaphores = []
        if inline_kv:
            semaphores = [ttnn.SemaphoreDescriptor(id=kv_flag, core_ranges=cores, initial_value=0)]
        if kv_write is not None and not inline_kv:
            everywhere = ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]
            )
            semaphores = [
                ttnn.SemaphoreDescriptor(id=sem_k, core_ranges=everywhere, initial_value=0),
                ttnn.SemaphoreDescriptor(id=sem_v, core_ranges=everywhere, initial_value=0),
            ]
        program = ttnn.ProgramDescriptor(
            kernels=[reader, writer, compute] + kv_kernels, semaphores=semaphores, cbs=cbs + kv_cbs
        )
        extra_t = [qkv_t] if inline_kv and all(qkv_t is not u for u in tensors) else []
        return ttnn.generic_op(
            tensors + [t for t in kv_tensors if all(t is not u for u in tensors)] + extra_t + [out], program
        )

    def _prefill_2d_cfg(self, x, w, in0_block_w, subblock_w):
        """Full-grid 2-D multicast config for a prefill projection x [.., M, K] @ w [K, N]."""
        grid = self.device.compute_with_storage_grid_size()
        m_t, n_t = x.padded_shape[-2] // 32, w.padded_shape[-1] // 32
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=subblock_w,
            per_core_M=-(-m_t // grid.y),
            per_core_N=-(-n_t // grid.x),
            transpose_mcast=False,
            fused_activation=None,
        )

    def _prefill_o_proj(self, attn, b, seq):
        """Prefill o_proj on the merged heads [1, B * S, n * D] -> [B, S, hidden] in L1 (the residual add input)."""
        grid = self.device.compute_with_storage_grid_size()
        m_t, n_t = attn.padded_shape[-2] // 32, self.wo.padded_shape[-1] // 32
        out = ttnn.linear(
            attn,
            self.wo,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_cfg,
            # Full-grid 2-D multicast with 8-tile K blocks (8 K steps instead of the auto config's).
            program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(grid.x, grid.y),
                in0_block_w=8,
                out_subblock_h=1,
                out_subblock_w=3,
                per_core_M=-(-m_t // grid.y),
                per_core_N=-(-n_t // grid.x),
                transpose_mcast=False,
                fused_activation=None,
            ),
        )
        ttnn.deallocate(attn)
        return ttnn.reshape(out, (b, seq, out.shape[-1]))

    def _attn_prefill_fused(self, q, k, v, mask, b, seq, memory_config, kv_cache):
        """Prefill attention for roped q [B, n, S, D], k / v [B, n_kv, S, D] and the causal mask [1, 1, S, S] as one
        generic_op (kernels/attn_prefill), output head-merged [1, B * S, n * D]. A unit is one query tile row of a
        (batch, KV head) block (q viewed as [B, n_kv, G * S, D]); each core takes a contiguous unit range, keeping
        the block's K / V tiles resident. Per unit: the attn_fused arithmetic over all S / 32 key tiles. The core
        holding a block's first row also stores its K / V into kv_cache (key tiles past S: the zero tile)."""
        n_kv, g = self.n_kv_heads, self.kv_groups
        d_t, s_t = self.head_dim // 32, seq // 32
        r_t = g * s_t
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, b * seq, self.n_heads * self.head_dim]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device,
            memory_config,
        )
        cores, rt = self._spread(b * n_kv * r_t)
        bf, f32 = ttnn.bfloat16, ttnn.float32
        tile, f32_tile = 2 * 32 * 32, 4 * 32 * 32
        (cb_q, cb_k, cb_scale, cb_mask, cb_t, cb_t2, cb_sm_in, cb_mx, cb_sm, cb_v, cb_probs) = range(11)
        cb_out, cb_zero, cb_max, cb_exps, cb_recip = 16, 17, 24, 25, 26
        spec = [
            (cb_zero, 1, bf),
            (cb_q, 2 * d_t, bf),
            (cb_k, s_t * d_t, bf),
            (cb_scale, 1, bf),
            (cb_mask, 2, bf),
            (cb_t, 2, bf),
            (cb_t2, 2, bf),
            (cb_sm_in, s_t, bf),
            (cb_mx, 1, bf),
            (cb_sm, 1, f32),
            (cb_v, s_t * d_t, bf),
            (cb_probs, s_t, bf),
            (cb_out, 2 * d_t, bf),
            (cb_max, 1, f32),
            (cb_exps, s_t, f32),
            (cb_recip, 1, f32),
        ]
        cbs = [
            ttnn.CBDescriptor(
                total_size=n * (f32_tile if fmt == f32 else tile),
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(idx, fmt, f32_tile if fmt == f32 else tile)],
            )
            for idx, n, fmt in spec
        ]
        tensors = [q, k, v, self.scale_tile, mask, self.max_scaler, self.sum_scaler]
        accessor_ct = []
        for t in tensors:
            accessor_ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        reader = ttnn.KernelDescriptor(
            kernel_source=os.path.join(_ATTN_PREFILL_DIR, "reader_attn_prefill.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[d_t, s_t, r_t, cb_q, cb_k, cb_scale, cb_mask, cb_mx, cb_sm, cb_v]
            + accessor_ct
            + [1, kv_cache.capacity // 32, cb_zero]
            + list(ttnn.TensorAccessorArgs(kv_cache.k).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(self.zero_tile).get_compile_time_args()),
            runtime_args=rt["dm"],
            common_runtime_args=[t.buffer_address() for t in tensors]
            + [kv_cache.k.buffer_address(), kv_cache.v.buffer_address(), self.zero_tile.buffer_address()],
            config=ttnn.ReaderConfigDescriptor(),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=os.path.join(_ATTN_PREFILL_DIR, "writer_attn_prefill.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[cb_out, d_t, s_t, r_t, n_kv, g] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
            runtime_args=rt["dm"],
            common_runtime_args=[out.buffer_address()],
            config=ttnn.WriterConfigDescriptor(),
        )
        compute_cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,  # as self.compute_cfg (scores, softmax and probs.V)
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            dst_full_sync_en=False,
        )
        # As _attn_fused: the score scale's SFPU multiply operands unpack to DEST as fp32.
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for idx in (cb_t, cb_scale):
            modes[idx] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_cfg.unpack_to_dest_mode = modes
        compute = ttnn.KernelDescriptor(
            kernel_source=os.path.join(_ATTN_PREFILL_DIR, "compute_attn_prefill.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[
                d_t, s_t, r_t, cb_q, cb_k, cb_scale, cb_mask, cb_t, cb_t2, cb_sm_in, cb_mx, cb_sm, cb_max, cb_exps,
                cb_recip, cb_probs, cb_v, cb_out,
            ],
            runtime_args=rt["dm"],
            config=compute_cfg,
        )
        program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
        return ttnn.generic_op(tensors + [kv_cache.k, kv_cache.v, self.zero_tile, out], program)

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

    def _prefill_mm(self, x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG):
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
            memory_config=memory_config,
            compute_kernel_config=self.compute_cfg,
        )

    def _kv_write(self, k, qkv, kv_cache, cur_pos, batch, k_row=0):
        """Decode KV-cache write as one generic_op: per (cache, b, kv head) unit, the slot row pos[b] of
        cache[b, h] becomes old + (new - old) (binary_ng's default bf16 FPU sub/add), every other row untouched."""
        grid = self.device.compute_with_storage_grid_size()
        all_cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(grid.x * grid.y)]
        kernels, cbs, _ = self._kv_write_parts(k, qkv, kv_cache, cur_pos, batch, k_row, all_cores)
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        # The caches are updated in place; generic_op's last tensor is its (here: in-place) output.
        ttnn.generic_op([k, qkv, cur_pos, kv_cache.k, kv_cache.v], program)

    def _kv_write_parts(self, k, qkv, kv_cache, cur_pos, batch, k_row, core_list, signal=None, k_from_rows=False):
        """The KV-write kernels and CBs on the first units-many cores of core_list (logical, in order; K units
        first). signal=(sem_k, sem_v, coords): after each unit the writer bumps that semaphore on row block
        b * n_kv + h, whose NoC coordinates are coords[2 * bh], coords[2 * bh + 1]."""
        n_kv, d_t = self.n_kv_heads, self.head_dim // 32
        units = 2 * batch * n_kv
        n_cores = min(units, len(core_list))
        base, extra = divmod(units, n_cores)
        used = core_list[:n_cores]
        cores = _core_rects(used)  # rectangles, not one range per core

        dm_rt, cmp_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        start = 0
        for i, c in enumerate(used):
            count = base + (1 if i < extra else 0)
            dm_rt[c.x][c.y] = [start, count]
            cmp_rt[c.x][c.y] = [count]
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
        sem_k, sem_v, coords = signal if signal is not None else (0, 0, [])
        v_col0 = (self.n_heads + n_kv) * self.head_dim // 32
        reader = ttnn.KernelDescriptor(
            kernel_source=os.path.join(_KV_WRITE_DIR, "reader_kv_write.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[*shape, v_col0, cb_old, cb_new, cb_scr, cb_pos, k_row, int(k_from_rows)]
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
            compile_time_args=[*shape[:4], cb_out, cb_wscr, cb_wpos]
            + cache_ct
            + pos_ct
            + [int(signal is not None), sem_k, sem_v],
            runtime_args=dm_rt,
            common_runtime_args=common + list(coords),
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
        return [reader, writer, compute], cbs, [k, qkv, cur_pos, kv_cache.k, kv_cache.v]

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
        self,
        hidden_states,
        position_embeddings=None,
        past_key_values=None,
        kv_cache=None,
        attn_ctx=None,
        prenorm=None,
        residual=None,
        **kwargs,
    ):
        """prenorm: the input norm, still to be applied to hidden_states (in the QKV program when bound).
        residual: the layer's residual stream; when the decode o_proj program adds it itself, the second return
        value is "residual_added" and the first is residual + attention."""
        x = hidden_states
        self._fused_prenorm = False
        self._prenorm = prenorm
        self._residual = residual
        self._residual_added = False
        if prenorm is not None:
            decode = kv_cache is not None and attn_ctx.mode == "decode"
            if decode and self.dmm_qkv_norm is not None and self.dmm_qkv_norm.norm_pro is prenorm:
                self._fused_prenorm = self.dmm_qkv_norm.accepts(x) and prenorm._kernel_ok(x)
            if not self._fused_prenorm:
                x = prenorm(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (x.shape[0] * x.shape[1], x.shape[-2], x.shape[-1]))
        b, seq = x.shape[0], x.shape[-2]
        if kv_cache is not None:
            if attn_ctx.mode == "decode":
                # Decode rows arrive as [1, B, hidden] so every linear sees B real rows, not B padded tiles.
                b, seq = x.shape[-2], 1
            rope_given = attn_ctx.rope is not None or attn_ctx.rope_rows is not None
            if attn_ctx.mode == "decode" and _USE_ROPE_KERNEL and rope_given:
                # The rope kernel reads the step's shared cos / signed sin (ctx.rope): skip the per-layer reshape.
                cos = sin = None
            else:
                cos, sin = self._cos_sin(position_embeddings, b, seq)
            out = self._cached(x, cos, sin, kv_cache, attn_ctx, b, seq)
            return out, ("residual_added" if self._residual_added else None)

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

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native, tensor-parallel TTNN port of the Qwen2.5-VL text decoder layer
(`Qwen2_5_VLDecoderLayer`, `model.language_model.layers[i]`), prefill (no KV cache).

    h = x + o_proj(attn(rope_mrope(q, k), v))       GQA: 28 q heads / 4 kv heads, head_dim 128, causal
    y = h + down(silu(gate(norm2(h))) * up(norm2(h)))

TP scheme (tt_transformers attention.py / mlp.py) on the TP (column) axis, DP axis replicated:
    * q/k/v fused and column-parallel by KV GROUP: chip d owns kv heads [d*kvl, (d+1)*kvl) and the q heads
      that attend to them, so GQA needs no cross-chip traffic. Biases shard with their columns.
    * o_proj row-parallel over the same local heads -> all_reduce.
    * MLP gate/up column-parallel, down row-parallel -> all_reduce.
    * norms replicated; the residual stream is kept in fp32.
mRoPE: the (t, h, w) cos/sin sections are combined on host (metadata, per HF
apply_multimodal_rotary_pos_emb) and applied on device with rotate_half == x @ R.
"""
from __future__ import annotations

import math

import numpy as np
import torch

import ttnn
from models.demos.qwen_image_edit_text_encoder._stubs import attention as _attention
from models.demos.qwen_image_edit_text_encoder._stubs.attention import (
    MASK_NEG,
    exact_all_reduce,
    exact_all_reduce_residual,
    hifi4_config,
    mesh_shape,
    pad_to_tile,
    rotate_half,
    shard_mapper,
    split_bf16,
    split_linear,
    split_matmul,
    upload,
    upload_rows,
)
from models.demos.qwen_image_edit_text_encoder._stubs.encoder_stack import TtRMSNorm, _bf16, _fp32

# the precise attention moves heads in and out of the fused projections by per-head slices + concat
# instead of reshape + permute (which pads the head count to a tile in the permuted layout)
HEADS_BY_SLICES = False
# with HEADS_BY_SLICES: the per-head slices of a head split / merge in L1 (freed once concatenated into DRAM) when
# together they take at most this many bytes (0: DRAM)
HEADS_L1_BYTES = 0


def _concat_heads(parts, dim):
    """concat of per-head slices into DRAM; L1 slices are freed once read."""
    out = ttnn.concat(parts, dim=dim, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for t in parts:
        if t.memory_config().buffer_type == ttnn.BufferType.L1:
            ttnn.deallocate(t)
    return out


def _heads_kw(n, nbytes):
    l1 = n > 1 and HEADS_L1_BYTES and nbytes <= HEADS_L1_BYTES
    return {"memory_config": ttnn.L1_MEMORY_CONFIG} if l1 else {}


# fn(o) -> the (hi, lo) bf16 limbs of the merged heads [B, 1, S, H * D] read straight from the head-major attention
# output o [B, H, S, D] (e.g. limb_split.fused_split_merge_heads), or None where it does not apply: the out
# projection then skips the per-head slices + concat and the split (None: slices + concat, then the split)
MERGE_SPLIT_FN = None
# the decoder layer's two residual adds made in their branch's exact TP reduce pass (exact_all_reduce_residual)
RESIDUAL_REDUCE = False
# MLP gate / up's shared bf16 limbs split straight into L1 when together they take at most this many bytes
# (the split hook's optional output memory config; freed after both products. 0: in x's memory)
MLP_LIMBS_L1_BYTES = 0
# the LM attention's logit scale applied as the mask add's input activation (one pass instead of multiply + add)
FUSED_SCALE_MASK = False
# the LM MLP's SiLU applied as the gate input's activation of the gate * up multiply (one op, no SiLU round trip)
MLP_FUSED_SILU = False
# the LM attention softmax's row-sized intermediates (scores - max, exp) kept in L1 when the scores take at most
# this many bytes (0: in DRAM)
SOFTMAX_L1_BYTES = 0
# the LM attention softmax's exp applied as the scores - max subtract's output activation (one op, no exp round trip)
SOFTMAX_FUSED_EXP = False
# with SOFTMAX_L1_BYTES: the masked scores and the probabilities in L1 too (the softmax reads and P @ V reads them)
SOFTMAX_IO_L1 = False
# with SOFTMAX_L1_BYTES: the raw QK^T scores in L1 too (read by the mask add; freed once it is made)
SCORES_L1 = False
# fn(t, cos, sin) -> the precise LM attention's RoPE t * cos + rotate_half(t) * sin in one call (e.g. a fused kernel),
# or None to run the stock ops (None: slices + neg + concat + multiplies + add)
ROPE_FN = None
# optional fn(qkv, lo, n, head_dim, cos, sin) -> ROPE_FN of heads [lo, lo + n) read straight from the fused qkv
# projection (no per-head slices + concat first), or None where it does not apply
ROPE_HEADS_FN = None
# with ROPE_HEADS_FN: q's / k's RoPE output in L1 (read by the QK^T product, freed after it) when q takes at most this
# many bytes (0: DRAM)
ROPE_L1_BYTES = 0
# the LM attention's fused float32 qkv projection (its bias add's output) in L1 when it takes at most this many bytes,
# read by the q / k RoPE and the v head slice and freed once they are made (0: its default placement)
QKV_L1_BYTES = 0


# mask_k_tiles of the additive mask the caller is about to run the LM layers with (set by the caller around its
# forward_padded calls, None outside them): the LM attention's P @ V then accumulates per output tile row only the
# key tiles that mask leaves unmasked (the probabilities past them are exactly 0), through LIMB_BMM_FN(..., k_tiles=)
PV_K_TILES = None


def mask_k_tiles(mask):
    """mask [B, 1, R, N] additive (MASK_NEG = masked) -> int [B, R / 32]: per tile row, 1 + the last key tile any of
    its rows leaves unmasked -- every key tile past it is masked in all of the tile's rows, so their softmax
    probabilities are exactly 0 (exp of ~MASK_NEG). A row with no unmasked key (uniform softmax) takes all N / 32."""
    B, _, R, N = mask.shape
    nt = N // 32
    open_ = mask[:, 0] > MASK_NEG / 2  # [B, R, N]
    any_open = open_.any(axis=-1)  # [B, R]
    last = np.where(any_open, N - 1 - np.argmax(open_[:, :, ::-1], axis=-1), N - 1)  # last open key (or all)
    per_row = last // 32 + 1
    return per_row.reshape(B, R // 32, 32).max(axis=-1).astype(np.int64).clip(1, nt)


# with SOFTMAX_FUSED_EXP: exp(scores - max) written height-sharded in L1 (each core's whole rows; read by the row sum
# and the divide) instead of L1 interleaved
SOFTMAX_EXP_SHARDED = False


def _height_shard(t):
    """A height-sharded L1 config of t ([..., R, C] TILE: whole rows per core, the most cores dividing R's tile
    rows), or None where it does not fit."""
    dims = [int(d) for d in t.padded_shape]
    rows, cols = math.prod(dims[:-1]), dims[-1]
    if rows % 32:
        return None
    grid = t.device().compute_with_storage_grid_size()
    rt = rows // 32
    n = max(c for c in range(1, min(rt, grid.x * grid.y) + 1) if rt % c == 0)
    return ttnn.create_sharded_memory_config(
        (rows // n, cols),
        ttnn.num_cores_to_corerangeset(n, grid, row_wise=True),
        ttnn.ShardStrategy.HEIGHT,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _t(lin):
    return lin.weight.detach().float().t().contiguous()


def rotate_half_matrix(d):
    half = d // 2
    r = np.zeros((d, d), dtype=np.float32)
    r[np.arange(half) + half, np.arange(half)] = -1.0
    r[np.arange(half), np.arange(half) + half] = 1.0
    return r.reshape(1, 1, d, d)


def mrope_tables(cos, sin, mrope_section):
    """HF apply_multimodal_rotary_pos_emb section select: [3, B, S, D] -> [B, S, D] (numpy fp32).
    Already-combined [B, S, D] tables pass through."""
    cos = cos.float().numpy() if isinstance(cos, torch.Tensor) else np.asarray(cos, dtype=np.float32)
    sin = sin.float().numpy() if isinstance(sin, torch.Tensor) else np.asarray(sin, dtype=np.float32)
    if cos.ndim == 3:
        return cos, sin
    bounds = np.cumsum([0] + list(mrope_section) * 2)
    idx = np.zeros(cos.shape[-1], dtype=np.int64)
    for i, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        idx[lo:hi] = i % 3
    cols = np.arange(cos.shape[-1])
    return cos[idx, :, :, cols].transpose(1, 2, 0), sin[idx, :, :, cols].transpose(1, 2, 0)


def text_attention_mask(attention_mask, b, s, s_pad, rep):
    """Additive [B, 1, rep*s_pad, s_pad] mask: causal (HF default when the mask is None) or the given
    4D mask (bool: True = attend; float: additive), tiled `rep` times along rows (GQA row packing)."""
    causal = np.where(np.arange(s_pad)[None, :] <= np.arange(s_pad)[:, None], 0.0, MASK_NEG).astype(np.float32)
    causal[:, s:] = MASK_NEG
    causal[s:, :] = 0.0
    mask = np.broadcast_to(causal, (b, 1, s_pad, s_pad)).copy()
    if attention_mask is not None:
        am = attention_mask.numpy() if attention_mask.dtype == torch.bool else attention_mask.float().numpy()
        if am.ndim == 2:  # [B, S] padding mask, nonzero = keep
            add = np.where(am.astype(bool), 0.0, MASK_NEG)[:, None, None, :]
        elif am.dtype == bool:  # [B, 1, S, S], True = attend
            add = np.where(am, 0.0, MASK_NEG)
        else:  # [B, 1, S, S] additive
            add = am
        add = np.broadcast_to(add, (b, 1, s, s)).astype(np.float32)
        mask[:, :, :s, :s] = np.maximum(mask[:, :, :s, :s] + add, MASK_NEG)
    return np.tile(mask, (1, 1, rep, 1))


class TtTextAttention:
    def __init__(self, device, torch_module, pair=None):
        """pair: a second attention module placed on the second row stage (row-staged layers); same TP split."""
        self.device = device
        _, self.tp = mesh_shape(device)
        cfg = torch_module.config
        self.num_heads = int(torch_module.num_heads)
        self.num_kv = int(torch_module.num_key_value_heads)
        self.head_dim = int(torch_module.head_dim)
        self.mrope_section = list(cfg.rope_parameters["mrope_section"])
        assert self.num_kv % self.tp == 0, f"{self.num_kv} kv heads not divisible by TP={self.tp}"
        self.kvl = self.num_kv // self.tp
        self.group = self.num_heads // self.num_kv
        self.hl = self.kvl * self.group
        self.scale = 1.0 / math.sqrt(self.head_dim)
        D, hl, kvl = self.head_dim, self.hl, self.kvl

        def _fused(m):
            wq, wk, wv = _t(m.q_proj), _t(m.k_proj), _t(m.v_proj)  # [C, out]
            bq, bk, bv = (p.bias.detach().float() for p in (m.q_proj, m.k_proj, m.v_proj))
            w_blocks, b_blocks = [], []
            for d in range(self.tp):
                qs, ks = slice(d * hl * D, (d + 1) * hl * D), slice(d * kvl * D, (d + 1) * kvl * D)
                w_blocks += [wq[:, qs], wk[:, ks], wv[:, ks]]
                b_blocks += [bq[qs], bk[ks], bv[ks]]
            return torch.cat(w_blocks, dim=1), torch.cat(b_blocks).reshape(1, 1, 1, -1), _t(m.o_proj)

        w, b, wo = _fused(torch_module)
        if pair is not None:
            w2, b2, wo2 = _fused(pair)
            self.wqkv = upload_rows(device, [w, w2], col_dim=-1)
            self.bqkv = upload_rows(device, [b, b2], col_dim=-1)
            self.wo = upload_rows(device, [wo, wo2], col_dim=0)
        else:
            col = shard_mapper(device, -1)
            self.wqkv = upload(device, w, mapper=col)
            self.bqkv = upload(device, b, mapper=col)
            self.wo = upload(device, wo, mapper=shard_mapper(device, 0))
        self.rot = upload(device, rotate_half_matrix(D), dtype=ttnn.float32)
        self.compute_cfg = hifi4_config()
        # precise: every matmul input carried as bf16 hi + lo (exact products), probs @ v in float32,
        # o_proj out in float32 with an exact TP reduce
        self.precise = False

    def rope_tables(self, position_embeddings, s_pad):
        cos, sin = mrope_tables(*position_embeddings, self.mrope_section)  # [B, S, D]
        b, s, d = cos.shape
        pad = np.zeros((b, s_pad - s, d), dtype=np.float32)
        cos = np.concatenate([cos, pad], axis=1).reshape(b, 1, s_pad, d)
        sin = np.concatenate([sin, pad], axis=1).reshape(b, 1, s_pad, d)
        return upload(self.device, cos, dtype=ttnn.float32), upload(self.device, sin, dtype=ttnn.float32)

    def mask(self, attention_mask, b, s, s_pad):
        return upload(self.device, text_attention_mask(attention_mask, b, s, s_pad, self.group), dtype=ttnn.float32)

    def forward_padded(self, x, tt_cos, tt_sin, tt_mask, kv=None, residual=None):
        """x: [B, 1, s_pad, C] bf16 replicated -> [B, 1, s_pad, C] bf16 replicated. kv: prefix K/V cache
        (see _forward_precise; precise path only). residual: see _forward_precise (precise path only)."""
        D, hl, kvl, G = self.head_dim, self.hl, self.kvl, self.group
        b, s_pad = x.shape[0], x.shape[-2]
        if self.precise:
            return self._forward_precise(x, tt_cos, tt_sin, tt_mask, kv=kv, residual=residual)
        assert kv is None, "the prefix K/V cache is implemented on the precise path only"
        # q/k carry large biases in Qwen2 (scores of O(100s)); keep the score path in fp32.
        qkv = ttnn.linear(x, self.wqkv, bias=self.bqkv, compute_kernel_config=self.compute_cfg, dtype=ttnn.float32)

        def _heads(lo, n):
            t = ttnn.slice(qkv, [0, 0, 0, lo * D], [b, 1, s_pad, (lo + n) * D])
            t = ttnn.reshape(t, (b, s_pad, n, D))
            return ttnn.permute(t, (0, 2, 1, 3))  # [B, n, S, D]

        q, k, v = _heads(0, hl), _heads(hl, kvl), _heads(hl + kvl, kvl)

        def _rope(t):
            rot = ttnn.matmul(t, self.rot, compute_kernel_config=self.compute_cfg)
            return ttnn.add(ttnn.multiply(t, tt_cos), ttnn.multiply(rot, tt_sin))

        q, k = _rope(q), _rope(k)
        # GQA: pack the G q heads sharing a kv head along rows -> [B, kvl, G*S, D]
        q = ttnn.reshape(q, (b, kvl, G * s_pad, D))
        scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1), compute_kernel_config=self.compute_cfg, dtype=ttnn.float32)
        scores = ttnn.add(ttnn.multiply(scores, self.scale), tt_mask)
        probs = ttnn.softmax(scores, dim=-1, numeric_stable=True, compute_kernel_config=self.compute_cfg)
        o = ttnn.matmul(
            ttnn.typecast(probs, ttnn.bfloat16), ttnn.typecast(v, ttnn.bfloat16), compute_kernel_config=self.compute_cfg
        )  # [B, kvl, G*S, D]
        o = ttnn.reshape(o, (b, hl, s_pad, D))
        o = ttnn.permute(o, (0, 2, 1, 3))
        o = ttnn.reshape(o, (b, 1, s_pad, hl * D))

        out = ttnn.linear(o, self.wo, compute_kernel_config=self.compute_cfg)
        if self.tp > 1:
            out = ttnn.all_reduce(out, cluster_axis=1, topology=ttnn.Topology.Linear)
        return out

    def _forward_precise(self, x, tt_cos, tt_sin, tt_mask, kv=None, residual=None):
        """kv = (mode, cache, key, P) or None. "store": keep this call's roped K and V of positions [0, P)
        in cache[key]. "use": x holds positions [P, P + s) only; the cached prefix K/V are prepended, so
        the queries attend to [0, P + s) (tt_cos/tt_sin/tt_mask are for those positions)."""
        D, hl, kvl, G = self.head_dim, self.hl, self.kvl, self.group
        b, s_pad = x.shape[0], x.shape[-2]
        cfg = self.compute_cfg
        ex = getattr(self, "exact", True)
        qkv_bytes = 4 * b * s_pad * int(self.wqkv.padded_shape[-1])
        qkv_l1 = 0 < qkv_bytes <= QKV_L1_BYTES
        qkv = split_linear(
            x,
            self.wqkv,
            bias=self.bqkv,
            compute_kernel_config=cfg,
            exact=ex,
            bias_out_mem=ttnn.L1_MEMORY_CONFIG if qkv_l1 else None,
        )

        def _heads(lo, n):
            if HEADS_BY_SLICES:  # one tile-aligned column block per head, stacked on dim 1 (no padded permute)
                kw = _heads_kw(n, b * s_pad * n * D * 4)
                parts = [
                    ttnn.slice(qkv, [0, 0, 0, (lo + h) * D], [b, 1, s_pad, (lo + h + 1) * D], **kw) for h in range(n)
                ]
                return parts[0] if n == 1 else _concat_heads(parts, 1)
            t = ttnn.slice(qkv, [0, 0, 0, lo * D], [b, 1, s_pad, (lo + n) * D])
            t = ttnn.reshape(t, (b, s_pad, n, D))
            return ttnn.permute(t, (0, 2, 1, 3))

        cos = tt_cos if tt_cos.dtype == ttnn.float32 else ttnn.typecast(tt_cos, ttnn.float32)
        sin = tt_sin if tt_sin.dtype == ttnn.float32 else ttnn.typecast(tt_sin, ttnn.float32)

        def _rope(t):
            y = ROPE_FN(t, cos, sin) if ROPE_FN is not None else None
            if y is not None:
                return y
            return ttnn.add(ttnn.multiply(t, cos), ttnn.multiply(rotate_half(t), sin))

        rope_l1 = 0 < b * hl * s_pad * D * 4 <= ROPE_L1_BYTES

        def _rope_heads(lo, n):
            """RoPE of heads [lo, lo + n): read straight from qkv by ROPE_HEADS_FN where it applies."""
            mem = ttnn.L1_MEMORY_CONFIG if rope_l1 else None
            y = ROPE_HEADS_FN(qkv, lo, n, D, cos, sin, mem) if ROPE_HEADS_FN is not None else None
            return y if y is not None else _rope(_heads(lo, n))

        q, k, v = _rope_heads(0, hl), _rope_heads(hl, kvl), _heads(hl + kvl, kvl)
        if qkv_l1:
            ttnn.deallocate(qkv)
        if kv is not None:
            mode, cache, key, P = kv
            if mode == "store":
                dram = {"memory_config": ttnn.DRAM_MEMORY_CONFIG}  # (k may be in L1: ROPE_L1_BYTES)
                cache[key] = (
                    ttnn.slice(k, [0, 0, 0, 0], [b, kvl, P, D], **dram),
                    ttnn.slice(v, [0, 0, 0, 0], [b, kvl, P, D], **dram),
                )
            else:
                kp, vp = cache[key]
                k, v = ttnn.concat([kp, k], dim=2), ttnn.concat([vp, v], dim=2)
        q = ttnn.reshape(q, (b, kvl, G * s_pad, D))
        n_keys = int(k.shape[-2])
        s_l1 = SCORES_L1 and SOFTMAX_L1_BYTES and 4 * b * kvl * G * s_pad * n_keys <= SOFTMAX_L1_BYTES
        scores = split_matmul(
            q, k, transpose_b=True, compute_kernel_config=cfg, exact=ex, out_mem=ttnn.L1_MEMORY_CONFIG if s_l1 else None
        )
        for t in (q, k):
            if t.memory_config().buffer_type == ttnn.BufferType.L1:
                ttnn.deallocate(t)
        nbytes = scores.volume() * (4 if scores.dtype == ttnn.float32 else 2)
        sm = ttnn.L1_MEMORY_CONFIG if SOFTMAX_L1_BYTES and nbytes <= SOFTMAX_L1_BYTES else None
        io = sm if SOFTMAX_IO_L1 else None
        if FUSED_SCALE_MASK:
            scale = [ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, self.scale)]
            scores = ttnn.add(scores, tt_mask, input_tensor_a_activations=scale, memory_config=io)
        else:
            scores = ttnn.add(ttnn.multiply(scores, self.scale), tt_mask, memory_config=io)
        mx = ttnn.max(scores, dim=-1, keepdim=True)
        if SOFTMAX_FUSED_EXP:
            esm = _height_shard(scores) if SOFTMAX_EXP_SHARDED and sm is not None else None
            e = ttnn.subtract(scores, mx, activations=[ttnn.UnaryOpType.EXP], memory_config=esm or sm)
        else:
            e = ttnn.exp(ttnn.subtract(scores, mx, memory_config=sm), memory_config=sm)
        probs = ttnn.divide(e, ttnn.sum(e, dim=-1, keepdim=True, compute_kernel_config=cfg), memory_config=io)
        kt = None
        tbl = PV_K_TILES
        if tbl is not None and tbl.shape[1] == G * s_pad // 32 and int(tt_mask.shape[-1]) == n_keys:
            if int(tbl.max()) <= n_keys // 32:  # (a table of this mask: its rows and key tiles)
                # one program runs every chip's sequences (a row-split batch puts different ones on different
                # chips): each tile row takes the most key tiles any sequence needs there
                kt = np.broadcast_to(tbl.max(axis=0), (b * kvl, tbl.shape[1]))  # [B * kvl, G * s_pad / 32]
        o = split_matmul(probs, v, compute_kernel_config=cfg, exact=ex, k_tiles=kt)  # [B, kvl, G*S, D]
        o = ttnn.reshape(o, (b, hl, s_pad, D))
        op = MERGE_SPLIT_FN(o) if MERGE_SPLIT_FN is not None and hl > 1 else None
        if op is not None and o.memory_config().buffer_type == ttnn.BufferType.L1:
            ttnn.deallocate(o)  # (an L1 P @ V output: read only by the split, freed before the out projection)
        if op is not None:  # the out projection's limbs read straight from the head-major o (no merged o)
            out = split_linear(op[0], self.wo, compute_kernel_config=cfg, exact=ex, parts=tuple(op))
            return _reduce(out, self.device, self.tp, residual)
        if HEADS_BY_SLICES:  # heads side by side along the width by slices + concat (no padded permute)
            kw = _heads_kw(hl, b * s_pad * hl * D * 4)
            o = _concat_heads([ttnn.slice(o, [0, h, 0, 0], [b, h + 1, s_pad, D], **kw) for h in range(hl)], -1)
        else:
            o = ttnn.permute(o, (0, 2, 1, 3))
            o = ttnn.reshape(o, (b, 1, s_pad, hl * D))
        out = split_linear(o, self.wo, compute_kernel_config=cfg, exact=ex)
        return _reduce(out, self.device, self.tp, residual)


def _reduce(out, device, tp, residual):
    """The row-parallel output's exact TP reduce; with a residual (RESIDUAL_REDUCE): (residual + reduce, True)
    with the add in the reduce's pass where exact_all_reduce_residual applies, else (reduce, False)."""
    if residual is not None and tp > 1:
        r = exact_all_reduce_residual(out, device, residual)
        if r is not None:
            return r, True
    if tp > 1:
        out = exact_all_reduce(out, device)
    return (out, False) if residual is not None else out


class TtTextMLP:
    """down(silu(gate(x)) * up(x)), no biases -- gate/up column-parallel, down row-parallel."""

    def __init__(self, device, torch_module, pair=None):
        """pair: a second MLP placed on the second row stage (row-staged layers); same TP split."""
        _, self.tp = mesh_shape(device)
        self.device = device
        if pair is not None:
            self.w_gate = upload_rows(device, [_t(torch_module.gate_proj), _t(pair.gate_proj)], col_dim=-1)
            self.w_up = upload_rows(device, [_t(torch_module.up_proj), _t(pair.up_proj)], col_dim=-1)
            self.w_down = upload_rows(device, [_t(torch_module.down_proj), _t(pair.down_proj)], col_dim=0)
        else:
            col = shard_mapper(device, -1)
            self.w_gate = upload(device, _t(torch_module.gate_proj), mapper=col)
            self.w_up = upload(device, _t(torch_module.up_proj), mapper=col)
            self.w_down = upload(device, _t(torch_module.down_proj), mapper=shard_mapper(device, 0))
        self.compute_cfg = hifi4_config()

    @staticmethod
    def _limbs_l1(x):
        """Whether gate / up's shared limbs of x go straight to L1 (MLP_LIMBS_L1_BYTES)."""
        if not MLP_LIMBS_L1_BYTES or _attention.SPLIT_FN is None:
            return False
        if x.dtype != ttnn.float32 or x.layout != ttnn.TILE_LAYOUT or x.is_sharded():
            return False
        return 2 * 2 * math.prod(int(d) for d in x.padded_shape) <= MLP_LIMBS_L1_BYTES

    def __call__(self, x, residual=None):
        if getattr(self, "precise", False):
            cfg = self.compute_cfg
            ex = getattr(self, "exact", True)
            xp = split_bf16(x, 2) if _attention.SHARE_SPLIT and not self._limbs_l1(x) else None  # shared limbs
            if _attention.SHARE_SPLIT and xp is None:
                xp = tuple(_attention.SPLIT_FN(x, ttnn.L1_MEMORY_CONFIG))
            gate = split_linear(x, self.w_gate, compute_kernel_config=cfg, exact=ex, parts=xp)
            up = split_linear(x, self.w_up, compute_kernel_config=cfg, exact=ex, parts=xp)
            if xp is not None and xp[0].memory_config().buffer_type == ttnn.BufferType.L1:
                for p in xp:
                    ttnn.deallocate(p)
            if MLP_FUSED_SILU:
                h = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
            else:
                h = ttnn.multiply(ttnn.silu(gate), up)
            # lo_fidelity: this MLP's own lower-limb fidelity for the down projection (None: the module default)
            lo = getattr(self, "lo_fidelity", None)
            out = split_linear(h, self.w_down, compute_kernel_config=cfg, exact=ex, lo_fidelity=lo)
            return _reduce(out, self.device, self.tp, residual)
        gate = ttnn.linear(x, self.w_gate, compute_kernel_config=self.compute_cfg)
        up = ttnn.linear(x, self.w_up, compute_kernel_config=self.compute_cfg)
        out = ttnn.linear(ttnn.multiply(ttnn.silu(gate), up), self.w_down, compute_kernel_config=self.compute_cfg)
        if self.tp > 1:
            out = ttnn.all_reduce(out, cluster_axis=1, topology=ttnn.Topology.Linear)
        return out


class TtTextDecoderLayer:
    def __init__(self, device, torch_module, mlp=None, pair=None):
        """pair: a second decoder layer whose weights live on the second row stage (row-staged layers)."""
        self.device = device
        pn = (lambda name: getattr(pair, name)) if pair is not None else (lambda name: None)
        self.input_layernorm = TtRMSNorm(device, torch_module.input_layernorm, pair=pn("input_layernorm"))
        self.post_attention_layernorm = TtRMSNorm(
            device, torch_module.post_attention_layernorm, pair=pn("post_attention_layernorm")
        )
        self.self_attn = TtTextAttention(device, torch_module.self_attn, pair=pn("self_attn"))
        self.mlp = mlp if mlp is not None else TtTextMLP(device, torch_module.mlp, pair=pn("mlp"))

    def forward_padded(self, x, tt_cos, tt_sin, tt_mask, kv=None):
        """x: [B, 1, s_pad, C] fp32 residual stream (branch inputs stay fp32 with precise_inputs).
        kv: the attention's prefix K/V cache argument (TtTextAttention._forward_precise)."""
        cast = _fp32 if getattr(self, "precise_inputs", False) else _bf16
        if RESIDUAL_REDUCE and self.self_attn.precise and getattr(self.mlp, "precise", False):
            # each branch's residual add in its exact reduce's pass where that applies (else the same ttnn.add)
            a, fused = self.self_attn.forward_padded(
                cast(self.input_layernorm(x)), tt_cos, tt_sin, tt_mask, kv=kv, residual=x
            )
            x = a if fused else ttnn.add(x, _fp32(a))
            m, fused = self.mlp(cast(self.post_attention_layernorm(x)), residual=x)
            return m if fused else ttnn.add(x, _fp32(m))
        a = self.self_attn.forward_padded(cast(self.input_layernorm(x)), tt_cos, tt_sin, tt_mask, kv=kv)
        x = ttnn.add(x, _fp32(a))
        return ttnn.add(x, _fp32(self.mlp(cast(self.post_attention_layernorm(x)))))

    def __call__(self, hidden_states, attention_mask=None, position_embeddings=None, **kwargs):
        b, s, c = hidden_states.shape
        s_pad = pad_to_tile(s)
        tt_cos, tt_sin = self.self_attn.rope_tables(position_embeddings, s_pad)
        tt_mask = self.self_attn.mask(attention_mask, b, s, s_pad)
        x = ttnn.reshape(hidden_states, (b, 1, s, c))
        if s_pad != s:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, s_pad - s), (0, 0)], 0.0)
        x = self.forward_padded(_fp32(x), tt_cos, tt_sin, tt_mask)
        if s_pad != s:
            x = ttnn.slice(x, [0, 0, 0, 0], [b, 1, s, c])
        return ttnn.reshape(x, (b, s, c))


def build(device, torch_module=None):
    return TtTextDecoderLayer(device, torch_module)


def layer(device, torch_module=None):
    return TtTextDecoderLayer(device, torch_module)

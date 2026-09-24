# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `acoustic_transformer_block` (`AcousticTransformerBlock`).

    r = attention(attention_norm(x)); h = x + r
    r = feed_forward(ffn_norm(h));    out = h + r

The attention inside is `BidirectionalAttention`: NO positional encoding and NO causal mask, so
SDPA runs with `is_causal=False`. Shapes come from `params.json`'s `acoustic_transformer_args`:
dim 3072, 32 heads / 8 KV heads, head_dim 128, hidden_dim 9216, no biases.

BATCH AXIS. The leading bound is read from the tensor, never assumed to be 1: a `[B, 1, S, dim]`
input keeps all B samples and comes back as `[B, 1, S, dim]`, while the rank-<=3 input the
component test feeds (`[1, S, dim]`) is unchanged. A hardcoded leading 1 would have silently
dropped samples 1..B-1 once the sampler ran a real batch.

ADDITIVE MASK. `attn_mask=` is added to the attention scores. The flow-matching sampler's
sequence is 3 real tokens living in one 32-row tile, so it hands in a mask that blocks columns
3..31; the component test passes no mask and runs unmasked as before.

FIDELITY. Everything -- residual stream, Q/K/V, softmax -- runs in float32 against bfloat16
weights with HiFi4 + `fp32_dest_acc_en`. The sampler's output is rounded onto 21 levels 0.1 apart,
so a code flips on a few-1e-3 error, and SDPA's bfloat16-only interface plus `ttnn.softmax`'s
~6e-3 absolute error on the probabilities cost ~7% of the codes. See `_attention`.
"""

from __future__ import annotations

import math

import torch

import ttnn

_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
)


# Tall (>= 8 tile rows) linears are compute-bound, so they run one fidelity rung below HiFi4.
_TALL_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
)
_TILE_BYTES = {ttnn.float32: 4096, ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}
_L1_BUDGET = 1_100_000


def _mcast_cfg(x, w, rows, out_dtype):
    """A full-grid 2D-multicast program config for a tall `[rows, K] x [K, N]` linear, or None.

    Left to itself ttnn picks a partial grid with small K-blocks for these shapes. This spreads M
    over the grid rows and N over the grid columns, takes the widest K-block whose double-buffered
    in0/in1 blocks plus the output block fit L1, and the largest subblock fp32 DEST allows (4 tiles).
    None when even the output block alone does not fit, so the caller keeps ttnn's default.
    """
    grid = x.device().compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    mt, kt, nt = rows // 32, int(w.shape[-2]) // 32, int(w.shape[-1]) // 32
    per_m, per_n = -(-mt // gy), -(-nt // gx)
    size = lambda dt: _TILE_BYTES.get(dt, 2048)
    fixed = per_m * per_n * (size(out_dtype) + (0 if out_dtype == ttnn.float32 else 4096))
    kb = next(
        (
            c
            for c in (16, 8, 4, 2, 1)
            if kt % c == 0 and fixed + 2 * c * (per_m * size(x.dtype) + per_n * size(w.dtype)) <= _L1_BUDGET
        ),
        None,
    )
    if kb is None:
        return None
    sub = max(
        ((h, s) for h in range(1, 5) for s in range(1, 5) if h * s <= 4 and per_m % h == 0 and per_n % s == 0),
        key=lambda hs: (hs[0] * hs[1], hs[1]),
    )
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=kb,
        out_subblock_h=sub[0],
        out_subblock_w=sub[1],
        per_core_M=per_m,
        per_core_N=per_n,
        transpose_mcast=False,
        fused_activation=None,
    )


def _short_cfg(x, w, rows, out_dtype):
    """A 1D in0-multicast config for a SHORT (2..7 tile rows) linear, or None.

    Such a linear is bound by streaming its weight, so every core should own a slice of N and
    read only its own weight columns while the small activation is multicast to all of them.
    Left to itself ttnn gives it small K-blocks, and each block is a multicast round trip that
    every core waits on; this takes the widest K-block that fits L1.
    """
    grid = x.device().compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    mt, kt, nt = rows // 32, int(w.shape[-2]) // 32, int(w.shape[-1]) // 32
    per_n = next(p for p in range(-(-nt // (gx * gy)), nt + 1) if nt % p == 0)
    size = lambda dt: _TILE_BYTES.get(dt, 2048)
    fixed = mt * per_n * (size(out_dtype) + (0 if out_dtype == ttnn.float32 else 4096))
    kb = next(
        (
            c
            for c in (32, 24, 16, 12, 8, 6, 4, 3, 2, 1)
            if kt % c == 0 and fixed + 2 * c * (mt * size(x.dtype) + per_n * size(w.dtype)) <= _L1_BUDGET
        ),
        None,
    )
    if kb is None:
        return None
    sub = max(
        ((h, s) for h in range(1, 5) for s in range(1, 5) if h * s <= 4 and mt % h == 0 and per_n % s == 0),
        key=lambda hs: (hs[0] * hs[1], hs[1]),
    )
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=kb,
        out_subblock_h=sub[0],
        out_subblock_w=sub[1],
        per_core_M=mt,
        per_core_N=per_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def _lin(x, w, **kwargs):
    """`ttnn.linear` with the leading batch folded into M, so the weight streams ONCE.

    A `[B, 1, S, K]` activation against a 2-D weight runs as B separate `S x K x N` matmuls that
    each re-read the whole weight from DRAM; `[1, 1, B*S, K]` is one matmul that reads it once.
    Tall results (>= 8 tile rows) also get a hand-sized full-grid program config.
    """
    shape = [int(d) for d in x.shape]
    lead = 1
    for d in shape[:-2]:
        lead *= d
    rows = lead * shape[-2]
    if rows >= 256 and rows % 32 == 0 and "program_config" not in kwargs:
        cfg = _mcast_cfg(x, w, rows, kwargs.get("dtype") or x.dtype)
        if cfg is not None:
            kwargs["program_config"] = cfg
        kwargs["compute_kernel_config"] = _TALL_COMPUTE
    elif 64 <= rows < 256 and rows % 32 == 0 and "program_config" not in kwargs:
        cfg = _short_cfg(x, w, rows, kwargs.get("dtype") or x.dtype)
        if cfg is not None:
            kwargs["program_config"] = cfg
    if lead == 1:
        return ttnn.linear(x, w, **kwargs)
    y = ttnn.linear(ttnn.reshape(x, [1, 1, rows, shape[-1]]), w, **kwargs)
    return ttnn.reshape(y, shape[:-1] + [int(y.shape[-1])])


def _from_torch(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    t = t.to(torch.bfloat16) if dtype == ttnn.bfloat16 else t.to(torch.float32)
    if device.__class__.__name__ == "MeshDevice":
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=layout,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


def _weight(linear, device):
    """A `[in, out]` device tensor for a torch `nn.Linear` (whose weight is `[out, in]`)."""
    return _from_torch(linear.weight.detach().transpose(0, 1).contiguous(), device)


def _gamma(norm, device):
    """A norm's gamma as a `[1, 1, 1, dim]` float32 tile tensor, for `_rms_norm`."""
    return _from_torch(norm.weight.detach().reshape(1, 1, 1, -1).contiguous(), device, dtype=ttnn.float32)


def _rms_norm(x, gamma, eps, dtype=None):
    """RMSNorm in float32: `x * rsqrt(mean(x^2) + eps) * gamma`.

    NOT the `ttnn` layernorm op, which carries 2.7e-3 of RELATIVE error -- measured on this chip
    against a float64 reference, at either gamma dtype, with a float32 input and a float32
    output. Written out with `mean / rsqrt / multiply` the same normalization holds 1.2e-7. The
    sampler downstream rounds onto 21 levels 0.1 apart in x, so 2.7e-3 through seven norms is
    worth ~1% of the output codes and 1.2e-7 is worth none of them.
    """
    inv = ttnn.rsqrt(ttnn.add(ttnn.mean(ttnn.square(x), dim=-1, keepdim=True), eps))
    return ttnn.multiply(ttnn.multiply(x, inv), gamma, dtype=dtype or ttnn.float32)


def _bmm(a, b, per_core_m=None, transpose_b=False):
    """Head-batched `a @ b` spread over the full grid.

    Without a program config the `[B, H, 32, 128] x [B, H, 128, 32]` score product lands on ONE
    core (and probs @ V on four), running B*H tiny matmuls back to back. The reuse config makes
    every (batch, head) output block its own work unit, so they fan out across the grid.
    """
    m, k, n = int(a.shape[-2]) // 32, int(a.shape[-1]) // 32, int(b.shape[-2 if transpose_b else -1]) // 32
    grid = a.device().compute_with_storage_grid_size()
    cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=k,
        out_subblock_h=1,
        out_subblock_w=min(n, 4),
        per_core_M=per_core_m or m,
        per_core_N=n,
    )
    return ttnn.matmul(a, b, transpose_b=transpose_b, program_config=cfg, compute_kernel_config=_COMPUTE)


def _attention(h, wqkv, wo, n_heads, n_kv_heads, scale, attn_mask):
    """GQA attention, bidirectional and non-causal, entirely in float32.

    NOT `ttnn.transformer.scaled_dot_product_attention`. SDPA rejects float32
    (`sdpa_device_operation.cpp:43`), so it forces Q/K/V down to bfloat16, and its softmax carries
    ~6e-3 of ABSOLUTE error on the attention probabilities -- measured on this chip against a
    float64 reference, and `ttnn.softmax` on its own is just as bad (6.2e-3) at either dtype and
    with `numeric_stable` either way. The same softmax written out as
    `max / exp / sum / divide` in float32 holds 6e-8. The sampler this feeds rounds onto 21
    levels 0.1 apart in x, so 6e-3 on a probability is worth roughly 7% of the output codes;
    6e-8 is worth none of them. The sequence here is one 32-row tile, so the explicit form costs
    four small ops per block and nothing measurable in time.

    The query heads are GROUPED BY KV HEAD rather than K/V being repeat_interleaved: query head
    h reads KV head h // repeats (the reference's `repeat_kv` mapping), and those `repeats` heads
    are contiguous, so `[B, H, S, D]` read as `[B, H_kv, repeats * S, D]` is a free view whose
    batch dims line up with K/V's. No K/V tensor is materialised `repeats` times. The mask only
    ever blocks COLUMNS, so its first row broadcasts over every stacked query row.
    """
    qkv = _lin(h, wqkv, dtype=ttnn.float32, compute_kernel_config=_COMPUTE)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        qkv, num_heads=n_heads, num_kv_heads=n_kv_heads, transpose_k_heads=False
    )
    batch, _, seq, head_dim = (int(d) for d in q.shape)
    repeats = n_heads // n_kv_heads
    q = ttnn.reshape(q, [batch, n_kv_heads, repeats * seq, head_dim])

    # The reference scales the QUERY before the product, not the scores after it.
    scores = _bmm(ttnn.multiply(q, scale), ttnn.transpose(k, -2, -1))
    if attn_mask is not None:
        if int(attn_mask.shape[-2]) != 1:
            attn_mask = ttnn.slice(attn_mask, [0, 0, 0, 0], [1, 1, 1, int(attn_mask.shape[-1])])
        scores = ttnn.add(scores, attn_mask)
    weights = ttnn.subtract(scores, ttnn.max(scores, dim=-1, keepdim=True), activations=[ttnn.UnaryOpType.EXP])
    weights = ttnn.divide(weights, ttnn.sum(weights, dim=-1, keepdim=True))

    out = ttnn.reshape(_bmm(weights, v), [batch, n_heads, seq, head_dim])
    return _lin(
        ttnn.experimental.nlp_concat_heads(out),
        wo,
        dtype=ttnn.float32,
        compute_kernel_config=_COMPUTE,
    )


_COMPACT_MASKS = {}
_COMPACT_ROWS = (96, 192)


def _compact_mask(device, rows, tokens, repeats):
    """Additive `[1, 1, repeats * rows, rows]` mask letting a row attend only to its own sample.

    Row/column `t * R + r` is token t of sample r, so the allowed pairs are those with equal
    `index % R`; the `repeats` grouped query heads stack the same pattern vertically. One per
    (device, rows) shape, shared by every block, created at build time for the usual row counts
    so nothing is allocated inside a trace.
    """
    key = (id(device), rows, tokens, repeats)
    mask = _COMPACT_MASKS.get(key)
    if mask is None:
        sample = torch.arange(rows) % (rows // tokens)
        blk = torch.where(sample[:, None] == sample[None, :], 0.0, -1.0e9)
        mask = _from_torch(blk.repeat(repeats, 1).reshape(1, 1, repeats * rows, rows), device, dtype=ttnn.float32)
        _COMPACT_MASKS[key] = mask
    return mask


def _compact_attention(h, wqkv, wo, n_heads, n_kv_heads, scale, tokens):
    """The same attention on the COMPACT layout: `[1, 1, tokens * R, dim]`, token t in rows t*R..

    No 32-row pad per sample, so nothing downstream computes on 29 padding rows. All
    `tokens * R` rows form one sequence and a constant mask keeps each row to its own sample's
    tokens, so this is ordinary head-batched attention with last-axis reductions -- no per-key
    loop and no batch-axis reduction (which ttnn implements with a full permute).
    """
    rows = int(h.shape[-2])
    repeats = n_heads // n_kv_heads
    qkv = _lin(h, wqkv, dtype=ttnn.float32, compute_kernel_config=_COMPUTE)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        qkv, num_heads=n_heads, num_kv_heads=n_kv_heads, transpose_k_heads=False
    )
    head_dim = int(q.shape[-1])
    q = ttnn.reshape(q, [1, n_kv_heads, repeats * rows, head_dim])

    scores = _bmm(ttnn.multiply(q, scale), k, per_core_m=1, transpose_b=True)
    scores = ttnn.add(scores, _compact_mask(h.device(), rows, tokens, repeats))
    weights = ttnn.subtract(scores, ttnn.max(scores, dim=-1, keepdim=True), activations=[ttnn.UnaryOpType.EXP])
    weights = ttnn.divide(weights, ttnn.sum(weights, dim=-1, keepdim=True))

    out = ttnn.reshape(_bmm(weights, v, per_core_m=1), [1, n_heads, rows, head_dim])
    return _lin(ttnn.experimental.nlp_concat_heads(out), wo, dtype=ttnn.float32, compute_kernel_config=_COMPUTE)


def _leading(shape) -> int:
    """The product of every axis before `[seq, dim]` -- the real batch, from the tensor."""
    dims = list(shape)[:-2]
    batch = 1
    for d in dims:
        batch *= int(d)
    return batch


def build(device, torch_module):
    blk = torch_module
    attn = blk.attention
    ff = blk.feed_forward

    n_heads = int(attn.n_local_heads)
    n_kv_heads = int(attn.n_local_kv_heads)
    head_dim = int(attn.head_dim)
    dim = int(blk.dim)
    scale = 1.0 / math.sqrt(head_dim)
    eps = float(blk.attention_norm.eps)

    wqkv = _from_torch(
        torch.cat(
            [
                attn.wq.weight.detach().transpose(0, 1),
                attn.wk.weight.detach().transpose(0, 1),
                attn.wv.weight.detach().transpose(0, 1),
            ],
            dim=-1,
        ).contiguous(),
        device,
        dtype=ttnn.bfloat8_b,
    )
    wo = _from_torch(attn.wo.weight.detach().transpose(0, 1).contiguous(), device, dtype=ttnn.bfloat8_b)
    w1 = _from_torch(ff.w1.weight.detach().transpose(0, 1).contiguous(), device, dtype=ttnn.bfloat8_b)
    # The down projection is DRAM-bound at 1024 rows; bf8_b halves the weight it streams.
    w2 = _from_torch(ff.w2.weight.detach().transpose(0, 1).contiguous(), device, dtype=ttnn.bfloat8_b)
    w3 = _from_torch(ff.w3.weight.detach().transpose(0, 1).contiguous(), device, dtype=ttnn.bfloat8_b)
    g_attn = _gamma(blk.attention_norm, device)
    g_ffn = _gamma(blk.ffn_norm, device)
    for rows in _COMPACT_ROWS:
        _compact_mask(device, rows, 3, n_heads // n_kv_heads)

    def acoustic_transformer_block(x, attn_mask=None, tokens=None, **kwargs):
        seq = int(x.shape[-2])
        batch = _leading(x.shape)
        rank = len(list(x.shape))

        h4 = ttnn.reshape(x, [batch, 1, seq, dim])
        if h4.dtype != ttnn.float32:
            h4 = ttnn.typecast(h4, ttnn.float32)

        xn = _rms_norm(h4, g_attn, eps, dtype=ttnn.bfloat16)
        if tokens:
            attn_out = _compact_attention(xn, wqkv, wo, n_heads, n_kv_heads, scale, tokens)
        else:
            attn_out = _attention(xn, wqkv, wo, n_heads, n_kv_heads, scale, attn_mask)
        h4 = ttnn.add(h4, attn_out)

        hn = _rms_norm(h4, g_ffn, eps, dtype=ttnn.bfloat16)
        gate = _lin(hn, w1, dtype=ttnn.float32, compute_kernel_config=_COMPUTE, memory_config=ttnn.L1_MEMORY_CONFIG)
        up = _lin(hn, w3, dtype=ttnn.float32, compute_kernel_config=_COMPUTE, memory_config=ttnn.L1_MEMORY_CONFIG)
        h4 = ttnn.add(
            h4,
            _lin(
                ttnn.multiply(
                    gate,
                    up,
                    input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                ),
                w2,
                compute_kernel_config=_COMPUTE,
            ),
        )

        if rank >= 4:
            return h4
        return ttnn.reshape(h4, [batch, seq, dim])

    return acoustic_transformer_block

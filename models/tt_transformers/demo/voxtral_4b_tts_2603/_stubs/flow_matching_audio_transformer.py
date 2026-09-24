# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `flow_matching_audio_transformer` (`acoustic_transformer`).

The deterministic core of the flow-matching sampler -- the velocity field plus the semantic head:

    t_proj   = time_projection(time_embedding(t))
    llm_proj = llm_projection(llm_hidden)
    seq      = [input_projection(x_t), t_proj, llm_proj]     # THREE tokens
    velocity = acoustic_codebook_output(norm(blocks(seq))[:, 0, :])
    semantic = semantic_codebook_output(llm_hidden)          # reads the RAW hidden state

The Euler loop around this is not ported here because it is not a function of its inputs: it draws
`x_0 = torch.randn(...)` inside the module. `x_t` is supplied instead, which leaves every weight
under test (see the note in `tests/pcc/test_flow_matching_audio_transformer.py`).

**The sequence is padded to one tile.** The real sequence is 3 tokens, which in TILE layout lives
inside a 32-row tile with 29 rows of padding. Rather than hope an op respects a sub-tile logical
length, the sequence is explicitly built 32 rows long and attention is given an additive mask that
blocks columns 3..31 -- so rows 0..2 attend to exactly the three real tokens, which is what the
reference computes. Only row 0 is ever read out.

The blocks are `AcousticTransformerBlock`s: bidirectional, RoPE-free, GQA 32/8, head_dim 128,
dim 3072, hidden 9216, `norm_eps` 1e-5, no biases.

The activation path runs in float32 (weights stay bfloat16 -- `ttnn.linear` takes a float32
activation against a bfloat16 weight). All-bfloat16 cleared the 0.99 target by only 0.0007, and
that margin is not worth holding: the residual accumulates over three blocks and the read-out is a
single row. The attention is written out rather than calling SDPA, which rejects float32
(`sdpa_device_operation.cpp:43`) -- see `_attention` -- because the sampler around this field
rounds onto 21 levels and SDPA's bfloat16 softmax was worth ~7% of the output codes.

THE 29-ROW TILE PAD IS A BUILD-TIME BUFFER. It used to be a `ttnn.zeros(...)` on every call,
which cannot live inside a captured trace (a trace replays kernels, it cannot allocate). It is
now a persistent device buffer, created once per distinct batch and reused; `build(..., batch=N)`
pre-creates the one the caller will actually use so nothing is allocated on the first traced
call. The numerics are unchanged -- same shape, same zeros, same concat.
"""

from __future__ import annotations

import math

import torch

import ttnn

_TILE = 32
_MASK_NEG = -1.0e9


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
    inv = ttnn.rsqrt(ttnn.add(ttnn.mean(ttnn.multiply(x, x), dim=-1, keepdim=True), eps))
    return ttnn.multiply(ttnn.multiply(x, inv), gamma, dtype=dtype or ttnn.float32)


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
    if lead == 1:
        return ttnn.linear(x, w, **kwargs)
    y = ttnn.linear(ttnn.reshape(x, [1, 1, rows, shape[-1]]), w, **kwargs)
    return ttnn.reshape(y, shape[:-1] + [int(y.shape[-1])])


def _bmm(a, b):
    """Head-batched `a @ b` spread over the full grid.

    Without a program config the `[B, H, 32, 128] x [B, H, 128, 32]` score product lands on ONE
    core (and probs @ V on four), running B*H tiny matmuls back to back. The reuse config makes
    every (batch, head) output block its own work unit, so they fan out across the grid.
    """
    m, k, n = int(a.shape[-2]) // 32, int(a.shape[-1]) // 32, int(b.shape[-1]) // 32
    grid = a.device().compute_with_storage_grid_size()
    cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=k,
        out_subblock_h=1,
        out_subblock_w=min(n, 4),
        per_core_M=m,
        per_core_N=n,
    )
    return ttnn.matmul(a, b, program_config=cfg, compute_kernel_config=_COMPUTE)


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
    scores = ttnn.subtract(scores, ttnn.max(scores, dim=-1, keepdim=True))
    weights = ttnn.exp(scores)
    weights = ttnn.divide(weights, ttnn.sum(weights, dim=-1, keepdim=True))

    out = ttnn.reshape(_bmm(weights, v), [batch, n_heads, seq, head_dim])
    return _lin(
        ttnn.experimental.nlp_concat_heads(out),
        wo,
        dtype=ttnn.float32,
        compute_kernel_config=_COMPUTE,
    )


def _compile_block(device, blk, mask):
    """One `AcousticTransformerBlock` as a callable on `[B, 1, TILE, dim]`."""
    attn = blk.attention
    ff = blk.feed_forward
    n_heads = int(attn.n_local_heads)
    n_kv_heads = int(attn.n_local_kv_heads)
    scale = 1.0 / math.sqrt(int(attn.head_dim))

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
    )
    wo = _weight(attn.wo, device)
    w1, w3 = (_weight(m, device) for m in (ff.w1, ff.w3))
    # The down projection is DRAM-bound at 1024 rows; bf8_b halves the weight it streams.
    w2 = _from_torch(ff.w2.weight.detach().transpose(0, 1).contiguous(), device, dtype=ttnn.bfloat8_b)
    g_attn = _gamma(blk.attention_norm, device)
    g_ffn = _gamma(blk.ffn_norm, device)
    eps = float(blk.attention_norm.eps)

    def run(h):
        xn = _rms_norm(h, g_attn, eps, dtype=ttnn.bfloat16)
        h = ttnn.add(h, _attention(xn, wqkv, wo, n_heads, n_kv_heads, scale, mask))

        hn = _rms_norm(h, g_ffn, eps, dtype=ttnn.bfloat16)
        gated = ttnn.multiply(
            _lin(hn, w1, dtype=ttnn.float32, compute_kernel_config=_COMPUTE),
            _lin(hn, w3, dtype=ttnn.float32, compute_kernel_config=_COMPUTE),
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            # Consumed once, by the down projection: hand it over in L1, not through DRAM.
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        return ttnn.add(h, _lin(gated, w2, compute_kernel_config=_COMPUTE))

    return run


def build(device, torch_module, batch=None):
    at = getattr(torch_module, "inner", torch_module)
    args = at.acoustic_transformer_args
    dim = int(args.dim)
    n_real_tokens = 3

    # float32, not bfloat16: `inv_freq` is COMPUTED by the module (exp of an arange), not a
    # checkpoint tensor, so bfloat16 would put 0.4% of error into the sinusoidal phase that
    # nothing downstream can recover.
    inv_freq = _from_torch(at.time_embedding.inv_freq.detach().reshape(1, -1).contiguous(), device, dtype=ttnn.float32)
    w_time = _weight(at.time_projection, device)
    w_llm = _weight(at.llm_projection, device)
    w_input = _weight(at.input_projection, device)
    w_acoustic = _weight(at.acoustic_codebook_output, device)
    w_semantic = _weight(at.semantic_codebook_output, device)
    semantic_bias = None
    if at.semantic_codebook_output.bias is not None:
        semantic_bias = _from_torch(at.semantic_codebook_output.bias.detach().reshape(1, 1, 1, -1), device)

    # Columns 3..31 of the padded tile are not real tokens; block them so rows 0..2 attend to
    # exactly the three the reference builds.
    mask_torch = torch.zeros(1, 1, 1, _TILE)
    mask_torch[:, :, :, n_real_tokens:] = _MASK_NEG
    mask = _from_torch(mask_torch, device, dtype=ttnn.float32)

    # The 29 pad rows of the one-tile sequence, as a PERSISTENT buffer rather than a per-call
    # `ttnn.zeros` (which a trace cannot replay). One buffer per distinct batch.
    pad_rows = _TILE - n_real_tokens
    _pads = {}

    def _pad_for(rows):
        buf = _pads.get(rows)
        if buf is None:
            buf = _from_torch(torch.zeros(rows, 1, pad_rows, dim), device, dtype=ttnn.float32)
            _pads[rows] = buf
        return buf

    if batch is not None:
        _pad_for(int(batch))

    blocks = [_compile_block(device, at.layers[str(i)], mask) for i in at.layers_ids]
    g_final = _gamma(at.norm, device)
    eps_final = float(at.norm.eps)

    acoustic_out = int(at.acoustic_codebook_output.out_features)
    semantic_out = int(at.semantic_codebook_output.out_features)

    def flow_matching_audio_transformer(llm_hidden, x_t=None, t=None, **kwargs):
        batch = int(llm_hidden.shape[0])
        h_in = ttnn.reshape(llm_hidden, [batch, 1, 1, dim])

        h_in = ttnn.typecast(h_in, ttnn.float32)
        semantic = _lin(h_in, w_semantic, compute_kernel_config=_COMPUTE)
        if semantic_bias is not None:
            semantic = ttnn.add(semantic, semantic_bias)
        semantic = ttnn.reshape(semantic, [batch, semantic_out])

        # TimeEmbedding: outer product t (x) inv_freq, then cat(cos, sin).
        freqs = ttnn.matmul(
            ttnn.typecast(ttnn.reshape(t, [batch, 1, 1, int(t.shape[-1])]), ttnn.float32),
            ttnn.typecast(inv_freq, ttnn.float32),
            compute_kernel_config=_COMPUTE,
        )
        t_emb = ttnn.concat([ttnn.cos(freqs), ttnn.sin(freqs)], dim=-1)
        t_proj = _lin(t_emb, w_time, compute_kernel_config=_COMPUTE)
        llm_proj = _lin(h_in, w_llm, compute_kernel_config=_COMPUTE)
        x_proj = _lin(
            ttnn.typecast(ttnn.reshape(x_t, [batch, 1, 1, int(x_t.shape[-1])]), ttnn.float32),
            w_input,
            compute_kernel_config=_COMPUTE,
        )

        h = ttnn.concat([x_proj, t_proj, llm_proj, _pad_for(batch)], dim=2)

        for block in blocks:
            h = block(h)
        h = _rms_norm(h, g_final, eps_final)

        first = ttnn.slice(h, [0, 0, 0, 0], [batch, 1, 1, dim])
        velocity = ttnn.reshape(
            _lin(first, w_acoustic, compute_kernel_config=_COMPUTE),
            [batch, acoustic_out],
        )
        return velocity, semantic

    return flow_matching_audio_transformer

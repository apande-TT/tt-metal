# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `codec_transformer` (`audio_tokenizer.decoder_blocks.1`).

A stack of `CodecTransformerBlock`s (two, for this decoder stage), each:

    r = attention_scale * attention(attention_norm(x));  h = x + r
    r = ffn_scale      * feed_forward(ffn_norm(h));       out = h + r

`attention_scale` / `ffn_scale` are LayerScale parameters -- per-channel `[dim]` vectors, not
scalars -- and this checkpoint's values are small and signed (~-5e-3, ~-1.5e-4), so dropping them
would not merely rescale the residual, it would flip its sign.

The attention is ALiBi + sliding-window causal + QK-norm with no RoPE; see `_stubs/codec_attention.py`
for the details. The window belongs to the decoder STAGE, not the model: the four stages run 2, 4,
8, 16 as each transposed convolution doubles it, so the mask is built from this stack's own
`args.attn_sliding_window_size`.

`norm_eps` here is **1e-2**, three orders of magnitude larger than the text backbone's 1e-5; it is
read off the modules rather than assumed.
"""

from __future__ import annotations

import math

import torch

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import common, cpp_addnorm, cpp_band_attn, cpp_softmax

_SHARD_HEIGHT = 32
# 2048 rows = 256 codec frames after the decoder's 8x upsampling, which is the whole stage's frame
# ceiling. The mask is translation-invariant, so a longer sequence needs a LARGER constant here
# (and nothing else); it cannot be rebuilt inside the forward, which must stay torch-free.
_MASK_MAX_SEQ = 2048
_MASK_NEG = -1.0e9

# fp32 accumulation in DEST. The activation path is float32 -- bfloat16 end to end put the full
# codec chain at PCC 0.9873 over eight residual blocks plus five convolutions.
_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
)


# FLOAT32 WEIGHTS. Measured on this device against a float64 reference, one matmul at M=32,
# K=N=3072, HiFi4 + `fp32_dest_acc_en`: a bfloat16 weight costs 1.738e-3 relative where a float32
# weight costs 1.169e-3. That 1.5x is small per op and this codec stacks eight residual blocks on
# top of five convolutions, where it is the last error source left after the softmax and the RMS
# norm were spelled out. No `ttnn.embedding` table is built here (those must stay bfloat16).
_TILE_BYTES = {ttnn.float32: 4096, ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}
_L1_BUDGET = 1_100_000


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def _mcast_cfg(x, w, rows, out_dtype):
    """A full-grid 2D-multicast program config for a tall `[rows, K] x [K, N]` linear, or None.

    M goes over the grid rows and N over the grid columns. Per-core M/N are searched a few tiles
    above the minimum (a slightly larger block often divides into better subblocks), and when the
    whole per-core output does not fit L1 it is split into out-blocks. Ranked by per-core work,
    then subblock area (fp32 DEST caps it at 4 tiles), then K-block width, then out-block area.
    """
    grid = x.device().compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    mt, kt, nt = rows // 32, int(w.shape[-2]) // 32, int(w.shape[-1]) // 32
    size = lambda dt: _TILE_BYTES.get(dt, 2048)
    xs, ws = size(x.dtype), size(w.dtype)
    os_ = size(out_dtype) + (0 if out_dtype == ttnn.float32 else 4096)
    best = None
    for pm in range(-(-mt // gy), -(-mt // gy) + 5):
        if -(-mt // pm) > gy:
            continue
        for pn in range(-(-nt // gx), -(-nt // gx) + 5):
            if -(-nt // pn) > gx:
                continue
            for bh in _divisors(pm):
                for bw in _divisors(pn):
                    kb = next(
                        (
                            c
                            # grid: a short M (<= 2 tile rows a core) or a long K (>= 64 tiles) takes a K block of 16 --
                            # half the K steps (half the float32 partial packs of the out block).
                            for c in ((16, 8, 4, 2, 1) if pm <= 2 or kt >= 64 else (8, 4, 2, 1))
                            if kt % c == 0 and bh * bw * os_ + 2 * c * (bh * xs + bw * ws) <= _L1_BUDGET
                        ),
                        None,
                    )
                    if kb is None:
                        continue
                    sub = max(
                        (
                            (h, s)
                            for h in range(1, 5)
                            for s in range(1, 5)
                            if h * s <= 4 and bh % h == 0 and bw % s == 0
                        ),
                        key=lambda hs: (hs[0] * hs[1], hs[1]),
                    )
                    # K block before out-block area: at in0_block_w=1 every K step re-packs the whole
                    # fp32 out block (the 4096 x 1024 x 4096 FFN up: 343 us at 13 x 12 / kb 1, 249 at
                    # 13 x 4 / kb 8).
                    # grid: past a K block of 4, fewer out blocks down M before a wider K block -- each extra
                    # M out block re-reads (and re-multicasts) the core column's whole weight block (the fp32
                    # 4096 x 1024 x 1792 conv product: 13 one-row blocks at kb 8 ran 326 us).
                    score = (pm * pn, -sub[0] * sub[1], -min(kb, 4), pm // bh, -kb, -bh * bw)
                    if best is None or score < best[0]:
                        best = (score, pm, pn, bh, bw, kb, sub)
    if best is None:
        return None
    _, pm, pn, bh, bw, kb, sub = best
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=kb,
        out_subblock_h=sub[0],
        out_subblock_w=sub[1],
        out_block_h=bh,
        out_block_w=bw,
        per_core_M=pm,
        per_core_N=pn,
        transpose_mcast=False,
        fused_activation=None,
    )


def _lin(x, w, **kwargs):
    """`ttnn.linear` with the leading batch folded into M, so the weight streams ONCE.

    A `[B, 1, S, K]` activation against a 2-D weight runs as B separate `S x K x N` matmuls that
    each re-read the whole weight from DRAM; `[1, 1, B*S, K]` is one matmul that reads it once.
    Tall results (>= 8 tile rows) also get a hand-sized full-grid program config.
    """
    # fidelity: a caller may ask for a lower fidelity than the tall linears' HiFi2 (the FFN gate / up: LoFi).
    fidelity = kwargs.pop("fidelity", ttnn.MathFidelity.HiFi2)
    shape = [int(d) for d in x.shape]
    lead = 1
    for d in shape[:-2]:
        lead *= d
    rows = lead * shape[-2]
    padded = lead * (-(-shape[-2] // 32) * 32)
    if lead > 1 and shape[-2] % 32 != 0 and padded >= 128 and "program_config" not in kwargs:
        # grid: an unaligned per-sample length -- the 4-D rows go straight into a full-grid 2D config (it folds the
        # batch over each sample's tile-padded rows), so neither side needs the fold's row-major relayout.
        cfg = _mcast_cfg(x, w, padded, kwargs.get("dtype") or x.dtype)
        if cfg is not None:
            # fidelity: HiFi2 unless the caller asks lower (LoFi on EVERY codec linear fails the e2e gate).
            kwargs["compute_kernel_config"] = ttnn.WormholeComputeKernelConfig(
                math_fidelity=fidelity, fp32_dest_acc_en=True, packer_l1_acc=True
            )
            return ttnn.linear(x, w, program_config=cfg, **kwargs)
    if rows >= 256 and rows % 32 == 0 and "program_config" not in kwargs:
        cfg = _mcast_cfg(x, w, rows, kwargs.get("dtype") or x.dtype)
        if cfg is not None:
            kwargs["program_config"] = cfg
            # Fidelity rung: the tall (compute-bound) codec linears at HiFi2.
            kwargs["compute_kernel_config"] = ttnn.WormholeComputeKernelConfig(
                math_fidelity=fidelity, fp32_dest_acc_en=True, packer_l1_acc=True
            )
    if lead == 1:
        return ttnn.linear(x, w, **kwargs)
    y = ttnn.linear(ttnn.reshape(x, [1, 1, rows, shape[-1]]), w, **kwargs)
    return ttnn.reshape(y, shape[:-1] + [int(y.shape[-1])])


def _from_torch(t, device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    t = t.to(torch.bfloat16) if dtype == ttnn.bfloat16 else t.to(torch.float32)
    if layout == ttnn.TILE_LAYOUT and t.numel() <= 262144:
        # A small tensor (a gamma, a bias, a layer scale) tilizes on the HOST: tilized on the device it is
        # a whole single-core op (~80 us for a 1 x 1024 float32 row) at load time.
        mapper = ttnn.ReplicateTensorToMesh(device) if device.__class__.__name__ == "MeshDevice" else None
        return ttnn.to_device(ttnn.from_torch(t, dtype=dtype, layout=layout, mesh_mapper=mapper), device)
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


def _folded_weight(linear, device, rows=None, cols=None):
    """`_weight` with a per-INPUT-channel scale `rows` and a per-OUTPUT-channel scale `cols` folded in
    (a float32 product on the host): `(x * g) @ W == x @ (g W)` and `(x @ W) * s == x @ (W s)`."""
    w = linear.weight.detach().float().transpose(0, 1)
    if rows is not None:
        w = w * rows.float().reshape(-1, 1)
    if cols is not None:
        w = w * cols.float().reshape(1, -1)
    # dtype rung: the part chain's codec linear weights as bf16 (the whole-section body's dtype).
    return _from_torch(w.contiguous(), device, dtype=ttnn.bfloat8_b)  # dtype rung: bf8_b codec weights


def _alibi_window_mask(slopes, window, seq):
    """`[1, H, seq, seq]`: ALiBi bias `slope[h] * (j - i)`, blocked where `j > i` or `j < i - window`.

    Depends only on `j - i`, so the top-left `[S, S]` corner of one big mask is exactly the mask for
    a length-`S` sequence -- which is what lets the forward stay free of torch calls (the runtime
    native probe graduates only at zero torch ops, so a per-call rebuild is not an option).
    """
    pos = torch.arange(seq)
    rel = pos.unsqueeze(0) - pos.unsqueeze(1)
    bias = slopes.reshape(-1, 1, 1).float() * rel.unsqueeze(0).float()
    blocked = (rel > 0) | (rel < -window)
    return bias.masked_fill(blocked.unsqueeze(0), _MASK_NEG).unsqueeze(0)


def _compile_block(device, blk, mask, window=None):
    """One `CodecTransformerBlock` as a callable on a `[1, 1, S, dim]` ttnn tensor."""
    attn = blk.attention
    ff = blk.feed_forward
    args = blk.args

    n_heads = int(attn.n_local_heads)
    n_kv_heads = int(attn.n_local_kv_heads)
    head_dim = int(args.head_dim)
    dim = int(blk.dim)
    scale = 1.0 / math.sqrt(head_dim)
    qk_norm = bool(args.qk_norm)

    # The norms' gammas and the LayerScales are folded into the weights beside them, so no
    # per-channel multiply of the [B, 1, T, dim] float32 stream runs in the forward.
    attn_gamma = ffn_gamma = None
    attn_g = blk.attention_norm.weight.detach()
    ffn_g = blk.ffn_norm.weight.detach()
    attn_ls = blk.attention_scale.detach() if blk.layer_scale else None
    ffn_ls = blk.ffn_scale.detach() if blk.layer_scale else None
    attn_eps = float(blk.attention_norm.eps)
    ffn_eps = float(blk.ffn_norm.eps)

    wq, wk, wv = (_folded_weight(m, device, rows=attn_g) for m in (attn.wq, attn.wk, attn.wv))
    wo = _folded_weight(attn.wo, device, cols=attn_ls)
    q_gamma = _norm_gamma(attn.q_norm, device) if qk_norm else None
    k_gamma = _norm_gamma(attn.k_norm, device) if qk_norm else None
    q_eps = float(attn.q_norm.eps) if qk_norm else 0.0
    k_eps = float(attn.k_norm.eps) if qk_norm else 0.0

    w1, w3 = (_folded_weight(m, device, rows=ffn_g) for m in (ff.w1, ff.w3))
    w2 = _folded_weight(ff.w2, device, cols=ffn_ls)

    attn_scale = ffn_scale = None  # folded into wo / w2

    if blk.post_attention_norm is not None or blk.post_ffn_norm is not None:
        raise NotImplementedError("post_attention_norm / post_ffn_norm are not ported")

    def block(h):
        seq = int(h.shape[-2])
        # shard: the float32 residual stream lives in L1 while it is <= 1024 rows (<= 4 MB, ~37 KB a core), so the
        # residual adds and the fused norms read and write L1 instead of DRAM (was: only a short sequence). At
        # 2048 rows its 8 MB on top of the FFN's L1 hidden clashed with the up projection's buffers. The group
        # runners hand it back in its input's memory (common.restore_memory).
        if common.l1_while_rows_fit(h, 1024) is not None and h.memory_config().buffer_type != ttnn.BufferType.L1:
            # shard: a short sequence's float32 residual stream (~19 KB a core) moves to L1, so the
            # residual adds and every step of the spelled-out RMS norms read and write L1 instead of DRAM.
            h = ttnn.to_memory_config(h, ttnn.L1_MEMORY_CONFIG)
        # bf16 normalised rows into the q / k / v linears (fp32 DEST; q / k come back bf16, v fp32).
        # shard: a short sequence's normed rows (q / k / v's input) in L1.
        xn_mem = ttnn.L1_MEMORY_CONFIG if int(h.shape[-2]) < 32 else None
        # shard: the attention norm's bf16 rows (q / k / v's in0) land in L1 up to 2048 rows (<= 4 MB).
        xn = _rms_norm(
            h, attn_gamma, attn_eps, dtype=ttnn.bfloat16, memory_config=xn_mem or common.l1_while_rows_fit(h, 2048)
        )

        # A sub-tile sequence (T < 32: one padded tile row a sample) keeps its small q / k / v in L1.
        qkv_mem = ttnn.L1_MEMORY_CONFIG if int(xn.shape[-2]) < 32 else None
        # dtype: q / k leave their projections as bf16 (half the write; the fp32 qk-norm reads half and writes
        # the float32 the band attention takes); v stays float32.
        qk_dt = ttnn.bfloat16 if qk_norm else ttnn.float32
        # shard: tall bf16 q / k (<= 8 MB each) stay in L1 for the qk-norm, which writes its float32 to DRAM.
        qk_rows = 1
        for d in list(xn.padded_shape)[:-1]:
            qk_rows *= int(d)
        qk_l1 = qkv_mem is None and qk_norm and qk_rows * int(wq.shape[-1]) * 2 <= (8 << 20)
        qk_mem = ttnn.L1_MEMORY_CONFIG if qk_l1 else qkv_mem
        norm_mem = ttnn.DRAM_MEMORY_CONFIG if qk_l1 else None
        q = _lin(xn, wq, compute_kernel_config=_COMPUTE, dtype=qk_dt, memory_config=qk_mem)
        k = _lin(xn, wk, compute_kernel_config=_COMPUTE, dtype=qk_dt, memory_config=qk_mem)
        # shard: a tall float32 v (<= 16 MB) stays in L1 for the band attention.
        v_mem = ttnn.L1_MEMORY_CONFIG if qkv_mem is None and qk_rows * int(wv.shape[-1]) * 4 <= (16 << 20) else qkv_mem
        v = _lin(xn, wv, compute_kernel_config=_COMPUTE, dtype=ttnn.float32, memory_config=v_mem)
        if qk_norm:
            # dtype: with the fused norm q / k leave it as bf16, where the projections put them (L1 while they
            # fit) -- the band attention takes bf16 q / k as its score product's operands, so no float32 copy.
            fused = common.codec_fused_norm()
            qk_out = ttnn.bfloat16 if fused else ttnn.float32
            out_mem = qk_mem if fused else norm_mem
            qn = _rms_norm(q, q_gamma, q_eps, dtype=qk_out, memory_config=out_mem)
            ttnn.deallocate(q)
            kn = _rms_norm(k, k_gamma, k_eps, dtype=qk_out, memory_config=out_mem)
            ttnn.deallocate(k)
            q, k = qn, kn

        # SDPA rejects float32 outright (`sdpa_device_operation.cpp:43`) -- so this does not call
        # it. Spelling the attention out as two matmuls and a softmax keeps Q/K/V, the ALiBi mask
        # and the whole reduction in FLOAT32, which SDPA cannot do at any fidelity. The codec runs
        # eight residual blocks over a few hundred rows, so the explicit form costs little, and
        # the bfloat16 narrowing it removes was compounding through all eight.
        if n_kv_heads == n_heads and cpp_band_attn.supports_merged(q, k, v, mask, window, n_heads):
            # structural: the banded attention reads each head's q / k / v tiles IN PLACE from the merged
            # [B, 1, T, H * D] linear outputs and writes the context back merged -- no q / k / v concat, no
            # nlp_create_qkv_heads, no nlp_concat_heads (pure data movement, the same values).
            # dtype: the context leaves as bf16 (o_proj reads half; the softmax and both products stay
            # float32 in DEST) and, when it is small (<= 8 MB), in L1.
            ctx_mem = (
                qkv_mem
                if qkv_mem is not None
                else (ttnn.L1_MEMORY_CONFIG if qk_rows * int(wo.shape[-2]) * 2 <= (8 << 20) else None)
            )
            a = cpp_band_attn.apply(
                q, k, v, mask, scale, memory_config=ctx_mem, merged_heads=n_heads, dtype=ttnn.bfloat16, real_rows=seq
            )
        else:
            qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads(
                ttnn.concat(
                    [ttnn.typecast(q, v.dtype), ttnn.typecast(k, v.dtype)] + [v] if q.dtype != v.dtype else [q, k, v],
                    dim=-1,
                ),
                num_heads=n_heads,
                num_kv_heads=n_kv_heads,
                transpose_k_heads=False,
            )
            if cpp_band_attn.supports(qh, kh, vh, mask, window):
                # structural: the sliding window (< one tile) as a BANDED attention -- each query tile row
                # against its own and the previous key tile row only (tt/cpp_band_attn, one generic_op for
                # scores, scale, mask, softmax and P@V). Every other tile of the full chain is masked to an
                # exact zero weight, so the context is the same.
                a = cpp_band_attn.apply(qh, kh, vh, mask, scale)
            else:
                scores = _bmm(qh, kh, transpose_b=True)
                scaled = ttnn.multiply(scores, scale)
                if cpp_softmax.supports(scaled, mask):
                    # Mask add + softmax in one pass, the mask read in place from the prebuilt one (no per-call
                    # slice copy): tt/cpp_softmax replays these same float32 SFPU ops.
                    weights = cpp_softmax.apply(scaled, mask)
                else:
                    weights = _softmax(ttnn.add(scaled, ttnn.slice(mask, [0, 0, 0, 0], [1, n_heads, seq, seq])))
                a = _bmm(weights, vh)
                ttnn.deallocate(scores)
            a = ttnn.experimental.nlp_concat_heads(a)
        # shard: q / k / v and the normed rows they came from are dead once the attention has run -- free their L1
        # here rather than at the block's return, so the o_proj / FFN buffers have it.
        for dead in (q, k, v, xn):
            ttnn.deallocate(dead)
        # shard: the float32 residual branches (o_proj, w2) land in L1 while the rows fit (<= 4096).
        res_mem = xn_mem if xn_mem is not None else common.l1_while_rows_fit(h)
        r = _lin(
            a,
            wo,
            memory_config=res_mem,  # shard: the residual branch stays in L1 while the rows fit
            dtype=ttnn.float32,
            compute_kernel_config=_COMPUTE,
        )
        ttnn.deallocate(a)
        if attn_scale is not None:
            r = ttnn.multiply(r, attn_scale)
        fused_hn = None
        if ffn_gamma is None and cpp_addnorm.supports(h, r):
            # cpp rung: the residual add and the FFN norm as one generic_op (tt/cpp_addnorm): h and r read
            # once, the stock float32 SFPU add, the norm's row kept in L1, the normed rows written bf16.
            h, fused_hn = cpp_addnorm.apply(
                h, r, ffn_eps, dtype=ttnn.bfloat16, memory_config=common.l1_while_rows_fit(h)
            )
        else:
            h = ttnn.add(h, r)
        ttnn.deallocate(r)  # shard: free the attention branch's L1 before the FFN norm

        # shard rung: the FFN's two activations -- the normed rows w1 / w3 read and the gated product w2
        # reads (the latter only to 2048 rows) -- live in L1 (interleaved) while they fit, so in0 reads skip DRAM.
        ffn_mem = common.l1_while_rows_fit(h)
        hn = (
            fused_hn
            if fused_hn is not None
            else _rms_norm(h, ffn_gamma, ffn_eps, dtype=ttnn.bfloat16, memory_config=ffn_mem)
        )
        # dtype: the FFN hidden (w1 / w3 outputs, the gate) in bf16; w2 still sums in fp32 DEST
        # and writes the fp32 residual branch.
        # dtype: a tall FFN hidden (>= 256 rows) is written bf8_b -- w1 / w3 write, and the gated multiply
        # reads, half the bytes; so is the gated product the multiply writes and w2 reads.
        hid_dt = ttnn.bfloat16 if common.l1_while_rows_fit(h, 255) else ttnn.bfloat8_b
        # shard: the bf8_b hidden (<= 2048 rows: ~9 MB each) lands in L1 for the gated multiply that reads it.
        hid_mem = common.l1_while_rows_fit(h, 2048)
        gate = _lin(
            hn, w1, compute_kernel_config=_COMPUTE, dtype=hid_dt, memory_config=hid_mem, fidelity=ttnn.MathFidelity.LoFi
        )
        up = _lin(
            hn, w3, compute_kernel_config=_COMPUTE, dtype=hid_dt, memory_config=hid_mem, fidelity=ttnn.MathFidelity.LoFi
        )
        ttnn.deallocate(hn)
        # shard: past 2048 rows the gated rows go to DRAM so w2's float32 output fits in L1 instead.
        gated_mem = common.l1_while_rows_fit(h, 2048)
        gated = ttnn.multiply(
            gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=hid_dt, memory_config=gated_mem
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        # fidelity: the FFN down at LoFi on its tall full-grid path (fp32 DEST kept).
        r = _lin(
            gated,
            w2,
            compute_kernel_config=_COMPUTE,
            dtype=ttnn.float32,
            memory_config=ffn_mem,
            fidelity=ttnn.MathFidelity.LoFi,
        )
        ttnn.deallocate(gated)
        if ffn_scale is not None:
            r = ttnn.multiply(r, ffn_scale)
        return ttnn.add(h, r)

    return block


def _norm_gamma(norm, device):
    """`[1, 1, 1, dim]` float32 TILE -- the form the spelled-out RMS norm's final multiply takes."""
    return _from_torch(norm.weight.detach().reshape(1, 1, 1, -1), device, dtype=ttnn.float32)


def _rms_norm(x, gamma, eps, dtype=None, memory_config=None):
    """`x * rsqrt(mean(x^2) + eps) * gamma`, spelled out, entirely in float32.

    NOT `ttnn.rms_norm`: on this model's real inputs the stock op sits at ~9.65e-4 relative error
    where these four ops sit at 6.6e-8. A norm error is RELATIVE -- it RESCALES the whole branch
    after it -- so it shows up as a NORM RATIO rather than as a PCC drop, and the codec stacks
    eight residual blocks with two or three norms each. Measured: the codec's first transformer
    group came back at norm ratio 1.029 against torch with the stock op, and no single stage of
    the chain looked broken. `tt/vocode_stage.py` spells out the same four ops for the same
    reason, so the two bodies agree.
    """
    shape, padded = x.shape, x.padded_shape
    dims, pdims = [int(v) for v in shape], [int(v) for v in padded]  # (a ttnn.Shape does not slice)
    if dims[:-1] != pdims[:-1] and dims[-1] == pdims[-1]:
        # datamove: only the ROWS carry tile padding (a short / unaligned length) -- the norm runs on the whole
        # tiles through a zero-cost view (rows are independent; the padding rows are finite), so the mean's reduce
        # does not FillPad the padding first, and the result is viewed back without a fill.
        out = _rms_norm(ttnn.reshape(x, padded, padded), gamma, eps, dtype=dtype, memory_config=memory_config)
        return ttnn.reshape(out, shape, padded, skip_padding_fill=True)
    if common.codec_fused_norm():
        # structural: the whole norm as ONE stock ttnn.rms_norm at HiFi4 / fp32 DEST / exact rsqrt
        # (common.codec_rms_norm) instead of five or six full passes over the rows.
        return common.codec_rms_norm(x, gamma, eps, dtype=dtype, memory_config=memory_config)
    scale = ttnn.rsqrt(ttnn.add(ttnn.mean(ttnn.square(x), dim=-1, keepdim=True), eps))
    # `dtype`: the normalised rows can be written narrower (bf16) for the linears that read them;
    # `memory_config`: and placed where they read them from.
    out = {"dtype": dtype} if dtype is not None else {}
    if memory_config is not None:
        out["memory_config"] = memory_config
    if gamma is None:
        return ttnn.multiply(x, scale, **out)
    y = ttnn.multiply(x, scale)
    return ttnn.multiply(y, gamma, **out)


def _softmax(x, dim=-1):
    """`exp(x - max) / sum(exp(x - max))`, spelled out in three ops.

    NOT `ttnn.softmax`. Measured on this build against a float64 reference, the stock op's rows do
    not sum to 1: mean 0.9943, worst 0.9611, for ~1.8e-2 relative error -- and NO flag changes it
    (`numeric_stable=True` and `compute_kernel_config` all return the identical tensor). These
    three ops sit at 5.5e-8, and renormalising the stock op's output only reaches 2.1e-2, so its
    per-element values are wrong too, not just its sum.

    A softmax that does not sum to 1 ATTENUATES the attention output it weights. That reads as a
    NORM RATIO below 1 at a PCC of 0.9999, so a PCC-only gate cannot see it, and it is invisible
    in any layer whose residual is already large. The acoustic stubs beside this file spell the
    same three ops out for the same reason.
    """
    e = ttnn.subtract(x, ttnn.max(x, dim=dim, keepdim=True), activations=[ttnn.UnaryOpType.EXP])
    return ttnn.divide(e, ttnn.sum(e, dim=dim, keepdim=True))


def _largest_divisor(n, cap):
    return max(d for d in range(1, min(n, cap) + 1) if n % d == 0)


# Fidelity: the codec attention's head-batched products at HiFi2 (fp32 DEST kept).
_BMM_HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
)


def _bmm(a, b, transpose_b=False):
    """Head-batched attention `a @ b` over the full grid, every (batch, head) its own work unit.
    With no program config these `[B, H, S, S]` products ran on 1-64 cores at 237-818 us."""
    m, k, n = (-(-int(d) // 32) for d in (a.shape[-2], a.shape[-1], b.shape[-2 if transpose_b else -1]))
    # NEVER SPLIT M. The reuse factory steps a WHOLE batch (M x K tiles of `a`) between the output
    # blocks one core owns, so once M is split into blocks and there are more blocks than cores a
    # core's second M block reads and writes the next head's rows. That is what broke the audio in
    # 27033d4462 (4-tile M blocks), and what a 1-tile split did on the long sequences the e2e test
    # decodes. Here every (batch, head) is one whole-M block -- B * H = 128 of them, more than the
    # grid, so a split is never safe -- the acoustic stubs' rule; a block that outgrows L1 (a longer
    # utterance) falls back to the default.
    tile = lambda t: 4096 if t.dtype == ttnn.float32 else 2048
    if 2 * m * k * tile(a) + 2 * k * n * tile(b) + 2 * m * n * 4096 > 1_200_000:
        return ttnn.matmul(a, b, transpose_b=transpose_b, compute_kernel_config=_BMM_HIFI2)
    grid = a.device().compute_with_storage_grid_size()
    cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=k,
        out_subblock_h=1,
        out_subblock_w=_largest_divisor(n, 4),
        per_core_M=m,
        per_core_N=n,
    )
    return ttnn.matmul(a, b, transpose_b=transpose_b, program_config=cfg, compute_kernel_config=_BMM_HIFI2)


def build(device, torch_module):
    stack = torch_module
    blocks_torch = [stack.layers[str(i)] for i in stack.layers_ids]
    dim = int(blocks_torch[0].dim)

    # One upload per distinct mask on this device (tt/common.shared_device_mask): every block build
    # with the same window shares it.
    mask = common.shared_device_mask(
        device,
        _alibi_window_mask(
            blocks_torch[0].attention.alibi_slopes.detach(),
            int(stack.args.attn_sliding_window_size),
            _MASK_MAX_SEQ,
        ),
        ttnn.float32,
        lambda m: _from_torch(m, device, dtype=ttnn.float32),
    )
    window = int(stack.args.attn_sliding_window_size)
    blocks = [_compile_block(device, blk, mask, window) for blk in blocks_torch]

    def codec_transformer(hidden_states, **kwargs):
        # The leading bound comes from the TENSOR, never from a literal 1: the pipeline drives this
        # with 32 independent samples stacked on axis 0, and a hardcoded 1 would silently decode
        # only the first of them.
        shape = [int(v) for v in hidden_states.shape]
        seq = shape[-2]
        batch = shape[0] if len(shape) >= 3 else 1
        if seq > _MASK_MAX_SEQ:
            raise NotImplementedError(
                f"sequence {seq} exceeds the prebuilt ALiBi mask ({_MASK_MAX_SEQ}); raise "
                f"_MASK_MAX_SEQ -- the mask cannot be rebuilt inside the forward"
            )
        h = ttnn.reshape(hidden_states, [batch, 1, seq, dim])
        for block in blocks:
            h = block(h)
        return ttnn.reshape(h, [batch, seq, dim])

    return codec_transformer

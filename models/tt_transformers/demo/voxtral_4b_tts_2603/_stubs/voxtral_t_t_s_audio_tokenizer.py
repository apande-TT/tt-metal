# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `voxtral_t_t_s_audio_tokenizer` (`audio_tokenizer`) -- the whole neural
codec decoder, from integer codes to a waveform.

    quantizer.decode -> conv(292->1024, k3) -> [transformer, upsample] x4 -> output_proj -> depatch

The open-source checkpoint ships the DECODER only (no `input_proj.*` / `encoder_blocks.*`), so
there is no encode path to port.

**Everything runs channels-LAST**, `[B, 1, T, C]`. The reference permutes to channels-first around
each convolution and back for each transformer; in ttnn the matmul-shaped convolution wants
channels last anyway, so the whole chain stays in one layout and the only transpose left is the one
that brings the acoustic codes' `[B, 36, T]` around.

Sliding windows DOUBLE up the decoder -- 2, 4, 8, 16 -- because each transposed convolution doubles
the frame rate. Each stage's ALiBi mask is built from that stage's own window.

Two padding modes appear, and they are different:
  * the first convolution pads `replicate` (the boundary sample repeated), and
  * `output_proj` pads `reflect` -- `[x6, x5, x4, x3, x2, x1, x0, x1, ...]`, a mirror about index 0
    that EXCLUDES index 0 itself.

**The activation path runs in float32.** With everything in bfloat16 the full chain landed at PCC
0.9873: each stage is fine on its own, but eight codec residual blocks plus five convolutions
compound. Weights stay bfloat16 -- `ttnn.linear` takes a float32 activation against a bfloat16
weight and returns float32 -- so only the activations widen. Q/K/V (and the ALiBi mask that has to
match them) are cast back down for SDPA, which rejects float32 outright
(`sdpa_device_operation.cpp:43`).

Frame arithmetic for a 64-frame input: 64 -> 64 -> 128 -> 256 -> 512 frames, then `output_proj`
emits 240 samples per frame, de-patched to 122880 samples (12.5 Hz frames, 1920 samples each after
the 8x upsampling, 24 kHz).
"""

from __future__ import annotations

import math

import torch

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import common, cpp_band_attn, cpp_shift_add, cpp_softmax, cpp_upsample2

# A PERSISTENT ZERO BUFFER, NOT A PER-CALL `ttnn.zeros`.
# `ttnn.zeros` builds its tensor on the host and enqueues a WRITE to land it on the device, and a
# captured trace cannot replay a write -- capturing this stage died on `TT_FATAL: Writes are not
# supported during trace capture`. The shape is a function of the input shape, which a trace pins,
# so the buffer is created once per (device, shape, dtype) and reused. It is READ-ONLY here and is
# therefore never deallocated by a caller.
_ZEROS = {}


def _zeros_like_buf(device, shape, dtype):
    key = (id(device), tuple(int(s) for s in shape), str(dtype))
    buf = _ZEROS.get(key)
    if buf is None:
        buf = ttnn.zeros(list(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        _ZEROS[key] = buf
    return buf


_SHARD_HEIGHT = 32
_MASK_MAX_SEQ = 2048
_MASK_NEG = -1.0e9


def _from_torch(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
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
    return _from_torch(w.contiguous(), device, dtype=ttnn.bfloat8_b)  # dtype rung: bf8_b codec weights


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


def _fold_linear(x, w, **kwargs):
    """`ttnn.linear` with the leading batch folded into M, so the weight streams ONCE.

    A `[B, 1, T, K]` activation against a 2-D weight runs as B separate matmuls that each re-read
    the whole weight; `[1, 1, B*T, K]` is one matmul. A T that is not tile-aligned makes the fold a
    real relayout each way, still far cheaper than re-reading the weight B times. Tall results get
    the same hand-sized full-grid config the part-chain stubs' `_lin` uses.
    """
    # fidelity: a caller may ask for a lower fidelity than the tall linears' HiFi2 (the FFN gate / up: LoFi).
    fidelity = kwargs.pop("fidelity", ttnn.MathFidelity.HiFi2)
    shape = [int(d) for d in x.shape]
    lead = 1
    for d in shape[:-2]:
        lead *= d
    if lead == 1:
        return ttnn.linear(x, w, **kwargs)
    rows = lead * shape[-2]
    padded = lead * (-(-shape[-2] // 32) * 32)
    if shape[-2] % 32 != 0 and padded >= 128 and "program_config" not in kwargs:
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
    y = ttnn.linear(ttnn.reshape(x, [1, 1, rows, shape[-1]]), w, **kwargs)
    return ttnn.reshape(y, shape[:-1] + [int(y.shape[-1])])


_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
)


def _row(x4, index):
    """One sequence row of a `[B, 1, T, C]` tensor, as `[B, 1, 1, C]`."""
    batch, channels = int(x4.shape[0]), int(x4.shape[-1])
    return ttnn.slice(x4, [0, 0, index, 0], [batch, 1, index + 1, channels])


def _alibi_window_mask(slopes, window, seq):
    """`[1, H, seq, seq]`: ALiBi bias `slope[h] * (j - i)`, blocked where `j > i` or `j < i - window`.

    Depends only on `j - i`, so the top-left `[S, S]` corner is exactly the mask for a length-`S`
    sequence -- which is what lets the forward stay free of torch calls (the runtime native probe
    graduates only at zero torch ops, so a per-call rebuild is not an option).
    """
    pos = torch.arange(seq)
    rel = pos.unsqueeze(0) - pos.unsqueeze(1)
    bias = slopes.reshape(-1, 1, 1).float() * rel.unsqueeze(0).float()
    blocked = (rel > 0) | (rel < -window)
    return bias.masked_fill(blocked.unsqueeze(0), _MASK_NEG).unsqueeze(0)


def _compile_codec_block(device, blk, mask, window=None):
    """One `CodecTransformerBlock` as a callable on a `[B, 1, T, dim]` ttnn tensor."""
    attn = blk.attention
    ff = blk.feed_forward
    args = blk.args
    n_heads = int(attn.n_local_heads)
    n_kv_heads = int(attn.n_local_kv_heads)
    scale = 1.0 / math.sqrt(int(args.head_dim))
    qk_norm = bool(args.qk_norm)
    dim = int(blk.dim)

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
        if seq < 32 and h.memory_config().buffer_type != ttnn.BufferType.L1:
            # shard: a short sequence's float32 residual stream (~19 KB a core) moves to L1, so the
            # residual adds and every step of the spelled-out RMS norms read and write L1 instead of DRAM.
            h = ttnn.to_memory_config(h, ttnn.L1_MEMORY_CONFIG)
        # bf16 normalised rows into the q / k / v linears (fp32 DEST; q / k come back bf16, v fp32).
        # shard: a short sequence's normed rows (q / k / v's input) in L1.
        xn_mem = ttnn.L1_MEMORY_CONFIG if int(h.shape[-2]) < 32 else None
        xn = _rms_norm(h, attn_gamma, attn_eps, dtype=ttnn.bfloat16, memory_config=xn_mem)
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
        q = _fold_linear(xn, wq, compute_kernel_config=_COMPUTE, dtype=qk_dt, memory_config=qk_mem)
        k = _fold_linear(xn, wk, compute_kernel_config=_COMPUTE, dtype=qk_dt, memory_config=qk_mem)
        # shard: a tall float32 v (<= 16 MB) stays in L1 for the band attention.
        v_mem = ttnn.L1_MEMORY_CONFIG if qkv_mem is None and qk_rows * int(wv.shape[-1]) * 4 <= (16 << 20) else qkv_mem
        v = _fold_linear(xn, wv, compute_kernel_config=_COMPUTE, dtype=ttnn.float32, memory_config=v_mem)
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
        # and the whole reduction in FLOAT32, which SDPA cannot do at any fidelity. Matches the
        # part-chain stubs (`codec_transformer`, `codec_transformer_block`, `codec_attention`)
        # exactly, so the two halves of the batch run the same arithmetic.
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
        # shard: the float32 residual branches (o_proj, w2) land in L1 while the rows fit (<= 4096).
        res_mem = xn_mem if xn_mem is not None else common.l1_while_rows_fit(h)
        r = _fold_linear(
            a,
            wo,
            memory_config=res_mem,  # shard: the residual branch stays in L1 while the rows fit
            dtype=ttnn.float32,
            compute_kernel_config=_COMPUTE,
        )
        if attn_scale is not None:
            r = ttnn.multiply(r, attn_scale)
        h = ttnn.add(h, r)
        ttnn.deallocate(r)  # shard: free the attention branch's L1 before the FFN norm

        # shard rung: the FFN's two activations -- the normed rows w1 / w3 read and the gated product w2
        # reads (the latter only to 2048 rows) -- live in L1 (interleaved) while they fit, so in0 reads skip DRAM.
        ffn_mem = common.l1_while_rows_fit(h)
        hn = _rms_norm(h, ffn_gamma, ffn_eps, dtype=ttnn.bfloat16, memory_config=ffn_mem)
        # dtype: the FFN hidden (w1 / w3 outputs, the gate) in bf16; w2 still sums in fp32 DEST
        # and writes the fp32 residual branch.
        # dtype: a tall FFN hidden (>= 256 rows) is written bf8_b -- w1 / w3 write, and the gated multiply
        # reads, half the bytes; so is the gated product the multiply writes and w2 reads.
        hid_dt = ttnn.bfloat16 if common.l1_while_rows_fit(h, 255) else ttnn.bfloat8_b
        # shard: the bf8_b hidden (<= 2048 rows: ~9 MB each) lands in L1 for the gated multiply that reads it.
        hid_mem = common.l1_while_rows_fit(h, 2048)
        gate = _fold_linear(
            hn, w1, compute_kernel_config=_COMPUTE, dtype=hid_dt, memory_config=hid_mem, fidelity=ttnn.MathFidelity.LoFi
        )
        up = _fold_linear(
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
        r = _fold_linear(
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


def _compile_causal_conv1d(device, mod):
    """A `CausalConv1d` as a callable on `[B, 1, T, C_in]` -> `[B, 1, T', C_out]`."""
    conv = mod.conv
    weight = conv.weight.detach()
    out_channels, in_channels, kernel = (int(v) for v in weight.shape)
    stride = int(conv.stride[0])
    dilation = int(conv.dilation[0])
    effective_kernel = int(mod._effective_kernel_size)
    padding_total = int(mod._padding_total)
    pad_mode = str(mod.pad_mode)
    if pad_mode not in ("replicate", "reflect"):
        raise NotImplementedError(f"pad_mode {pad_mode!r} is not ported")
    reflect = pad_mode == "reflect"

    taps = [_from_torch(weight[:, :, i].transpose(0, 1).contiguous(), device) for i in range(kernel)]
    bias = None
    if conv.bias is not None:
        bias = _from_torch(conv.bias.detach().reshape(1, 1, 1, out_channels), device)
    # structural: the K taps side by side, for one wide product over the padded rows (tt/cpp_shift_add).
    wide = cpp_shift_add.wide_weight(taps) if kernel > 1 and stride == 1 and dilation == 1 else None

    def _wide_linear(x, w, **kwargs):
        # The tall codec linears' own full-grid config and HiFi2 (what each per-tap product ran), sized by
        # the product's real output dtype.
        cfg = _mcast_cfg(x, w, int(x.shape[-2]), kwargs.get("dtype") or ttnn.float32)
        if cfg is not None:
            kwargs["program_config"] = cfg
            kwargs["compute_kernel_config"] = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
            )
        return ttnn.linear(x, w, **kwargs)

    def run(x4):
        batch, length = int(x4.shape[0]), int(x4.shape[-2])
        n_frames = (length - effective_kernel + padding_total) / stride + 1
        target = (math.ceil(n_frames) - 1) * stride + (effective_kernel - padding_total)
        extra = target - length

        mode = "reflect" if reflect else "replicate"
        if wide is not None and (padding_total > 0 or extra > 0) and cpp_shift_add.supports_unpadded(x4, mode):
            # structural: the wide product straight from the UNPADDED rows, the padding resolved in the shift-add
            # (a padded row's product is the product of the row it copies) -- no row-major copy, no concat.
            out_len = (length + padding_total + extra - effective_kernel) // stride + 1
            acc = cpp_shift_add.conv_wide_unpadded(
                x4, wide, mode, padding_total, out_len, _COMPUTE, linear=_wide_linear
            )
            return acc if bias is None else ttnn.add(acc, bias)

        # Edge rows cut and joined ROW_MAJOR, the padded block tilized once (a one-row TILE slice is
        # re-tilized row by row, and a TILE concat of unaligned pieces untilizes them all again).
        xr = ttnn.to_layout(x4, ttnn.ROW_MAJOR_LAYOUT) if padding_total > 0 or extra > 0 else x4
        pieces = []
        if padding_total > 0:
            if reflect:
                pieces.extend(_row(xr, i) for i in range(padding_total, 0, -1))
            else:
                first = _row(xr, 0)
                pieces.append(first if padding_total == 1 else ttnn.repeat(first, [1, 1, padding_total, 1]))
        pieces.append(xr)
        if extra > 0:
            if reflect:
                pieces.extend(_row(xr, length - 1 - i) for i in range(1, extra + 1))
            else:
                last = _row(xr, length - 1)
                pieces.append(last if extra == 1 else ttnn.repeat(last, [1, 1, extra, 1]))
        padded_rm = None if len(pieces) == 1 else ttnn.concat(pieces, dim=2)
        padded = pieces[0] if padded_rm is None else None

        padded_len = length + padding_total + extra
        out_len = (padded_len - effective_kernel) // stride + 1

        if wide is not None and padded_rm is not None and cpp_shift_add.supports(padded_rm, taps, stride, dilation):
            # structural: ONE tilize of the padded rows and ONE product against the taps side by side; the
            # row shift happens on the narrow tap outputs (tt/cpp_shift_add), not as K copies of the input.
            acc = cpp_shift_add.conv_wide(padded_rm, wide, out_len, _COMPUTE, linear=_wide_linear)
            return acc if bias is None else ttnn.add(acc, bias)
        acc = None
        for i, tap in enumerate(taps):
            begin = i * dilation
            end = begin + (out_len - 1) * stride + 1
            step = [1, 1, stride, 1] if stride > 1 else None
            if padded_rm is not None:
                # Cut from the row-major rows, then tilized: a row-offset slice of a TILE tensor
                # untilizes it whole for every tap.
                seg = ttnn.to_layout(
                    ttnn.slice(padded_rm, [0, 0, begin, 0], [batch, 1, end, in_channels], step), ttnn.TILE_LAYOUT
                )
            else:
                seg = ttnn.slice(padded, [0, 0, begin, 0], [batch, 1, end, in_channels], step)
            term = _fold_linear(seg, tap, compute_kernel_config=_COMPUTE)
            acc = term if acc is None else ttnn.add(acc, term)
        return acc if bias is None else ttnn.add(acc, bias)

    return run


def _compile_causal_conv_transpose1d(device, mod):
    """A `CausalConvTranspose1d` as a callable on `[B, 1, T, C]` -> `[B, 1, T * 2, C]`."""
    conv = mod.conv
    weight = conv.weight.detach()
    in_channels, out_channels, kernel = (int(v) for v in weight.shape)
    stride = int(conv.stride[0])
    if stride != 2 or kernel != 4:
        raise NotImplementedError(f"only kernel 4 / stride 2 is ported, got {kernel}/{stride}")
    right_trim = math.ceil((kernel - stride) * float(mod.trim_ratio))
    if (kernel - stride) - right_trim != 0:
        raise NotImplementedError("a non-zero left trim is not ported")

    taps = [_from_torch(weight[:, :, i].contiguous(), device) for i in range(kernel)]
    bias = None
    if conv.bias is not None:
        bias = _from_torch(conv.bias.detach().reshape(1, 1, 1, out_channels), device)
    # structural: the four taps side by side, for ONE product and ONE interleaving shift-add (tt/cpp_upsample2).
    # dtype: the wide weight built on the HOST as bf8_b (the taps side by side, as wide_weight lays
    # them out) instead of a device concat of float32 / bf16 taps.
    wide = (
        _from_torch(
            torch.cat([weight[:, :, i] for i in range(kernel)], dim=-1).contiguous(), device, dtype=ttnn.bfloat8_b
        )
        if cpp_upsample2.enabled() and out_channels % 32 == 0
        else None
    )

    def _wide_linear(x, w, **kwargs):
        # The tall codec linears' own full-grid config, over the batch-folded, tile-padded rows of the 4-D input.
        # fidelity: the transposed-conv products at LoFi (fp32 DEST kept).
        cfg = _mcast_cfg(x, w, cpp_upsample2.tile_rows(x), kwargs.get("dtype") or ttnn.float32)
        if cfg is not None:
            kwargs["program_config"] = cfg
            kwargs["compute_kernel_config"] = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True
            )
        return ttnn.linear(x, w, **kwargs)

    def run(x4):
        batch, length = int(x4.shape[0]), int(x4.shape[-2])
        if wide is not None and cpp_upsample2.supports(x4, taps):
            # out[2m + j] = x[m] @ W_j + x[m - 1] @ W_{2+j}, written interleaved and trimmed to 2L in one pass:
            # no zero-row concats (untilize + concat + tilize each), no even | odd concat, no relayout reshape.
            out = cpp_upsample2.apply(x4, wide, out_channels, length * stride, _COMPUTE, linear=_wide_linear)
            return out if bias is None else ttnn.add(out, bias)
        zero_row = _zeros_like_buf(device, [batch, 1, 1, out_channels], x4.dtype)

        def _delayed(tap):
            """`tap` applied to the PREVIOUS input step: a zero row, then steps 0..L-2."""
            head = ttnn.slice(x4, [0, 0, 0, 0], [batch, 1, length - 1, in_channels])
            # grid: the unaligned rows miss `_fold_linear`'s config -- the hand full-grid one (rows padded to a tile).
            rows = -(-batch * (length - 1) // 32) * 32
            cfg = _mcast_cfg(head, tap, rows, head.dtype) if rows >= 128 else None
            extra = {} if cfg is None else {"program_config": cfg}
            return ttnn.concat([zero_row, _fold_linear(head, tap, compute_kernel_config=_COMPUTE, **extra)], dim=2)

        even = ttnn.add(_fold_linear(x4, taps[0], compute_kernel_config=_COMPUTE), _delayed(taps[2]))
        odd = ttnn.add(_fold_linear(x4, taps[1], compute_kernel_config=_COMPUTE), _delayed(taps[3]))
        out = ttnn.reshape(ttnn.concat([even, odd], dim=-1), [batch, 1, length * stride, out_channels])
        return out if bias is None else ttnn.add(out, bias)

    return run


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

    NOT `ttnn.softmax`: measured on this build against a float64 reference, the stock op's rows do
    not sum to 1 (mean 0.9943, worst 0.9611, ~1.8e-2 relative error) and no flag changes it. These
    three ops sit at 5.5e-8. A softmax that does not sum to 1 ATTENUATES the attention output it
    weights, which reads as a norm ratio below 1 at a PCC of 0.9999.
    """
    e = ttnn.subtract(x, ttnn.max(x, dim=dim, keepdim=True), activations=[ttnn.UnaryOpType.EXP])
    return ttnn.divide(e, ttnn.sum(e, dim=dim, keepdim=True))


# THE SEMANTIC CODEBOOK IS GATHERED IN TWO HALVES.
# `ttnn.embedding` requires a bfloat16 table (`embedding_device_operation.cpp:36`), and this table
# is the codec's INPUT -- everything downstream amplifies whatever it gets wrong. Measured on this
# checkpoint: a single bfloat16 table put the quantizer latent at 1.613e-3 relative, and the
# 292->1024 convolution that consumes it amplified that to 7.7e-3, which then dominated every
# later stage. So the table is split the way a compensated matmul splits a weight --
# `hi = bf16(t)`, `lo = bf16(t - hi)` -- and the two gathers are added back in float32. Two
# bfloat16 mantissas end to end is ~1e-5 on a table this size, for one extra gather and one add
# on 416 rows.
def _split_table(weight, device):
    hi = weight.to(torch.bfloat16)
    lo = (weight - hi.float()).to(torch.bfloat16)
    return (
        _from_torch(hi.contiguous(), device, layout=ttnn.ROW_MAJOR_LAYOUT),
        _from_torch(lo.contiguous(), device, layout=ttnn.ROW_MAJOR_LAYOUT),
    )


def _split_embedding(ids, tables, layout=None):
    hi, lo = tables
    layout = ttnn.TILE_LAYOUT if layout is None else layout
    return ttnn.add(
        ttnn.typecast(ttnn.embedding(ids, hi, layout=layout), ttnn.float32),
        ttnn.typecast(ttnn.embedding(ids, lo, layout=layout), ttnn.float32),
    )


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
    codec = torch_module
    quant = codec.quantizer
    semantic = quant.semantic_codebook
    acoustic = quant.acoustic_codebook

    n_semantic = int(semantic.num_codebooks)
    n_acoustic = int(acoustic.num_codebooks)
    n_levels = int(acoustic.n_levels)
    shift = (n_levels - 1) / 2.0
    scale = 2.0 / (n_levels - 1)
    latent_dim = int(codec.latent_dim)
    patch_size = int(codec.patch_size)

    table = _split_table(semantic.embedding.detach(), device)

    stages = []
    for blk in codec.decoder_blocks:
        name = type(blk).__name__
        if name == "CausalConv1d":
            stages.append(_compile_causal_conv1d(device, blk))
        elif name == "CausalConvTranspose1d":
            stages.append(_compile_causal_conv_transpose1d(device, blk))
        elif name == "CodecTransformer":
            # One upload per distinct mask on this device (tt/common.shared_device_mask): every block build
            # with the same window shares it.
            mask = common.shared_device_mask(
                device,
                _alibi_window_mask(
                    blk.layers["0"].attention.alibi_slopes.detach(),
                    int(blk.args.attn_sliding_window_size),
                    _MASK_MAX_SEQ,
                ),
                ttnn.float32,
                lambda m: _from_torch(m, device, dtype=ttnn.float32),
            )
            window = int(blk.args.attn_sliding_window_size)
            blocks = [_compile_codec_block(device, blk.layers[str(i)], mask, window) for i in blk.layers_ids]

            def _stack(x4, _blocks=blocks):
                for b in _blocks:
                    x4 = b(x4)
                return x4

            stages.append(_stack)
        else:
            raise NotImplementedError(f"decoder block {name} is not ported")
    output_proj = _compile_causal_conv1d(device, codec.output_proj)

    def voxtral_t_t_s_audio_tokenizer(codes, **kwargs):
        batch, rows, frames = (int(v) for v in codes.shape)
        if frames * 8 > _MASK_MAX_SEQ:
            raise NotImplementedError(
                f"{frames} frames upsample to {frames * 8}, past the prebuilt ALiBi mask "
                f"({_MASK_MAX_SEQ}); raise _MASK_MAX_SEQ -- the mask cannot be rebuilt inside the "
                f"forward"
            )

        sem_codes = ttnn.reshape(ttnn.slice(codes, [0, 0, 0], [batch, n_semantic, frames]), [batch, frames])
        # `ttnn.embedding` requires a bfloat16 table (`embedding_device_operation.cpp:36`);
        # widen once here so every residual add downstream happens in float32.
        sem = ttnn.typecast(_split_embedding(sem_codes, table, layout=ttnn.TILE_LAYOUT), ttnn.float32)
        aco_codes = ttnn.typecast(
            ttnn.to_layout(
                ttnn.slice(codes, [0, n_semantic, 0], [batch, n_semantic + n_acoustic, frames]),
                ttnn.TILE_LAYOUT,
            ),
            ttnn.float32,
        )
        aco = ttnn.transpose(ttnn.multiply(ttnn.subtract(aco_codes, shift), scale), -2, -1)

        # float32 on BOTH sides: the semantic half is a two-gather split now and comes back
        # float32, and `ttnn.concat` requires a single dtype.
        h = ttnn.reshape(
            ttnn.concat([sem, ttnn.typecast(aco, ttnn.float32)], dim=-1),
            [batch, 1, frames, latent_dim],
        )
        for stage in stages:
            h = stage(h)
        h = output_proj(h)

        # "b (c h) t -> b c (t h)" with h = patch_size: channels-last, that is just a flatten.
        # flat=False hands back the channels-last [B, 1, T, patch] for a caller that flattens itself.
        if not kwargs.get("flat", True):
            return h
        out_frames = int(h.shape[-2])
        return ttnn.reshape(h, [batch, 1, out_frames * patch_size])

    return voxtral_t_t_s_audio_tokenizer

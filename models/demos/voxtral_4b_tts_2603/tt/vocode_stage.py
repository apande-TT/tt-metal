# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The `vocode` section: integer audio codes -> a 24 kHz waveform, on the device.

This is the neural audio CODEC DECODER of `mistralai/Voxtral-4B-TTS-2603`
(`hf_model.audio_tokenizer`), composed from the graduated stubs in Source B. The chain the
reference runs, and the chain this stage runs:

    codes [B, 37, T]
      -> quantizer.decode        semantic row 0: an 8192 x 256 Euclidean table (built as
                                 embedding_sum / cluster_usage, both BUFFERS);
                                 acoustic rows 1..36: 21 FSQ levels rescaled to [-1, 1]
      -> latent [B, 292, T]
      -> CausalConv1d(292 -> 1024, k3, s1, pad_mode="replicate")
      -> CodecTransformer(window 2)    x n_layers
      -> CausalConvTranspose1d(k4, s2) -> CodecTransformer(window 4)
      -> CausalConvTranspose1d(k4, s2) -> CodecTransformer(window 8)
      -> CausalConvTranspose1d(k4, s2) -> CodecTransformer(window 16)
      -> output_proj CausalConv1d(1024 -> 240, k7, pad_mode="reflect", WEIGHT-NORMED)
      -> de-patch "b (c h) t -> b c (t h)", h = 240
      -> waveform [B, 1, T * 1920]

The windows DOUBLE because each transposed convolution doubles the frame rate, so each group's
ALiBi mask is built from its OWN window. The two padding modes are different and both matter: the
first convolution replicates the boundary sample, `output_proj` REFLECTS -- a mirror about index 0
that excludes index 0 itself.

TWO IMPLEMENTATIONS, SPLIT BY BATCH ROW. Source B graduated the whole section
(`voxtral_t_t_s_audio_tokenizer`) AND twelve of its parts, which cover the same arithmetic. Running
both at the same position would compute every sample twice, so the batch is SPLIT: rows
`[0:split]` go through an explicit chain assembled here from the part stubs, rows `[split:B]`
through the whole-section body, and the two waveforms are concatenated. Every sample is decoded
exactly once, and the per-sample PCC gate covers both implementations. `split` defaults to `B // 2`;
below B=2 the whole-section body takes everything.

The part chain routes each position to exactly one stub:

    decoder_blocks[0]      causal_conv1d
    group 0 (window 2)     codec_transformer            (one stub per block, a stack of one)
    upsample 0, 1          causal_conv_transpose1d
    group 1 (window 4)     codec_transformer_block
    upsample 2             parametrized_conv_transpose1d + the causal right trim
    group 2 (window 8)     codec_attention              (inside the residual block assembled here)
    group 3 (window 16)    codec_transformer
    output_proj            parametrized_conv1d, fed the weight that
                           parametrization_list(weight_norm(g, v)) reconstructs ON THE DEVICE
    quantizer              mistral_audio_codebook for the first half of the part rows, and
                           semantic_codebook + acoustic_codebook -- the two codebooks that
                           `MistralAudioCodebook` IS -- for the second half

`output_proj`'s weight is a torch parametrization, `w = g * v / ||v||`, recomputed on every
reference forward; `g` is signed (23 of 1024 channels are negative in the sibling layer), so the
identity is `||w|| == |g|`. The two weight-space stubs run in the forward for the same reason torch
does: the tensor they return IS the weight the convolution consumes.

FLOAT32 ACTIVATIONS, BFLOAT16 WEIGHTS. All-bfloat16 put the full chain at PCC 0.9873 -- every stage
is fine alone, but eight residual blocks plus five convolutions compound. Q/K/V and the ALiBi mask
narrow to bfloat16 for SDPA, which rejects float32 outright.

The codec attention's two head-batched products take a full-grid reuse program config with each
(batch, head) one whole-M block (the codec stubs' `_bmm`): splitting M across blocks is what broke
the audio before (27033d4462).

No device is ever opened here.
"""

from __future__ import annotations

import math

import torch

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import common, cpp_addnorm, cpp_shift_add, cpp_wave

# fp32 accumulation in DEST for the ops this module owns (the stubs carry their own copy).
_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
)


# The stubs that prebuild an ALiBi mask, and therefore set the frame ceiling. Their
# `_MASK_MAX_SEQ` is READ (never copied) so this stage's `max_frames` moves when the constant does.
_MASK_OWNERS = (
    "voxtral_t_t_s_audio_tokenizer",
    "codec_transformer",
    "codec_transformer_block",
    "codec_attention",
)

# Which part stub runs each transformer GROUP. Cycled over the groups, so all three routes are
# live no matter how many groups the config declares (this checkpoint declares four).
_GROUP_ROUTES = ("codec_transformer", "codec_transformer_block", "codec_attention")


# ----------------------------------------------------------------------------------------
# build-time weight prep (host side -- never reached from a forward)
# ----------------------------------------------------------------------------------------


# FLOAT32 WEIGHTS, to match the stubs. The codec's part chain routes one transformer group
# through `_attention_block` below, whose feed-forward weights are built HERE rather than inside a
# stub -- so while the stub-owned groups moved to float32 weights this one silently stayed at
# bfloat16, and it was the group where the worst sample lost its accuracy (PCC 0.99995 going in,
# 0.99511 coming out, while every other sample held 0.9998). Measured on this device, one matmul:
# a bfloat16 weight costs 1.738e-3 relative against float64 where a float32 weight costs 1.169e-3.
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
            # Fidelity: the tall (compute-bound) codec linears at HiFi2 (the attention products too).
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


def _matmul_weight(linear, device):
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


def _norm_gamma(norm, device):
    """`[1, 1, 1, dim]` float32 TILE -- the form `_rms_norm`'s final multiply takes."""
    return _from_torch(norm.weight.detach().reshape(1, 1, 1, -1), device, dtype=ttnn.float32)


def _rms_norm(x, gamma, eps, dtype=None, memory_config=None):
    """`x * rsqrt(mean(x^2) + eps) * gamma`, spelled out, entirely in float32.

    NOT `ttnn.rms_norm`: on this model's real inputs the stock op sits at 9.65e-4 relative error
    against the reference where these four ops sit at 6.6e-8. A norm error is RELATIVE, so it
    rescales the whole branch that follows it, and this stage's output is rounded onto 21 levels
    0.1 apart in x -- a fifth of all values land within 0.01 of a code edge, so a error of that
    size is the difference between the right audio code and the one next to it. The graduated
    stubs beside this file (`flow_matching_audio_transformer`, `acoustic_transformer_block`)
    already spell it out; this is the same four ops so the two bodies agree.
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


class _StackOfOne:
    """A `CodecTransformer`-shaped view over a SUBSET of its blocks.

    `codec_transformer`'s `build` reads `stack.args`, `stack.layers_ids` and `stack.layers[str(i)]`.
    Handing it a view holding one block makes the stub run exactly one block position, which is
    what lets every element of `VocodeStage.blocks` own its own callable -- and it is also how the
    `layers` cap is applied without mutating the reference model.
    """

    def __init__(self, stack, layer_ids):
        self.args = stack.args
        self.layers_ids = list(layer_ids)
        self.layers = {str(i): stack.layers[str(i)] for i in self.layers_ids}


# ----------------------------------------------------------------------------------------
# the block wrapper the structural walk reads
# ----------------------------------------------------------------------------------------


class CodecBlock:
    """ONE codec transformer block position, whatever stub route runs it.

    The four groups carry different sliding windows (2/4/8/16) and three different stub routes, but
    every member of `VocodeStage.blocks` is one of these: a plain class with a `kind` tag and one
    `__call__`, and deliberately NO `__slots__` -- the structural stack walk reads `__dict__`.
    """

    def __init__(self, kind, group, layer_id, window, run):
        self.kind = kind
        self.group = group
        self.layer_id = layer_id
        self.window = window
        self.run = run

    def __call__(self, hidden, **kwargs):
        """`[B, S, dim]` -> `[B, S, dim]` -- the reference `CodecTransformerBlock` contract."""
        return self.run(hidden, **kwargs)

    def __repr__(self):
        return f"CodecBlock(kind={self.kind!r}, group={self.group}, layer_id={self.layer_id}, " f"window={self.window})"


def _attention_block(device, blk, attention_stub):
    """A `CodecTransformerBlock` built around the `codec_attention` stub.

    Everything the attention does is the stub's; what is assembled here is the residual skeleton
    the block is -- two RMS norms, the two LayerScale vectors and the SwiGLU feed-forward:

        r = attention_scale * attention(attention_norm(x));  h = x + r
        r = ffn_scale      * feed_forward(ffn_norm(h));       out = h + r

    `attention_scale` / `ffn_scale` are per-CHANNEL `[dim]` parameters and this checkpoint's values
    are small and SIGNED, so dropping them would flip the residual's sign rather than rescale it.
    `norm_eps` is 1e-2 here, three orders of magnitude above the text backbone's, and is read off
    the modules.
    """
    if blk.post_attention_norm is not None or blk.post_ffn_norm is not None:
        raise NotImplementedError("post_attention_norm / post_ffn_norm are not ported")

    dim = int(blk.dim)
    attn_gamma = _norm_gamma(blk.attention_norm, device)
    attn_eps = float(blk.attention_norm.eps)
    # The FFN norm's gamma and LayerScale are folded into w1 / w3 and w2 (the attention side's
    # belong to the codec_attention stub's weights, which this skeleton does not build).
    ffn_gamma = None
    ffn_g = blk.ffn_norm.weight.detach()
    ffn_ls = blk.ffn_scale.detach() if blk.layer_scale else None
    ffn_eps = float(blk.ffn_norm.eps)
    w1 = _folded_weight(blk.feed_forward.w1, device, rows=ffn_g)
    w2 = _folded_weight(blk.feed_forward.w2, device, cols=ffn_ls)
    w3 = _folded_weight(blk.feed_forward.w3, device, rows=ffn_g)
    attn_scale = ffn_scale = None
    if blk.layer_scale:
        attn_scale = _from_torch(blk.attention_scale.detach().reshape(1, 1, 1, dim), device, dtype=ttnn.float32)

    def run(x3, **kwargs):
        batch, seq = int(x3.shape[0]), int(x3.shape[-2])
        h = ttnn.reshape(x3, [batch, 1, seq, dim])
        # shard: the float32 residual stream lives in L1 while it is <= 1024 rows (<= 4 MB, ~37 KB a core), so the
        # residual adds and the fused norms read and write L1 instead of DRAM (was: only a short sequence). At
        # 2048 rows its 8 MB on top of the FFN's L1 hidden clashed with the up projection's buffers. The group
        # runners hand it back in its input's memory (common.restore_memory).
        if common.l1_while_rows_fit(h, 1024) is not None and h.memory_config().buffer_type != ttnn.BufferType.L1:
            # shard: a short sequence's float32 residual stream moves to L1 (residual adds + norms in L1).
            h = ttnn.to_memory_config(h, ttnn.L1_MEMORY_CONFIG)

        # bf16 rows into the attention stub's q / k / v; shard: a short sequence's normed rows in L1.
        xn_mem = ttnn.L1_MEMORY_CONFIG if int(h.shape[-2]) < 32 else None
        # shard: the attention norm's bf16 rows (q / k / v's in0) land in L1 up to 2048 rows (<= 4 MB).
        xn = _rms_norm(
            h, attn_gamma, attn_eps, dtype=ttnn.bfloat16, memory_config=xn_mem or common.l1_while_rows_fit(h, 2048)
        )
        # dtype: the codec q / k projections write bf16; the fp32 qk-norm writes the float32 the band attention takes.
        # shard: the stubs keep tall bf16 q / k (<= 8 MB each) and a float32 v (<= 16 MB) in L1.
        # dtype: the conv-transpose stubs build their wide [C_in, 4 C_out] weight on the host as bf8_b
        # (and run its product at LoFi, fp32 DEST).
        r = attention_stub(ttnn.reshape(xn, [batch, seq, dim]))
        r = ttnn.reshape(r, [batch, 1, seq, dim])
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
        # shard: the float32 w2 residual branch lands in L1 while the rows fit (<= 4096).
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
        return ttnn.reshape(ttnn.add(h, r), [batch, seq, dim])

    return run


# ----------------------------------------------------------------------------------------
# the chain steps the part implementation is assembled from
# ----------------------------------------------------------------------------------------


def _group_runner(blocks):
    """The transformer group as a channels-FIRST step, so every chain step has one signature.

    The convolutions run on `[B, C, L]` (the reference's own layout for them) and the transformer
    on `[B, L, C]`; the reference permutes around every convolution for exactly the same reason.
    """

    def run(x_cf):
        return ttnn.transpose(run_cl(x_cf), -2, -1)

    def run_cl(x_cf):
        """The group's output left channels-LAST (`[B, L, C]`), for a next step that takes it that way."""
        h = ttnn.transpose(x_cf, -2, -1)
        mem = h.memory_config()
        for block in blocks:
            h = block(h)
        return common.restore_memory(h, mem)

    def cl(x_cl):
        """The group CHANNELS-LAST in and out (`[B, L, C]`): a neighbour that hands over / takes channels-last
        rows saves the transpose pair a channels-first handoff costs (pure data movement)."""
        h = x_cl
        mem = h.memory_config()
        for block in blocks:
            h = block(h)
        return common.restore_memory(h, mem)

    run.run_cl = run_cl
    run.cl = cl
    return run


def _cl_conv(conv_stub):
    """A conv stub run CHANNELS-LAST in and out: `[B, L, C]` -> `[B, L', C_out]` through its `x_cl` /
    `cl_out` keywords (the stub reads and writes its rows channels-last anyway)."""

    def cl(x_cl):
        batch, length, channels = (int(v) for v in x_cl.shape)
        return conv_stub(x_cl, x_cl=ttnn.reshape(x_cl, [batch, 1, length, channels]), cl_out=True)

    return cl


def _trimmed_upsample(mod, conv_stub):
    """`parametrized_conv_transpose1d` plus the causal trim its wrapper does.

    The bare transposed convolution emits `2L + (k - s)`; `CausalConvTranspose1d` drops
    `ceil((k - s) * trim_ratio)` samples off the RIGHT (nothing off the left at trim_ratio 1.0).
    """
    kernel = int(mod.conv.kernel_size[0])
    stride = int(mod.conv.stride[0])
    total = kernel - stride
    right = math.ceil(total * float(mod.trim_ratio))
    if total - right != 0:
        raise NotImplementedError(f"a non-zero left trim ({total - right}) is not ported")

    def run(x_cf):
        # The stub writes the trimmed length itself when it can (`trim_right`); otherwise it is cut here.
        out = conv_stub(x_cf, trim_right=right)
        batch, channels, length = (int(v) for v in out.shape)
        full = int(x_cf.shape[-1]) * stride + total
        if right == 0 or length == full - right:
            return out
        return ttnn.slice(out, [0, 0, 0], [batch, channels, length - right])

    def cl(x_cl):
        """`run` CHANNELS-LAST in and out (`[B, L, C]` -> `[B, L', C_out]`)."""
        batch, length, channels = (int(v) for v in x_cl.shape)
        out = conv_stub(x_cl, trim_right=right, x_cl=ttnn.reshape(x_cl, [batch, 1, length, channels]), cl_out=True)
        rows, out_channels = int(out.shape[-2]), int(out.shape[-1])
        full = length * stride + total
        if right == 0 or rows == full - right:
            return out
        return ttnn.slice(out, [0, 0, 0], [batch, rows - right, out_channels])

    run.cl = cl
    return run


def _reflect_padded_conv(mod, conv_stub, weight_fn, wide_fn=None):
    """`output_proj`: the REFLECT causal padding, then the bare conv fed a device-side weight.

    `pad_mode="reflect"` mirrors about index 0 and EXCLUDES it -- the six padded rows are
    `x[6], x[5], x[4], x[3], x[2], x[1]`, not `x[0]` six times (that is `replicate`, which the
    FIRST convolution uses). The padding is the `CausalConv1d` wrapper's job, and
    `parametrized_conv1d` is the bare `Conv1d` inside it, so the wrapper's arithmetic is replayed
    here from the module's own `_padding_total` / `_effective_kernel_size`.

    `weight_fn()` is `parametrization_list(weight_norm=...)`: the weight-norm reconstruction, run on
    the device, handed straight to the convolution that consumes it.
    """
    if str(mod.pad_mode) != "reflect":
        raise NotImplementedError(f"pad_mode {mod.pad_mode!r} is not the reflect path")
    stride = int(mod.conv.stride[0])
    effective_kernel = int(mod._effective_kernel_size)
    padding_total = int(mod._padding_total)

    def run(x_cf):
        batch, channels, length = (int(v) for v in x_cf.shape)
        if length <= padding_total:
            raise NotImplementedError(
                f"reflect padding {padding_total} needs more than {length} rows; the reference's "
                f"`pad1d` grows the tensor first, which this port does not do"
            )
        x4 = ttnn.reshape(ttnn.transpose(x_cf, -2, -1), [batch, 1, length, channels])

        n_frames = (length - effective_kernel + padding_total) / stride + 1
        target = (math.ceil(n_frames) - 1) * stride + (effective_kernel - padding_total)
        extra = target - length

        if stride == 1 and cpp_shift_add.supports_unpadded(x4, "reflect"):
            # The conv takes the UNPADDED rows and resolves the reflect padding itself (a padded row's product
            # is the product of the row it mirrors): no row-major copy, no edge-row concat, no tilize.
            return conv_stub(
                x_cf, weight=weight_fn(), x_cl=x4, pad_front=padding_total, pad_back=extra, pad_mode="reflect"
            )

        pieces = _reflect_rows(x4, batch, length, channels, extra)
        padded_rm = None if len(pieces) == 1 else ttnn.concat(pieces, dim=2)
        padded_len = length + padding_total + extra
        if padded_rm is not None:
            # The conv works from the row-major rows alone (no tilize of them, no channels-first transpose that
            # the stub would only transpose back): it takes its length from `x_rm`, and `x` is not read.
            # (A [B, C, L'] reshape of the rows would NOT be a view: ROW_MAJOR rows change length.)
            return conv_stub(x_cf, weight=weight_fn(), x_rm=padded_rm)
        cf = ttnn.transpose(ttnn.reshape(pieces[0], [batch, padded_len, channels]), -2, -1)
        return conv_stub(cf, weight=weight_fn())

    def _reflect_rows(x4, batch, length, channels, extra):
        """The reflect-padded rows as ROW_MAJOR pieces `[pad rows..., x, tail rows...]` (one piece: no padding).

        The edge rows are cut and joined ROW_MAJOR, and the padded block tilized once: a one-row slice of a
        TILE tensor comes back row-major and is re-tilized row by row, and a TILE concat of non-tile-aligned
        pieces untilizes them all again."""
        x_rm = ttnn.to_layout(x4, ttnn.ROW_MAJOR_LAYOUT) if padding_total + extra > 0 else x4

        def row(index):
            return ttnn.slice(x_rm, [0, 0, index, 0], [batch, 1, index + 1, channels])

        pieces = [row(i) for i in range(padding_total, 0, -1)]
        pieces.append(x_rm)
        pieces.extend(row(length - 1 - i) for i in range(1, extra + 1))
        return pieces

    def run_cl(x3):
        """`run` on CHANNELS-LAST rows `[B, L, C]` (a transformer group's own layout), returning channels-last
        `[B, L', C_out]`: the conv reads its rows channels-last anyway, so neither side transposes."""
        batch, length, channels = (int(v) for v in x3.shape)
        x4 = ttnn.reshape(x3, [batch, 1, length, channels])
        n_frames = (length - effective_kernel + padding_total) / stride + 1
        extra = (math.ceil(n_frames) - 1) * stride + (effective_kernel - padding_total) - length
        # structural: `wide_fn()` reconstructs the weight-normed weight straight in the conv's wide layout
        # (parametrization_list(wide=True) -> weight_norm's wide path), so no k-last permute reaches the conv.
        weight = {"wide_weight": wide_fn()} if wide_fn is not None else {"weight": weight_fn()}
        if stride == 1 and length > padding_total and cpp_shift_add.supports_unpadded(x4, "reflect"):
            return conv_stub(
                x3,
                x_cl=x4,
                pad_front=padding_total,
                pad_back=extra,
                pad_mode="reflect",
                cl_out=True,
                **weight,
            )
        if stride == 1 and length > padding_total and padding_total + extra > 0:
            # An unaligned length: the padded rows built ROW_MAJOR from the channels-last rows, the result taken
            # channels-last -- no transpose out to channels-first and back.
            padded_rm = ttnn.concat(_reflect_rows(x4, batch, length, channels, extra), dim=2)
            return conv_stub(x3, x_rm=padded_rm, cl_out=True, **weight)
        return ttnn.transpose(run(ttnn.transpose(x3, -2, -1)), -2, -1)

    run.run_cl = run_cl
    return run


# ----------------------------------------------------------------------------------------
# the stage
# ----------------------------------------------------------------------------------------


class VocodeStage:
    """The resident codec decoder. `decode` is the section's forward; nothing here is host compute."""

    def __init__(
        self,
        device,
        codec,
        blocks,
        n_layers,
        part_chain,
        whole_section,
        multi_vocab,
        code_offset,
        max_frames,
        max_frames_provenance,
        split_at=None,
    ):
        self.device = device
        self.reference_module = codec
        self.blocks = blocks
        self.n_layers = int(n_layers)
        self.group_windows = tuple(sorted({b.window for b in blocks})) if blocks else ()
        self.frame_rate = float(codec.frame_rate)
        self.sampling_rate = int(codec.sampling_rate)
        self.samples_per_frame = int(codec.downsample_factor)
        self.n_codebooks = int(codec.num_codebooks)
        self.code_offset = int(code_offset)
        self.max_frames = int(max_frames)
        self.max_frames_provenance = max_frames_provenance
        self.split_at = None if split_at is None else int(split_at)
        self._part_chain = part_chain
        self._whole_section = whole_section
        self._multi_vocab = multi_vocab

    # -- the section's two entry points ---------------------------------------------------

    def audio_token_embedding(self, codes):
        """`[B, 37, T]` SHIFTED codes -> `[B, 1, T, 3072]`, the SUM over the 37 codebooks.

        This is what the text decode loop feeds back as its next input. The codes stay in the
        SHIFTED space here -- `MultiVocabEmbeddings`'s 37 packed codebooks are indexed with the two
        audio special tokens included; only the quantizer wants them unshifted.

        The time axis is required, not optional: the reference broadcasts its per-codebook offsets
        as `[1, 37, 1]`, so a rank-2 frame would add them along the wrong axis. A rank-2 input is
        given the axis here rather than silently mis-indexed.
        """
        shape = [int(v) for v in codes.shape]
        if len(shape) == 2:
            codes = ttnn.reshape(codes, [shape[0], shape[1], 1])
            shape = shape + [1]
        if len(shape) != 3:
            raise ValueError(f"audio_token_embedding wants [B, {self.n_codebooks}, T], got {shape}")
        if shape[1] != self.n_codebooks:
            raise ValueError(f"audio_token_embedding wants {self.n_codebooks} codebook rows, got {shape[1]}")
        # The lookup table is bfloat16 (`ttnn.embedding` requires it), so the 37-term sum is done
        # widened: `input_embedding_concat_type: "sum"` adds 37 of them and the text stack consumes
        # the result in float32.
        if shape[2] == 1:
            # One frame (the decode feedback): the 37 lookups come back as the ROWS of one
            # `[B, 1, 37, D]` tile tensor and are summed over rows, 1/16 the tiles of `[B, 37, 1, D]`.
            emb = self._multi_vocab(codes, codebooks_as_rows=True)
            return ttnn.sum(ttnn.typecast(emb, ttnn.float32), dim=2, keepdim=True, compute_kernel_config=_COMPUTE)
        emb = self._multi_vocab(codes)
        return ttnn.sum(ttnn.typecast(emb, ttnn.float32), dim=1, keepdim=True)

    def decode(self, codes, split_at=None):
        """`[B, 37, T]` SHIFTED codes -> `[B, 1, T * 1920]` float32 waveform.

        The offset between the acoustic transformer's output space and the codec's own code space
        is subtracted ON DEVICE. Rows `[0:split]` are decoded by the part chain and rows
        `[split:B]` by the whole-section body, so every sample is decoded exactly once.
        """
        shape = [int(v) for v in codes.shape]
        if len(shape) != 3:
            raise ValueError(f"decode wants [B, {self.n_codebooks}, T], got {shape}")
        batch, rows, frames = shape
        if rows != self.n_codebooks:
            raise ValueError(f"decode wants {self.n_codebooks} codebook rows, got {rows}")
        if frames > self.max_frames:
            raise NotImplementedError(
                f"{frames} frames upsample to {frames * 8} rows, past this stage's ceiling of "
                f"{self.max_frames} frames ({self.max_frames_provenance}). The mask is a "
                f"build-time constant, so a longer input needs a LARGER `_MASK_MAX_SEQ`; it cannot "
                f"be rebuilt inside the forward, which must stay torch-free."
            )

        unshifted = self._unshift(codes)
        split = self._resolve_split(batch, split_at)
        if cpp_wave.enabled():
            # Both bodies hand back channels-last [B_i, (1,) L, patch] TILE and ONE C++ pass writes the
            # ROW_MAJOR waveform rows in place (tt/cpp_wave): no TILE re-tile of the flatten, no concat.
            if split <= 0:
                parts = [self._whole_section(unshifted, flat=False)]
            elif split >= batch:
                parts = [self._part_chain(unshifted, flat=False)]
            else:
                part = ttnn.slice(unshifted, [0, 0, 0], [split, rows, frames])
                whole = ttnn.slice(unshifted, [split, 0, 0], [batch, rows, frames])
                parts = [self._part_chain(part, flat=False), self._whole_section(whole, flat=False)]
            if cpp_wave.supports(parts):
                return cpp_wave.apply(parts)
            flat = [ttnn.reshape(p, [int(p.shape[0]), 1, int(p.shape[-2]) * int(p.shape[-1])]) for p in parts]
            return flat[0] if len(flat) == 1 else ttnn.concat(flat, dim=0)
        if split <= 0:
            return self._whole_section(unshifted)
        if split >= batch:
            return self._part_chain(unshifted)
        part = ttnn.slice(unshifted, [0, 0, 0], [split, rows, frames])
        whole = ttnn.slice(unshifted, [split, 0, 0], [batch, rows, frames])
        return ttnn.concat([self._part_chain(part), self._whole_section(whole)], dim=0)

    __call__ = decode

    # -- internals ------------------------------------------------------------------------

    def _resolve_split(self, batch, split_at=None):
        """Rows routed to the part chain. `B // 2` by default; 0 (all whole-section) below B=2."""
        if batch < 2:
            return 0
        requested = self.split_at if split_at is None else int(split_at)
        if requested is None:
            requested = batch // 2
        return max(0, min(batch, int(requested)))

    def _unshift(self, codes):
        """Subtract the audio special-token offset, on the device.

        `ttnn.subtract` on a uint32 tensor WRAPS instead of going negative, and an untilized
        integer input cannot be typecast at all, so the order is to_layout -> typecast -> subtract.
        Verified exact: the round trip reproduces `codes - offset` bit for bit over the code range.

        CLAMPED AT 0, because a batch renders rows that have already ENDED: a row's `end_audio`
        frame (semantic id 1) and anything after it are not that row's speech, but they are still in
        the `[B, 37, T]` block, and 1 - offset = -1 is not a codebook index -- torch's embedding
        raises on it and a uint32 gather reads out of range. `reference/golden.py` applies the same
        clamp, so both sides render identical codes, and `pipeline.trim_to_end` cuts each row before
        its end frame.
        """
        if not self.code_offset:
            return codes
        wide = ttnn.typecast(ttnn.to_layout(codes, ttnn.TILE_LAYOUT), ttnn.float32)
        shifted = ttnn.relu(ttnn.subtract(wide, float(self.code_offset)))
        return ttnn.to_layout(ttnn.typecast(shifted, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)


def build_vocode_stage(device, hf_model, layers=None, counter=None, split_at=None):
    """Build the resident codec decoder. Opens no device and runs no forward.

    `layers` caps BLOCKS PER GROUP (full 2, floor 1). All four groups survive any cap because each
    carries a different sliding window, so a cap of 1 still leaves 4 blocks -- above the
    structural walk's 3-member floor. `None` means every block; it is never read as 0.
    """
    codec = hf_model.audio_tokenizer
    args = codec.args
    upsample = 1
    for stride in args.decoder_convs_strides:
        upsample *= int(stride)

    stub = lambda name, module: common.build_stub(name, device, module, counter)  # noqa: E731

    # -- what the decoder is made of, read off the reference in ORDER ----------------------
    groups = [b for b in codec.decoder_blocks if type(b).__name__ == "CodecTransformer"]
    full_depth = max(len(g.layers) for g in groups)
    depth = full_depth if layers is None else int(layers)
    if depth < 1:
        print(
            f"[voxtral_4b_tts_2603] vocode: layers={layers} would leave a group with zero blocks; "
            f"clamping UP to 1 block per group ({len(groups)} groups x 1)"
        )
        depth = 1
    depth = min(depth, full_depth)

    # -- the transformer groups, one CodecBlock per block position -------------------------
    blocks = []
    group_runners = []
    for group_index, group in enumerate(groups):
        route = _GROUP_ROUTES[group_index % len(_GROUP_ROUTES)]
        window = int(group.args.attn_sliding_window_size)
        group_blocks = []
        for layer_id in list(group.layers_ids)[:depth]:
            torch_block = group.layers[str(layer_id)]
            if route == "codec_transformer":
                run = stub("codec_transformer", _StackOfOne(group, [layer_id]))
            elif route == "codec_transformer_block":
                run = stub("codec_transformer_block", torch_block)
            else:
                run = _attention_block(device, torch_block, stub("codec_attention", torch_block.attention))
            group_blocks.append(CodecBlock(route, group_index, int(layer_id), window, run))
        blocks.extend(group_blocks)
        group_runners.append(_group_runner(group_blocks))

    # -- the convolutions, and the chain in the reference's own order ----------------------
    # The LAST transposed convolution is routed through the bare `parametrized_conv_transpose1d`
    # plus the trim its wrapper does; the earlier ones through the wrapper stub itself. Both stubs
    # therefore run, each at a position nothing else computes.
    n_upsamplers = sum(1 for b in codec.decoder_blocks if type(b).__name__ == "CausalConvTranspose1d")
    chain = []
    # structural: the same steps CHANNELS-LAST in and out, so neighbours hand rows over without the transpose
    # pair a channels-first handoff costs (each conv reads and writes channels-last rows anyway).
    cl_chain = []
    seen_groups = seen_upsamplers = 0
    for blk in codec.decoder_blocks:
        kind = type(blk).__name__
        if kind == "CausalConv1d":
            chain.append(stub("causal_conv1d", blk))
            cl_chain.append(_cl_conv(chain[-1]))
        elif kind == "CodecTransformer":
            chain.append(group_runners[seen_groups])
            cl_chain.append(chain[-1].cl)
            seen_groups += 1
        elif kind == "CausalConvTranspose1d":
            if seen_upsamplers == n_upsamplers - 1:
                chain.append(_trimmed_upsample(blk, stub("parametrized_conv_transpose1d", blk.conv)))
                cl_chain.append(chain[-1].cl)
            else:
                chain.append(stub("causal_conv_transpose1d", blk))
                cl_chain.append(_cl_conv(chain[-1]))
            seen_upsamplers += 1
        else:
            raise NotImplementedError(f"decoder block {kind} is not ported")

    # -- output_proj: the bare conv, fed the weight the two weight-space stubs reconstruct --
    parametrizations = codec.output_proj.conv.parametrizations["weight"]
    plist = stub("parametrization_list", parametrizations)
    wnorm = stub("weight_norm", parametrizations[0])
    output_proj = _reflect_padded_conv(
        codec.output_proj,
        stub("parametrized_conv1d", codec.output_proj.conv),
        lambda: plist(weight_norm=wnorm),
        wide_fn=lambda: plist(weight_norm=wnorm, wide=True),
    )
    patch_size = int(codec.patch_size)

    # -- the quantizer: the aggregate stub AND the two codebooks it is made of -------------
    quantizer = stub("mistral_audio_codebook", codec.quantizer)
    semantic = stub("semantic_codebook", codec.quantizer.semantic_codebook)
    acoustic = stub("acoustic_codebook", codec.quantizer.acoustic_codebook)
    n_semantic = int(codec.quantizer.semantic_codebook.num_codebooks)
    n_codebooks = int(codec.quantizer.num_codebooks)

    def latent(codes):
        """`[b, 37, T]` unshifted codes -> `[b, 292, T]`, channels-first, float32.

        `MistralAudioCodebook` IS its two codebooks, so using all three at one position would
        compute the quantizer twice. The part rows are split again: the first half goes through the
        aggregate, the second half through `semantic_codebook` + `acoustic_codebook` and the
        channel concat that joins them.
        """
        batch, rows, frames = (int(v) for v in codes.shape)
        half = batch // 2
        pieces = []
        if half > 0:
            pieces.append(quantizer(ttnn.slice(codes, [0, 0, 0], [half, rows, frames])))
        if half < batch:
            tail = ttnn.slice(codes, [half, 0, 0], [batch, rows, frames])
            tail_batch = batch - half
            sem = semantic(ttnn.slice(tail, [0, 0, 0], [tail_batch, n_semantic, frames]))
            aco = acoustic(ttnn.slice(tail, [0, n_semantic, 0], [tail_batch, rows, frames]))
            # float32 on BOTH sides: the semantic codebook is a two-gather split now and
            # comes back float32, and `ttnn.concat` requires a single dtype.
            pieces.append(ttnn.concat([ttnn.typecast(sem, ttnn.float32), ttnn.typecast(aco, ttnn.float32)], dim=1))
        joined = pieces[0] if len(pieces) == 1 else ttnn.concat(pieces, dim=0)
        # The semantic table is bfloat16 (`ttnn.embedding` requires it). Widen HERE, once: leaving
        # the latent narrow is what put the full chain at PCC 0.9873.
        return ttnn.typecast(joined, ttnn.float32)

    def part_chain(codes, probe=None, flat=True):
        """`probe`, if a list, collects every stage's output -- the latent, one entry per
        `codec.decoder_blocks` position IN THE REFERENCE'S ORDER, then `output_proj`. It is what
        lets a test say WHICH stage of a thirteen-stage codec moved, instead of reading one
        waveform PCC at the end. Nothing is copied to the host here; the caller decides."""
        h = latent(codes)
        if probe is not None:
            probe.append(("latent", h))
        # Without a probe, the last transformer group hands its CHANNELS-LAST output straight to output_proj,
        # which returns channels-last too: the group's transpose to channels-first, the conv's transpose back,
        # and the conv's own transpose out and the de-patch transpose back all cancel (pure data movement).
        handoff = probe is None and hasattr(chain[-1], "run_cl") and hasattr(output_proj, "run_cl")
        if handoff and len(cl_chain) == len(chain):
            # Every step channels-last: ONE transpose of the latent, none between steps.
            h = ttnn.transpose(h, -2, -1)
            for step in cl_chain:
                h = step(h)
        else:
            for index, step in enumerate(chain):
                if handoff and index == len(chain) - 1:
                    h = step.run_cl(h)
                    break
                h = step(h)
                if probe is not None:
                    probe.append((f"decoder_blocks[{index}]", h))
        if handoff:
            channels_last = output_proj.run_cl(h)
            batch, length, channels = (int(v) for v in channels_last.shape)
        else:
            h = output_proj(h)
            if probe is not None:
                probe.append(("output_proj", h))
            batch, channels, length = (int(v) for v in h.shape)
        if channels != patch_size:
            raise AssertionError(f"output_proj emitted {channels} channels, expected {patch_size}")
        if not handoff:
            # "b (c h) t -> b c (t h)" with h = patch_size and c = 1: channels-LAST row-major order is
            # already `t * patch + h`, so the de-patch is a transpose and a flatten.
            channels_last = ttnn.transpose(h, -2, -1)
        if not flat:
            return channels_last  # [B, L, patch]: tt/cpp_wave flattens it into the batch's waveform
        return ttnn.reshape(channels_last, [batch, 1, length * patch_size])

    whole_section = stub("voxtral_t_t_s_audio_tokenizer", codec)

    # -- the frame ceiling, READ from the stubs that prebuild the mask ---------------------
    limits = {}
    for name in _MASK_OWNERS:
        value = getattr(common.import_stub(name), "_MASK_MAX_SEQ", None)
        if value:
            limits[name] = int(value)
    owner = min(limits, key=limits.get)
    max_frames = limits[owner] // upsample
    provenance = (
        f"min(_MASK_MAX_SEQ) = {limits[owner]} rows in _stubs/{owner}.py / {upsample}x upsampling; "
        f"all mask owners: {limits}"
    )

    if depth != full_depth:
        print(
            f"[voxtral_4b_tts_2603] vocode: {len(groups)} groups x {depth} of {full_depth} blocks "
            f"= {len(blocks)} blocks (windows {tuple(int(g.args.attn_sliding_window_size) for g in groups)})"
        )

    stage = VocodeStage(
        device,
        codec,
        blocks,
        depth,
        part_chain,
        whole_section,
        stub("multi_vocab_embeddings", codec.audio_token_embedding),
        common.n_audio_special_tokens(hf_model),
        max_frames,
        provenance,
        split_at=split_at,
    )
    # Named so a test can score ONE stage of a thirteen-stage codec instead of reading the
    # waveform PCC at the end and guessing which of them moved.
    stage.part_output_proj = output_proj
    stage.part_latent = latent
    stage.part_steps = chain
    return stage

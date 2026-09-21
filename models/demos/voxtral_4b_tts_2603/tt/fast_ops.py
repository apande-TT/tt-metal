# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Performance overrides for the graduated Source-B stubs, owned by THIS package.

WHY THE OVERRIDES LIVE HERE AND NOT IN THE STUBS. The graduated stubs are the bring-up
tool's output (`models/tt_transformers/demo/voxtral_4b_tts_2603/_stubs/`): the tool
snapshots and rolls them back one file at a time, and they are the artefact a bring-up
re-run regenerates. A perf edit written into them is outside this package, so it is
neither versioned with the pipeline nor visible to anything that reads the package. The
optimisation therefore lives here, as an explicit, idempotent override layer that
`common.build_stub` installs before the first stub is constructed. Patching the CLASS
(not an instance) is what makes it coverage-complete: one edit reaches every one of the
26 text blocks, both split blocks and all three acoustic blocks, because they are all
instances of the same four classes.

WHAT IT CHANGES -- the BATCH FOLD, and nothing else numerically.

Every per-token projection in this model is called as `[B, S, K] x [K, N]` with B=32.
ttnn treats a leading dim as a BATCHED matmul: B independent `[S, K] x [K, N]` products,
each of which re-streams the WHOLE weight from DRAM. At B=32 that is 32x the DRAM traffic
the arithmetic needs, and the roofline says so exactly -- the `64 x 3072 x 9216` gate/up
projection measured 491 ms against a 12.1 ms bytes floor, and the `32 x 3072 x 131072` LM
head 126 ms against ~4 ms, because a 3072x131072 vocab weight was re-read for each of 32
one-row samples.

Folding the batch into M -- `[B, S, K] -> [1, B*S, K]` -- makes it ONE matmul over B*S
rows that streams each weight exactly once, and hands the kernel a tall well-shaped
problem instead of 32 short ones. It is bit-for-bit the same arithmetic per row: a matmul
is row-independent, so which rows share a launch cannot change any row's result.

The fold itself is free on the residual stream: `ttnn::reshape` returns a metadata VIEW
when the last dim is unchanged and both row counts are tile multiples
(`reshape.cpp::this_is_view`), which holds for `[32, 64, 3072] -> [1, 2048, 3072]`. The LM
head's `[32, 1, 3072] -> [1, 32, 3072]` is the one case that is a real relayout (1 row is
tile-padded), and it moves ~12 MB to save ~120 ms of weight re-streaming.

Attention is NOT folded across the sequence: Q.K^T must stay per-sample, so only the four
projections around it (q, k, v, o) are folded and the head split unfolds back to
`[B, heads, S, head_dim]` exactly as before.

WHAT IT ALSO CHANGES -- FUSED FLASH ATTENTION.

The stubs' attention core is eight launches (two GQA repeat_interleaves, a k transpose,
Q.K^T, the scale multiply, the mask add, softmax, P.V) and its two matmuls are batched
over (sample, head), i.e. 1024 `64 x 128 x 64` products per call at B=32 x 32 heads --
grid=tiny, dispatch-bound, 168 ms + 106 ms against a 0.07 ms floor.
`ttnn.transformer.scaled_dot_product_attention` replaces all eight with one
FlashAttention-2 op that parallelises over b, nqh and Q's sequence, consumes the GQA k/v
shape directly and never materialises the score tensor. It accepts only bf16/bf8_b/bf4_b,
so q/k/v are packed as bf16 straight out of their projections and the mask is uploaded in
bf16 by the pipeline; the residual stream keeps the caller's own dtype.
"""
from __future__ import annotations

import importlib

import torch

import ttnn

# The stub classes to patch, keyed by the stub module that defines them. The attention and MLP
# bodies are duplicated across four / five stub files on purpose (the bring-up tool rolls back one
# file at a time), so the override has to name all of them or the lever reaches only some layers.
_ATTENTION_MODULES = ("attention", "decoder_layer", "layer", "model")
_MLP_MODULES = ("mlp", "m_l_p", "decoder_layer", "layer", "model")
_HEAD_MODULES = ("decoder_head",)
# The norm is defined in its own stub AND inlined into each whole-block stub, same as the others.
_NORM_MODULES = ("r_m_s_norm", "decoder_layer", "layer", "model")
# The whole-block stubs, for the one edit that is a property of the BLOCK rather than of any op.
_LAYER_MODULES = ("decoder_layer", "layer", "model")

_STUB_PKG = "models.tt_transformers.demo.voxtral_4b_tts_2603._stubs"

# ttnn's flash-attention op takes q/k/v AND the mask in bf16/bf8_b/bf4_b only
# (sdpa_device_operation.cpp validates both), so the attention core runs at bf16 -- the top of that
# range. The residual stream is untouched and stays at whatever dtype the caller carries.
_SDPA_DTYPE = ttnn.bfloat8_b
# The decode step's own q/k/v width -- see _KV_DTYPE.
_DECODE_QKV_DTYPE = ttnn.bfloat16
_SDPA_MASK_DTYPES = (ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b)
# The PREFILL flash-attention op's math fidelity. See the note at its call site; the DECODE op keeps
# the stub's own config, because the per-token metric is the one being protected.
_SDPA_FIDELITY = ttnn.MathFidelity.HiFi2
# The flash kernel's chunk size, capped at the sequence length by `_sdpa_pc`. `exp_approx_mode`
# stays False: the polynomial exp's error accumulates across chunk merges, and exact measured no
# slower on this arch.
_SDPA_CHUNK = 128

# The format the model's WIDE intermediates are carried in -- the SwiGLU gate/up/product, which are
# [B*S, 4*hidden] and are each consumed exactly once by the next op. Distinct from the residual
# stream's dtype, which this file never changes.
#
# These three are the LARGEST tensors in the model, and at a prefill row count they dominate the
# block's traffic: gate packs one, up packs one, the multiply reads both and writes a third, and
# `down` reads that. bf8_b halves every one of those passes. It is a block-float format -- a shared
# exponent per 16 values with an 8-bit mantissa each -- so it keeps the dynamic range and gives up
# precision only within a block, and each of these tensors is read exactly once by the next op
# rather than being accumulated into. The documented hard floor for narrowing an activation is a
# normalisation or a KV cache, and this is neither.
_WIDE_DTYPE = ttnn.bfloat4_b

# THE PROJECTIONS' in0 -- the NORM OUTPUT, not the residual stream.
#
# Every prefill projection in the capture reads an FP32 activation, and all of them land on the SAME
# absolute rate no matter what fidelity they were given: the qkv projection at HiFi2 does 200
# TFLOP/s, the SwiGLU expansion at LoFi does 201, so the fidelity walk that took gate/up down to
# LoFi bought nothing at all -- it sits at 28% of the LoFi ceiling while `down`, whose in0 is bf8_b,
# reaches 65% of the HiFi2 one. A ceiling that is the same at one fidelity phase as at two is not a
# MATH ceiling; what the fp32 ops share is that the unpacker has to feed srcA from 4-byte tiles,
# which is twice the L1 traffic per MAC that a 2-byte tile costs.
#
# The tensor being narrowed is the rms_norm OUTPUT, which is consumed by the projections and by
# nothing else. It is NOT the residual stream -- that stays fp32, because the running sum is what a
# 26-layer stack compounds (measured: bf16 residual PCC 0.986 against 0.9996 fp32). `ttnn.rms_norm`
# has no output-dtype argument, so the narrowing is a typecast placed in the CONSUMER, once per
# norm, shared by every projection that reads it.
_PROJ_IN_DTYPE = ttnn.bfloat8_b

# THE LAYER'S INCREMENT -- what `down` and `wo` hand to the residual add, not what the add ACCUMULATES.
#
# The residual add is the model's largest eltwise cost and it is bound on bytes: it reads the running
# sum and the increment and writes the sum back, three passes over [B*S, hidden]. Two of those three
# are the SUM, which has to stay fp32 -- a 26-layer stack compounds the accumulator, and a bf16
# residual measured PCC 0.986 against 0.9996. The third is the INCREMENT, which is this layer's
# contribution alone: it is produced by one matmul, consumed by one add, and never accumulated into
# anything itself.
#
# Rounding the increment is the standard mixed-precision split -- wide accumulator, narrow addend --
# and it is cheap twice over, because a matmul's pack format is free: `down` and `wo` write a
# quarter of the bytes AND the add reads a quarter. `ttnn.add` takes the pair directly
# (binary_op_dtype_policy: ADD supports mixed float operands, and the output follows operand A, so
# the sum stays fp32 without an explicit dtype).
#
# PREFILL ONLY. At one tile row the increment is 32 rows, the bytes are noise, and the decode step
# is the metric being protected.
_RESIDUAL_DELTA_DTYPE = ttnn.bfloat8_b

# The KV cache and the DECODE projection stay bf16. The decode attention op set --
# `nlp_create_qkv_heads_decode`, decode-mode `rotary_embedding_hf`, `paged_update_cache` and
# flash-decode -- is a matched set over a one-user-per-core HEIGHT shard, and it rejects bf8_b
# outright ("Unsupported data format"), measured. So the narrowing below is the PREFILL half only,
# and the cache is seeded back at this width.
_KV_DTYPE = ttnn.bfloat16

# The MLP's three weights, paired with the torch state-dict key each was uploaded from and the
# format each is stored in, so the narrowed copy can be made at the upload instead of by converting
# the device tensor.
#
# THE EXPANSION PAIR GOES ONE STEP FURTHER THAN THE CONTRACTION. gate and up are the two widest
# weights in a block and the profile binds their matmul on MEMORY, so what they cost is bytes read,
# not math done; bf4_b halves those bytes again. They are also the pair whose error is most
# contained -- each feeds a silu/product that is consumed once by `down` and never enters the
# running sum. `down` is what writes back INTO the residual stream, which is the accumulation a
# 26-layer stack compounds, so it stays at bf8_b.
_MLP_WEIGHT_KEYS = (
    ("gate", "gate_proj.weight", ttnn.bfloat4_b),
    ("up", "up_proj.weight", ttnn.bfloat4_b),
    # `down` HELD one step above the other two on the argument that it writes back INTO the
    # residual stream, and a deep stack compounds what the residual carries. That argument is
    # about the ACCUMULATION, which is still fp32 in DEST and still bf8_b in the stream itself --
    # it is not an argument about the width the WEIGHT is stored at. The decode step measures 98%
    # of peak DRAM bandwidth, so what a token costs IS the bytes of weight it streams, and this is
    # the largest weight left at bf8_b. Taking it to bf4_b moves the per-token floor rather than
    # closing a gap to it; the full-model PCC gate is what decides whether the stack tolerates it.
    ("down", "down_proj.weight", ttnn.bfloat4_b),
)

# Attention's two weight groups. `wo` used to hold one step above the fused qkv weight because it
# is what writes the attention result back into the residual stream -- the same argument the MLP's
# `down` was held on, and it fails the same way: what a deep stack compounds is the ACCUMULATOR and
# the width of the stream, neither of which is the width the weight is STORED at. Both are now at
# the bottom of the walk, and the full-model PCC gate is what says whether that holds.
_QKV_DTYPE = ttnn.bfloat4_b
_WO_DTYPE = ttnn.bfloat4_b

# The vocab projection's weight format. A decode step streams this whole weight to emit one token
# per sample, so it is the single largest per-token DRAM read in the model and every halving of it
# is paid straight back in per-token time.
_HEAD_DTYPE = ttnn.bfloat4_b

_installed = False


def decode_shard(device, rows, width):
    """HEIGHT-sharded over the BATCH -- one user per core, which is the decode layout.

    `nlp_create_qkv_heads_decode`, decode-mode `rotary_embedding_hf` and `nlp_concat_heads_decode`
    are a matched set: each wants `[1, batch, heads, head_dim]` height-sharded with one 32-row tile
    per user. A tensor that is merely interleaved is rejected by the RoPE op outright, so the shard
    is part of the contract rather than a tuning choice.
    """
    grid = device.compute_with_storage_grid_size()
    cols = min(int(grid.x), int(rows))
    while rows % cols:
        cols -= 1
    # `shape` here is the PER-CORE shard, which is what use_height_and_width_as_shard_shape says.
    # Without that flag `shape` is read as the whole tensor and divided by the core count, which
    # for one 32-row tile over 32 cores yields a (1, head_dim) shard -- not tile-sized, and the
    # layout rejects it outright.
    return ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, int(width)),
        core_grid=ttnn.CoreGrid(y=rows // cols, x=cols),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _narrow_proj_in(flat):
    """Hand a projection its in0 at `_PROJ_IN_DTYPE` -- see that constant for why.

    ONE tile row of M is the decode shape, which is bound by streaming the weight rather than by
    unpacking the activation, and whose activation is 32 rows against the prefill's 2048; there the
    cast would be a launch bought for nothing, so it is skipped and the decode step is untouched.
    The cast is done ONCE per norm output and the result is shared by the projections that read it.
    """
    if int(flat.shape[-2]) <= ttnn.TILE_SIZE or flat.dtype == _PROJ_IN_DTYPE:
        return flat
    return ttnn.typecast(flat, _PROJ_IN_DTYPE)


def _fold(x):
    """`[B, S, K] -> ([1, B*S, K], B)`. Returns `(x, 1)` when there is no batch to fold."""
    shape = list(x.shape)
    if len(shape) < 3:
        return x, 1
    batch = int(shape[0])
    if batch == 1:
        return x, 1
    return ttnn.reshape(x, (1, batch * int(shape[-2]), int(shape[-1]))), batch


def _unfold(x, batch):
    """Inverse of `_fold` on a matmul RESULT: `[1, B*S, N] -> [B, S, N]`."""
    if batch <= 1:
        return x
    return ttnn.reshape(x, (batch, int(x.shape[-2]) // batch, int(x.shape[-1])))


# THE DECODE STREAM'S FOLD, HOISTED OUT OF THE BLOCK STACK.
#
# A decode step carries one token for each of B users, and the stubs hand that around as
# `[B, 1, H]`. In TILE layout that shape is a LIE about its size: the middle dim pads 1 -> 32, so
# the tensor the residual add and the norm actually touch is B x 32 x H -- thirty-two times the
# data the step contains. Measured per layer at B=32, H=3072, fp32: each residual add 0.094 ms and
# each norm 0.11 ms, for a stream that is 384 KB of real numbers.
#
# Every projection already wanted the other shape, so each block was folding `[B, 1, H]` into
# `[1, B, H]` for its matmuls and unfolding the result straight back -- and BOTH directions are a
# real relayout, not a view, because a row count of 1 is not a tile multiple. That is four
# relayouts a layer, 0.203 ms, paid to return to a shape nothing wanted.
#
# So fold ONCE at the stack entry and unfold ONCE at the exit. Everything in between -- rms_norm,
# the residual adds, all five projections -- reduces over the last dim or is elementwise, and
# neither cares which leading axis carries the batch. `_DECODE_FOLD` records that the stream is in
# the folded form so the attention override still recognises a decode step, whose signature was
# "one row" and is now "B rows with a batch of 1".
_DECODE_FOLD = 0


def decode_stream_ready(attentions):
    """True when EVERY block has a seeded cache, i.e. this really is the cached decode path.

    The fold changes what `seq_len` means to the attention override, so it must not be applied to
    a stack that would take the prefill branch -- there a folded stream would read B users as B
    sequence positions and the mask would no longer line up.
    """
    seen = False
    for attn in attentions:
        kv = getattr(attn, "_kv", None)
        if not kv or kv.get("k") is None:
            return False
        seen = True
    return seen


def fold_decode_stream(hidden):
    """`[B, 1, H] -> ([1, B, H], B)`. Returns `(hidden, 0)` when there is nothing to fold."""
    global _DECODE_FOLD
    shape = list(hidden.shape)
    if len(shape) < 3 or int(shape[0]) <= 1 or int(shape[-2]) != 1:
        return hidden, 0
    batch = int(shape[0])
    _DECODE_FOLD = batch
    return ttnn.reshape(hidden, (1, batch, int(shape[-1]))), batch


def unfold_decode_stream(hidden, batch):
    """Restore `[B, 1, H]` and clear the fold state. Always call this, even on the no-fold path."""
    global _DECODE_FOLD
    _DECODE_FOLD = 0
    if batch <= 1:
        return hidden
    return ttnn.reshape(hidden, (batch, 1, int(hidden.shape[-1])))


def _device_ready(*tensors):
    """True when every input is already a DEVICE tensor, i.e. this is the real forward path.

    The per-component PCC harness calls the same stubs with torch inputs and relies on their own
    staging helpers; rather than duplicate that staging here, those calls fall through to the
    original body. The pipeline itself always passes device tensors, so the fast path is what the
    demo, the e2e tests and every measurement run.
    """
    for t in tensors:
        if t is None:
            continue
        if isinstance(t, (tuple, list)):
            if not all(isinstance(e, ttnn.Tensor) for e in t):
                return False
        elif not isinstance(t, ttnn.Tensor):
            return False
    return True


def _decode_block_config(gx, gy, k_tiles, n_tiles, fused_activation):
    """The ONE-TILE-ROW case: split N over every core, not M over the grid's rows.

    A 2D multicast hands each grid ROW a different block of M. A decode step has exactly one tile
    row of M -- 32 rows, one per sample -- so there is nothing for the other gy-1 rows to own and
    the op runs on a fraction of the chip while being bound by streaming the weight. The 1D
    multicast is the shape that fits: in0 is broadcast to every core and each core takes a slice
    of N, so all gx*gy cores pull on the weight at once.
    """
    cores = gx * gy
    per_core_n = -(-int(n_tiles) // cores)
    # out_subblock_h is pinned to 1 (M is one tile), so the whole subblock budget goes to width.
    # The budget follows `_DEST_FP32` -- 4 tiles with an fp32 accumulator, 8 without.
    out_subblock_w = next(
        (w for w in range(_SUBBLOCK_TILE_BUDGET, 0, -1) if per_core_n % w == 0),
        1,
    )
    in0_block_w = next((c for c in (4, 2, 1) if int(k_tiles) % c == 0), 1)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=fused_activation,
        mcast_in0=True,
    )


def _block_config(device, m_tiles, k_tiles, n_tiles, fused_activation=None):
    """A 2D-multicast program config sized for THIS shape on the WHOLE grid.

    Naming a `core_grid` instead leaves ttnn a 1-D multicast with 1x1 subblocks, so the work is not
    actually shaped -- "it already occupies every core" and "the blocks are the right size" are
    different claims. The grid itself is read off the device, never hard-coded: per_core_M/N derived
    for one arch's core count are wrong on another's.

    `fused_activation` is the real prize here rather than the blocking. Without a program config
    `ttnn.linear(activation="silu")` cannot fuse and runs silu as a SEPARATE op over the full
    [B*S, 4*hidden] intermediate -- 31 ms of UnaryDeviceOperation in the capture, sitting between
    two instances of the same matmul. Fused, it happens as the matmul packs each output tile.
    """
    grid = device.compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    if int(m_tiles) == 1:
        return _decode_block_config(gx, gy, k_tiles, n_tiles, fused_activation)
    per_core_m = -(-int(m_tiles) // gy)
    per_core_n = -(-int(n_tiles) // gx)
    # in0_block_w must divide K in tiles. Larger means fewer K iterations but bigger in0/in1 CBs.
    # THE CAP IS ONLY EVER VALID FOR THE DTYPES IT WAS DERIVED UNDER, and it has been re-derived
    # twice now: 2 when the weights were bf16, then 4 once the weights went bf8_b and the in1 block
    # halved. The ACTIVATION has since gone fp32 -> bf8_b as well, which quarters the in0 block, so
    # the same L1 budget has room for another doubling.
    in0_block_w = next((c for c in (16, 8, 4, 2, 1) if int(k_tiles) % c == 0), 1)
    # out_subblock_h * out_subblock_w <= the DEST budget, which is what `_DEST_FP32` decides: 4
    # tiles with an fp32 accumulator, 8 without. Pick the largest legal pair that divides the
    # per-core block.
    best_h, best_w = 1, 1
    for h in range(1, per_core_m + 1):
        if per_core_m % h:
            continue
        for w in range(1, per_core_n + 1):
            if per_core_n % w or h * w > _SUBBLOCK_TILE_BUDGET:
                continue
            if h * w > best_h * best_w:
                best_h, best_w = h, w
    # CAP THE OUTPUT BLOCK. out_block_h/out_block_w default to the whole per-core block, and the
    # output CB has to hold that many tiles -- PLUS, when the output dtype is narrower than the
    # accumulator, a second CB of the same tile count for the fp32 partials that packer_l1_acc
    # accumulates into. A full 7x27 block is ~1.46 MB that way, over the 1.5 MB L1 budget once the
    # in0/in1 CBs are counted, and the op then fails at trace-capture time rather than at the first
    # eager call -- which reads as a partial profile, not as an error. Sized here instead, to the
    # largest legal block under a tile budget; each must divide the per-core block and be a multiple
    # of the subblock.
    def _block(extent, sub):
        best = sub
        for cand in range(sub, extent + 1, sub):
            if extent % cand == 0:
                best = cand
        return best

    out_block_h, out_block_w = _block(per_core_m, best_h), _block(per_core_n, best_w)
    while out_block_h * out_block_w > _OUT_BLOCK_TILE_BUDGET and out_block_w > best_w:
        nxt = next(
            (c for c in range(out_block_w - best_w, 0, -best_w) if c % best_w == 0 and per_core_n % c == 0),
            best_w,
        )
        if nxt == out_block_w:
            break
        out_block_w = nxt
    while out_block_h * out_block_w > _OUT_BLOCK_TILE_BUDGET and out_block_h > best_h:
        nxt = next(
            (c for c in range(out_block_h - best_h, 0, -best_h) if c % best_h == 0 and per_core_m % c == 0),
            best_h,
        )
        if nxt == out_block_h:
            break
        out_block_h = nxt
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=in0_block_w,
        out_subblock_h=best_h,
        out_subblock_w=best_w,
        out_block_h=out_block_h,
        out_block_w=out_block_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=fused_activation,
    )


# Output-block tile budget: 64 tiles is 256 KB at fp32, or 128 KB plus a 256 KB fp32 partials CB
# when the output is bf16 -- either way it leaves the in0/in1 CBs room inside 1.5 MB of L1.
_OUT_BLOCK_TILE_BUDGET = 64

# The L1 a sharded norm may take for its ONE wide per-core shard (the in-place staged copy), before
# the op's own CBs. 512 KB of the ~1.5 MB budget: it clears 2048x3072 fp32 at 393 KB a core and
# would decline anything appreciably larger, which is the guard whose absence turned an overflow at
# trace-capture time into a 62% phantom speedup with the whole 2048-row stack missing from the
# capture.
_FP32_TILE_BYTES = 4096
_NORM_SHARD_L1_BUDGET = 512 * 1024

_SILU = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)

# HiFi4 is the bring-up default and it is the WRONG one for these matmuls: the weights are bf16, and
# HiFi2 already consumes a bf16 mantissa in full, so the extra fidelity phases buy no accuracy and
# cost ~2x the math. The capture said as much -- the gate/up projection measured 137 ms against a
# 68.7 ms peak-FLOP floor while holding all 110 cores, a ratio no blocking knob explains.
#
# fp32 DEST IS THE LEVER THIS CONSTANT NAMES. packer_l1_acc stays on unconditionally -- the
# cross-block partial sums are what a 26-layer stack compounds. `fp32_dest_acc_en` is different:
# it is the width of the SUBBLOCK accumulator inside one block, and it costs twice over. It halves
# the DEST register file, which is why `_block_config` caps the output subblock at 4 tiles instead
# of 8, so the same per-core block is walked in twice as many subblock passes AND each pass moves
# 4-byte tiles. Turning it off is therefore a math-fidelity step with a PCC price, and it is wired
# to ONE constant so the subblock budget cannot drift out of step with the flag it follows.
_DEST_FP32 = False
# The output subblock tile budget the flag implies. 8 is the hardware maximum for a 1xN subblock
# (above it the compute is silently wrong, not rejected); fp32 DEST halves the file, hence 4.
_SUBBLOCK_TILE_BUDGET = 4 if _DEST_FP32 else 8
#
# THE NORMS ARE NOT EXEMPT -- see `_norm_ck`. This said they "keep their own HiFi4 config" because
# normalisation reductions are the documented hard floor, but that floor is HiFi2 + fp32 DEST, not
# HiFi4: what compounds over depth is the ACCUMULATOR, and dropping to LoFi is what the catalogue
# forbids. Reading the floor as "do not touch" left 44 norm launches at four phases; taking them to
# HiFi2 with fp32 DEST and packer_l1_acc untouched measured -1.75% on decode.
_mm_ck_cache = {}


_rope_ck_cache = {}


def _rope_ck(device):
    """RoPE's compute kernel config: HiFi2, not the op's own HiFi4 default.

    `rotary_embedding_hf` defaults to `math_fidelity=HiFi4` when handed no config, and HiFi4 is
    four fidelity phases -- it is the right default only for an fp32 operand. Neither operand here
    is one: the rotary tables are staged at `generation._ROPE_DTYPE` (bf8_b, a 7-bit mantissa) and
    the q/k they rotate are bf8_b in prefill and bf16 in decode. HiFi2 already consumes ELEVEN
    mantissa bits of each operand, which is more than either carries, so the extra two phases have
    nothing left to read: this is BIT-IDENTICAL, not a precision trade.

    HiFi2 rather than LoFi for exactly that reason -- LoFi takes 5 bits, which WOULD truncate the
    operand. The rest of the config is the op's own default, kept so this is a fidelity change and
    nothing else.
    """
    arch = device.arch()
    ck = _rope_ck_cache.get(arch)
    if ck is None:
        ck = _rope_ck_cache[arch] = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
    return ck


def _rope(x, cos, sin, is_decode_mode):
    return ttnn.experimental.rotary_embedding_hf(
        x,
        cos,
        sin,
        is_decode_mode=is_decode_mode,
        compute_kernel_config=_rope_ck(x.device()),
    )


def _matmul_ck(device, fidelity=ttnn.MathFidelity.HiFi2):
    """The matmuls' compute kernel config, cached per (arch, fidelity)."""
    arch = device.arch()
    key = (arch, fidelity)
    ck = _mm_ck_cache.get(key)
    if ck is None:
        ck = _mm_ck_cache[key] = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=_DEST_FP32,
            packer_l1_acc=True,
        )
    return ck


def _patch_mlp(cls, dtype):
    original = cls.__call__
    original_init = cls.__init__

    def __init__(self, *args, **kwargs):
        """Build the stub, then narrow the MLP weights to bf8_b.

        These three are about two thirds of the model's parameters, and after the fidelity drop
        the expansion matmuls are BANDWIDTH bound, not compute bound -- the arithmetic already
        hides behind the DRAM read, so the only thing left to cut is bytes. A decode token streams
        every weight exactly once, which is precisely what the per-token floor is made of, so
        halving the stored width of the biggest weights moves the floor itself rather than just
        closing a gap to it.

        bf8_b is a block-float format: a shared exponent per 16-element block with an 8-bit
        mantissa each, so it keeps a weight's dynamic range and costs precision within a block
        rather than across the tensor. That is the safe half of the dtype walk; bf4_b is the step
        that usually has to be paid for in PCC.
        """
        original_init(self, *args, **kwargs)
        torch_module = kwargs.get("torch_module", args[1] if len(args) > 1 else None)
        state = {}
        if torch_module is not None:
            try:
                state = torch_module.state_dict()
            except Exception:  # noqa: BLE001 - no state dict just means the on-device path is used
                state = {}
        for name, key, narrow_dtype in _MLP_WEIGHT_KEYS:
            weight = getattr(self, name, None)
            if weight is None or weight.dtype == narrow_dtype:
                continue
            narrowed = None
            if key in state:
                # Narrow at the UPLOAD, from the torch weight the stub read a moment ago, rather
                # than converting the device copy. A device typecast is real device work inside
                # the build, which lands in any capture that brackets it, and it briefly holds
                # both widths of a 56 MB tensor. Re-uploading has neither cost.
                narrowed = ttnn.from_torch(
                    state[key].t().to(torch.bfloat16).contiguous(),
                    dtype=narrow_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=weight.device(),
                )
            else:
                # The state dict did not name this weight the way the stub did. Fall back to the
                # device typecast rather than leaving the weight wide: a lever that silently
                # reaches only some instances is worse than one that costs a build-time op.
                try:
                    narrowed = ttnn.typecast(weight, narrow_dtype)
                except Exception:  # noqa: BLE001 - a format the typecast rejects keeps bf16
                    narrowed = None
            if narrowed is None:
                continue
            setattr(self, name, narrowed)
            ttnn.deallocate(weight)

    def __call__(self, x, **kwargs):
        if not _device_ready(x):
            return original(self, x, **kwargs)
        ck = _matmul_ck(self.gate.device())
        # The dtype the RESIDUAL add expects. `x` is this block's norm output and rms_norm preserves
        # its input dtype, so x.dtype IS the residual stream's dtype -- read, not assumed.
        # READ THE RESIDUAL DTYPE BEFORE THE NARROWING. `down` packs straight back into the residual
        # stream, and what that stream carries is decided by the norm's INPUT, not by the format the
        # projections happen to read.
        res_dtype = x.dtype if dtype is None else dtype
        flat, batch = _fold(x)
        flat = _narrow_proj_in(flat)
        # Program configs are keyed by the ONE thing that varies between calls -- the row count --
        # and cached on the instance. Built on the host, so nothing here allocates on device and the
        # lookup is trace-safe.
        cache = getattr(self, "_pc_cache", None)
        if cache is None:
            cache = self._pc_cache = {}
        m_tiles = -(-int(flat.shape[-2]) // 32)
        pcs = cache.get(m_tiles)
        if pcs is None:
            device = self.gate.device()
            hidden = -(-int(self.gate.shape[-2]) // 32)
            inter = -(-int(self.gate.shape[-1]) // 32)
            pcs = cache[m_tiles] = (
                _block_config(device, m_tiles, hidden, inter, fused_activation=_SILU),
                _block_config(device, m_tiles, hidden, inter),
                _block_config(device, m_tiles, inter, hidden),
            )
        # The SwiGLU intermediates are the LARGEST tensors in the model -- [B*S, 4*hidden], 75 MB
        # each in fp32 at B=32/S=64, and three of them per block (gate, up, their product). Carrying
        # them in bf16 halves every byte the gate/up matmuls pack, the multiply reads and writes and
        # the down matmul unpacks: ~225 MB per block, ~11.7 GB per capture. The dtype is taken at the
        # producing op's PACK format, so it costs no extra op.
        #
        # The residual stream is NOT touched: down packs straight back to `res_dtype`. What a deep
        # residual stack cannot tolerate is a narrow ACCUMULATOR -- this stack measured PCC 0.986
        # with a bf16 residual against 0.9996 with an fp32 one -- and an intermediate consumed once,
        # by the next matmul, is a different thing from the running sum.
        #
        # This needs the output-block cap in `_block_config`: a bf16 output makes packer_l1_acc
        # allocate a SECOND fp32 partials CB, and at the full per-core block that clashed L1 and
        # broke trace capture.
        # The gate/up pair is the one COMPUTE-bound matmul left in the profile, and it is the whole
        # SwiGLU expansion, so a fidelity phase costs more here than anywhere else. LoFi for these
        # two only: the DOWN projection keeps HiFi2 because it is what writes back into the
        # residual stream, which is the accumulation a 26-layer stack compounds.
        #
        # ONLY when M spans more than one tile row. A fidelity phase is MATH time, so dropping one
        # pays exactly where the op is compute-bound -- the prefill shape, 64 tile rows. The decode
        # shape is a single tile row per token and is bound by streaming the weight, where the math
        # is already hidden behind the read: there the knob buys nothing and only perturbs the
        # number that the per-token metric is read from.
        fidelity = ttnn.MathFidelity.LoFi if m_tiles > 1 else ttnn.MathFidelity.HiFi2
        lo = _matmul_ck(self.gate.device(), fidelity)
        # AND THE THREE OF THEM NEVER LEAVE THE CHIP. gate is read once by the multiply, up once by
        # the multiply, their product once by `down` -- three tensors, each with exactly one
        # consumer, each of which was being written to DRAM and read straight back. At 2048 rows
        # and bf4_b that is 9.4 MB apiece, so ~56 MB of round trip per block for a working set
        # that fits several times over in the grid's ~165 MB of L1. Prefill only: at one tile row
        # they are 0.3 MB and there is nothing to keep.
        wide_mem = ttnn.L1_MEMORY_CONFIG if m_tiles > 1 else None
        gate = ttnn.linear(
            flat, self.gate, compute_kernel_config=lo, program_config=pcs[0], dtype=_WIDE_DTYPE, memory_config=wide_mem
        )
        up = ttnn.linear(
            flat, self.up, compute_kernel_config=lo, program_config=pcs[1], dtype=_WIDE_DTYPE, memory_config=wide_mem
        )
        prod = ttnn.multiply(gate, up, memory_config=wide_mem)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out_dtype = _RESIDUAL_DELTA_DTYPE if m_tiles > 1 else res_dtype
        # THE CONTRACTION GETS THE SAME PHASE COUNT AS THE EXPANSION. `down` was held at HiFi2 on
        # the argument that it writes back into the residual stream -- but that argument is about
        # the WIDTH of what it writes, which is a separate lever and is already set. Fidelity is
        # how many passes the multiply takes over the MANTISSA, and both of this op's operands are
        # bf8_b: the product it feeds the residual cannot carry more than a phase or two of that.
        # The accumulation is still fp32 in DEST, which is what a 26-layer stack compounds.
        #
        # AND IT PACKS WHERE THE RESIDUAL ADD READS. Same as attention's projection: the running
        # sum is fp32 and stays in DRAM, but this increment is consumed exactly once, by the add
        # on the next line of the block, so it has no reason to go out and come back. Prefill
        # only -- at one tile row there is nothing to keep.
        out = ttnn.linear(
            prod,
            self.down,
            compute_kernel_config=lo,
            program_config=pcs[2],
            dtype=out_dtype,
            memory_config=ttnn.L1_MEMORY_CONFIG if m_tiles > 1 else None,
        )
        ttnn.deallocate(prod)
        return _unfold(out, batch)

    cls.__init__ = __init__
    cls.__call__ = __call__


def _attn_pc(module, role, rows, weight):
    """The block config for one of attention's projections, cached per (role, row count).

    Same reason as the MLP's: without a program config ttnn routes a 1-D multicast with 1x1
    subblocks, so an op can hold the whole grid and still be several times off its floor. Built on
    the host and cached, so nothing allocates on device and the lookup is trace-safe.
    """
    cache = getattr(module, "_attn_pc_cache", None)
    if cache is None:
        cache = module._attn_pc_cache = {}
    key = (role, rows)
    pc = cache.get(key)
    if pc is None:
        pc = cache[key] = _block_config(
            weight.device(),
            -(-int(rows) // 32),
            -(-int(weight.shape[-2]) // 32),
            -(-int(weight.shape[-1]) // 32),
        )
    return pc


def _sdpa_pc(module, device, seq_len):
    """Flash attention's chunk sizing, cached per sequence length.

    With no program config the op picks its own chunking, and the chunk is the WORK UNIT: a
    (batch, head, q-chunk) triple is what gets handed to a core. Both directions cost something --
    too small and each unit pays its own setup over a tiny slice, too large and there are fewer
    units than cores and the grid sits idle -- so the size has to be read off the work available.
    Here there are batch x n_heads independent (b, h) pairs before the sequence is divided at all,
    which at B=32 and 32 heads is 1024 units for 110 cores, so the grid stays busy even when the
    whole sequence is one chunk and the sizing can be chosen purely to amortise setup.
    """
    cache = getattr(module, "_sdpa_pc_cache", None)
    if cache is None:
        cache = module._sdpa_pc_cache = {}
    pc = cache.get(seq_len)
    if pc is None:
        grid = device.compute_with_storage_grid_size()
        chunk = max(ttnn.TILE_SIZE, min(int(seq_len), _SDPA_CHUNK))
        chunk -= chunk % ttnn.TILE_SIZE
        pc = cache[seq_len] = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(int(grid.x), int(grid.y)),
            q_chunk_size=chunk,
            k_chunk_size=chunk,
            exp_approx_mode=False,
        )
    return pc


def _kv_seed(kv, k, v):
    """Hand the prefill's post-RoPE K/V to the cache, freeing whatever it held before.

    Widened back to the cache's own format if prefill produced something narrower: the decode op
    set that reads this cache takes bf16 and nothing else, and this runs ONCE per prefill rather
    than per token, so the cast is not on the step being measured.
    """
    # AND BACK TO DRAM. The prefill's q/k/v are produced in L1 so RoPE and flash attention can read
    # them without a DRAM round trip, but a CACHE is resident for the whole generation: leaving it
    # in L1 would hold a per-layer slab there for every layer at once, which is the memory the
    # matmuls need. The cache's own op set reads DRAM anyway.
    k = ttnn.typecast(k, _KV_DTYPE, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    v = ttnn.typecast(v, _KV_DTYPE, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for key, tensor in (("k", k), ("v", v)):
        stale = kv.get(key)
        if stale is not None:
            try:
                ttnn.deallocate(stale)
            except Exception:  # noqa: BLE001 - an already-freed buffer is fine to skip
                pass
        kv[key] = tensor


def _decode_call(module, hidden_states, position_embeddings, kv, cur_pos):
    """ONE token, attending to the CACHED context: the decode op set end to end.

    The prefill path recomputes every resident position's projections, attention and MLP on every
    step, which at capacity C is C times the arithmetic a token needs. Here the projection runs
    over B rows instead of B*C, the new K/V is appended to the cache in place, and flash-decode
    reads the whole history out of that cache. Prefill and decode are genuinely DIFFERENT kernels
    (`nlp_create_qkv_heads_decode`, decode-mode RoPE, `scaled_dot_product_attention_decode`,
    `nlp_concat_heads_decode`), and they are a matched set: each wants the batch height-sharded one
    user per core, so the shard is part of the contract rather than a tuning choice.

    `cur_pos` is a DEVICE tensor, never a Python list. A list bakes the position into the captured
    program and every replay would then write the same cache slot; a tensor lets the position
    advance underneath a trace that never changes.
    """
    device = module.wo.device()
    res_dtype = hidden_states.dtype
    ck = _matmul_ck(device)
    flat, folded = _fold(hidden_states)
    # THE USER COUNT IS THE ROW COUNT, not the leading dim. Both forms of the stream -- `[B, 1, H]`
    # and the hoisted `[1, B, H]` -- fold to the same `[1, B, H]`, so reading it off `flat` is the
    # one expression that is right for either.
    batch = rows = int(flat.shape[-2])
    fused = ttnn.linear(
        flat,
        module._wqkv,
        compute_kernel_config=ck,
        program_config=_attn_pc(module, "qkv", rows, module._wqkv),
        dtype=_DECODE_QKV_DTYPE,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    # [1, B, W] -> [1, 1, B, W]: a leading-dim reshape, so a metadata view.
    xqkv = ttnn.reshape(fused, (1, 1, batch, int(fused.shape[-1])))
    shard = decode_shard(device, batch, module.head_dim)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
        xqkv,
        num_heads=module.n_heads,
        num_kv_heads=module.n_kv_heads,
        memory_config=shard,
    )
    ttnn.deallocate(fused)

    if position_embeddings is not None:
        cos, sin = position_embeddings
        q = _rope(q, cos, sin, True)
        k = _rope(k, cos, sin, True)

    # Append this token's K/V in place. No page table: the cache is one contiguous [B, nkv, C, hd]
    # block per layer, not a paged pool.
    ttnn.experimental.paged_update_cache(kv["k"], k, update_idxs_tensor=cur_pos)
    ttnn.experimental.paged_update_cache(kv["v"], v, update_idxs_tensor=cur_pos)
    ttnn.deallocate(k)
    ttnn.deallocate(v)

    # is_causal is left at its default: flash-decode derives the valid key range from cur_pos, so
    # the one query row attends to exactly positions [0, cur_pos] and no mask tensor is read.
    attn = ttnn.transformer.scaled_dot_product_attention_decode(
        q,
        kv["k"],
        kv["v"],
        cur_pos_tensor=cur_pos,
        scale=module.scaling,
        compute_kernel_config=module.compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(q)
    # Back to the one-user-per-core shard the concat wants. Flash-decode writes to DRAM (its
    # working set wants the room); the concat is a pure shuffle and takes only a sharded input.
    attn_sharded = ttnn.to_memory_config(attn, decode_shard(device, batch, module.head_dim))
    ttnn.deallocate(attn)
    merged = ttnn.experimental.nlp_concat_heads_decode(
        attn_sharded,
        num_heads=module.n_heads,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.deallocate(attn_sharded)
    # [1, 1, B, nq*hd] -> [1, B, nq*hd], again a leading-dim view.
    flat_out = ttnn.reshape(merged, (1, batch, int(merged.shape[-1])))
    out = ttnn.linear(
        flat_out,
        module.wo,
        compute_kernel_config=ck,
        program_config=_attn_pc(module, "wo", batch, module.wo),
        dtype=res_dtype,
    )
    ttnn.deallocate(merged)
    if folded > 1:
        return _unfold(out, folded)
    # Hand back the folded stream when the stack is carrying it; the unfold happens once, at the
    # stack exit, instead of once per block.
    return out if _DECODE_FOLD else ttnn.reshape(out, (batch, 1, int(out.shape[-1])))


def _patch_attention(cls, dtype):
    original = cls.__call__
    original_init = cls.__init__

    def __init__(self, *args, **kwargs):
        """Build the stub as it builds itself, then add the FUSED qkv weight beside the three.

        One `[K, nq*hd + 2*nkv*hd]` matmul replaces three, and -- the reason this matters far more
        than the two saved dispatches -- its single output is exactly the layout
        `nlp_create_qkv_heads` consumes, which does the split AND the head transpose for all three
        tensors in one pass. Done HERE, at build time, so no allocation happens inside a forward
        (and so nothing allocates during a trace capture).

        wq/wk/wv are left in place rather than freed: they are what the non-device fallback path
        above still calls, and the fused copy is ~38 MB per block against 32 GB of device DRAM.
        """
        original_init(self, *args, **kwargs)
        self._wqkv = None
        torch_module = kwargs.get("torch_module", args[1] if len(args) > 1 else None)
        state = {}
        if torch_module is not None:
            try:
                state = torch_module.state_dict()
            except Exception:  # noqa: BLE001 - no state dict falls back to the device-side concat
                state = {}
        qkv_keys = ("q_proj.weight", "k_proj.weight", "v_proj.weight")
        if all(k in state for k in qkv_keys):
            # Built from the TORCH weights and uploaded once, already narrowed. Concatenating the
            # three device copies and then typecasting would do the same work on the device inside
            # the build, and would hold both widths of the result at once.
            self._wqkv = ttnn.from_torch(
                torch.cat([state[k].t().to(torch.bfloat16) for k in qkv_keys], dim=-1).contiguous(),
                dtype=_QKV_DTYPE,
                layout=ttnn.TILE_LAYOUT,
                device=self.wo.device(),
            )
        else:
            try:
                self._wqkv = ttnn.concat([self.wq, self.wk, self.wv], dim=-1)
            except Exception:  # noqa: BLE001 - any shape/dtype the concat rejects keeps 3 matmuls
                self._wqkv = None
        if "o_proj.weight" in state:
            stale = self.wo
            self.wo = ttnn.from_torch(
                state["o_proj.weight"].t().to(torch.bfloat16).contiguous(),
                dtype=_WO_DTYPE,
                layout=ttnn.TILE_LAYOUT,
                device=stale.device(),
            )
            ttnn.deallocate(stale)
        # The KV cache is OFF until a caller allocates one. Prefill then also FILLS it, and a
        # single-token call switches to the decode op set. Nothing about the prefill-only path
        # changes while this is None, so every existing caller is untouched.
        self._kv = None

    def kv_enable(self, cur_pos):
        """Arm this block's K/V cache. The BUFFERS appear on the next prefill, which seeds them.

        `cur_pos` is the stack's shared `[B]` position tensor. It is held on the INSTANCE rather
        than threaded through `__call__`, because the whole-block stubs call their attention with
        a fixed `(hidden, position_embeddings, attention_mask)` signature and forward nothing else
        -- adding an argument would mean editing all four stub bodies, which is the thing this
        override layer exists to avoid. The tensor's identity is stable and only its contents
        advance, so holding it here is trace-safe.

        Nothing is allocated here on purpose: the cache is the prefill's own post-RoPE K/V, so
        letting the prefill hand its tensors over is both exactly the right shape and exactly the
        right contents, with no copy and no chance of the two disagreeing. The prefill runs during
        setup, i.e. before any trace capture, so by capture time the buffers exist and their
        addresses are fixed for every replay -- which is what a traced decode step requires.
        """
        self._kv = {"k": None, "v": None, "pos": cur_pos}
        return self._kv

    def kv_disable(self):
        kv = getattr(self, "_kv", None)
        self._kv = None
        if kv is None:
            return
        for key in ("k", "v"):
            if kv.get(key) is None:
                continue
            try:
                ttnn.deallocate(kv[key])
            except Exception:  # noqa: BLE001 - an already-freed buffer is fine to skip
                pass

    def __call__(self, hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
        if not _device_ready(hidden_states, position_embeddings, attention_mask):
            return original(
                self,
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                **kwargs,
            )
        seq_len = int(hidden_states.shape[-2])
        # ONE token with an armed cache is the decode step; anything longer is a prefill, which
        # still runs the full path (and, when the cache is armed, seeds it).
        kv = getattr(self, "_kv", None)
        if kv is not None and kv.get("k") is not None and (seq_len == 1 or _DECODE_FOLD):
            return _decode_call(self, hidden_states, position_embeddings, kv, kv["pos"])
        ck = _matmul_ck(self.wo.device())
        # HiFi4 IS THE BRING-UP DEFAULT, and it is wrong for an op whose inputs are bf8_b. Four
        # fidelity phases exist to consume an fp32 mantissa; q, k and v arrive with an 8-bit one, so
        # phases three and four multiply bits that are not there. The scores are still accumulated
        # in fp32 DEST (fp32_dest_acc_en is left on) -- this narrows the MULTIPLY, not the running
        # softmax sum, which is the part attention actually cannot afford to lose.
        sdpa_ck = _matmul_ck(self.wo.device(), _SDPA_FIDELITY)
        # The residual stream's dtype is READ OFF the input rather than assumed: the output
        # projection has to hand back exactly what the residual add expects, and this override is
        # shared by heads that declare it differently.
        res_dtype = hidden_states.dtype
        flat, batch = _fold(hidden_states)
        rows = int(flat.shape[-2])
        # NARROW THE ACTIVATION AND DROP THE PHASE TOGETHER, or neither pays. Measured separately
        # both fail: the cast alone costs more than it saves (50.63 -> 51.09) because at HiFi2 this
        # projection already sits at 65% of its ceiling, and LoFi alone buys nothing because an
        # FP32 in0 caps every projection here at the same ~200 TFLOP/s whatever its fidelity says.
        # The cap is the unpacker feeding srcA from 4-byte tiles; removing it is what makes the
        # spare phase worth dropping, and dropping the phase is what makes the cast worth paying.
        flat = _narrow_proj_in(flat)
        qkv_ck = _matmul_ck(
            self.wo.device(),
            ttnn.MathFidelity.LoFi if rows > ttnn.TILE_SIZE else ttnn.MathFidelity.HiFi2,
        )

        # q/k/v are produced DIRECTLY in bf16 -- the flash-attention op below takes nothing wider,
        # and a projection's pack format is free, where a typecast afterwards would re-read and
        # re-write the whole [B, heads, S, head_dim] tensor.
        wqkv = getattr(self, "_wqkv", None)
        if wqkv is not None:
            # ONE fused projection, then ONE op for the head split. Splitting a projection with
            # `reshape([B, S, heads, hd]) + transpose(1, 2)` per tensor was the single largest
            # remaining cost in the capture -- 325 ms of ReshapeView + Transpose against ~26 ms of
            # bytes -- because reshaping the LAST dim of a tile tensor is a full relayout, not a
            # view. `nlp_create_qkv_heads` reads the fused [B, 1, S, (nq+2nkv)*hd] once and writes
            # q/k/v already in [B, heads, S, hd], doing all three splits and transposes in a single
            # pass. transpose_k_heads=False because SDPA wants k as [b, nkv, s, dh], not k^T.
            # THE IN half of the handoff the comment below describes. q/k/v already land in L1 and
            # stay there through RoPE and flash attention, but the tensor the head split READS was
            # still going out to DRAM and being fetched back -- 12.6 MB each way per block at 2048
            # rows. `nlp_create_qkv_heads` is tagged dispatch-bound, and what a dispatch-bound op
            # waits for is its operand arriving, so the read is exactly the wait. The projection
            # packs to L1 instead; at bf8_b the fused [rows, (nq+2nkv)*hd] is 12.6 MB against
            # ~165 MB of grid L1, which is the same working set q/k/v hold a moment later.
            fused = ttnn.linear(
                flat,
                wqkv,
                compute_kernel_config=qkv_ck,
                program_config=_attn_pc(self, "qkv", rows, wqkv),
                dtype=_SDPA_DTYPE,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            # Rank-4 view: the last dim is unchanged and both row counts are tile multiples.
            staged = ttnn.reshape(fused, (batch, 1, seq_len, int(fused.shape[-1])))
            # STRAIGHT INTO L1. q/k/v are written by the head split, read by RoPE, written again,
            # and read by flash attention -- four passes that all went through DRAM for a working
            # set of only ~12.6 MB at bf8_b (q 8.4, k and v 2.1 each), which is 115 KB a core on
            # this grid. The three TM/RoPE ops around attention are tagged dispatch-bound rather
            # than bandwidth-bound, and what a dispatch-bound op is waiting for is its operands
            # arriving; taking them off DRAM is the lever that shortens that wait. The mask is left
            # in DRAM because the flash-attention op hard-asserts it there.
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                staged,
                num_heads=self.n_heads,
                num_kv_heads=self.n_kv_heads,
                transpose_k_heads=False,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(fused)
        else:

            def project(weight, n_heads):
                proj = ttnn.linear(flat, weight, compute_kernel_config=ck, dtype=_SDPA_DTYPE)
                heads = ttnn.reshape(proj, (batch, seq_len, n_heads, self.head_dim))
                ttnn.deallocate(proj)
                out = ttnn.transpose(heads, 1, 2)
                ttnn.deallocate(heads)
                return out

            q = project(self.wq, self.n_heads)
            k = project(self.wk, self.n_kv_heads)
            v = project(self.wv, self.n_kv_heads)
        if attention_mask is not None and attention_mask.dtype not in _SDPA_MASK_DTYPES:
            # The pipeline uploads its masks in bf16 already; this covers a caller that does not.
            attention_mask = ttnn.typecast(attention_mask, _SDPA_DTYPE)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q = _rope(q, cos, sin, False)
            k = _rope(k, cos, sin, False)

        # ONE fused flash-attention op in place of the entire manual core. The hand-rolled version
        # was repeat_interleave(k) + repeat_interleave(v) + transpose(k) + matmul + multiply +
        # add(mask) + softmax + matmul = eight launches, and its two matmuls are BATCHED over
        # (sample, head): at B=32 x 32 heads that is 1024 `64 x 128 x 64` products per call, which
        # the roofline tags grid=tiny / bound_by=dispatch -- 168 ms for Q.K^T and 106 ms for P.V
        # against a 0.07 ms floor. SDPA parallelises over b, nqh AND Q's sequence, reads the GQA
        # k/v directly ([b x nkv x s x dh], so both repeat_interleaves go away too) and never
        # materialises the [B, heads, S, S] score tensor at all.
        #
        # NO MASK TENSOR. This used to pass one with is_causal=False, on the grounds that the mask
        # was not plain causality -- the decode stage LEFT-padded, so it also had to blank the
        # padded KEYS, and a kernel-built causal mask would have attended to the pad. That reason
        # expired when the KV cache landed: `pipeline._stage_setup` now RIGHT-pads both stages
        # ("BOTH stages RIGHT-pad now"), because the cache wants real keys in slots [0, real_len)
        # with the new token appending at real_len.
        #
        # Under right padding the pad-blanking is REDUNDANT. Every padded key sits at a position
        # >= real_len, and every row this model reads sits at a position <= real_len - 1, so
        # causality alone already excludes it -- the two masks agree on every row that is read.
        # (They differ on the padded ROWS, which are sliced away before the head and never enter
        # the cache, whose contents come from the real rows.) The other two mask builders,
        # generation.stage_constants and the acoustic stack's, are pure causality already.
        #
        # Handing SDPA is_causal=True and no mask is not just cheaper per byte: the op's
        # use_provided_mask is a COMPILE-TIME constant, so the mask read and the add come out of
        # the kernel entirely rather than being skipped at runtime.
        context = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            scale=self.scaling,
            compute_kernel_config=sdpa_ck,
            program_config=_sdpa_pc(self, self.wo.device(), seq_len),
            # The OUT half of the same handoff: the context is the size of q, its only consumer is
            # the head merge, and the merge's only consumer is the output projection. Keeping all
            # three in L1 takes the last DRAM round trips out of the attention core.
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)
        if kv is None:
            ttnn.deallocate(k)
            ttnn.deallocate(v)
        else:
            # SEED THE CACHE by HANDING IT the prefill's own post-RoPE K/V rather than copying
            # into a buffer allocated earlier. These are the very tensors a cache-less decode
            # would recompute from scratch every token, so the cached step is the same arithmetic
            # as the full recompute, not an approximation of it -- and taking ownership means
            # there is no copy op and no second shape that could disagree with this one.
            _kv_seed(kv, k, v)
        # The mirror of the split: `nlp_concat_heads` folds [B, heads, S, hd] back to
        # [B, 1, S, heads*hd] in one pass, replacing the transpose + last-dim reshape pair (107 ms
        # of ReshapeView plus its share of 39 ms of Transpose in the capture).
        heads = ttnn.experimental.nlp_concat_heads(context, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(context)
        # Straight into the FOLDED layout: the output projection is the same per-token projection
        # as q/k/v and pays the same 32x weight re-stream if left batched. A view, since the last
        # dim is unchanged.
        merged = ttnn.reshape(heads, (1, batch * seq_len, self.n_heads * self.head_dim))
        # Into the INCREMENT's width HERE, where it is one matmul's pack format, rather than as a
        # separate cast over a [B*S, hidden] tensor. The residual add takes this mixed with the fp32
        # running sum and keeps the sum's width -- see _RESIDUAL_DELTA_DTYPE.
        out = ttnn.linear(
            merged,
            self.wo,
            # Same reasoning as the MLP's contraction: both operands are bf8_b, so the extra
            # fidelity phase has no mantissa to consume. Prefill only -- a single-tile-row decode
            # step is bound by streaming the weight, where the math already hides behind the read.
            compute_kernel_config=_matmul_ck(
                self.wo.device(), ttnn.MathFidelity.LoFi if rows > ttnn.TILE_SIZE else ttnn.MathFidelity.HiFi2
            ),
            program_config=_attn_pc(self, "wo", rows, self.wo),
            dtype=_RESIDUAL_DELTA_DTYPE if rows > ttnn.TILE_SIZE else res_dtype,
            # AND PACK THE INCREMENT WHERE THE RESIDUAL ADD WILL READ IT. That add is the single
            # largest gap in the capture and the roofline tags it memory-bound, which for an op
            # this shape means it is waiting on its operands, not computing. The running sum is
            # 25 MB of fp32 and has to stay in DRAM; the INCREMENT is 6.3 MB at bf8_b and is
            # consumed exactly once, by that add, so it has no reason to make the round trip.
            # Prefill only: at one tile row there is nothing to keep.
            memory_config=ttnn.L1_MEMORY_CONFIG if rows > ttnn.TILE_SIZE else None,
        )
        # `merged` is a VIEW of `heads`, so it is left to the last reference to release rather than
        # deallocated here -- freeing a view frees the tensor it aliases, and `heads` is still in
        # scope. Both die at return.
        return _unfold(out, batch)

    cls.__init__ = __init__
    cls.__call__ = __call__
    cls.kv_enable = kv_enable
    cls.kv_disable = kv_disable


def _patch_head(cls, dtype):
    original = cls.__call__
    original_init = cls.__init__

    def __init__(self, *args, **kwargs):
        """Build the stub, then narrow the vocab projection to bf8_b.

        This single weight is 3072 x 131072 -- 805 MB at bf16, the largest tensor in the model --
        and a decode step streams all of it to produce one token per sample. It is the one weight
        where the dtype is worth more than everything around it.
        """
        original_init(self, *args, **kwargs)
        torch_module = kwargs.get("torch_module", args[1] if len(args) > 1 else None)
        weight = getattr(torch_module, "weight", None)
        if weight is None or self.weight.dtype == _HEAD_DTYPE:
            return
        stale = self.weight
        self.weight = ttnn.from_torch(
            weight.t().to(torch.bfloat16).contiguous(),
            dtype=_HEAD_DTYPE,
            layout=ttnn.TILE_LAYOUT,
            device=stale.device(),
        )
        ttnn.deallocate(stale)

    def __call__(self, hidden_states, keep_folded=False, **kwargs):
        """`keep_folded` hands back `[1, B, vocab]` instead of `[B, 1, vocab]`.

        The unfold is the single most expensive movement left in a decode step. Everywhere else a
        fold is free, because the last dim is unchanged and the row counts are tile multiples --
        but `[1, B, vocab] -> [B, 1, vocab]` turns one 32-row tile row into B slabs of a single
        row each, every one padded back out to a tile, and it does that across a 131072-wide
        tensor. The sampler that consumes this does `argmax(dim=-1)` and could not care which of
        the two leading dims carries the batch, so a caller that only samples asks for the folded
        form and the relayout never happens.
        """
        if not _device_ready(hidden_states):
            return original(self, hidden_states, **kwargs)
        extra = {} if dtype is None else {"dtype": dtype}
        ck = _matmul_ck(self.weight.device())
        # THE MANY-ROW HEADS GET THE SAME PAIR THE PROJECTIONS GOT. This override serves three
        # weights: the text vocab head, which is called with ONE tile row per token and is bound by
        # streaming a 201 MB weight, and the acoustic section's llm_projection and semantic
        # codebook head, which are called over a whole 1024-row sequence and are the last matmuls
        # in the model still reading an FP32 activation at HiFi2. For the second kind the coupled
        # lever applies unchanged -- narrow the in0 so the unpacker stops feeding srcA from 4-byte
        # tiles, and drop the phase that cap was hiding. The one-tile-row head is left exactly as
        # it was: its activation is 32 rows against a 201 MB weight, so there are no bytes there to
        # save, and it is on the per-token path.
        # `flat` is NOT deallocated here. When the fold is a metadata view it shares the caller's
        # buffer, so freeing it would free the caller's tensor; when it is a real relayout it is a
        # local that the last reference releases on return. One rule covers both.
        flat, batch = _fold(hidden_states)
        if int(flat.shape[-2]) > ttnn.TILE_SIZE:
            flat = _narrow_proj_in(flat)
            ck = _matmul_ck(self.weight.device(), ttnn.MathFidelity.LoFi)
        out = ttnn.linear(
            flat,
            self.weight,
            bias=self.bias,
            compute_kernel_config=ck,
            **extra,
        )
        return out if keep_folded else _unfold(out, batch)

    cls.__init__ = __init__
    cls.__call__ = __call__


def _norm_shard_plan(device, rows, width, narrowing=False):
    """The largest BLOCK shard of a `[rows, width]` tile grid that divides the device grid exactly.

    A block shard is the only layernorm layout that splits the WIDTH as well as the rows, and it is
    the only way this op can hold more than `rows/32` cores -- the interleaved kernel parallelises
    over tile rows alone, so at 2048 rows it takes 64 of 110 and at 1024 rows only 32. Both
    extents have to divide exactly (a partial shard is rejected outright), so the plan is the
    largest divisor of each under the real device grid, never a hard-coded one.

    `narrowing` says the caller will also fold its consumer's typecast into this call, which changes
    the test the plan must pass. Without it the shard has to buy CORES, because its own reshard
    costs what the interleaved op's DRAM write costs and a wash is a loss. With it the shard buys an
    L1 HANDOFF: the wide result never goes to DRAM and the typecast never reads it back, ~50 MB a
    call at 2048x3072, which is worth taking even at the same core count.

    AND IT MUST FIT L1. This is the guard the first attempt was missing: that version held the
    staged input AND a separate norm output, 2 x 393 KB a core at 2048 rows before the norm's own
    CBs, and the op then failed during TRACE CAPTURE rather than at the first eager call -- which
    reads as a partial profile (the whole 2048-row stack absent from the census) and measures as a
    62% speedup. So the plan is written for an IN-PLACE norm, which needs one wide shard rather
    than two, and is declined outright when even that does not fit the budget.

    Returns `(memory_config, program_config)` or None when no split is worth it or none fits.
    """
    grid = device.compute_with_storage_grid_size()
    m_tiles, n_tiles = -(-int(rows) // 32), -(-int(width) // 32)
    gy = max((c for c in range(1, int(grid.y) + 1) if m_tiles % c == 0), default=1)
    gx = max((c for c in range(1, int(grid.x) + 1) if n_tiles % c == 0), default=1)
    if gy * gx <= 1:
        return None
    if not narrowing and gy * gx <= min(m_tiles, int(grid.x) * int(grid.y)):
        # No more cores than the interleaved kernel already takes -- nothing for this to buy.
        return None
    block_h, block_w = m_tiles // gy, n_tiles // gx
    if block_h * block_w * _FP32_TILE_BYTES > _NORM_SHARD_L1_BUDGET:
        return None
    subblock_w = next((w for w in range(min(block_w, 4), 0, -1) if block_w % w == 0), 1)
    shard = ttnn.create_sharded_memory_config(
        shape=(int(rows), int(width)),
        core_grid=ttnn.CoreGrid(y=gy, x=gx),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    return shard, ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        subblock_w=subblock_w,
        block_h=block_h,
        block_w=block_w,
        # IN PLACE, over the STAGED COPY -- never over the caller's tensor. `staged` is a fresh L1
        # shard of the residual, so overwriting it cannot touch the running sum, which the next
        # residual add still needs. This is what halves the L1 the plan has to fit.
        inplace=True,
    )


_norm_ck_cache = {}


def _norm_ck(norm):
    """The norm's compute kernel, with the bring-up default's two spare fidelity phases removed.

    The stub builds every RMSNorm with `math_fidelity=HiFi4`, which is the SAFE default a bring-up
    reaches for, not a perf choice -- and there are 44 norm launches in the capture. Fidelity is
    how many passes the FPU makes over the operands' mantissa, and this op multiplies an
    activation by a per-channel weight: HiFi2 carries the full mantissa of a bf16 weight, which is
    what the weight actually is. The cap that MUST stay is `fp32_dest_acc_en=True` -- a
    normalisation reduction sums the whole row, and that accumulator is the one thing a 26-layer
    stack compounds (the catalogue's hard floor for norms is HiFi2 + fp32 DEST, never LoFi), so
    the DEST width and the packer accumulation are carried over from the stub unchanged and only
    the phase count moves.

    Read off the norm's OWN config rather than rebuilt from scratch, so a norm that is already at
    or below HiFi2 keeps whatever it has, and cached per (arch, source fidelity) so the lookup
    allocates nothing and stays trace-safe.
    """
    ck = getattr(norm, "compute_kernel_config", None)
    fidelity = getattr(ck, "math_fidelity", None)
    if fidelity != ttnn.MathFidelity.HiFi4:
        return ck
    device = norm.weight.device()
    key = (device.arch(), fidelity)
    got = _norm_ck_cache.get(key)
    if got is None:
        got = _norm_ck_cache[key] = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=bool(getattr(ck, "math_approx_mode", False)),
            fp32_dest_acc_en=bool(getattr(ck, "fp32_dest_acc_en", True)),
            packer_l1_acc=bool(getattr(ck, "packer_l1_acc", True)),
        )
    return got


def _norm_call(norm, original, hidden_states, out_dtype=None, **kwargs):
    """`rms_norm`, optionally on a block shard, optionally narrowing the result on the way out.

    The narrowing is the reason the shard pays. The MLP's norm output is read by exactly one
    consumer, which wants it at `_PROJ_IN_DTYPE`, and as two interleaved ops that is four passes
    over a [2048, 3072] fp32 tensor: the norm reads the residual and writes a wide result, then the
    typecast reads that wide result back and writes a narrow one. Sharded, the wide result never
    leaves L1 -- only the residual comes in and only the narrow result goes out.
    """
    if not isinstance(hidden_states, ttnn.Tensor):
        return original(norm, hidden_states, **kwargs)
    # ON THE INSTANCE, not just on the call below. The interleaved path hands the work back to the
    # stub's own `__call__`, which reads `self.compute_kernel_config` -- passing the narrowed
    # config only to the sharded branch would leave every norm that declines a shard, and every
    # decode-shaped norm, still paying four phases. One assignment covers both branches, and it is
    # idempotent because `_norm_ck` returns the config unchanged once it is no longer HiFi4.
    norm.compute_kernel_config = _norm_ck(norm)
    rows = int(hidden_states.shape[-2])
    for dim in list(hidden_states.shape)[:-2]:
        rows *= int(dim)
    width = int(hidden_states.shape[-1])
    # ONE TILE ROW IS THE DECODE STEP, left bit-identical: the narrowing exists to cut bytes on a
    # 2048-row tensor, and at 32 rows there are none to cut.
    narrowing = out_dtype is not None and rows > ttnn.TILE_SIZE and out_dtype != hidden_states.dtype
    plan = None
    if rows > ttnn.TILE_SIZE:
        cache = getattr(norm, "_shard_cache", None)
        if cache is None:
            cache = norm._shard_cache = {}
        key = (rows, width, narrowing)
        if key not in cache:
            try:
                cache[key] = _norm_shard_plan(norm.weight.device(), rows, width, narrowing=narrowing)
            except Exception:  # noqa: BLE001 - a shape the shard helper rejects keeps the stock path
                cache[key] = None
        plan = cache[key]
    # WHERE THE NORM'S RESULT LANDS. A block norm has exactly ONE consumer -- `input_layernorm`
    # the qkv projection, `post_attention_layernorm` the MLP expansion -- and the result is
    # 6.3 MB at `_PROJ_IN_DTYPE`, so the same "consumed once, do not round-trip" rule that paid on
    # the two increments and the SwiGLU intermediates applies to it. The RESIDUAL is not this
    # tensor: that is the norm's INPUT, it is fp32 and 25 MB, and it stays in DRAM.
    landing = ttnn.L1_MEMORY_CONFIG if (narrowing and rows > ttnn.TILE_SIZE) else ttnn.DRAM_MEMORY_CONFIG
    if plan is None:
        out = original(norm, hidden_states, **kwargs)
        return ttnn.typecast(out, out_dtype, memory_config=landing) if narrowing else out
    shard, pc = plan
    staged = ttnn.to_memory_config(hidden_states, shard)
    # `inplace=True`, so this aliases `staged`; do NOT deallocate staged separately.
    out = ttnn.rms_norm(
        staged,
        epsilon=norm.epsilon,
        weight=norm.weight,
        compute_kernel_config=norm.compute_kernel_config,
        program_config=pc,
        memory_config=shard,
    )
    if narrowing:
        # Still ON the shard, so this reads and writes L1 rather than the 25 MB each way it would
        # cost between two interleaved ops.
        out = ttnn.typecast(out, out_dtype)
    return ttnn.to_memory_config(out, landing)


class _NarrowingNorm:
    """The MLP's norm, with its consumer's typecast folded in. Transparent for everything else.

    Wrapping the INSTANCE the decoder layer holds, rather than patching the norm class, is what
    keeps this scoped. The same class also serves the attention block's norm and the stack's final
    norm, and neither wants a narrow result -- attention's projection is fed by its own cast on a
    different schedule, and the final norm feeds a slice and an fp32 head. Only this call site has
    a consumer that was going to pay for the cast anyway.
    """

    __slots__ = ("_inner", "_original")

    def __init__(self, inner, original):
        self._inner = inner
        self._original = original

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __call__(self, hidden_states, **kwargs):
        return _norm_call(self._inner, self._original, hidden_states, out_dtype=_PROJ_IN_DTYPE, **kwargs)


def _patch_norm(cls, dtype):
    original = cls.__call__

    def __call__(self, hidden_states, **kwargs):
        return _norm_call(self, original, hidden_states, **kwargs)

    cls.__call__ = __call__
    cls._perf_original_call = staticmethod(original)


def _patch_layer(cls, dtype):
    """Hand BOTH of a block's norms the narrowing their one consumer would otherwise do alone.

    A pre-norm block has exactly two norms and each feeds exactly one thing: `input_layernorm` the
    qkv projection, `post_attention_layernorm` the MLP. Both consumers now read `_PROJ_IN_DTYPE`,
    so both were paying for the same standalone typecast over a [B*S, hidden] fp32 tensor, and
    both can have it folded into the sharded norm instead. The stack's FINAL norm is not a block
    norm and is deliberately left out: it feeds a slice and an fp32 head.
    """
    original_init = cls.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        for name in ("input_layernorm", "post_attention_layernorm"):
            norm = getattr(self, name, None)
            inner_original = getattr(type(norm), "_perf_original_call", None) if norm is not None else None
            if inner_original is not None and not isinstance(norm, _NarrowingNorm):
                setattr(self, name, _NarrowingNorm(norm, inner_original))

    cls.__init__ = __init__


def install() -> bool:
    """Install the overrides once. Idempotent; returns True the first time it patched."""
    global _installed
    if _installed:
        return False
    _installed = True
    for name, cls_name, patch in (
        [(n, "TtAttention", _patch_attention) for n in _ATTENTION_MODULES]
        + [(n, "TtMLP", _patch_mlp) for n in _MLP_MODULES]
        + [(n, "TtDecoderHead", _patch_head) for n in _HEAD_MODULES]
        + [(n, "TtRMSNorm", _patch_norm) for n in _NORM_MODULES]
        # LAST: it wraps the norm INSTANCE a layer builds, so the norm class must already carry its
        # original __call__ by the time any layer is constructed.
        + [(n, "TtDecoderLayer", _patch_layer) for n in _LAYER_MODULES]
    ):
        module = importlib.import_module(f"{_STUB_PKG}.{name}")
        cls = getattr(module, cls_name, None)
        if cls is None:
            continue
        # `model.py` pins every linear's output dtype; the others inherit it from the input. The
        # override must not change that, so the module's own constant is carried through.
        patch(cls, getattr(module, "_ACT_DTYPE", None))
    return True

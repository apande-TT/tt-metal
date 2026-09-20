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

_STUB_PKG = "models.tt_transformers.demo.voxtral_4b_tts_2603._stubs"

# ttnn's flash-attention op takes q/k/v AND the mask in bf16/bf8_b/bf4_b only
# (sdpa_device_operation.cpp validates both), so the attention core runs at bf16 -- the top of that
# range. The residual stream is untouched and stays at whatever dtype the caller carries.
_SDPA_DTYPE = ttnn.bfloat16
_SDPA_MASK_DTYPES = (ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b)

# The format the model's WIDE intermediates are carried in -- the SwiGLU gate/up/product, which are
# [B*S, 4*hidden] and are each consumed exactly once by the next op. Distinct from the residual
# stream's dtype, which this file never changes.
_WIDE_DTYPE = ttnn.bfloat16

# The WEIGHT format for every projection. bfloat8_b is a block format: a shared 8-bit exponent per
# 16-datum block with an 8-bit mantissa each, so it holds a weight's dynamic range while halving the
# bytes. Two things follow, and both matter here:
#   * DRAM: the checkpoint's served width drops from ~8.4 GB to ~4.7 GB, which is what every
#     memory-bound op in the capture is waiting on.
#   * MATH: bf8_b is the format LoFi is FOR -- one fidelity phase covers its mantissa -- so the
#     matmul rate doubles as well. Dtype and fidelity are therefore ONE lever: bf8_b at HiFi2 buys
#     the bytes and none of the math, and measured 153.13 ms/token against LoFi's 137.52.
#   * L1: a bf8_b tile is 1088 B against bf16's 2048, which frees the in1 circular buffer enough to
#     widen in0_block_w -- see `_block_config`.
_WEIGHT_DTYPE = ttnn.bfloat8_b


def _narrow(tensor):
    """Return `tensor` re-packed as `_WEIGHT_DTYPE`, or the original if that is not possible."""
    if tensor is None or tensor.dtype == _WEIGHT_DTYPE:
        return tensor
    try:
        return ttnn.typecast(tensor, _WEIGHT_DTYPE)
    except Exception:  # noqa: BLE001 - a format ttnn will not re-pack just keeps the wide weight
        return tensor


def _narrowing_upload(fn):
    """Wrap a stub's weight-upload helper so a PROJECTION weight is created narrow.

    Re-packing a weight after it is already on device works, but it is ~235 device typecasts over
    8.4 GB at build time, and those land INSIDE the profiled region -- they cost 31 ms of the
    capture's device time and slowed the build, for a re-pack that production pays once. Uploading
    at the target format never allocates the wide copy at all.

    Only 2-D tensors with both dims at least a tile wide are narrowed: that is exactly the
    projection matrices. The RMSNorm weight, the rope tables and the mask arrive 4-D and a bias
    arrives as [1, N], so they keep the helper's own format and pass straight through.
    """

    def upload(t, device, *args, **kwargs):
        if getattr(t, "dim", None) and t.dim() == 2 and min(t.shape) >= 32 and "dtype" not in kwargs:
            return ttnn.from_torch(
                t.to(torch.bfloat16).contiguous(),
                dtype=_WEIGHT_DTYPE,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
        return fn(t, device, *args, **kwargs)

    upload._voxtral_narrowed = True
    return upload


def _narrow_in_place(module, names):
    """Re-pack any weight that still arrived wide, and FREE the original.

    A safety net rather than the main path: `_narrowing_upload` means these are normally already
    narrow and every call here is a no-op. It covers a weight built through some other route.
    """
    for name in names:
        wide = getattr(module, name, None)
        if wide is None:
            continue
        narrow = _narrow(wide)
        if narrow is not wide:
            setattr(module, name, narrow)
            try:
                ttnn.deallocate(wide)
            except Exception:  # noqa: BLE001
                pass


_installed = False


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
    per_core_m = -(-int(m_tiles) // gy)
    per_core_n = -(-int(n_tiles) // gx)
    # in0_block_w must divide K in tiles. Larger means fewer trips round the K loop but a bigger in0
    # circular buffer. 2 was the ceiling while the weights were bf16; a bf8_b weight tile is 1088 B
    # against 2048, which frees enough of the in1 buffer to afford 4 -- at per_core_M=7 that is a
    # 224 KB in0 CB and a 235 KB in1 CB, which leaves the capped output block room inside 1.5 MB.
    in0_block_w = next((c for c in (4, 2, 1) if int(k_tiles) % c == 0), 1)
    # out_subblock_h * out_subblock_w <= 4 because the compute kernel runs fp32_dest_acc_en=True,
    # which halves DEST. Pick the largest legal pair that divides the per-core block.
    best_h, best_w = 1, 1
    for h in range(1, per_core_m + 1):
        if per_core_m % h:
            continue
        for w in range(1, per_core_n + 1):
            if per_core_n % w or h * w > 4:
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

_SILU = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)

# The fidelity FOLLOWS THE WEIGHT FORMAT: one phase covers a bf8_b mantissa exactly as two cover
# bf16's, and every phase beyond that is math for no accuracy. HiFi4 was the bring-up default and
# the capture showed its cost directly -- the gate/up projection measured 137 ms against a 68.7 ms
# peak-FLOP floor while holding all 110 cores, a ratio no blocking knob explains.
#
# The walk was measured a step at a time on the production per-token metric: HiFi4 -> HiFi2 over
# bf16 weights 229.95 -> 177.04 ms, bf8_b weights at HiFi2 161.61 -> 153.13, and this last step to
# LoFi 153.13 -> 137.52. It costs PCC 0.9999 -> 0.9989 and it buys BOTH stages ~15%, decode
# included -- which is the stage still short of its band, and the reason the spend is the right
# one here.
#
# fp32_dest_acc_en and packer_l1_acc are deliberately LEFT ON: the accumulation, not the multiply,
# is what a 26-layer residual stack compounds, and turning fp32 DEST off is a separate lever with
# its own PCC price (it would also relax the subblock limit in _block_config).
#
# The NORMS keep their own HiFi4 config. Normalisation reductions are the documented hard floor --
# they compound to a PCC failure over depth in a way a projection does not.
_mm_ck_cache = {}


def _matmul_ck(device):
    """The matmuls' compute kernel config: LoFi over bf8_b weights. Cached per arch."""
    arch = device.arch()
    ck = _mm_ck_cache.get(arch)
    if ck is None:
        ck = _mm_ck_cache[arch] = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    return ck


def _patch_mlp(cls, dtype):
    original = cls.__call__
    original_init = cls.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _narrow_in_place(self, ("gate", "up", "down"))

    def __call__(self, x, **kwargs):
        if not _device_ready(x):
            return original(self, x, **kwargs)
        ck = _matmul_ck(self.gate.device())
        # The dtype the RESIDUAL add expects. `x` is this block's norm output and rms_norm preserves
        # its input dtype, so x.dtype IS the residual stream's dtype -- read, not assumed.
        res_dtype = x.dtype if dtype is None else dtype
        flat, batch = _fold(x)
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
        gate = ttnn.linear(flat, self.gate, compute_kernel_config=ck, program_config=pcs[0], dtype=_WIDE_DTYPE)
        up = ttnn.linear(flat, self.up, compute_kernel_config=ck, program_config=pcs[1], dtype=_WIDE_DTYPE)
        prod = ttnn.multiply(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = ttnn.linear(prod, self.down, compute_kernel_config=ck, program_config=pcs[2], dtype=res_dtype)
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
        try:
            self._wqkv = _narrow(ttnn.concat([self.wq, self.wk, self.wv], dim=-1))
        except Exception:  # noqa: BLE001 - any shape/dtype the concat rejects just keeps 3 matmuls
            self._wqkv = None
        # The three unfused copies are narrowed too rather than left wide: the fused weight is what
        # the fast path reads, so leaving them in bf16 would ADD storage instead of saving it.
        _narrow_in_place(self, ("wq", "wk", "wv", "wo"))

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
        ck = _matmul_ck(self.wo.device())
        sdpa_ck = self.compute_kernel_config
        # The residual stream's dtype is READ OFF the input rather than assumed: the output
        # projection has to hand back exactly what the residual add expects, and this override is
        # shared by heads that declare it differently.
        res_dtype = hidden_states.dtype
        flat, batch = _fold(hidden_states)
        rows = int(flat.shape[-2])

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
            fused = ttnn.linear(
                flat,
                wqkv,
                compute_kernel_config=ck,
                program_config=_attn_pc(self, "qkv", rows, wqkv),
                dtype=_SDPA_DTYPE,
            )
            # Rank-4 view: the last dim is unchanged and both row counts are tile multiples.
            staged = ttnn.reshape(fused, (batch, 1, seq_len, int(fused.shape[-1])))
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                staged,
                num_heads=self.n_heads,
                num_kv_heads=self.n_kv_heads,
                transpose_k_heads=False,
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
            q = ttnn.experimental.rotary_embedding_hf(q, cos, sin, is_decode_mode=False)
            k = ttnn.experimental.rotary_embedding_hf(k, cos, sin, is_decode_mode=False)

        # ONE fused flash-attention op in place of the entire manual core. The hand-rolled version
        # was repeat_interleave(k) + repeat_interleave(v) + transpose(k) + matmul + multiply +
        # add(mask) + softmax + matmul = eight launches, and its two matmuls are BATCHED over
        # (sample, head): at B=32 x 32 heads that is 1024 `64 x 128 x 64` products per call, which
        # the roofline tags grid=tiny / bound_by=dispatch -- 168 ms for Q.K^T and 106 ms for P.V
        # against a 0.07 ms floor. SDPA parallelises over b, nqh AND Q's sequence, reads the GQA
        # k/v directly ([b x nkv x s x dh], so both repeat_interleaves go away too) and never
        # materialises the [B, heads, S, S] score tensor at all.
        #
        # is_causal=False ON PURPOSE: the mask is not plain causality. The decode stage LEFT-pads,
        # so the mask also blanks the padded KEYS; letting the kernel build its own causal mask
        # would silently attend to the pad.
        context = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            is_causal=False,
            scale=self.scaling,
            compute_kernel_config=sdpa_ck,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        # The mirror of the split: `nlp_concat_heads` folds [B, heads, S, hd] back to
        # [B, 1, S, heads*hd] in one pass, replacing the transpose + last-dim reshape pair (107 ms
        # of ReshapeView plus its share of 39 ms of Transpose in the capture).
        heads = ttnn.experimental.nlp_concat_heads(context)
        ttnn.deallocate(context)
        # Straight into the FOLDED layout: the output projection is the same per-token projection
        # as q/k/v and pays the same 32x weight re-stream if left batched. A view, since the last
        # dim is unchanged.
        merged = ttnn.reshape(heads, (1, batch * seq_len, self.n_heads * self.head_dim))
        # Back to the residual stream's dtype HERE, where it is one matmul's pack format, rather
        # than as a separate cast over a [B*S, hidden] tensor.
        out = ttnn.linear(
            merged,
            self.wo,
            compute_kernel_config=ck,
            program_config=_attn_pc(self, "wo", rows, self.wo),
            dtype=res_dtype,
        )
        # `merged` is a VIEW of `heads`, so it is left to the last reference to release rather than
        # deallocated here -- freeing a view frees the tensor it aliases, and `heads` is still in
        # scope. Both die at return.
        return _unfold(out, batch)

    cls.__init__ = __init__
    cls.__call__ = __call__


def _patch_head(cls, dtype):
    original = cls.__call__
    original_init = cls.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # The vocab weight is the single largest tensor in the checkpoint (3072 x 131072 = 805 MB in
        # bf16) and the head is memory-bound, so this one re-pack is worth ~400 MB of DRAM per read.
        _narrow_in_place(self, ("weight",))

    def __call__(self, hidden_states, **kwargs):
        if not _device_ready(hidden_states):
            return original(self, hidden_states, **kwargs)
        extra = {} if dtype is None else {"dtype": dtype}
        ck = _matmul_ck(self.weight.device())
        # `flat` is NOT deallocated here. When the fold is a metadata view it shares the caller's
        # buffer, so freeing it would free the caller's tensor; when it is a real relayout it is a
        # local that the last reference releases on return. One rule covers both.
        flat, batch = _fold(hidden_states)
        out = ttnn.linear(
            flat,
            self.weight,
            bias=self.bias,
            compute_kernel_config=ck,
            **extra,
        )
        return _unfold(out, batch)

    cls.__init__ = __init__
    cls.__call__ = __call__


def install() -> bool:
    """Install the overrides once. Idempotent; returns True the first time it patched."""
    global _installed
    if _installed:
        return False
    _installed = True
    # FIRST, so the weights are CREATED narrow rather than re-packed on device afterwards.
    for name in dict.fromkeys(_ATTENTION_MODULES + _MLP_MODULES + _HEAD_MODULES):
        module = importlib.import_module(f"{_STUB_PKG}.{name}")
        for helper in ("_to_device", "_weight"):
            fn = getattr(module, helper, None)
            if fn is None or getattr(fn, "_voxtral_narrowed", False):
                continue
            setattr(module, helper, _narrowing_upload(fn))
    for name, cls_name, patch in (
        [(n, "TtAttention", _patch_attention) for n in _ATTENTION_MODULES]
        + [(n, "TtMLP", _patch_mlp) for n in _MLP_MODULES]
        + [(n, "TtDecoderHead", _patch_head) for n in _HEAD_MODULES]
    ):
        module = importlib.import_module(f"{_STUB_PKG}.{name}")
        cls = getattr(module, cls_name, None)
        if cls is None:
            continue
        # `model.py` pins every linear's output dtype; the others inherit it from the input. The
        # override must not change that, so the module's own constant is carried through.
        patch(cls, getattr(module, "_ACT_DTYPE", None))
    return True

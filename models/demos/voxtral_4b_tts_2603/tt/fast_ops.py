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


def _patch_mlp(cls, dtype):
    original = cls.__call__

    def __call__(self, x, **kwargs):
        if not _device_ready(x):
            return original(self, x, **kwargs)
        ck = self.compute_kernel_config
        extra = {} if dtype is None else {"dtype": dtype}
        flat, batch = _fold(x)
        gate = ttnn.linear(flat, self.gate, compute_kernel_config=ck, activation="silu", **extra)
        up = ttnn.linear(flat, self.up, compute_kernel_config=ck, **extra)
        prod = ttnn.multiply(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = ttnn.linear(prod, self.down, compute_kernel_config=ck, **extra)
        ttnn.deallocate(prod)
        return _unfold(out, batch)

    cls.__call__ = __call__


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
            self._wqkv = ttnn.concat([self.wq, self.wk, self.wv], dim=-1)
        except Exception:  # noqa: BLE001 - any shape/dtype the concat rejects just keeps 3 matmuls
            self._wqkv = None

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
        ck = self.compute_kernel_config
        # The residual stream's dtype is READ OFF the input rather than assumed: the output
        # projection has to hand back exactly what the residual add expects, and this override is
        # shared by heads that declare it differently.
        res_dtype = hidden_states.dtype
        flat, batch = _fold(hidden_states)

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
            fused = ttnn.linear(flat, wqkv, compute_kernel_config=ck, dtype=_SDPA_DTYPE)
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
            compute_kernel_config=ck,
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
        out = ttnn.linear(merged, self.wo, compute_kernel_config=ck, dtype=res_dtype)
        # `merged` is a VIEW of `heads`, so it is left to the last reference to release rather than
        # deallocated here -- freeing a view frees the tensor it aliases, and `heads` is still in
        # scope. Both die at return.
        return _unfold(out, batch)

    cls.__init__ = __init__
    cls.__call__ = __call__


def _patch_head(cls, dtype):
    original = cls.__call__

    def __call__(self, hidden_states, **kwargs):
        if not _device_ready(hidden_states):
            return original(self, hidden_states, **kwargs)
        extra = {} if dtype is None else {"dtype": dtype}
        # `flat` is NOT deallocated here. When the fold is a metadata view it shares the caller's
        # buffer, so freeing it would free the caller's tensor; when it is a real relayout it is a
        # local that the last reference releases on return. One rule covers both.
        flat, batch = _fold(hidden_states)
        out = ttnn.linear(
            flat,
            self.weight,
            bias=self.bias,
            compute_kernel_config=self.compute_kernel_config,
            **extra,
        )
        return _unfold(out, batch)

    cls.__call__ = __call__


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
    ):
        module = importlib.import_module(f"{_STUB_PKG}.{name}")
        cls = getattr(module, cls_name, None)
        if cls is None:
            continue
        # `model.py` pins every linear's output dtype; the others inherit it from the input. The
        # override must not change that, so the module's own constant is carried through.
        patch(cls, getattr(module, "_ACT_DTYPE", None))
    return True

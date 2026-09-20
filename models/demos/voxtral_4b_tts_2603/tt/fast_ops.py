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
        extra = {} if dtype is None else {"dtype": dtype}
        flat, batch = _fold(hidden_states)

        def project(weight, n_heads):
            proj = ttnn.linear(flat, weight, compute_kernel_config=ck, **extra)
            heads = ttnn.reshape(proj, (batch, seq_len, n_heads, self.head_dim))
            ttnn.deallocate(proj)
            out = ttnn.transpose(heads, 1, 2)
            ttnn.deallocate(heads)
            return out

        q = project(self.wq, self.n_heads)
        k = project(self.wk, self.n_kv_heads)
        v = project(self.wv, self.n_kv_heads)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q = ttnn.experimental.rotary_embedding_hf(q, cos, sin, is_decode_mode=False)
            k = ttnn.experimental.rotary_embedding_hf(k, cos, sin, is_decode_mode=False)

        if self.n_rep > 1:
            k = ttnn.repeat_interleave(k, self.n_rep, dim=1)
            v = ttnn.repeat_interleave(v, self.n_rep, dim=1)

        k_t = ttnn.transpose(k, -2, -1)
        ttnn.deallocate(k)
        scores = ttnn.matmul(q, k_t, compute_kernel_config=ck, **extra)
        ttnn.deallocate(q)
        ttnn.deallocate(k_t)
        scores = ttnn.multiply(scores, self.scaling)
        if attention_mask is not None:
            scores = ttnn.add(scores, attention_mask)
        probs = ttnn.softmax(scores, dim=-1, compute_kernel_config=ck)
        ttnn.deallocate(scores)

        context = ttnn.matmul(probs, v, compute_kernel_config=ck, **extra)
        ttnn.deallocate(probs)
        ttnn.deallocate(v)
        heads = ttnn.transpose(context, 1, 2)
        ttnn.deallocate(context)
        # Straight into the FOLDED layout: the output projection is the same per-token
        # projection as q/k/v and pays the same 32x weight re-stream if left batched.
        merged = ttnn.reshape(heads, (1, batch * seq_len, self.n_heads * self.head_dim))
        ttnn.deallocate(heads)
        out = ttnn.linear(merged, self.wo, compute_kernel_config=ck, **extra)
        ttnn.deallocate(merged)
        return _unfold(out, batch)

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

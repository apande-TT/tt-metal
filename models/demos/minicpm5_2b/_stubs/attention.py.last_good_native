# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn port of `LlamaAttention` for `openbmb/MiniCPM5-2B`.

GQA attention: 16 query heads, 2 KV heads, head_dim 128, no projection biases.
HF semantics reproduced:
  q/k/v = x @ W^T, split into heads
  q, k  = q*cos + rotate_half(q)*sin   (cos/sin are the model's `position_embeddings`)
  [k, v] = cat(past, new) when a cache is passed (DynamicCache.update appends)
  out   = softmax(q k^T * head_dim^-0.5 + causal) v, merged heads, @ Wo^T
rotate_half is a fixed [head_dim, head_dim] matmul so the whole forward stays on device.

Batch is read from the activation ([B, S, hidden]); cos/sin may be torch (the HF capture) or ttnn
([B, S, head_dim], from the rotary stub). Passing `kv_cache` (a `KVCache`) together with `attn_ctx`
(an `AttnContext`) selects the resident-cache path the e2e pipeline uses: prefill writes the cache
and attends causally over its own keys, decode writes one position per batch row and attends over the
whole fixed-capacity cache. Every mask and position lives in a persistent device tensor, so neither
path creates a tensor from host and both are trace-capturable.
"""
from __future__ import annotations

import torch

import ttnn


def _rotate_half_matrix(head_dim):
    # x @ R == cat(-x[..., d/2:], x[..., :d/2])
    half = head_dim // 2
    rot = torch.zeros(head_dim, head_dim)
    rot[half:, :half] = -torch.eye(half)
    rot[:half, half:] = torch.eye(half)
    return rot


class KVCache:
    """One layer's resident cache: k and v, each [B, n_kv_heads, capacity, head_dim] on device."""

    def __init__(self, device, batch, n_kv_heads, capacity, head_dim):
        zeros = torch.zeros(batch, n_kv_heads, capacity, head_dim)
        self.k = ttnn.from_torch(zeros, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.v = ttnn.from_torch(zeros, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.capacity = capacity


class AttnContext:
    """Per-forward device tensors shared by every layer.

    mode "prefill": `mask` is the additive causal mask [1, 1, S, S].
    mode "decode":  `mask` is the additive key mask [B, 1, 1, capacity] (keys past each row's position
                    are -inf) and `write` the one-hot [B, 1, capacity, 1] of each row's write slot.
    """

    def __init__(self, mode, mask, write=None):
        self.mode, self.mask, self.write = mode, mask, write


class TtAttention:
    def __init__(self, device, torch_module) -> None:
        cfg = torch_module.config
        self.device = device
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.kv_groups = self.n_heads // self.n_kv_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.n_heads
        self.scale = self.head_dim**-0.5
        self.layer_idx = getattr(torch_module, "layer_idx", 0)

        self.compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            # packer_l1_acc with fp32 dest accumulation intermittently stalled the final readback here.
            packer_l1_acc=False,
        )

        def _w(linear):
            return ttnn.from_torch(
                linear.weight.detach().to(torch.float32).t().contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.wq = _w(torch_module.q_proj)
        self.wk = _w(torch_module.k_proj)
        self.wv = _w(torch_module.v_proj)
        self.wo = _w(torch_module.o_proj)
        self.rot = ttnn.from_torch(
            _rotate_half_matrix(self.head_dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

    def _to_tt(self, t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

    def _heads(self, x, n_heads, seq):
        # [B, S, n*D] -> [B, n, S, D]
        x = ttnn.reshape(x, (x.shape[0], seq, n_heads, self.head_dim))
        return ttnn.permute(x, (0, 2, 1, 3))

    def _cos_sin(self, position_embeddings, batch, seq):
        cos, sin = position_embeddings
        if isinstance(cos, ttnn.Tensor):
            return ttnn.reshape(cos, (batch, 1, seq, self.head_dim)), ttnn.reshape(sin, (batch, 1, seq, self.head_dim))
        cb = cos.shape[0] if cos.dim() == 3 else 1
        cos = ttnn.reshape(self._to_tt(cos), (cb, 1, seq, self.head_dim))
        sin = ttnn.reshape(self._to_tt(sin), (cb, 1, seq, self.head_dim))
        return cos, sin

    def _cached(self, x, cos, sin, kv_cache, ctx):
        """Resident-cache attention (prefill or one decode step) for a [B, S, hidden] activation."""
        b, seq = x.shape[0], x.shape[-2]
        q = ttnn.linear(x, self.wq, compute_kernel_config=self.compute_cfg)
        k = ttnn.linear(x, self.wk, compute_kernel_config=self.compute_cfg)
        v = ttnn.linear(x, self.wv, compute_kernel_config=self.compute_cfg)
        if ctx.mode == "prefill":
            q = self._heads(q, self.n_heads, seq)
            k = self._heads(k, self.n_kv_heads, seq)
            v = self._heads(v, self.n_kv_heads, seq)
            q = self._rope(q, cos, sin)
            k = self._rope(k, cos, sin)
            pad = kv_cache.capacity - seq
            ttnn.copy(ttnn.pad(k, ((0, 0), (0, 0), (0, pad), (0, 0)), 0.0) if pad else k, kv_cache.k)
            ttnn.copy(ttnn.pad(v, ((0, 0), (0, 0), (0, pad), (0, 0)), 0.0) if pad else v, kv_cache.v)
            k = ttnn.repeat_interleave(k, self.kv_groups, dim=1)
            v = ttnn.repeat_interleave(v, self.kv_groups, dim=1)
            scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1), compute_kernel_config=self.compute_cfg)
            scores = ttnn.add(ttnn.multiply(scores, self.scale), ctx.mask)
            probs = ttnn.softmax(scores, dim=-1, compute_kernel_config=self.compute_cfg)
            attn = ttnn.matmul(probs, v, compute_kernel_config=self.compute_cfg)
            attn = ttnn.permute(attn, (0, 2, 1, 3))
            attn = ttnn.reshape(attn, (b, seq, self.n_heads * self.head_dim))
        else:
            # One token per row: [B, 1, n*D] splits straight into the GQA layout [B, kv, groups, D]
            # (head h = g * groups + j), so the cache is attended per KV head with no repeat.
            q = self._rope(ttnn.reshape(q, (b, self.n_kv_heads, self.kv_groups, self.head_dim)), cos, sin)
            k = self._rope(ttnn.reshape(k, (b, self.n_kv_heads, 1, self.head_dim)), cos, sin)
            v = ttnn.reshape(v, (b, self.n_kv_heads, 1, self.head_dim))
            # cache <- cache + onehot(pos) * (new - cache): writes row b's slot pos[b] only.
            ttnn.copy(ttnn.add(kv_cache.k, ttnn.multiply(ctx.write, ttnn.subtract(k, kv_cache.k))), kv_cache.k)
            ttnn.copy(ttnn.add(kv_cache.v, ttnn.multiply(ctx.write, ttnn.subtract(v, kv_cache.v))), kv_cache.v)
            scores = ttnn.matmul(q, ttnn.transpose(kv_cache.k, -2, -1), compute_kernel_config=self.compute_cfg)
            scores = ttnn.add(ttnn.multiply(scores, self.scale), ctx.mask)
            probs = ttnn.softmax(scores, dim=-1, compute_kernel_config=self.compute_cfg)
            attn = ttnn.matmul(probs, kv_cache.v, compute_kernel_config=self.compute_cfg)
            attn = ttnn.reshape(attn, (b, seq, self.n_heads * self.head_dim))
        return ttnn.linear(attn, self.wo, compute_kernel_config=self.compute_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _rope(self, x, cos, sin):
        x_rot = ttnn.matmul(x, self.rot, compute_kernel_config=self.compute_cfg)
        return ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(x_rot, sin))

    def _past_kv(self, past_key_values):
        if past_key_values is None:
            return None
        layers = getattr(past_key_values, "layers", None)
        if layers is None or self.layer_idx >= len(layers):
            return None
        layer = layers[self.layer_idx]
        keys, values = getattr(layer, "keys", None), getattr(layer, "values", None)
        if keys is None or values is None or keys.numel() == 0:
            return None
        return self._to_tt(keys), self._to_tt(values), keys.shape[2]

    def __call__(
        self, hidden_states, position_embeddings=None, past_key_values=None, kv_cache=None, attn_ctx=None, **kwargs
    ):
        x = hidden_states
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (x.shape[0] * x.shape[1], x.shape[-2], x.shape[-1]))
        b, seq = x.shape[0], x.shape[-2]
        if kv_cache is not None:
            cos, sin = self._cos_sin(position_embeddings, b, seq)
            return self._cached(x, cos, sin, kv_cache, attn_ctx), None

        q = ttnn.linear(x, self.wq, compute_kernel_config=self.compute_cfg)
        k = ttnn.linear(x, self.wk, compute_kernel_config=self.compute_cfg)
        v = ttnn.linear(x, self.wv, compute_kernel_config=self.compute_cfg)
        q = self._heads(q, self.n_heads, seq)
        k = self._heads(k, self.n_kv_heads, seq)
        v = self._heads(v, self.n_kv_heads, seq)

        cos, sin = self._cos_sin(position_embeddings, b, seq)
        q = self._rope(q, cos, sin)
        k = self._rope(k, cos, sin)

        # Explicit GQA attention for both prefill and decode (SDPA's default chunking hung at S=64).
        past = self._past_kv(past_key_values)
        past_len = 0
        if past is not None:
            past_k, past_v, past_len = past
            k = ttnn.concat([past_k, k], dim=2)
            v = ttnn.concat([past_v, v], dim=2)
        k = ttnn.repeat_interleave(k, self.kv_groups, dim=1)
        v = ttnn.repeat_interleave(v, self.kv_groups, dim=1)
        scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1), compute_kernel_config=self.compute_cfg)
        scores = ttnn.multiply(scores, self.scale)
        if seq > 1:
            # Query i sits at absolute position past_len + i and sees keys 0..past_len + i.
            total = past_len + seq
            mask = ttnn.full((1, 1, seq, total), -1e9, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
            mask = ttnn.triu(mask, diagonal=past_len + 1)
            scores = ttnn.add(scores, mask)
            ttnn.deallocate(mask)
        probs = ttnn.softmax(scores, dim=-1, compute_kernel_config=self.compute_cfg)
        attn = ttnn.matmul(probs, v, compute_kernel_config=self.compute_cfg)
        for t in (q, k, v):
            ttnn.deallocate(t)

        attn = ttnn.permute(attn, (0, 2, 1, 3))
        attn = ttnn.reshape(attn, (b, seq, self.n_heads * self.head_dim))
        out = ttnn.linear(attn, self.wo, compute_kernel_config=self.compute_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn)
        return out, None

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)


# Module-level `build` — primary test entry point.
def build(device, torch_module=None):
    return TtAttention.build(device, torch_module)


# Module-level shim with the component's lowercase slug name.
def attention(device, torch_module=None):
    return TtAttention.build(device, torch_module)

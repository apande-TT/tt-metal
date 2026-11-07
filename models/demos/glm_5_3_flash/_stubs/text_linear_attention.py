# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn, head-parallel port of `Glm5NextTextLinearAttention` (GLM-5.3-Flash KDA).

The KDA implementation lives in the decoder-layer stub (`_KDA`), where it was validated inside the
full layer; this component wraps that same class so the two can never drift apart.

Math (HF): q/k/v projections -> depthwise causal conv(K=4) + SiLU -> l2-normalised q/k ->
chunked Kimi delta rule with per-channel decay g = lb * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))
and beta = sigmoid(b_proj(x)) -> gated RMSNorm (o_norm * sigmoid(g_b(g_a(x)))) -> o_proj.
See `_stubs/text_decoder_layer.py` for the exact device reformulation of the chunk math.

Tensor parallelism over N chips (num_heads/N whole heads per chip; the recurrence is never split
over time): q/k/v_proj, conv channels, f_b_proj, dt_bias, A_log, b_proj, g_b_proj are sharded by
head (column-parallel); f_a_proj, g_a_proj, o_norm are replicated; o_proj is row-parallel followed
by an all-reduce (all_gather + on-device sum).
"""
from __future__ import annotations

import ttnn
from models.demos.glm_5_3_flash._stubs.text_decoder_layer import _KDA, _Ctx


class TtTextLinearAttention:
    def __init__(self, ctx, m):
        self.ctx = ctx
        self.kda = _KDA(ctx, m)

    @classmethod
    def build(cls, device, torch_module):
        return cls(_Ctx(device), torch_module)

    def _mask(self, attention_mask, S):
        if attention_mask is None:
            return None
        # HF multiplies hidden_states by the 2-D padding mask; do it on device.
        return (
            self.ctx.f32(attention_mask.reshape(1, 1, S, 1))
            if not isinstance(attention_mask, ttnn.Tensor)
            else (
                ttnn.typecast(
                    ttnn.reshape(ttnn.to_layout(attention_mask, ttnn.TILE_LAYOUT), [1, 1, S, 1]), ttnn.float32
                )
            )
        )

    def prefill(self, h, B, S, valid=None, real_len=None):
        """Batched prefill: h [1,1,B*S,D] fp32 -> (out [1,1,B*S,D] fp32, resident recurrent + conv state)."""
        return self.kda.prefill(h, B, S, valid, real_len)

    def decode(self, h, state):
        """One token per sample: h [1,1,B,D] fp32 + resident state -> (out [1,1,B,D] fp32, new state)."""
        return self.kda.decode(h, state)

    def __call__(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        if cache_params is not None:
            raise NotImplementedError("KDA cache path not ported in this component")
        x = hidden_states
        in_shape = list(x.shape)
        B, S, D = in_shape
        if B != 1:
            raise NotImplementedError("batch > 1 not supported by this port")
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = ttnn.typecast(ttnn.reshape(x, [1, 1, S, D]), ttnn.float32)
        out = self.kda(x, S, self._mask(attention_mask, S))  # [1,1,S,D] replicated
        return ttnn.reshape(out, in_shape)


def build(device, torch_module=None):
    return TtTextLinearAttention.build(device, torch_module)


def text_linear_attention(device, torch_module=None):
    return TtTextLinearAttention.build(device, torch_module)

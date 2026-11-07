# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn, tensor-parallel port of `Glm5NextTextModel` (GLM-5.3-Flash language model).

Reference (from inputs_embeds):
    H = inputs_embeds.unsqueeze(2).expand(hc_mult)          # hc_mult residual streams
    for layer in layers[:num_hidden_layers]:
        H = layer(H)          # mHC( KDA | DSA attention ) then mHC( dense MLP | MoE )
    out = norm(hc_head(H))   # mean over streams, RMSNorm

Composed of the graduated ports (one implementation each, so they cannot drift):
  * every layer          <- _stubs/text_decoder_layer.py (itself built from the leaf ports)
  * hc_head              <- _stubs/text_hyper_head.py
  * final norm           <- _stubs/text_r_m_s_norm.py
Residual streams are [1, hc_mult, R, D] fp32 replicated (stream-major), R = batch * tokens. ``forward(x, step)``
serves both prefill (step.mode == "prefill", B x S tokens) and decode (one token per sample) with the
per-layer resident state (KDA recurrent + conv state, DSA KV cache) kept in ``step.state``.
"""
from __future__ import annotations

import ttnn
from models.demos.glm_5_3_flash._stubs.text_decoder_layer import Step, TtTextDecoderLayer, _Ctx, _rs
from models.demos.glm_5_3_flash._stubs.text_hyper_head import TtTextHyperHead
from models.demos.glm_5_3_flash._stubs.text_r_m_s_norm import TtTextRMSNorm


class TtTextModel:
    def __init__(self, device, layers, norm_module, hc_mult, progress=None):
        self.device = device
        self.ctx = _Ctx(device)
        self.hc = int(hc_mult)
        self.layers = list(layers)  # same-typed TtTextDecoderLayer elements (discoverable stack)
        self.hc_head = TtTextHyperHead.build(device)
        self.norm = TtTextRMSNorm.build(device, norm_module)
        self.progress = progress

    @classmethod
    def build(cls, device, torch_module):
        m = torch_module
        ctx = _Ctx(device)
        n_layers = min(len(m.layers), int(m.config.num_hidden_layers))
        layers = [TtTextDecoderLayer(ctx, m.layers[i]) for i in range(n_layers)]
        return cls(device, layers, m.norm, m.config.hc_mult)

    @classmethod
    def from_layers(cls, device, layer_iter, norm_module, hc_mult, progress=None):
        """Build one layer at a time from an iterator of HF decoder layers (each freed after its upload),
        so a model larger than host RAM never has to exist on the host at once."""
        ctx = _Ctx(device)
        layers = []
        for i, m in enumerate(layer_iter):
            layers.append(TtTextDecoderLayer(ctx, m))
            if progress:
                progress(f"built text layer {i} ({m.block_type}, {'moe' if layers[-1].is_moe else 'dense'})")
        return cls(device, layers, norm_module, hc_mult, progress=progress)

    def forward(self, x, step, layer_progress=False):
        """x [1,1,R,D] embeddings (any float dtype) -> normed hidden [1,1,R,D] fp32."""
        R, D = x.shape[-2], x.shape[-1]
        x = ttnn.typecast(x, ttnn.float32)
        X = ttnn.repeat(x, ttnn.Shape([1, self.hc, 1, 1]))  # [1,hc,R,D] stream-major residual streams
        for i, layer in enumerate(self.layers):
            X = layer.forward(X, step)
            if layer_progress and self.progress:
                ttnn.synchronize_device(self.device)
                self.progress(f"{step.mode} layer {i + 1}/{len(self.layers)} done")
        h = self.hc_head.streams(X)  # [1,1,R,D] (graduated text_hyper_head)
        h = self.norm(h)
        return ttnn.typecast(h, ttnn.float32)

    def __call__(self, inputs_embeds, attention_mask=None, past_key_values=None, **kwargs):
        if past_key_values is not None:
            raise NotImplementedError("use forward(x, step) for the cached path")
        if attention_mask is not None:
            raise NotImplementedError("padding masks: pass step.valid / real_len via forward()")
        x = inputs_embeds
        B, S, D = x.shape
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        h = self.forward(_rs(x, [1, 1, B * S, D]), Step("prefill", B, S))
        return ttnn.reshape(h, [B, S, D])


def build(device, torch_module=None):
    return TtTextModel.build(device, torch_module)


def text_model(device, torch_module=None):
    return TtTextModel.build(device, torch_module)

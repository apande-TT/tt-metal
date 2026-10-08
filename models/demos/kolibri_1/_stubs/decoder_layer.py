# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure-TTNN Kolibri-1 decoder layer (`decoder_layer`, `model.layers.0`), tensor/expert-parallel.

Reference (`Kolibri1DecoderLayer` in tests/pcc/_reference_loader.py) -- sandwich norms around both
sublayers:

    h = h + post_attn_norm(self_attn(input_layernorm(h), position_ids))
    h = h + post_ffn_norm(mlp(post_attention_layernorm(h)))

The attention is the graduated `attention` stub (heads split across the TP chips, o_proj all_reduce)
and the MoE is the shared expert-parallel block in `_kolibri_moe.py` (local experts per chip, one
all_reduce). Both hand back the full hidden state on every chip, so the four RMSNorms and the
residual adds run replicated with no further collectives.
"""
from __future__ import annotations

import ttnn
from models.demos.kolibri_1._stubs._kolibri_moe import TtKolibri1MoE
from models.demos.kolibri_1._stubs.attention import TtKolibri1Attention

_NORMS = ("input_layernorm", "post_attn_norm", "post_attention_layernorm", "post_ffn_norm")


class TtKolibri1DecoderLayer:
    def __init__(self, device, torch_module) -> None:
        self.device = device
        self.attn = TtKolibri1Attention(device, torch_module.self_attn)
        self.moe = TtKolibri1MoE(device, torch_module.mlp)
        self.eps = float(torch_module.input_layernorm.variance_epsilon)
        mapper = ttnn.ReplicateTensorToMesh(device) if isinstance(device, ttnn.MeshDevice) else None
        self.norm_w = {}
        for name in _NORMS:
            w = getattr(torch_module, name).weight.float()
            self.norm_w[name] = ttnn.from_torch(
                w.reshape(1, 1, -1, 32),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                mesh_mapper=mapper,
            )
        self.norm_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def _norm(self, x, name):
        return ttnn.rms_norm(x, epsilon=self.eps, weight=self.norm_w[name], compute_kernel_config=self.norm_cfg)

    def __call__(self, hidden_states, position_ids=None, attention_mask=None, past_key_value=None, **kwargs):
        # The residual stream stays in the caller's dtype: the full model keeps it fp32 across all 50
        # layers (bf16 rounding of h every layer compounds); the sublayers compute in bf16 regardless.
        h = hidden_states
        a = self.attn(self._norm(h, "input_layernorm"), position_ids, attention_mask, past_key_value)
        h = ttnn.add(h, self._norm(self._like(a, h), "post_attn_norm"))
        ttnn.deallocate(a)
        m = self.moe(self._norm(h, "post_attention_layernorm"))
        h = ttnn.add(h, self._norm(self._like(m, h), "post_ffn_norm"))
        ttnn.deallocate(m)
        return h

    @staticmethod
    def _like(x, ref):
        return x if x.dtype == ref.dtype else ttnn.typecast(x, ref.dtype)


# Module-level `build` — primary test entry point.
def build(device, torch_module=None):
    return TtKolibri1DecoderLayer.build(device, torch_module)


# Backward-compatible slug shim.
def decoder_layer(device, torch_module=None):
    return TtKolibri1DecoderLayer.build(device, torch_module)

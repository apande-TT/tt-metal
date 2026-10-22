# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native tensor-parallel TTNN port of `mlp` (PixtralMLP, `vision_tower.transformer.layers.0.feed_forward`)
of `mistralai/Devstral-Small-2-24B-Instruct-2512`.

    out = down(silu(gate(x)) * up(x))            (1024 -> 4096 -> 1024)

Same SwiGLU structure as Ministral3MLP, so it reuses that tensor-parallel port: gate/up column-parallel
on the intermediate dim, down row-parallel then all_reduce.
"""
from __future__ import annotations

import ttnn
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs import pixtral_attention
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs.ministral3_m_l_p import TtMinistral3MLP


class TtMLP(TtMinistral3MLP):
    # The vision residual stream is fp32 (see attention.py); its down projection lands there unrounded.
    OUT_DTYPE = ttnn.float32
    DECODE_WEIGHTS = False  # the vision tower never decodes
    GATHER_DTYPE = None  # the vision residual adds expect the fp32 all_reduce
    # Block-float8 fused gate/up weight (the many-row SwiGLU). Not bf4: a bf4 SwiGLU weight attenuated the hidden
    # state around its massive activations in another model, and here it would save only 7.6 us a layer.
    GATE_UP_DTYPE = ttnn.bfloat8_b
    # HiFi2, not LoFi (optimize run 2): LoFi took the image features' PCC (tests/e2e, vs the fp32 HF vision tower) from
    # 0.999906 to 0.999268 (norm ratio 1.0009), and the language model amplifies that input error ~60x on its image-
    # token rows (test_e2e_image_text_to_text's prefill hidden PCC 0.9926 -> 0.9904 under a 0.995 floor). HiFi2 reads
    # 0.999866 (ratio 1.00006); HiFi4 gate/up + HiFi4 down 0.999903 for twice the passes. A bf16 weight does not help:
    # at LoFi 0.998266 with a 0.8% norm deficit, at HiFi4 0.999922 with a 0.1% deficit.
    GATE_UP_FIDELITY = ttnn.MathFidelity.HiFi2
    PREFILL_REDUCE_DTYPE = None
    PREFILL_DOWN_FIDELITY = ttnn.MathFidelity.HiFi2
    DOWN_DTYPE = ttnn.bfloat16
    # fp32 down output into the L1-resident residual stream (8 x 256 x 1024 fp32: 76 KB a core)
    DOWN_OUT_L1_BYTES = 96 * 1024
    FUSE_RESIDUAL_DOWN = True  # the layer's ffn residual add folded into the down (see TtMinistral3MLP)

    def _tp_degree(self, device, mesh):
        return pixtral_attention.vision_tp(device)[0]

    def __init__(self, device, torch_module, in_scale=None) -> None:
        """in_scale: the ffn_norm gamma, folded into gate/up (see TtAttention.FOLD_NORM_GAMMA)."""
        act = type(torch_module.act_fn).__name__
        assert act in ("SiLU", "SiLUActivation"), f"expected silu, got {act}"
        super().__init__(device, torch_module, in_scale=in_scale)


def build(device, torch_module=None):
    return TtMLP.build(device, torch_module)


def mlp(device, torch_module=None):
    return TtMLP.build(device, torch_module)

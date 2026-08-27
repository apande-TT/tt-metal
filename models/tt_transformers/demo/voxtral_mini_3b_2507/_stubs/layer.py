# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN stub for VoxtralEncoderLayer (audio_tower.layers[i]).

Pre-norm transformer block: LayerNorm + Attention + Add + LayerNorm + FFN + Add.
"""
from __future__ import annotations

import ttnn

_HIFI4_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


def _as_bf16(t):
    """Narrow a weight to bfloat16 ON THE HOST, before it is ever uploaded.

    ``from_torch(t, dtype=ttnn.bfloat16, ...)`` does NOT convert a float32 `t` on the host: it
    uploads the fp32 bytes, converts the LAYOUT on device in fp32, and only then emits a device
    Typecast to bf16.  Every call site here hands over a `.float()` tensor, so the layout
    conversion was moving 4 bytes per element to produce a 2-byte tensor and paying for a whole
    extra device op to do it.  Handing over bf16 makes the requested dtype the dtype that arrives:
    the conversion moves half the bytes and the typecast has nothing left to do.

    The VALUES are the same either way -- the fp32 -> bf16 rounding happens regardless, on the host
    here instead of on the device one op later.
    """
    return t.bfloat16() if hasattr(t, "bfloat16") else t


def _to_device(t, device):
    """Upload the weight ALREADY TILED, so the device emits no layout-conversion op at all.

    Passing ``device=`` to from_torch is what puts the conversion on the device: the ROW_MAJOR
    bytes go up and a Tilize (or TilizeWithValPadding, for a bias that is not a whole tile) runs
    there.  Building the tensor with NO device argument tilizes on the host instead, and
    ``ttnn.to_device`` is then a plain DMA of bytes that are already in the layout the consumer
    wants -- the conversion does not move to a cheaper kernel, it stops existing.

    This is a WEIGHT path, so the host cost is paid once at build and never in a forward, whereas
    the device op it replaces was on the critical path of the measured region.  Values are
    untouched: the same host-side bf16 tensor, the same tiling, just assembled before the copy
    rather than after it.
    """
    t = _as_bf16(t)
    kw = {"dtype": ttnn.bfloat16, "layout": ttnn.TILE_LAYOUT}
    try:
        if isinstance(device, ttnn.MeshDevice):
            kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
    except (AttributeError, TypeError):
        pass
    # NO `device=` HERE, and that is the whole point. `ttnn.open_device()` returns a MeshDevice on
    # this build, so a `isinstance(device, MeshDevice)` branch that kept `device=` was the branch
    # ALWAYS taken -- the host-tilize path below it was dead code. The mapper does not need the
    # tensor placed to describe the replication, so it composes with a host build.
    return ttnn.to_device(ttnn.from_torch(t, **kw), device)


class TtVoxtralEncoderLayer:
    def __init__(self, device, torch_module):
        self.device = device
        attn = torch_module.self_attn
        self.num_heads = attn.num_heads
        self.head_dim = attn.head_dim
        self.scaling = attn.head_dim**-0.5

        self.q_weight = _to_device(attn.q_proj.weight.T.contiguous().float(), device)
        self.q_bias = _to_device(attn.q_proj.bias.unsqueeze(0).float(), device)
        self.k_weight = _to_device(attn.k_proj.weight.T.contiguous().float(), device)
        self.v_weight = _to_device(attn.v_proj.weight.T.contiguous().float(), device)
        self.v_bias = _to_device(attn.v_proj.bias.unsqueeze(0).float(), device)
        self.out_weight = _to_device(attn.out_proj.weight.T.contiguous().float(), device)
        self.out_bias = _to_device(attn.out_proj.bias.unsqueeze(0).float(), device)

        self.attn_ln_w = _to_device(torch_module.self_attn_layer_norm.weight.unsqueeze(0).unsqueeze(0).float(), device)
        self.attn_ln_b = _to_device(torch_module.self_attn_layer_norm.bias.unsqueeze(0).unsqueeze(0).float(), device)
        self.attn_ln_eps = torch_module.self_attn_layer_norm.eps

        self.fc1_weight = _to_device(torch_module.fc1.weight.T.contiguous().float(), device)
        self.fc1_bias = _to_device(torch_module.fc1.bias.unsqueeze(0).float(), device)
        self.fc2_weight = _to_device(torch_module.fc2.weight.T.contiguous().float(), device)
        self.fc2_bias = _to_device(torch_module.fc2.bias.unsqueeze(0).float(), device)

        self.ffn_ln_w = _to_device(torch_module.final_layer_norm.weight.unsqueeze(0).unsqueeze(0).float(), device)
        self.ffn_ln_b = _to_device(torch_module.final_layer_norm.bias.unsqueeze(0).unsqueeze(0).float(), device)
        self.ffn_ln_eps = torch_module.final_layer_norm.eps

    def __call__(self, x, **kwargs):
        B = x.shape[0]
        S = x.shape[1] if len(x.shape) == 3 else x.shape[-2]

        residual = x
        x = ttnn.layer_norm(
            x, weight=self.attn_ln_w, bias=self.attn_ln_b, epsilon=self.attn_ln_eps, compute_kernel_config=_HIFI4_CFG
        )

        q = ttnn.linear(x, self.q_weight, bias=self.q_bias, compute_kernel_config=_HIFI4_CFG)
        q = ttnn.multiply(q, self.scaling)
        k = ttnn.linear(x, self.k_weight, compute_kernel_config=_HIFI4_CFG)
        v = ttnn.linear(x, self.v_weight, bias=self.v_bias, compute_kernel_config=_HIFI4_CFG)

        q = ttnn.reshape(q, (B, S, self.num_heads, self.head_dim))
        q = ttnn.transpose(q, 1, 2)
        k = ttnn.reshape(k, (B, S, self.num_heads, self.head_dim))
        k = ttnn.transpose(k, 1, 2)
        v = ttnn.reshape(v, (B, S, self.num_heads, self.head_dim))
        v = ttnn.transpose(v, 1, 2)

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=False, scale=1.0, compute_kernel_config=_HIFI4_CFG
        )
        attn_out = ttnn.transformer.concatenate_heads(attn_out)
        attn_out = ttnn.linear(attn_out, self.out_weight, bias=self.out_bias, compute_kernel_config=_HIFI4_CFG)

        x = ttnn.add(residual, attn_out)

        residual = x
        x = ttnn.layer_norm(
            x, weight=self.ffn_ln_w, bias=self.ffn_ln_b, epsilon=self.ffn_ln_eps, compute_kernel_config=_HIFI4_CFG
        )
        x = ttnn.linear(x, self.fc1_weight, bias=self.fc1_bias, compute_kernel_config=_HIFI4_CFG)
        x = ttnn.gelu(x)
        x = ttnn.linear(x, self.fc2_weight, bias=self.fc2_bias, compute_kernel_config=_HIFI4_CFG)
        x = ttnn.add(residual, x)

        return x


def build(device, torch_module=None):
    return TtVoxtralEncoderLayer(device, torch_module)

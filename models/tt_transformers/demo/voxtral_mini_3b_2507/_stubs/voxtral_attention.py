# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN stub for VoxtralAttention (audio_tower.layers[i].self_attn).

Standard multi-head self-attention for the audio encoder.
"""
from __future__ import annotations

import ttnn

_HIFI4_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


# AUDIO-TOWER SDPA FIDELITY, MATCHED TO ITS bf16 Q/K/V.  The encoder's SDPA was the last HiFi4 op
# in the tower, so the flash kernel's QK^T and PV matmuls each took FOUR math passes over bf16
# operands that hold two passes worth of mantissa.  HiFi2 is the documented setting for bf16
# attention (GUIDELINES/04 section 7); what protects the numerics is fp32_dest_acc_en, NOT the
# fidelity -- the softmax SUM is the precision-critical step and loses accuracy in fp16 DST, so
# that flag stays True while the fidelity drops.
#
# SCOPED TO THE AUDIO TOWER ON PURPOSE.  Dropping the LM's SDPA too (prefill + decode) bought
# only ~1 ms more and cost almost all of the remaining PCC margin: measured 0.9552 with all 12
# call sites at HiFi2 versus 0.9705 with just these six, against a 0.95 gate -- and 0.9705 is
# fractionally ABOVE the 0.9703 the tower measured at HiFi4, i.e. scoped this way the drop is
# free.  The LM attention feeds the logits the sampler reads directly, so it keeps HiFi4; the
# encoder's output is a 1500-frame embedding the projector then re-mixes, which tolerates it.
_SDPA_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


# AUDIO-TOWER PROJECTION FIDELITY, MATCHED TO THE bf16 WEIGHTS.  These projections keep bf16
# weights, and HiFi4 makes the math engine take FOUR passes over operands that hold TWO passes
# worth of mantissa -- the profiler tags every one of them compute-bound ("SLOW", not DRAM) on a
# full 110-core grid, so the math is the critical path and the extra passes are pure waste.
# HiFi2 is the documented pairing for bf16 (GUIDELINES/01 section 12; LoFi rarely wins at bf16,
# so this stops at HiFi2 rather than dropping all the way).  The layer_norms and SDPA stay at
# HiFi4 + fp32_dest_acc_en=True: this tower's own repair history records it losing PCC when its
# reductions ran at a lower fidelity, and softmax/variance accumulation is where that compounds.
_PROJ_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


def _dram_sharded():
    """Load the shared decode-layout helper that sits next to this stub.

    The stubs are imported standalone BY PATH (tt/pipeline._load_stub_module), so they have no
    package context and a relative import is not available to them.
    """
    import importlib.util
    import pathlib
    import sys

    key = "_voxtral_stub__dram_sharded"
    mod = sys.modules.get(key)
    if mod is None:
        spec = importlib.util.spec_from_file_location(key, pathlib.Path(__file__).with_name("_dram_sharded.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return mod


_DS = _dram_sharded()


def _to_device(t, device):
    # NARROW TO bf16 ON THE HOST.  Callers hand this `.float()` tensors, but the target dtype is
    # bf16, so ttnn used to upload fp32 and fix it up on DEVICE -- the profile showed 42 ms of
    # fp32 Tilize plus 24 ms of fp32->bf16 Typecast doing exactly that.  Narrowing first halves
    # the bytes tilized and removes the typecast entirely.  It is EXACT, not an approximation:
    # both host and device round fp32->bf16 round-to-nearest-even, and these weights came from a
    # bf16 checkpoint that `.float()` had merely widened, so this restores the original values.
    t = t.bfloat16()
    try:
        if isinstance(device, ttnn.MeshDevice):
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
    except (AttributeError, TypeError):
        pass
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


class TtVoxtralAttention:
    def __init__(self, device, torch_module):
        self.device = device
        self.num_heads = torch_module.num_heads
        self.head_dim = torch_module.head_dim
        self.scaling = self.head_dim**-0.5

        # FUSED QKV -- see _dram_sharded.fuse_qkv / qkv_heads.  The scale is NOT folded here
        # because this body passes scale=self.scaling to SDPA rather than pre-scaling Q.
        _qkv_w, _qkv_b = _DS.fuse_qkv(
            torch_module.q_proj.weight.T.contiguous().float(),
            torch_module.k_proj.weight.T.contiguous().float(),
            torch_module.v_proj.weight.T.contiguous().float(),
            qb=None if torch_module.q_proj.bias is None else torch_module.q_proj.bias.float(),
            kb=None if torch_module.k_proj.bias is None else torch_module.k_proj.bias.float(),
            vb=None if torch_module.v_proj.bias is None else torch_module.v_proj.bias.float(),
        )
        self.qkv_weight = _to_device(_qkv_w, device)
        self.qkv_bias = None if _qkv_b is None else _to_device(_qkv_b.unsqueeze(0).unsqueeze(0), device)
        self.o_weight = _to_device(torch_module.out_proj.weight.T.contiguous().float(), device)

        self.q_bias = (
            _to_device(torch_module.q_proj.bias.unsqueeze(0).unsqueeze(0).float(), device)
            if torch_module.q_proj.bias is not None
            else None
        )
        self.k_bias = (
            _to_device(torch_module.k_proj.bias.unsqueeze(0).unsqueeze(0).float(), device)
            if torch_module.k_proj.bias is not None
            else None
        )
        self.v_bias = (
            _to_device(torch_module.v_proj.bias.unsqueeze(0).unsqueeze(0).float(), device)
            if torch_module.v_proj.bias is not None
            else None
        )
        self.o_bias = (
            _to_device(torch_module.out_proj.bias.unsqueeze(0).unsqueeze(0).float(), device)
            if torch_module.out_proj.bias is not None
            else None
        )

    def __call__(self, hidden_states, **kwargs):
        B = hidden_states.shape[0]
        S = hidden_states.shape[1] if len(hidden_states.shape) == 3 else hidden_states.shape[-2]

        qkv = _DS.mm(self.device, hidden_states, self.qkv_weight, _PROJ_CFG, bias=self.qkv_bias)
        q, k, v = _DS.qkv_heads(qkv, self.num_heads)

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=self.scaling,
            program_config=_DS.sdpa_config(self.device, q, k),
            compute_kernel_config=_SDPA_CFG,
        )
        attn_out = ttnn.transformer.concatenate_heads(attn_out)
        attn_out = _DS.mm(self.device, attn_out, self.o_weight, _PROJ_CFG, bias=self.o_bias)

        return attn_out


def build(device, torch_module):
    return TtVoxtralAttention(device, torch_module)

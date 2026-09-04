# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN stub for VoxtralAttention (audio encoder self-attention).

Maps to: audio_tower.layers[i].self_attn
Standard multi-head self-attention: Q/K/V projections -> SDPA -> output projection.
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
# LoFi, NOT HiFi2: THE OPERANDS ARE BLOCK-FLOAT NOW, NOT bf16.  The note above was written when
# this tower carried bf16 Q/K/V; the projections have since narrowed to bfloat8_b, and a bf8_b
# operand holds ONE pass worth of mantissa, so HiFi2's two passes are the same waste HiFi4's four
# were (GUIDELINES/01 section 12: bf8b matmul -> LoFi).  The measurement says the flash kernel is
# where it matters: 238.6 us/call for 11.6 GFLOP is 48 TFLOP/s, ~6% of this part's block-float
# peak, against 121 GB/s of traffic -- it is math/overhead bound, not byte bound, so the passes
# are the critical path.  fp32_dest_acc_en stays True; that, not the fidelity, is what protects
# the softmax sum.
_SDPA_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
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
# THE AUDIO TOWER'S PROJECTION WEIGHTS.  qkv/out are memory-bound on a full grid, so halving the
# stored width halves the bytes each launch must pull.  Biases stay bf16.
_PROJ_DTYPE = ttnn.bfloat8_b


# WEIGHTS ARE NOW bf8_b, SO THE PAIRING IS LoFi (GUIDELINES/01 section 12).
_PROJ_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
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


class TtVoxtralAttention:
    def __init__(self, device, torch_module):
        self.device = device
        self.num_heads = torch_module.num_heads
        self.head_dim = torch_module.head_dim
        self.embed_dim = torch_module.embed_dim
        self.scaling = torch_module.head_dim**-0.5

        # FUSED QKV -- see _dram_sharded.fuse_qkv / qkv_heads.  This body pre-scales Q and passes
        # scale=1.0 to SDPA, so the scale folds into the Q columns and the multiply disappears.
        _qkv_w, _qkv_b = _DS.fuse_qkv(
            torch_module.q_proj.weight.T.contiguous().float(),
            torch_module.k_proj.weight.T.contiguous().float(),
            torch_module.v_proj.weight.T.contiguous().float(),
            qb=torch_module.q_proj.bias.float(),
            kb=None,
            vb=torch_module.v_proj.bias.float(),
            scale=torch_module.head_dim**-0.5,
        )
        self.qkv_weight = ttnn.from_torch(_qkv_w, dtype=_PROJ_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
        self.qkv_bias = ttnn.from_torch(
            _qkv_b.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        self.out_weight = ttnn.from_torch(
            torch_module.out_proj.weight.T.contiguous().float(),
            dtype=_PROJ_DTYPE,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.out_bias = ttnn.from_torch(
            torch_module.out_proj.bias.unsqueeze(0).float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    def __call__(self, hidden_states, **kwargs):
        bsz = hidden_states.shape[0]
        seq_len = hidden_states.shape[1] if len(hidden_states.shape) == 3 else hidden_states.shape[-2]

        qkv = _DS.mm(self.device, hidden_states, self.qkv_weight, _PROJ_CFG, bias=self.qkv_bias)
        q, k, v = _DS.qkv_heads(qkv, self.num_heads)

        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=1.0,
            program_config=_DS.sdpa_config(self.device, q, k),
            compute_kernel_config=_SDPA_CFG,
        )

        attn_output = ttnn.transformer.concatenate_heads(attn_output)

        attn_output = _DS.mm(self.device, attn_output, self.out_weight, _PROJ_CFG, bias=self.out_bias)

        return attn_output


def build(device, torch_module=None):
    return TtVoxtralAttention(device, torch_module)

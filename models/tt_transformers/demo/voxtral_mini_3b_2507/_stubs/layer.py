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
# THE AUDIO TOWER'S PROJECTION WEIGHTS.  qkv/out/fc1/fc2 are the whole parameter mass of this
# tower and the profile tags every one of them memory-bound on a full grid, so halving the stored
# width halves the bytes each launch must pull.  The norms, the biases and SDPA's operands are NOT
# narrowed -- normalization statistics are where a block-float rounding compounds over depth.
_PROJ_DTYPE = ttnn.bfloat8_b


# WEIGHTS ARE NOW bf8_b, SO THE PAIRING IS LoFi.  8-bit operands through a HiFi2 kernel make the
# math engine take two passes over one pass worth of mantissa, which cancels the bandwidth saving
# the narrower weight just bought (GUIDELINES/01 section 12: LoFi is the documented pairing for
# block-float matmul operands).  The layer_norms and SDPA keep their own configs -- only the four
# projections narrowed.
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

# THE ENCODER'S RESIDUAL STREAM IS ITS LAST bf16 TENSOR, exactly as prefill's was.  The projections
# already consume bf8_b weights and hand back bf8_b, but ttnn.add returns the WIDER of its two
# inputs, so the running sum came back bf16 -- and ttnn.layer_norm has NO output-dtype argument
# (its output dtype MATCHES its input), so the norm could never narrow it back and qkv/fc1 were
# handed a bf16 in0 (visible in the capture as `LoFi BF16 x BFP8` beside `BFP8 x BFP8` on the two
# projections fed by the residual instead of by a norm).  Narrowing the accumulator moves the two
# adds, the two layer_norms and two of the four projections at once, on a 3.85 MB activation
# carried through 32 blocks.  bf8_b is the FLOOR (GUIDELINES/01 section 13 names normalization
# activations), and the increments already arrive at exactly this granularity, so this rounds a sum
# at a width it already had.  The norms keep HiFi4 + fp32_dest_acc_en=True: this tower's repair
# history is about the ACCUMULATOR precision inside the reduction, which is untouched here.
_ACT_DTYPE = ttnn.bfloat8_b


def _to_device(t, device, dtype=ttnn.bfloat16):
    # BLOCK-FLOAT TARGETS SKIP THE HOST NARROWING.  bf8_b/bf4_b derive their mantissa from a
    # per-block shared exponent, so rounding to bf16 first can change the packed result; only the
    # bf16 path below is a pure round-trip removal.
    if dtype != ttnn.bfloat16:
        try:
            if isinstance(device, ttnn.MeshDevice):
                return ttnn.from_torch(
                    t,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(device),
                )
        except (AttributeError, TypeError):
            pass
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
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


class TtVoxtralEncoderLayer:
    def __init__(self, device, torch_module):
        self.device = device
        attn = torch_module.self_attn
        self.num_heads = attn.num_heads
        self.head_dim = attn.head_dim
        self.scaling = attn.head_dim**-0.5

        # FUSED QKV.  One [1280, 3*1280] weight instead of three, so the projection is one launch
        # and one weight read, and -- more importantly -- the fused output is the exact layout
        # nlp_create_qkv_heads consumes, which replaces the three reshape+transpose pairs below.
        # The attention scaling is folded into the Q columns, so the runtime multiply disappears
        # and SDPA keeps scale=1.0.  k_proj has no bias in this model; fuse_qkv zero-fills it.
        _qkv_w, _qkv_b = _DS.fuse_qkv(
            attn.q_proj.weight.T.contiguous().float(),
            attn.k_proj.weight.T.contiguous().float(),
            attn.v_proj.weight.T.contiguous().float(),
            qb=attn.q_proj.bias.float(),
            kb=None,
            vb=attn.v_proj.bias.float(),
            scale=attn.head_dim**-0.5,
        )
        self.qkv_weight = _to_device(_qkv_w, device, _PROJ_DTYPE)
        self.qkv_bias = _to_device(_qkv_b.unsqueeze(0), device)
        self.out_weight = _to_device(attn.out_proj.weight.T.contiguous().float(), device, _PROJ_DTYPE)
        self.out_bias = _to_device(attn.out_proj.bias.unsqueeze(0).float(), device)

        self.attn_ln_w = _to_device(torch_module.self_attn_layer_norm.weight.unsqueeze(0).unsqueeze(0).float(), device)
        self.attn_ln_b = _to_device(torch_module.self_attn_layer_norm.bias.unsqueeze(0).unsqueeze(0).float(), device)
        self.attn_ln_eps = torch_module.self_attn_layer_norm.eps

        self.fc1_weight = _to_device(torch_module.fc1.weight.T.contiguous().float(), device, _PROJ_DTYPE)
        self.fc1_bias = _to_device(torch_module.fc1.bias.unsqueeze(0).float(), device)
        self.fc2_weight = _to_device(torch_module.fc2.weight.T.contiguous().float(), device, _PROJ_DTYPE)
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

        qkv = _DS.mm(self.device, x, self.qkv_weight, _PROJ_CFG, bias=self.qkv_bias)
        q, k, v = _DS.qkv_heads(qkv, self.num_heads)

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=1.0,
            program_config=_DS.sdpa_config(self.device, q, k),
            compute_kernel_config=_SDPA_CFG,
        )
        attn_out = ttnn.transformer.concatenate_heads(attn_out)
        attn_out = _DS.mm(self.device, attn_out, self.out_weight, _PROJ_CFG, bias=self.out_bias)

        x = ttnn.add(residual, attn_out, dtype=_ACT_DTYPE)

        residual = x
        x = ttnn.layer_norm(
            x, weight=self.ffn_ln_w, bias=self.ffn_ln_b, epsilon=self.ffn_ln_eps, compute_kernel_config=_HIFI4_CFG
        )
        x = _DS.mm(self.device, x, self.fc1_weight, _PROJ_CFG, bias=self.fc1_bias, activation="gelu")
        x = _DS.mm(self.device, x, self.fc2_weight, _PROJ_CFG, bias=self.fc2_bias)
        x = ttnn.add(residual, x, dtype=_ACT_DTYPE)

        return x


def build(device, torch_module=None):
    return TtVoxtralEncoderLayer(device, torch_module)

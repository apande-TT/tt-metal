# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN stub for VoxtralEncoder (audio_tower).

Full encoder: conv1 + conv2 + positional_embedding + 32 transformer layers + layer_norm.
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

# THE FRONT-END CONVS WERE THE LAST HiFi4 MATH IN THE ENCODE STACK.  ttnn.conv1d takes its compute
# config as `compute_config` and defaults to HiFi4 when it is None, which both calls left it -- so
# the mel front-end ran the math engine over FOUR passes of mantissa against bf16 weights and a
# bf16 activation, which hold TWO.  The profiler tags both convs HiFi4 at 160.5 us and 52.9 us per
# call, 3.63 ms of the stage.  HiFi2 is the documented pairing for bf16 (GUIDELINES/01 section 12);
# it stops there rather than at LoFi because these are the FIRST ops on the mel input, so their
# error is the one every later layer inherits.  fp32_dest_acc_en stays True for the same reason.
_CONV_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)

# THE BIAS AND THE GELU BELONG TO THE CONVOLUTION, NOT AFTER IT.  Both front-end convs were followed
# by a standalone `ttnn.add(x, bias)` and a standalone `ttnn.gelu(x)` over the conv's own output --
# [1, 1, 3000, 1280] bf16 is 7.7 MB, so that pair re-reads and re-writes ~31 MB per conv purely to
# apply a channel vector and a pointwise function the convolution kernel can do inside its own pack
# loop.  conv1d takes the bias as an operand (and hands back the PREPARED copy alongside the
# prepared weights, so it is prepared once like they are) and the activation through Conv2dConfig.
#
# THE GELU IS NOT FUSED, AND THAT IS MEASURED -- ONLY THE BIAS IS.  Conv2dConfig takes an
# activation, so folding the gelu in as well looked like the same lever; it is not.  Even in the
# APPROXIMATE (tanh) form -- the one that is supposed to make fusing worth doing -- it cost
# encode 16.94 -> 17.25 ms (+1.8%), measured 2026-09-05.  Same shape of result as this model's
# fused-matmul gelu: the activation runs inside the convolution's pack loop, on the cores the conv
# happens to hold, where a standalone unary is free to spread over the whole grid.  The bias is
# different in kind and does pay -- it is a channel vector the kernel applies as it packs, not a
# transcendental -- so it stays fused and the gelu stays a separate op.
#
# conv1 HOLDS 100 CORES AND STILL RUNS AT A THIRD OF ITS MATH PEAK -- IT IS SHORT OF OVERLAP, NOT
# OF CORES.  The capture prices this call at 40.7 us for 2.96 GFLOP, i.e. 72.7 TFLOP/s on a part
# whose bf16 HiFi2 ceiling is several times that, on a full-width 100-core shard.  conv2d streams
# the halo'd activation and the prepared weight through circular buffers this config sizes, and
# with ONE block in each the reader and the math engine take turns: the matrix engine idles while
# the next act block arrives, then the NoC idles while it multiplies.  Double-buffering both
# operand CBs lets the next act block and the next weight slice land during the current multiply.
# It is affordable HERE specifically: conv1's inner dimension is 128 channels x 3 taps = 384, and
# its prepared weight is 983 kB across the grid, so a second block of each is a small L1 tenant --
# on conv2 (1280 channels, 9.8 MB of kernel) the same flags would be a real capacity trade, which
# is why this sits on conv1's config and not on the shared one below.
_CONV2D_CFG = ttnn.Conv2dConfig(
    enable_act_double_buffer=True,
    enable_weights_double_buffer=True,
)

# conv2 IS THE LAST bf16 WEIGHT MASS IN THE ENCODE STACK.  Its kernel is 1280 x 1280 x 3 = 4.9 M
# parameters, 9.8 MB at bf16, and the profiler prices the call at 105.2 us for 1.5 GFLOP -- 46% of
# the bf16 FLOP peak but ~93 GB/s against the weight alone, i.e. the read is a real share of the
# time.  Every projection in the tower behind it already stores bf8_b (GUIDELINES/01 section 12),
# and the encoder is the part of this model that absorbs block-float rounding: its output is a
# 1500-frame embedding the projector re-mixes, and narrowing the tower's fused QKV the same way
# measured e2e PCC going UP rather than down.  conv1 is deliberately NOT narrowed -- it has 128
# input channels against conv2's 1280, so it is a twentieth of the bytes and all of the dynamic
# range of the raw mel.  The compute config stays HiFi2 either way; this changes what is stored,
# not how many passes the math engine takes.
_CONV2D_CFG_BF8_W = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat8_b)


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


class TtEncoderLayer:
    def __init__(self, device, torch_layer):
        self.device = device
        attn = torch_layer.self_attn
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

        self.attn_ln_w = _to_device(torch_layer.self_attn_layer_norm.weight.unsqueeze(0).unsqueeze(0).float(), device)
        self.attn_ln_b = _to_device(torch_layer.self_attn_layer_norm.bias.unsqueeze(0).unsqueeze(0).float(), device)
        self.attn_ln_eps = torch_layer.self_attn_layer_norm.eps

        self.fc1_weight = _to_device(torch_layer.fc1.weight.T.contiguous().float(), device, _PROJ_DTYPE)
        self.fc1_bias = _to_device(torch_layer.fc1.bias.unsqueeze(0).float(), device)
        self.fc2_weight = _to_device(torch_layer.fc2.weight.T.contiguous().float(), device, _PROJ_DTYPE)
        self.fc2_bias = _to_device(torch_layer.fc2.bias.unsqueeze(0).float(), device)

        self.ffn_ln_w = _to_device(torch_layer.final_layer_norm.weight.unsqueeze(0).unsqueeze(0).float(), device)
        self.ffn_ln_b = _to_device(torch_layer.final_layer_norm.bias.unsqueeze(0).unsqueeze(0).float(), device)
        self.ffn_ln_eps = torch_layer.final_layer_norm.eps

    def __call__(self, x):
        B = x.shape[0]
        S = x.shape[1] if len(x.shape) == 3 else x.shape[-2]

        residual = x
        x = ttnn.layer_norm(
            x,
            weight=self.attn_ln_w,
            bias=self.attn_ln_b,
            epsilon=self.attn_ln_eps,
            compute_kernel_config=_HIFI4_CFG,
            # PLACEMENT, NOT MATH -- this pass is bandwidth-bound on the full-width stream; see _DS.stream_config.
            memory_config=_DS.stream_config(x),
        )

        # KEEP THE CHAIN IN L1 -- the fused projection's only consumer is the head split, one op
        # later, and that op already asks for an L1 output of its own.  ffn_config rather than
        # stream_config because this tensor is 3x the stream width (q + k + v).
        qkv = _DS.mm(
            self.device,
            x,
            self.qkv_weight,
            _PROJ_CFG,
            bias=self.qkv_bias,
            memory_config=_DS.ffn_config(S, int(self.qkv_weight.shape[-1]), _ACT_DTYPE),
        )
        q, k, v = _DS.qkv_heads(qkv, self.num_heads)
        # RELEASE THE FUSED PROJECTION AS SOON AS THE SPLIT HAS IT.  Now that qkv lands in L1 it is
        # 6.1 MB, and Python would hold that binding through the whole attention AND the FFN -- L1
        # the SDPA and matmul circular buffers underneath it have to work around.  The split has
        # already produced its own q/k/v, so nothing reads this again.
        ttnn.deallocate(qkv)

        # FLASH WRITES WHERE THE CONCAT READS -- see _DS.attn_out_config.  q/k/v come out of the
        # head split in L1, but this call defaulted its output to DRAM, so the attention wrote
        # [b, nqh, s, hd] out through the DRAM controller, concatenate_heads read it back and
        # wrote the same bytes again, and o_proj read them a third time: three full passes over a
        # value whose only consumers are the two ops after it.  The LM's bodies have taken this
        # placement for several rounds; the tower could not reach it only because the helper had
        # is_causal baked in, and this encoder is bidirectional.
        attn_out = _DS.sdpa_prefill(
            q,
            k,
            v,
            scale=1.0,
            program_config=_DS.sdpa_config(self.device, q, k, wide_k=True),
            compute_kernel_config=_SDPA_CFG,
            causal=False,
        )
        attn_out = _DS.concat_heads(attn_out)
        # Same: the attention output goes straight into the residual add, which is already L1.
        attn_out = _DS.mm(
            self.device,
            attn_out,
            self.out_weight,
            _PROJ_CFG,
            bias=self.out_bias,
            memory_config=_DS.ffn_config(S, int(self.out_weight.shape[-1]), _ACT_DTYPE),
        )

        x = ttnn.add(residual, attn_out, dtype=_ACT_DTYPE, memory_config=_DS.stream_config(residual, _ACT_DTYPE))

        residual = x
        x = ttnn.layer_norm(
            x,
            weight=self.ffn_ln_w,
            bias=self.ffn_ln_b,
            epsilon=self.ffn_ln_eps,
            compute_kernel_config=_HIFI4_CFG,
            # PLACEMENT, NOT MATH -- this pass is bandwidth-bound on the full-width stream; see _DS.stream_config.
            memory_config=_DS.stream_config(x),
        )
        # HAND FF1 TO FF2 THROUGH L1 -- see _DS.ffn_config.  The intermediate has exactly one
        # consumer, one op later, so a DRAM round trip is a full write plus a full read for nothing.
        x = _DS.mm(
            self.device,
            x,
            self.fc1_weight,
            _PROJ_CFG,
            bias=self.fc1_bias,
            activation="gelu",
            memory_config=_DS.ffn_config(S, int(self.fc1_weight.shape[-1]), _ACT_DTYPE),
        )
        # FF2 WRITES WHERE THE RESIDUAL ADD READS.  Every other projection in this block already
        # names its consumer's placement -- out_proj hands the attention residual an L1 tensor and
        # fc1 hands fc2 one -- and fc2 was the single call still defaulting to DRAM.  Its only
        # consumer is the add on the very next line, which asks for the L1 stream, so an
        # interleaved-DRAM output costs a full write plus a full read of a 2.05 MB value nothing
        # else reads, once per layer across the tower's 32 blocks.  Same helper and same width
        # argument as out_proj above, because it is the same [rows, 1280] layer stream.
        x = _DS.mm(
            self.device,
            x,
            self.fc2_weight,
            _PROJ_CFG,
            bias=self.fc2_bias,
            memory_config=_DS.ffn_config(S, int(self.fc2_weight.shape[-1]), _ACT_DTYPE),
        )
        x = ttnn.add(residual, x, dtype=_ACT_DTYPE, memory_config=_DS.stream_config(residual, _ACT_DTYPE))

        return x


class TtVoxtralEncoder:
    def __init__(self, device, torch_module):
        self.device = device
        self._prepared_w = {}
        self.max_source_positions = torch_module.config.max_source_positions

        self.conv1_weight = ttnn.from_torch(torch_module.conv1.weight.data.float(), dtype=ttnn.bfloat16)
        self.conv1_bias_tt = (
            _to_device(torch_module.conv1.bias.data.reshape(1, 1, 1, -1).float(), device)
            if torch_module.conv1.bias is not None
            else None
        )
        self.conv1_in_ch = torch_module.conv1.in_channels
        self.conv1_out_ch = torch_module.conv1.out_channels
        self.conv1_ks = torch_module.conv1.kernel_size[0]
        self.conv1_stride = torch_module.conv1.stride[0]
        self.conv1_padding = torch_module.conv1.padding[0]

        self.conv2_weight = ttnn.from_torch(torch_module.conv2.weight.data.float(), dtype=ttnn.bfloat16)
        # HOST, ROW_MAJOR, LIKE THE WEIGHT BESIDE IT -- required by the narrowing path, see
        # _CONV2D_CFG_BF8_W.  A weights_dtype that differs from the bias's own sends conv2d through
        # prepare_conv_bias_internal, which asserts `bias_tensor.layout() == Layout::ROW_MAJOR`
        # ("Host conv bias layout should be in row_major layout"); the bf16 path never reached that
        # check, so a device-resident TILE bias had been fine until now. conv1 keeps its tiled
        # device bias because its weights are not narrowed. Both are prepared once and cached on
        # device by the first call, so nothing here is uploaded inside the trace region.
        self.conv2_bias_tt = (
            ttnn.from_torch(
                torch_module.conv2.bias.data.reshape(1, 1, 1, -1).bfloat16(),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            if torch_module.conv2.bias is not None
            else None
        )
        self.conv2_in_ch = torch_module.conv2.in_channels
        self.conv2_out_ch = torch_module.conv2.out_channels
        self.conv2_ks = torch_module.conv2.kernel_size[0]
        self.conv2_stride = torch_module.conv2.stride[0]
        self.conv2_padding = torch_module.conv2.padding[0]

        self.embed_positions = _to_device(torch_module.embed_positions.weight.unsqueeze(0).float(), device)

        self.layers = [TtEncoderLayer(device, layer) for layer in torch_module.layers]

        self.ln_weight = _to_device(torch_module.layer_norm.weight.unsqueeze(0).unsqueeze(0).float(), device)
        self.ln_bias = _to_device(torch_module.layer_norm.bias.unsqueeze(0).unsqueeze(0).float(), device)
        self.ln_eps = torch_module.layer_norm.eps

    def _conv1d_cached(self, x, idx, weight, bias, in_ch, out_ch, ks, stride, pad, length, conv_config=None):
        """conv1d with the PREPROCESSED weights AND BIAS cached on device, gelu fused.

        The graduated body kept the conv weights on host and let every call
        upload/prepare them.  That host transfer is illegal inside
        ttnn.begin_trace_capture (TT_FATAL !trace_id_.has_value()), so the encode
        stage could not be traced.  Preparing once and reusing the device-resident
        weights is also strictly faster; numerics are unchanged.

        The bias rides the same cache: conv1d returns the prepared weight AND the prepared bias
        from the first call, so handing it the bias costs one preparation, not one per call, and
        removes the full-width `add` that used to follow.  See _CONV2D_CFG for the fused gelu.
        """
        prepared = self._prepared_w.get(idx)
        res = ttnn.conv1d(
            input_tensor=x,
            weight_tensor=prepared[0] if prepared is not None else weight,
            bias_tensor=prepared[1] if prepared is not None else bias,
            device=self.device,
            in_channels=in_ch,
            out_channels=out_ch,
            batch_size=1,
            input_length=length,
            kernel_size=ks,
            stride=stride,
            padding=pad,
            dilation=1,
            groups=1,
            conv_config=_CONV2D_CFG if conv_config is None else conv_config,
            compute_config=_CONV_CFG,
            return_weights_and_bias=prepared is None,
        )
        if prepared is None:
            out = res[0]
            wb = res[-1]
            self._prepared_w[idx] = tuple(wb) if isinstance(wb, (tuple, list)) else (wb, None)
        else:
            out = res[0] if isinstance(res, tuple) else res
        return out

    def __call__(self, input_features, **kwargs):
        # input_features: ttnn tensor (1, 128, 3000) TILE_LAYOUT on device
        # conv1d expects (N, input_length, 1, C) format
        #
        # THE CALLER CAN HAND US [N, L, C] AND THEN NEITHER OF THE TWO PREP OPS HAS TO RUN.  The mel
        # frame is UPLOADED -- it is the one tensor in this stage that comes from the host, and the
        # host can transpose it for free -- so producing [N, C, L] TILE and then untilizing and
        # transposing it on device is work the model chose to do to itself.  Both ops are pure
        # layout: profiled at 0.288 ms (untilize_with_unpadding, because L=3000 is not tile-aligned)
        # plus 0.338 ms (the permute) of roofline gap per capture, on a 768 kB tensor, and they run
        # once per stream.  Detect the shape the convolution actually wants and skip straight to the
        # placement.  A caller that still passes [N, C, L] -- the PCC harness builds the HF argument
        # shape and knows nothing about this -- keeps the original path, so the contract is widened
        # rather than moved.
        if int(input_features.shape[-1]) == self.conv1_in_ch:
            x = ttnn.to_memory_config(input_features, ttnn.L1_MEMORY_CONFIG)
            return self._forward_from_nlc(x)
        x = ttnn.to_layout(input_features, ttnn.ROW_MAJOR_LAYOUT)
        # LANDING THE MEL FRAME IN L1 IS WHAT PICKS conv2d's EXECUTION PATH.
        # ttnn::determine_conv2d_execution_path routes on ONE predicate --
        # `input_tensor.memory_config().is_l1()` -- and with no slice_config a DRAM input takes the
        # DRAM-slicing path, which is defined to write its result back to DRAM interleaved
        # ("Conv2D DRAM doesn't support specifying memory config, as the output will always be DRAM
        # Interleaved").  That is why the front-end paid for a round trip PER CONV: reshard in,
        # convolve, shard OUT to DRAM, run the gelu against DRAM, reshard back in for the next conv.
        # This tensor is [1, 3000, 128] bf16 = 768 kB and L1 holds 1.5 MB per core, so there is no
        # reason for it to be in DRAM at all.  Naming L1 here flips BOTH convs onto the L1 path in
        # one edit -- conv1 because this is its input, conv2 because conv1's output now stays in L1
        # -- so the two ShardedToInterleaved copies and the InterleavedToSharded between them
        # disappear and both gelus run against L1-resident shards instead of DRAM.
        # It is a placement change: the convolution reads the same values in the same order.
        x = ttnn.permute(x, (0, 2, 1), memory_config=ttnn.L1_MEMORY_CONFIG)  # (1, 3000, 128)
        return self._forward_from_nlc(x)

    def _forward_from_nlc(self, x):
        """The tower proper, from an L1-resident [N, L, C] mel frame onwards.

        Split out so the two ways of arriving at that frame -- uploaded in this layout, or
        untilized and transposed on device from [N, C, L] -- share one body.
        """
        # THE CANONICAL conv1d INPUT SHAPE IS [N, 1, L, C], NOT [N, L, 1, C].  ttnn.conv1d only
        # reshapes for you when the input is rank < 4; hand it a rank-4 tensor and it forwards the
        # shape straight to conv2d, which takes H/W from its explicit input_height=1 /
        # input_width=input_length arguments and treats the tensor as a flat NHWC buffer.  Both
        # shapes hold the same elements in the same row-major order, so the numerics are identical
        # -- but they are NOT the same physical layout: with the length in dim 1, the trailing dims
        # are (1, C), so in TILE layout every one-row slice pads out to a full 32-row tile and
        # conv2d then has to re-flatten it back to [1, 1, L, C].  Measured on the conv1 output,
        # that pair of re-tilizations cost 1.52 ms + 0.82 ms per encode on a 7.7 MB activation
        # (~10 GB/s) -- more than the convolution itself.  Naming the shape conv2d already wants
        # makes both reshapes leading-dim regroupings, i.e. views.
        x = ttnn.reshape(x, (1, 1, 3000, 128))  # (N, 1, L, C)

        # conv1: (1, 1, 3000, 128) -> (1, 1, 3000, 1280)
        x = self._conv1d_cached(
            x,
            1,
            self.conv1_weight,
            self.conv1_bias_tt,
            self.conv1_in_ch,
            self.conv1_out_ch,
            self.conv1_ks,
            self.conv1_stride,
            self.conv1_padding,
            3000,
        )
        # the bias is fused into the conv above; the gelu is not (see _CONV2D_CFG).
        # THE TANH VARIANT, LIKE EVERY OTHER GELU IN THIS MODEL.  A bare `ttnn.gelu(x)` is the
        # EXACT erf form -- the capture records it as `UnaryOpType::GELU;param={0}` -- which the
        # SFPU evaluates as a full erf polynomial per element.  On this tensor that is 76.0 us for
        # 3000 x 1280 elements spread over 100 cores, nearly twice conv1's own 41.7 us, on an op
        # that only has to touch each element once.  `_dram_sharded._apply` already routes every
        # OTHER gelu in the tower through ttnn.GeluVariant.Tanh for exactly this reason; these two
        # front-end calls were the ones it does not pass through.  The tanh form is the standard
        # gelu approximation (max abs error ~1e-3 against erf), and this feeds a convolution whose
        # own operands are bf16, which resolves ~4e-3.
        x = ttnn.gelu(x, variant=ttnn.GeluVariant.Tanh)

        # conv2: stride=2, so 3000 -> 1500
        x = ttnn.reshape(x, (1, 1, 3000, 1280))  # see the shape note above: [N, 1, L, C]
        x = self._conv1d_cached(
            x,
            2,
            self.conv2_weight,
            self.conv2_bias_tt,
            self.conv2_in_ch,
            self.conv2_out_ch,
            self.conv2_ks,
            self.conv2_stride,
            self.conv2_padding,
            3000,
            conv_config=_CONV2D_CFG_BF8_W,  # see _CONV2D_CFG_BF8_W: 9.8 MB of bf16 kernel, halved
        )
        # the bias is fused into the conv above; the gelu is not (see _CONV2D_CFG).
        # Tanh variant, for the reason given on conv1's gelu above.
        x = ttnn.gelu(x, variant=ttnn.GeluVariant.Tanh)

        # HAND THE STACK BACK ITS DRAM TENSOR.  Everything above now runs L1-resident and sharded,
        # but the positional-embedding add and the 32 encoder layers below are written against a
        # DRAM interleaved activation, so the shard is resolved once here rather than op by op.
        # This is the ONE copy the L1 path still pays, in place of the three it removes.
        x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        # Reshape to (1, 1500, 1280)
        x = ttnn.reshape(x, (1, 1500, 1280))
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # Add positional embedding
        x = ttnn.add(x, self.embed_positions)

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Final layer norm
        x = ttnn.layer_norm(
            x, weight=self.ln_weight, bias=self.ln_bias, epsilon=self.ln_eps, compute_kernel_config=_HIFI4_CFG
        )

        return x


def build(device, torch_module=None):
    return TtVoxtralEncoder(device, torch_module)

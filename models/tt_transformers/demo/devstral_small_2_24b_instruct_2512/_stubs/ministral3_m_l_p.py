# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native tensor-parallel TTNN port of `ministral3_m_l_p` (Ministral3MLP, `language_model.layers.0.mlp`)
of `mistralai/Devstral-Small-2-24B-Instruct-2512`.

    out = down(silu(gate(x)) * up(x))            (5120 -> 32768 -> 5120)

Tensor-parallel over the mesh (TP = number of devices; TP = 1 on a single device):
  * gate/up    column-parallel on the intermediate dim (the SwiGLU product stays per-device), folded into
               ONE weight per device whose column tiles alternate gate/up, so a single minimal_matmul with
               fuse_swiglu emits silu(gate) * up directly (no separate gate/up outputs, slices, silu or mul).
               That op splits M over one grid axis, so at decode's single tile row it would idle most of the
               grid: decode (mode="decode") keeps separate gate / up weights on ttnn.linear instead.
  * down_proj  row-parallel on the intermediate dim, then all_reduce.
"""

from __future__ import annotations

import torch

import ttnn
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs.ministral3_attention import (
    fold_input_scale,
    tp_ccl_args,
)

# Prefill token rows (batch rows x sequence) one MLP chunk processes at a time. With gate/up fused into one
# SwiGLU op the only intermediate is the (rows, intermediate/TP) product: 268 MB/chip for a full 32 x 512
# prefill, so that whole batch runs as ONE call (one down matmul, one all_reduce, no slices / concat).
PREFILL_ROWS_PER_CHUNK = 16384
# dtype of the decode-only gate / up / down weight copies.
DECODE_WEIGHT_DTYPE = ttnn.bfloat8_b
# Decoder layers whose decode-only gate / up copies are block-float4 (a quarter of the bf16 bytes the single-token
# step streams); the other layers keep DECODE_WEIGHT_DTYPE.
DECODE_GATE_UP_BF4_DTYPE = ttnn.bfloat4_b
DECODE_GATE_UP_BF4_LAYERS = frozenset(range(5, 36))
# Layers whose many-row (prefill) fused gate/up computes at LoFi instead of GATE_UP_FIDELITY. None: LoFi on layers
# 14-25 (optimize run 6, ~47 ms of trace prefill) tripled those layers' per-layer error (teacher-forced 1-PCC 1.1e-4 ->
# 3e-4), and the image prompt's 64 image-token rows amplify every layer's error ~10x more than text rows do, so
# test_e2e_image_text_to_text's prefill hidden PCC (floor 0.995) fell to 0.9922 with it. With HiFi2 gate/up on every
# layer, the HiFi4 down and prefill SDPA and the HiFi2 vision gate/up it reads 0.995511 (text rows 0.9985, image rows
# 0.9742); the LoFi layers kept on top of those 0.994102. Earlier text-only readings: LoFi on every layer 0.99334;
# 18-22 0.995137; 16-24 0.995147; 8-31 0.99509; 12-27 0.995806 fail test_e2e_text_generation's token agreement.
PREFILL_GATE_UP_LOFI_LAYERS = frozenset()
# The text layers' many-row (prefill) down projection at LoFi (optimize run 7: trace prefill 1028 -> 937 ms, text PCC
# 0.995097 -> 0.995002) halved the per-layer error budget the image-token rows cannot afford (see above): off, the down
# computes at PREFILL_DOWN_FIDELITY (HiFi4). The vision MLPs (subclasses) keep their own PREFILL_DOWN_FIDELITY.
PREFILL_DOWN_LOFI = False
# K tiles per activation multicast block of the bf4 decode gate / up projections (divides K = 160 tiles).
DECODE_GATE_UP_K_BLOCK = 8
# Math fidelity of the bf4 decode gate / up and down (HiFi3 == HiFi4 bit for bit on a block-float4 weight, see __init__).
# HiFi2: the bf4 gate / up program is compute-bound (~100 us of matmul after its prologue vs ~95 us of weight stream),
# so the pass fewer is decode time (trace 15.09 -> 14.72 ms); not bit-exact, text logits PCC 0.995037 -> 0.995223.
DECODE_GATE_UP_BF4_FIDELITY = ttnn.MathFidelity.HiFi2
# K tiles per activation multicast block of the decode down projection (divides K = 256 tiles).
DOWN_DECODE_K_BLOCK = 8
# Decode down projection as one C++ program streaming a bank-major weight copy (tt/down_decode.py).
DOWN_DECODE_STREAM = True
# Decoder layers whose decode-only down copy is block-float4 (the others read the bf8 prefill weight). 16-26 (layer 26
# added once the bf4 HiFi2 matmuls raised the gate PCC 0.995037 -> 0.995223: 0.995097 at 16-26, 0.994848 at 16-28).
# Before: 16-25 (layer 25
# added once the HiFi4 lm_head raised the gate PCC 0.995070 -> 0.995097: 0.995078 at 16-25). 16-24: layer
# 15 went back to bf8 (gate PCC 0.995034 at 15-24, 0.995374 at 16-23), headroom the fixed-chunk decode SDPA
# (ministral3_attention DECODE_SDPA_K_CHUNK, -0.22 ms for -0.00018 PCC) spends.
DECODE_DOWN_BF4_LAYERS = frozenset(range(16, 27))
# The decode down program folds its all_reduce and the residual add in (tt/down_decode.py StreamDownDecode.ar_call):
# its fabric-link workers send the partial shards of each core's first column while the second columns still stream.
DECODE_DOWN_AR_FOLD = True
# Many-row (prefill / vision) fused gate/up through tt/minimal_mm.py: minimal_matmul's own program with its
# output stage regrouped -- silu * up as ONE SFPU pass (the SFPU is ~1.4 ms of the 416-row gate/up's 9.8), bit-identical.
PREFILL_MM_PROGRAM = True
# Decode gate / up / silu-multiply as one C++ program (tt/gate_up_decode.py).
DECODE_GATE_UP_FUSED = True
# ... with the post-attention RMSNorm folded into that program's prologue (tt/gate_up_decode.py NormGateUpDecode): it
# takes the decode residual's own width shards, so its weight streams start while the row is normed.
DECODE_GATE_UP_NORM = True


class TtMinistral3MLP:
    # dtype of the down projection's output and its all_reduce: fp32, added unrounded into the fp32 residual
    # stream of the text decoder (and of the vision layers, whose MLPs subclass this one).
    OUT_DTYPE = ttnn.float32
    # Separate gate / up weights for the single-token decode path (the vision MLP subclasses never decode).
    DECODE_WEIGHTS = True
    fused_gate_up = None
    # dtype the reduced output is all_gathered in (None: plain fp32 all_reduce). The text decoder reduce-scatters
    # in fp32 and gathers the reduced shard in bf16 (half the bytes); its residual add takes the bf16 unrounded.
    GATHER_DTYPE = ttnn.bfloat16
    # Set by the decoder layer: the decode all_gather writes straight into its residual add's width shards.
    decode_gather_mem = None
    # Set by the decoder layer: decode's down all_reduce as one collective (tt/decode_all_reduce.py).
    decode_all_reduce = None
    # dtype of the many-row (prefill) down projection's partials and their reduce-scatter (None: OUT_DTYPE). The text
    # decoder reduce-scatters prefill's partials in bf16 (half the bytes, no cast before the gather).
    PREFILL_REDUCE_DTYPE = ttnn.bfloat16
    # dtype of the single-row (decode) down projection's partials and their reduce-scatter (None: OUT_DTYPE).
    DECODE_REDUCE_DTYPE = ttnn.bfloat16
    # Math fidelity of the many-row (prefill) down projection (None: the HiFi4 compute_cfg decode uses too). HiFi4:
    # with the HiFi2 gate/up, the HiFi4 prefill SDPA and the HiFi2 vision gate/up, test_e2e_image_text_to_text's
    # prefill hidden PCC reads 0.995511 (floor 0.995); HiFi2 here 0.994830, HiFi3 0.994959 (not bit-exact with HiFi4
    # on this block-float8 weight), LoFi (run 7) 0.9904. The vision MLPs (subclasses) override it.
    PREFILL_DOWN_FIDELITY = ttnn.MathFidelity.HiFi4
    # dtype of the down weight the many-row (prefill) path multiplies with.
    DOWN_DTYPE = ttnn.bfloat8_b
    # False: prefill returns the down projection's reduce-scattered 1/TP hidden shard (see the decoder layer).
    prefill_gather = True
    # dtype of the fused gate/up weight the many-row (prefill) path multiplies with.
    GATE_UP_DTYPE = ttnn.bfloat8_b
    # Math fidelity of the many-row fused gate/up.
    GATE_UP_FIDELITY = ttnn.MathFidelity.HiFi2
    # The many-row SwiGLU product goes to the down projection through L1 (interleaved, freed right after) while it
    # takes at most this many bytes per core (the vision MLP's 8192 x 1024 bf16: 152 KB; the text MLP's is far larger).
    PREFILL_INTER_L1_BYTES = 192 * 1024
    # The many-row down projection's output lands in L1 while it takes at most this many bytes per core (0: DRAM). The
    # vision MLPs set it: their fp32 output is added into the L1-resident vision residual stream (see attention.py).
    DOWN_OUT_L1_BYTES = 0
    # The many-row down projection folds a fp32 residual add into its output stage (tt/minimal_mm.py residual=,
    # bit-identical to ttnn.add(residual, down)): __call__(x, residual=h) returns h + mlp(x). The vision MLPs set it.
    FUSE_RESIDUAL_DOWN = False
    # A short prompt's (<= 128 tile rows) down projection runs on the full grid with blocks cut to each core's exact
    # M / N ranges (see _forward) instead of the 80-core grid of whole 16 x 16-tile blocks.
    SHORT_DOWN_FULL_GRID = True
    # The TP text MLP's short-prompt (<= 128 tile rows) down as tt/minimal_mm.matmul(..., swapped=True): the weight in
    # srcB (a bf16 copy of the bf8 weight, transposed once at load: exact), so HiFi2 multiplies the full operands.
    SHORT_DOWN_SWAPPED = True
    # ... and the long prompt's (> 128 tile rows) down too, in N blocks of LONG_DOWN_SWAPPED_N_BLOCK prompt tile rows
    LONG_DOWN_SWAPPED = True
    LONG_DOWN_SWAPPED_N_BLOCK = 10
    # ... on the last thirty text layers only: the image prompt's [IMG] rows amplify a rounding-order change through the
    # layers after it (image prefill-hidden PCC: layers 0-39 swapped 0.9944 < 0.995; 10-39 0.99525; 20-39 / 30-39 0.995503)
    LONG_DOWN_SWAPPED_LAYERS = frozenset(range(10, 40))

    def __init__(self, device, torch_module, in_scale=None, layer_idx=None, norm_eps=None) -> None:
        """in_scale: optional per-input-feature scale (the post-attention RMSNorm's gamma) folded into gate/up.
        layer_idx: the decoder layer's index (selects the decode gate / up precision, DECODE_GATE_UP_BF4_LAYERS).
        norm_eps: the eps of the unit-gamma RMSNorm feeding gate / up (its gamma is in_scale), when the decode gate / up
        may norm the residual itself (DECODE_GATE_UP_NORM)."""
        self.device = device
        try:
            mesh = int(device.get_num_devices())
        except Exception:
            mesh = 1
        self.tp = tp = self._tp_degree(device, mesh)
        assert (
            torch_module.intermediate_size % tp == 0
        ), f"intermediate {torch_module.intermediate_size} not divisible by TP={tp}"
        repl = ttnn.ReplicateTensorToMesh(device) if mesh > 1 else None
        shard = lambda dim: ttnn.ShardTensorToMesh(device, dim=dim) if tp > 1 else repl

        def _put(t, mapper, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG):
            return ttnn.from_torch(
                t.detach().to(torch.bfloat16).contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=mapper,
                memory_config=memory_config,
            )

        # Device d's slice of the folded weight holds its `inter` gate and `inter` up columns as alternating
        # 32-wide tiles [g_0, u_0, g_1, u_1, ...]: the layout minimal_matmul(fuse_swiglu=True) pairs up.
        self.inter = inter = torch_module.intermediate_size // tp
        w_gate = fold_input_scale(torch_module.gate_proj.weight.T, in_scale)
        w_up = fold_input_scale(torch_module.up_proj.weight.T, in_scale)
        k = w_gate.shape[0]
        chunks = []
        for d in range(tp):
            g = w_gate[:, d * inter : (d + 1) * inter].reshape(k, inter // 32, 1, 32)
            u = w_up[:, d * inter : (d + 1) * inter].reshape(k, inter // 32, 1, 32)
            chunks.append(torch.concat([g, u], dim=2).reshape(k, 2 * inter))
        self.w_gate_up = _put(torch.concat(chunks, dim=-1), shard(-1), self.GATE_UP_DTYPE)
        del chunks
        if self.DECODE_WEIGHTS:
            # Decode streams every MLP weight once per token (DRAM-bound): it reads its own block-float8 copies, half
            # the bytes of the bf16 weights prefill multiplies with; the layers in DECODE_GATE_UP_BF4_LAYERS read
            # block-float4 gate / up copies (the whole stack in bf4 costs more accuracy than the gate allows).
            lowp = layer_idx is not None and layer_idx in DECODE_GATE_UP_BF4_LAYERS
            gu_dtype = DECODE_GATE_UP_BF4_DTYPE if lowp else DECODE_WEIGHT_DTYPE
            # DECODE_GATE_UP_FUSED: gate, up and silu * up as one C++ program; the activation multicast once and both
            # data-movement RISCs streaming the weights independently of it, from bank-major copies the program owns
            # (tt/gate_up_decode.py). Otherwise two ttnn.linear calls on interleaved copies and the silu-multiply.
            if DECODE_GATE_UP_FUSED:
                from models.demos.devstral_small_2_24b_instruct_2512.tt.gate_up_decode import (
                    FusedGateUpDecode,
                    NormGateUpDecode,
                )

                fid = DECODE_GATE_UP_BF4_FIDELITY if lowp else ttnn.MathFidelity.HiFi4
                put = lambda t, dtype: _put(t, shard(-1), dtype)
                if (
                    DECODE_GATE_UP_NORM
                    and norm_eps is not None
                    and NormGateUpDecode.supports_norm(device, k, inter, gu_dtype)
                ):
                    self.fused_gate_up = NormGateUpDecode(device, w_gate, w_up, gu_dtype, fid, put, tp, norm_eps, k)
                elif FusedGateUpDecode.supports(device, k, inter, gu_dtype):
                    self.fused_gate_up = FusedGateUpDecode(device, w_gate, w_up, gu_dtype, fid, put, tp)
            if self.fused_gate_up is None:
                self.w_gate = _put(w_gate, shard(-1), gu_dtype)
                self.w_up = _put(w_up, shard(-1), gu_dtype)
            # bf4 layers: 1D multicast config with DECODE_GATE_UP_K_BLOCK-tile activation blocks. Each step costs
            # ~1.3 us of multicast latency, so the auto config's 2-tile blocks (80 steps over K = 160 tiles) are
            # latency-bound once the block-float4 weight halves the DRAM bytes behind each step (bf8 keeps the auto
            # config: wider blocks are DRAM-bound there, 4-tile 116 us vs 112).
            self.gate_up_decode_pc = None
            if lowp:
                grid = device.compute_with_storage_grid_size()
                n_gu = -(-(inter // 32) // (grid.x * grid.y))
                self.gate_up_decode_pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(grid.x, grid.y),
                    in0_block_w=DECODE_GATE_UP_K_BLOCK,
                    out_subblock_h=1,
                    out_subblock_w=max(w for w in (1, 2, 3, 4) if n_gu % w == 0),
                    per_core_M=1,
                    per_core_N=n_gu,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                )
            # bf4 layers multiply at DECODE_GATE_UP_BF4_FIDELITY. Matmul feeds the weight (in1) to SrcA, and a block-float4
            # value has at most 3 significant mantissa bits, so every fidelity phase over SrcA's low bits multiplies
            # zeros: HiFi3 accumulates exactly what HiFi4 does with one pass fewer.
            self.gate_up_decode_cfg = None
            if lowp:
                self.gate_up_decode_cfg = ttnn.init_device_compute_kernel_config(
                    device.arch(),
                    math_fidelity=DECODE_GATE_UP_BF4_FIDELITY,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=False,
                )
        del w_gate, w_up
        self.w_down = _put(torch_module.down_proj.weight.T, shard(-2), self.DOWN_DTYPE)
        if self.DECODE_WEIGHTS:
            # Decode down projection: 1D multicast matmul over the interleaved bf8 copy with DOWN_DECODE_K_BLOCK-tile K
            # blocks. The auto config's 2-tile blocks made 128 serialised activation multicasts (~1.3 us each, 168 us)
            # for only ~0.35 MB of weight per step; wider blocks put enough weight behind each step that the op
            # streams at the DRAM rate instead.
            n_out = torch_module.down_proj.weight.shape[0]
            # (the prefill weight itself when it is already in that dtype). Layers in DECODE_DOWN_BF4_LAYERS read a
            # block-float4 copy at DECODE_GATE_UP_BF4_FIDELITY (exact on a bf4 weight, see the gate / up above).
            self.down_decode_cfg = None
            lowp_down = layer_idx is not None and layer_idx in DECODE_DOWN_BF4_LAYERS
            # DOWN_DECODE_STREAM: the decode down as one C++ program streaming its own bank-major weight copy
            # (tt/down_decode.py: bit-identical to the 1D-multicast matmul); then no interleaved decode copy is made.
            self.stream_down = None
            if DOWN_DECODE_STREAM:
                from models.demos.devstral_small_2_24b_instruct_2512.tt.down_decode import StreamDownDecode

                wdt = DECODE_GATE_UP_BF4_DTYPE if lowp_down else DECODE_WEIGHT_DTYPE
                if StreamDownDecode.supports(device, self.inter, n_out, wdt):
                    fid = DECODE_GATE_UP_BF4_FIDELITY if lowp_down else ttnn.MathFidelity.HiFi4
                    put = lambda t, dtype: _put(t, shard(-2), dtype)
                    self.stream_down = StreamDownDecode(
                        device, torch_module.down_proj.weight.T, wdt, fid, put, tp, ar_fold=DECODE_DOWN_AR_FOLD
                    )
            if self.stream_down is not None:
                self.w_down_decode = None
            elif lowp_down:
                self.w_down_decode = _put(torch_module.down_proj.weight.T, shard(-2), DECODE_GATE_UP_BF4_DTYPE)
                self.down_decode_cfg = ttnn.init_device_compute_kernel_config(
                    device.arch(),
                    math_fidelity=DECODE_GATE_UP_BF4_FIDELITY,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=False,
                )
            elif self.DOWN_DTYPE == DECODE_WEIGHT_DTYPE:
                self.w_down_decode = self.w_down
            else:
                self.w_down_decode = _put(torch_module.down_proj.weight.T, shard(-2), DECODE_WEIGHT_DTYPE)
            grid = device.compute_with_storage_grid_size()
            # On exactly the cores the N split occupies (160 tiles / 2 = 80: an 8 x 10 rectangle), so the activation
            # multicast spans no idle columns of the full grid.
            n_per_core = -(-(n_out // 32) // (grid.x * grid.y))
            n_cores = -(-(n_out // 32) // n_per_core)
            down_grid = (8, n_cores // 8) if n_cores % 8 == 0 and n_cores // 8 <= grid.y else (grid.x, grid.y)
            self.down_decode_pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=down_grid,
                in0_block_w=DOWN_DECODE_K_BLOCK,
                out_subblock_h=1,
                out_subblock_w=n_per_core,
                per_core_M=1,
                per_core_N=n_per_core,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
        self.compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,  # L1 accumulation made batched prefill matmuls non-repeatable
        )
        # The many-row down projection is compute-bound as well (fp32 accumulation kept).
        self.down_cfg = self.compute_cfg
        down_lofi = PREFILL_DOWN_LOFI and type(self) is TtMinistral3MLP and layer_idx is not None
        if self.PREFILL_DOWN_FIDELITY is not None or down_lofi:
            self.down_cfg = ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=ttnn.MathFidelity.LoFi if down_lofi else self.PREFILL_DOWN_FIDELITY,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
        # SHORT_DOWN_SWAPPED: the bf8 weight's values (<= 7 significant bits: exact in bf16) transposed to (N x K) for
        # the swapped program, whose srcB multiplier pass takes all 7 -- HiFi2 over srcA's two passes is HiFi4's product.
        self.w_down_t = None
        self.long_down_swapped = self.LONG_DOWN_SWAPPED and layer_idx in self.LONG_DOWN_SWAPPED_LAYERS
        if self.SHORT_DOWN_SWAPPED and type(self) is TtMinistral3MLP and tp > 1 and self.DOWN_DTYPE == ttnn.bfloat8_b:
            # (built on the host: _put's bf8 quantization of the same matrix, dequantized, transposed and tilized there,
            # then uploaded -- no device ops)
            wq = ttnn.to_torch(
                ttnn.from_torch(
                    torch_module.down_proj.weight.T.detach().to(torch.bfloat16).contiguous(),
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                )
            )
            wt = ttnn.from_torch(
                wq.T.contiguous().to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=shard(-1),
            )
            del wq
            self.w_down_t = ttnn.to_device(wt, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            del wt
            self.down_swapped_cfg = ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
        # The many-row fused gate/up is compute-bound; it runs at HiFi2 (fp32 accumulation kept).
        lofi = type(self) is TtMinistral3MLP and layer_idx is not None and layer_idx in PREFILL_GATE_UP_LOFI_LAYERS
        self.gate_up_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi if lofi else self.GATE_UP_FIDELITY,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _tp_degree(self, device, mesh):
        """Tensor-parallel degree over a mesh of `mesh` devices (the vision MLPs follow the vision data-parallel
        switch, see pixtral_attention.DATA_PARALLEL)."""
        return mesh

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def decode_norms(self, h) -> bool:
        """Whether the decode gate / up can take the residual h itself and norm it (NormGateUpDecode)."""
        return getattr(self.fused_gate_up, "accepts", None) is not None and self.fused_gate_up.accepts(h)

    def __call__(self, x, *args, mode="prefill", residual_norm=False, residual=None, **kwargs):
        """residual_norm (decode, decode_norms(x)): x is the residual, normed here by the gate / up program.
        residual (many-row, TP = 1): return residual + mlp(x) (FUSE_RESIDUAL_DOWN: the add folded into the down)."""
        shape = tuple(x.shape)
        if len(shape) != 4:
            x = ttnn.reshape(x, (1,) * (4 - len(shape)) + shape)
        if mode == "decode" and self.DECODE_WEIGHTS:
            return ttnn.reshape(self._forward_decode(x, residual_norm), shape)
        b, _, s, d = x.shape
        step = max(1, PREFILL_ROWS_PER_CHUNK // max(1, s))
        if b > step:
            # Prefill of a large batch: run the 32768-wide intermediate a few batch rows at a time so its
            # peak DRAM stays small (rows are independent, so the result is identical).
            parts = [self._forward(ttnn.slice(x, [i, 0, 0, 0], [min(i + step, b), 1, s, d])) for i in range(0, b, step)]
            out = ttnn.concat(parts, dim=0)
            if residual is not None:
                out = ttnn.add(residual, out)
            return ttnn.reshape(out, shape[:-1] + (out.shape[-1],))
        out = self._forward(x, residual)
        return ttnn.reshape(out, shape[:-1] + (out.shape[-1],))

    def _forward_decode(self, x, residual_norm=False):
        # The single-row gate / up / product stay in L1 for the down matmul (no DRAM round trip of the 32 x inter
        # intermediates); silu is fused into the multiply (ttnn.linear's activation= runs as a separate unary op).
        l1 = ttnn.L1_MEMORY_CONFIG
        if residual_norm:  # one program: the norm of the residual x, both projections, silu * up
            m = self.fused_gate_up.norm_call(x)
        elif self.fused_gate_up is not None:  # one program: both projections + silu * up (tt/gate_up_decode.py)
            m = self.fused_gate_up(x)
        else:
            pc = self.gate_up_decode_pc
            cfg = self.compute_cfg if self.gate_up_decode_cfg is None else self.gate_up_decode_cfg
            g = ttnn.linear(x, self.w_gate, program_config=pc, memory_config=l1, compute_kernel_config=cfg)
            u = ttnn.linear(x, self.w_up, program_config=pc, memory_config=l1, compute_kernel_config=cfg)
            m = ttnn.multiply(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=l1)
        return self._decode_down(m)

    def decode_ar(self, x, p):
        """Decode with the attention's o_proj all_reduce and the residual add folded into the gate / up program
        (tt/gate_up_decode.py NormGateUpDecode.ar_norm_call): x the residual, p this device's o_proj partials. Returns
        (h, mlp(norm(h))) with h = x + all_reduce(p) -- or (None, h + mlp(norm(h))) when the down program folds its own
        all_reduce and the residual add in too (DECODE_DOWN_AR_FOLD, tt/down_decode.py StreamDownDecode.ar_call)."""
        h, m = self.fused_gate_up.ar_norm_call(x, p, self.decode_all_reduce)
        sd = self.stream_down
        if sd is not None and sd.ar_ok(m, h, self.decode_all_reduce):
            return None, sd.ar_call(m, h, self.decode_all_reduce)
        return h, self._decode_down(m)

    def ar_folds(self, x, p) -> bool:
        """Whether decode_ar can take (x, p)."""
        fn = getattr(self.fused_gate_up, "ar_accepts", None)
        return fn is not None and fn(x, p, self.decode_all_reduce)

    def _decode_down(self, m):
        if self.stream_down is not None:  # one program streaming the bank-major down copy (tt/down_decode.py)
            if self.decode_all_reduce is not None:  # partials in its width shards, one collective
                return self.decode_all_reduce(self.stream_down(m, memory_config=self.decode_all_reduce.mem))
            return self._reduce(self.stream_down(m))
        out = ttnn.linear(
            m,
            self.w_down_decode,
            program_config=self.down_decode_pc,
            dtype=self.DECODE_REDUCE_DTYPE if (self.DECODE_REDUCE_DTYPE and self.tp > 1) else self.OUT_DTYPE,
            compute_kernel_config=self.compute_cfg if self.down_decode_cfg is None else self.down_decode_cfg,
        )
        return self._reduce(out)

    def _gate_up_n_block(self, x, grid):
        """N block of the fused gate/up: 12 tiles, or the whole per-core N range when that is smaller. minimal_matmul
        hands each core 2 * ceil(pairs / cores) weight tiles along N (the grid's y axis when M > N), and a block wider
        than that range is computed padded: the vision MLP's 8-tile range ran 12-tile blocks (half of them padding)."""
        shape = tuple(int(d) for d in x.shape)
        mt = shape[-2] // ttnn.TILE_SIZE
        for d in shape[:-2]:
            mt *= d
        nt = int(self.w_gate_up.shape[-1]) // ttnn.TILE_SIZE
        from models.demos.devstral_small_2_24b_instruct_2512.tt import minimal_mm

        n_cores = grid.y if minimal_mm.transposed(mt, nt, grid.x, grid.y, swiglu=True) else grid.x
        n_core = 2 * -(-(nt // 2) // n_cores)
        return n_core if n_core < 12 else 12

    @staticmethod
    def _m_range(mt, nt, grid, swiglu=False):
        """Each core's M tile range in minimal_matmul (M along the grid's x axis when it splits transposed, else y)."""
        from models.demos.devstral_small_2_24b_instruct_2512.tt import minimal_mm

        return -(-mt // (grid.x if minimal_mm.transposed(mt, nt, grid.x, grid.y, swiglu) else grid.y))

    def _forward(self, x, residual=None):
        grid = self.device.compute_with_storage_grid_size()
        rows = 1
        for d in tuple(int(v) for v in x.shape)[:-1]:
            rows *= d
        x_l1 = x.memory_config().buffer_type == ttnn.BufferType.L1
        inter_l1 = 0 < 2 * rows * self.inter // (grid.x * grid.y) <= self.PREFILL_INTER_L1_BYTES
        # M blocks that tile each core's M range with the least padding (no taller than 8 tiles): the SwiGLU epilogue
        # runs whole blocks, so the 416-row prompt's 42 tiles a core take six 7-tile blocks (8-tile blocks padded the
        # last to 8 for 2 real rows), the data-parallel vision MLP's 7 one, the 128-row prompt's 13 two. Paired with
        # 1 x 4 subblocks: 9844 vs 10166 us at HiFi2 and 5947 vs 7310 us at LoFi on the 416-row prompt (8-tile M, 2 x 2).
        # The fp32 intermediate accumulates every output tile's K blocks in the same order whatever the blocking.
        nt_gu = int(self.w_gate_up.shape[-1]) // ttnn.TILE_SIZE
        m_rng = self._m_range(rows // ttnn.TILE_SIZE, nt_gu, grid, swiglu=True)
        m_gu = min(range(min(8, m_rng), max(1, min(8, m_rng) // 2) - 1, -1), key=lambda b: -(-m_rng // b) * b - m_rng)
        n_gu = self._gate_up_n_block(x, grid)
        gu_cfg = ttnn.MinimalMatmulConfig(
            M_block_size=m_gu,
            # (an L1 fp32 input + the L1 product leave room only for half-depth activation blocks)
            K_block_size=4 if x_l1 and inter_l1 and x.dtype == ttnn.float32 else 8,
            N_block_size=n_gu,
            subblock_h=1 if n_gu % 4 == 0 else 2 if m_gu % 2 == 0 else 1,
            subblock_w=4 if n_gu % 4 == 0 else 2,
            compute_with_storage_grid_size=grid,
        )
        from models.demos.devstral_small_2_24b_instruct_2512.tt import minimal_mm

        gu_kw = dict(
            config=gu_cfg,
            compute_kernel_config=self.gate_up_cfg,
            memory_config=ttnn.L1_MEMORY_CONFIG if inter_l1 else ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            fuse_swiglu=True,
        )
        if PREFILL_MM_PROGRAM and minimal_mm.supported(x, self.w_gate_up, self.gate_up_cfg, fuse_swiglu=True):
            # the same program with silu * up as one SFPU pass (tt/minimal_mm.py), bit-identical
            m = minimal_mm.matmul(x, self.w_gate_up, **gu_kw)
        else:
            m = ttnn.experimental.minimal_matmul(x, self.w_gate_up, **gu_kw)
        if x_l1:  # the caller's norm output, read only here (see TtAttention)
            ttnn.deallocate(x)
        # Many-row down projection on the full grid as well (ttnn.linear's auto config left it at ~45 TFLOP/s). A short
        # prompt (<= 128 tile rows) runs on the grid that gives every core whole 16 x 16-tile output blocks: on the
        # full 11 x 10 grid its 128 x 160 output tiles split 13 x 15 per core, padded to 16 x 16 by the 8-tile blocks.
        m_shape = tuple(int(d) for d in m.shape)
        mt = m_shape[-2] // ttnn.TILE_SIZE
        for d in m_shape[:-2]:
            mt *= d
        nt = int(self.w_down.shape[-1]) // ttnn.TILE_SIZE
        down_cfg = None
        # the down keeps minimal_matmul's stock M > N split (stock_orient below): transposed it measured no better
        n_rng_dn = -(-nt // (grid.y if mt > nt else grid.x))
        m_rng_dn = -(-mt // (grid.x if mt > nt else grid.y))
        n_sub = max(w for w in (1, 2, 3, 4) if n_rng_dn % w == 0)
        if self.SHORT_DOWN_FULL_GRID and mt <= 128 and mt < nt and n_sub >= 3:
            # A short prompt on the FULL grid with blocks cut to each core's exact ranges (M blocks of half the M
            # range, N blocks of the whole N range, 1 x n_sub subblocks): the 128-row prompt's 128 x 160 output tiles
            # are 13 x 15 a core over 110 cores instead of 16 x 16 over the 80-core grid of whole 16-tile blocks.
            down_cfg = ttnn.MinimalMatmulConfig(
                M_block_size=-(-m_rng_dn // 2),
                K_block_size=8,
                N_block_size=n_rng_dn,
                subblock_h=1,
                subblock_w=n_sub,
                compute_with_storage_grid_size=grid,
            )
        elif mt <= 128 and mt % 16 == 0 and nt % 16 == 0 and mt // 16 <= grid.y and grid.x - 1 <= nt // 16 <= grid.x:
            down_cfg = ttnn.MinimalMatmulConfig(
                M_block_size=8,
                K_block_size=8,
                N_block_size=8,
                subblock_h=2,
                subblock_w=2,
                compute_with_storage_grid_size=ttnn.CoreCoord(nt // 16, mt // 16),
            )
        elif n_rng_dn in (2, 4, 6):
            # A narrow per-core N range (the vision MLP's 32 output tiles over 10 cores: 4) runs N blocks of exactly
            # that width; the default 8-tile block would be half padding.
            m_dn = min(8, -(-mt // (grid.x if mt > nt else grid.y)))
            down_cfg = ttnn.MinimalMatmulConfig(
                M_block_size=m_dn if m_dn % 2 == 0 else 8,
                K_block_size=8,
                N_block_size=n_rng_dn,
                subblock_h=2,
                subblock_w=2,
                compute_with_storage_grid_size=grid,
            )
        elif n_rng_dn % 8 == 0:
            # A wide per-core N range (the 416-row prompt's 160 output tiles over 10 cores: 16) in 8-tile blocks of 1 x 4
            # subblocks (4142 -> 4108 us; the default config runs 2 x 2), and M blocks of a third of each core's M range
            # (38 rows: 13, 13, 12): the weight blocks are re-read once per M block, 3 times instead of 5 (LoFi: 2.51 ->
            # 2.42 ms a call; the per-tile K order depends only on the N block, so bit-identical).
            m_rng = -(-mt // (grid.x if mt > nt else grid.y))
            down_cfg = ttnn.MinimalMatmulConfig(
                M_block_size=-(-m_rng // 3),
                K_block_size=8,
                N_block_size=8,
                subblock_h=1,
                subblock_w=4,
                compute_with_storage_grid_size=grid,
            )
        down_dtype = self.PREFILL_REDUCE_DTYPE if (self.PREFILL_REDUCE_DTYPE and self.tp > 1) else self.OUT_DTYPE
        out_bytes = (4 if down_dtype == ttnn.float32 else 2) * rows * nt * ttnn.TILE_SIZE // (grid.x * grid.y)
        down_l1 = self.tp == 1 and 0 < out_bytes <= self.DOWN_OUT_L1_BYTES
        # (stock op: its copy-out epilogue costs ~nothing, measured 4110 vs 4109 us through tt/minimal_mm.py)
        down_kw = dict(
            config=down_cfg,
            compute_kernel_config=self.down_cfg,
            memory_config=ttnn.L1_MEMORY_CONFIG if down_l1 else ttnn.DRAM_MEMORY_CONFIG,
            dtype=down_dtype,
        )
        n_sw = -(-mt // grid.x)
        if (
            self.w_down_t is not None
            and ((mt <= 128 and n_sw % 2 == 0) or (self.long_down_swapped and mt > 128))
            and m.layout == ttnn.TILE_LAYOUT
            and not m.is_sharded()
            and int(m.padded_shape[-1]) == int(self.w_down_t.padded_shape[-1])
        ):
            # swapped (SHORT_DOWN_SWAPPED / LONG_DOWN_SWAPPED): the weight's 160 rows over the grid's 10 rows (16 a
            # core, M blocks of 8), the prompt's tile rows over its 11 columns (128: 12 a core, one N block; 416: 38 a
            # core in N blocks of 10), 2 x 2 subblocks, at HiFi2
            sw_cfg = ttnn.MinimalMatmulConfig(
                M_block_size=8,
                K_block_size=8,
                N_block_size=n_sw if mt <= 128 else self.LONG_DOWN_SWAPPED_N_BLOCK,
                subblock_h=2,
                subblock_w=2,
                compute_with_storage_grid_size=grid,
            )
            out = minimal_mm.matmul(
                m,
                self.w_down_t,
                config=sw_cfg,
                compute_kernel_config=self.down_swapped_cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=down_dtype,
                swapped=True,
                orient=False,
            )
        elif (
            residual is not None
            and self.FUSE_RESIDUAL_DOWN
            and self.tp == 1
            and down_cfg is not None
            and minimal_mm.supported(m, self.w_down, self.down_cfg)
            and minimal_mm.supports_residual(m, self.w_down, residual, down_dtype)
        ):
            # residual + down(m) in one program (tt/minimal_mm.py residual=), bit-identical to the add below
            out = minimal_mm.matmul(m, self.w_down, residual=residual, stock_orient=True, **down_kw)
            residual = None
        elif down_cfg is not None and minimal_mm.supported(m, self.w_down, self.down_cfg):
            # the same program, its core ranges' padding skipped (tt/minimal_mm.py SKIP_PADDING), bit-identical
            out = minimal_mm.matmul(m, self.w_down, stock_orient=True, **down_kw)
        else:
            out = ttnn.experimental.minimal_matmul(m, self.w_down, **down_kw)
        if inter_l1:
            ttnn.deallocate(m)
        out = self._reduce(out, gather=self.prefill_gather)
        if residual is not None:
            f, out = out, ttnn.add(residual, out, memory_config=out.memory_config())
            ttnn.deallocate(f)
        return out

    def _reduce(self, out, gather=True):
        if self.tp == 1:
            return out
        if self.GATHER_DTYPE is None:
            return ttnn.all_reduce(out, topology=ttnn.Topology.Linear)
        ccl = tp_ccl_args()
        out = ttnn.reduce_scatter(out, dim=3, **ccl)
        if not gather:
            return out
        if out.dtype != self.GATHER_DTYPE:
            out = ttnn.typecast(out, self.GATHER_DTYPE)
        if self.decode_gather_mem is not None and int(out.shape[-2]) == ttnn.TILE_SIZE:
            return ttnn.all_gather(out, dim=3, memory_config=self.decode_gather_mem, **ccl)
        return ttnn.all_gather(out, dim=3, **ccl)


def build(device, torch_module=None):
    return TtMinistral3MLP.build(device, torch_module)


def ministral3_m_l_p(device, torch_module=None):
    return TtMinistral3MLP.build(device, torch_module)

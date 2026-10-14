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
# K tiles per activation multicast block of the decode down projection (divides K = 256 tiles).
DOWN_DECODE_K_BLOCK = 8


class TtMinistral3MLP:
    # dtype of the down projection's output and its all_reduce: fp32, added unrounded into the fp32 residual
    # stream of the text decoder (and of the vision layers, whose MLPs subclass this one).
    OUT_DTYPE = ttnn.float32
    # Separate gate / up weights for the single-token decode path (the vision MLP subclasses never decode).
    DECODE_WEIGHTS = True
    # dtype the reduced output is all_gathered in (None: plain fp32 all_reduce). The text decoder reduce-scatters
    # in fp32 and gathers the reduced shard in bf16 (half the bytes); its residual add takes the bf16 unrounded.
    GATHER_DTYPE = ttnn.bfloat16
    # dtype of the many-row (prefill) down projection's partials and their reduce-scatter (None: OUT_DTYPE). The text
    # decoder reduce-scatters prefill's partials in bf16 (half the bytes, no cast before the gather); decode keeps fp32.
    PREFILL_REDUCE_DTYPE = ttnn.bfloat16
    # Math fidelity of the many-row (prefill) down projection (None: the HiFi4 compute_cfg decode uses too).
    PREFILL_DOWN_FIDELITY = ttnn.MathFidelity.HiFi2
    # dtype of the down weight the many-row (prefill) path multiplies with.
    DOWN_DTYPE = ttnn.bfloat8_b
    # False: prefill returns the down projection's reduce-scattered 1/TP hidden shard (see the decoder layer).
    prefill_gather = True
    # dtype of the fused gate/up weight the many-row (prefill) path multiplies with.
    GATE_UP_DTYPE = ttnn.bfloat8_b
    # The many-row SwiGLU product goes to the down projection through L1 (interleaved, freed right after) while it
    # takes at most this many bytes per core (the vision MLP's 8192 x 1024 bf16: 152 KB; the text MLP's is far larger).
    PREFILL_INTER_L1_BYTES = 192 * 1024

    def __init__(self, device, torch_module, in_scale=None) -> None:
        """in_scale: optional per-input-feature scale (the post-attention RMSNorm's gamma) folded into gate/up."""
        self.device = device
        try:
            self.tp = tp = int(device.get_num_devices())
        except Exception:
            self.tp = tp = 1
        assert (
            torch_module.intermediate_size % tp == 0
        ), f"intermediate {torch_module.intermediate_size} not divisible by TP={tp}"
        shard = lambda dim: ttnn.ShardTensorToMesh(device, dim=dim) if tp > 1 else None

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
            # the bytes of the bf16 weights prefill multiplies with.
            self.w_gate = _put(w_gate, shard(-1), DECODE_WEIGHT_DTYPE)
            self.w_up = _put(w_up, shard(-1), DECODE_WEIGHT_DTYPE)
        del w_gate, w_up
        self.w_down = _put(torch_module.down_proj.weight.T, shard(-2), self.DOWN_DTYPE)
        if self.DECODE_WEIGHTS:
            # Decode down projection: 1D multicast matmul over the interleaved bf8 copy with DOWN_DECODE_K_BLOCK-tile K
            # blocks. The auto config's 2-tile blocks made 128 serialised activation multicasts (~1.3 us each, 168 us)
            # for only ~0.35 MB of weight per step; wider blocks put enough weight behind each step that the op
            # streams at the DRAM rate instead.
            n_out = torch_module.down_proj.weight.shape[0]
            # (the prefill weight itself when it is already in that dtype)
            if self.DOWN_DTYPE == DECODE_WEIGHT_DTYPE:
                self.w_down_decode = self.w_down
            else:
                self.w_down_decode = _put(torch_module.down_proj.weight.T, shard(-2), DECODE_WEIGHT_DTYPE)
            grid = device.compute_with_storage_grid_size()
            n_per_core = -(-(n_out // 32) // (grid.x * grid.y))
            self.down_decode_pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(grid.x, grid.y),
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
        if self.PREFILL_DOWN_FIDELITY is not None:
            self.down_cfg = ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=self.PREFILL_DOWN_FIDELITY,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
        # The many-row fused gate/up is compute-bound; it runs at HiFi2 (fp32 accumulation kept).
        self.gate_up_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def __call__(self, x, *args, mode="prefill", **kwargs):
        shape = tuple(x.shape)
        if len(shape) != 4:
            x = ttnn.reshape(x, (1,) * (4 - len(shape)) + shape)
        if mode == "decode" and self.DECODE_WEIGHTS:
            return ttnn.reshape(self._forward_decode(x), shape)
        b, _, s, d = x.shape
        step = max(1, PREFILL_ROWS_PER_CHUNK // max(1, s))
        if b > step:
            # Prefill of a large batch: run the 32768-wide intermediate a few batch rows at a time so its
            # peak DRAM stays small (rows are independent, so the result is identical).
            parts = [self._forward(ttnn.slice(x, [i, 0, 0, 0], [min(i + step, b), 1, s, d])) for i in range(0, b, step)]
            out = ttnn.concat(parts, dim=0)
            return ttnn.reshape(out, shape[:-1] + (out.shape[-1],))
        out = self._forward(x)
        return ttnn.reshape(out, shape[:-1] + (out.shape[-1],))

    def _forward_decode(self, x):
        # The single-row gate / up / product stay in L1 for the down matmul (no DRAM round trip of the 32 x inter
        # intermediates); silu is fused into the multiply (ttnn.linear's activation= runs as a separate unary op).
        l1 = ttnn.L1_MEMORY_CONFIG
        g = ttnn.linear(x, self.w_gate, memory_config=l1, compute_kernel_config=self.compute_cfg)
        u = ttnn.linear(x, self.w_up, memory_config=l1, compute_kernel_config=self.compute_cfg)
        m = ttnn.multiply(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=l1)
        out = ttnn.linear(
            m,
            self.w_down_decode,
            program_config=self.down_decode_pc,
            dtype=self.OUT_DTYPE,
            compute_kernel_config=self.compute_cfg,
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
        n_cores = grid.y if mt > nt else grid.x
        n_core = 2 * -(-(nt // 2) // n_cores)
        return n_core if n_core < 12 else 12

    def _forward(self, x):
        grid = self.device.compute_with_storage_grid_size()
        rows = 1
        for d in tuple(int(v) for v in x.shape)[:-1]:
            rows *= d
        inter_l1 = 0 < 2 * rows * self.inter // (grid.x * grid.y) <= self.PREFILL_INTER_L1_BYTES
        m = ttnn.experimental.minimal_matmul(
            x,
            self.w_gate_up,
            config=ttnn.MinimalMatmulConfig(
                M_block_size=8,
                K_block_size=8,
                N_block_size=self._gate_up_n_block(x, grid),
                subblock_h=2,
                subblock_w=2,
                compute_with_storage_grid_size=grid,
            ),
            compute_kernel_config=self.gate_up_cfg,
            memory_config=ttnn.L1_MEMORY_CONFIG if inter_l1 else ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            fuse_swiglu=True,
        )
        # Many-row down projection on the full grid as well (ttnn.linear's auto config left it at ~45 TFLOP/s). A short
        # prompt (<= 128 tile rows) runs on the grid that gives every core whole 16 x 16-tile output blocks: on the
        # full 11 x 10 grid its 128 x 160 output tiles split 13 x 15 per core, padded to 16 x 16 by the 8-tile blocks.
        m_shape = tuple(int(d) for d in m.shape)
        mt = m_shape[-2] // ttnn.TILE_SIZE
        for d in m_shape[:-2]:
            mt *= d
        nt = int(self.w_down.shape[-1]) // ttnn.TILE_SIZE
        down_cfg = None
        if mt <= 128 and mt % 16 == 0 and nt % 16 == 0 and mt // 16 <= grid.y and grid.x - 1 <= nt // 16 <= grid.x:
            down_cfg = ttnn.MinimalMatmulConfig(
                M_block_size=8,
                K_block_size=8,
                N_block_size=8,
                subblock_h=2,
                subblock_w=2,
                compute_with_storage_grid_size=ttnn.CoreCoord(nt // 16, mt // 16),
            )
        elif -(-nt // (grid.y if mt > nt else grid.x)) in (2, 4, 6):
            # A narrow per-core N range (the vision MLP's 32 output tiles over 10 cores: 4) runs N blocks of exactly
            # that width; the default 8-tile block would be half padding.
            down_cfg = ttnn.MinimalMatmulConfig(
                M_block_size=8,
                K_block_size=8,
                N_block_size=-(-nt // (grid.y if mt > nt else grid.x)),
                subblock_h=2,
                subblock_w=2,
                compute_with_storage_grid_size=grid,
            )
        out = ttnn.experimental.minimal_matmul(
            m,
            self.w_down,
            config=down_cfg,
            compute_kernel_config=self.down_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.PREFILL_REDUCE_DTYPE if (self.PREFILL_REDUCE_DTYPE and self.tp > 1) else self.OUT_DTYPE,
        )
        if inter_l1:
            ttnn.deallocate(m)
        return self._reduce(out, gather=self.prefill_gather)

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
        return ttnn.all_gather(out, dim=3, **ccl)


def build(device, torch_module=None):
    return TtMinistral3MLP.build(device, torch_module)


def ministral3_m_l_p(device, torch_module=None):
    return TtMinistral3MLP.build(device, torch_module)

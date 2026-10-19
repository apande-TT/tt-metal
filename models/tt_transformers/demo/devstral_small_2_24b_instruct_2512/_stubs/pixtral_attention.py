# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native tensor-parallel TTNN port of `pixtral_attention` (PixtralAttention,
`vision_tower.transformer.layers.0.attention`) of `mistralai/Devstral-Small-2-24B-Instruct-2512`.

    q, k, v = x Wq, x Wk, x Wv                     (16 heads, head_dim 64, MHA)
    out     = sdpa(rope(q), rope(k), v, mask, scale=head_dim**-0.5, non-causal) Wo

Tensor-parallel over the mesh (TP = number of devices; TP = 1 on a single device):
  * q/k/v   column-parallel by HEAD: each device holds num_heads/TP heads, fused into one wqkv.
  * o_proj  row-parallel (input features = that device's heads), then all_reduce.
  * rotary tables and the mask are replicated.
"""
from __future__ import annotations

import torch

import ttnn
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs.ministral3_attention import (
    TtMinistral3Attention,
    fold_input_scale,
)


def _mesh_size(device):
    try:
        return int(device.get_num_devices())
    except Exception:
        return 1


# Vision data parallelism. True while the pipeline builds the vision tower and projector: those modules then keep
# their FULL weights replicated on every device of the mesh and run without any collective, and the caller hands each
# device its own slice of the image batch (and gathers the projector output). False: tensor-parallel over the mesh.
DATA_PARALLEL = False

# Vision q/k in interleaved-pair head_dim order (2j, 2j+1 <- HF's j, j + head_dim/2). True while the vision model builds
# its transformer, whose rotary tables then come in the same order (TtPixtralRotaryEmbedding(interleaved=True)): the
# q/k weight columns are permuted per head and rotate_half becomes a pair swap inside every 32-wide tile, so q's and
# k's RoPE each run as ONE fused rotary_embedding_llama instead of the rotate-half matmul + multiply / multiply / add.
# q . k is unchanged by a permutation applied to both, and V is not permuted.
META_ROPE = False


def vision_tp(device):
    """(tensor-parallel degree of a vision module built now, mapper for its replicated tensors)."""
    mesh = _mesh_size(device)
    return (1 if DATA_PARALLEL else mesh), (ttnn.ReplicateTensorToMesh(device) if mesh > 1 else None)


class TtPixtralAttention:
    # The fused q/k/v output (read once, by the head split, which frees it) lands in L1 while it takes at most this many
    # bytes per core (the data-parallel 8 x 256-row batch: 114 KB).
    QKV_OUT_L1_BYTES = 128 * 1024

    def __init__(self, device, torch_module, in_scale=None) -> None:
        """in_scale: the attention_norm gamma, folded into q/k/v (see TtAttention.FOLD_NORM_GAMMA)."""
        self.device = device
        self.meta_rope = META_ROPE
        self.tp, repl = vision_tp(device)
        tp = self.tp
        self.num_heads = torch_module.num_heads
        self.head_dim = hd = torch_module.head_dim
        self.scale = float(torch_module.scaling)
        assert self.num_heads % tp == 0, f"{self.num_heads} heads not divisible by TP={tp}"
        self.local_heads = lh = self.num_heads // tp

        is_mesh = tp > 1
        shard = lambda dim: ttnn.ShardTensorToMesh(device, dim=dim) if is_mesh else repl
        self._repl = repl

        def _put(t, mapper):
            return ttnn.from_torch(
                t.detach().to(torch.bfloat16).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Fused per-device [q_heads | k_heads | v_heads] so a dim=-1 shard hands each device its own heads.
        wq, wk, wv = (
            fold_input_scale(w.weight.T, in_scale)
            for w in (torch_module.q_proj, torch_module.k_proj, torch_module.v_proj)
        )
        self._perm = torch.arange(hd).reshape(2, hd // 2).T.reshape(-1)  # [0, hd/2, 1, hd/2 + 1, ...]
        if self.meta_rope:
            per_head = lambda w: w.reshape(w.shape[0], -1, hd)[:, :, self._perm].reshape(w.shape[0], -1)
            wq, wk = per_head(wq), per_head(wk)
        chunks = []
        for d in range(tp):
            sl = slice(d * lh * hd, (d + 1) * lh * hd)
            chunks += [wq[:, sl], wk[:, sl], wv[:, sl]]
        self.wqkv = _put(torch.hstack(chunks), shard(-1))  # 2-D slices: hstack == concat on dim -1
        self.wo = _put(torch_module.o_proj.weight.T, shard(-2))

        # rotate_half(x) == x @ R  (HF convention: cat(-x2, x1)).
        half = hd // 2
        rot = torch.zeros(hd, hd)
        rot[torch.arange(half) + half, torch.arange(half)] = -1.0
        rot[torch.arange(half), torch.arange(half) + half] = 1.0
        self.rot = _put(rot.reshape(1, 1, hd, hd), repl)
        # meta_rope: the 32 x 32 pair-swap tile the fused RoPE applies to every tile of a head.
        self.trans_mat = (
            _put(TtMinistral3Attention._pair_swap(ttnn.TILE_SIZE).reshape(1, 1, ttnn.TILE_SIZE, -1), repl)
            if self.meta_rope
            else None
        )
        # The fused RoPE replays the rotate-half chain's arithmetic: the pair swap is exact, and x * cos, rot * sin and
        # their sum are bf16 FPU products / sum at HiFi4 in a 16-bit DEST (the setting BinaryNg runs for bf16 operands).
        self.rope_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        self.compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,  # L1 accumulation made batched prefill matmuls non-repeatable
        )

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def _to_tt(self, t):
        return ttnn.from_torch(
            t.to(torch.bfloat16).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=self._repl,
        )

    def _narrow_n_cfg(self, x, w):
        """minimal_matmul config whose blocks are the per-core M / N ranges where those are under 8 tiles (M splits
        along the grid's x axis and N along y when M > N, else the other way round): the default 8-tile blocks would
        compute mostly padding (TP o_proj: 4 of 8 N tiles valid, TP QKV: 3 of 8; the data-parallel QKV's 7 x 9-tile
        range ran 8 x 16). None (the default 8 x 8 blocks) otherwise."""
        grid = self.device.compute_with_storage_grid_size()
        shape = tuple(int(d) for d in x.shape)
        mt = shape[-2] // ttnn.TILE_SIZE
        for d in shape[:-2]:
            mt *= d
        nt = int(w.shape[-1]) // ttnn.TILE_SIZE
        m = min(8, -(-mt // (grid.x if mt > nt else grid.y)))
        n = -(-nt // (grid.y if mt > nt else grid.x))
        if (n >= 8 and m == 8) or n > 12 or (n % 2 and n > 4 and n % 3):
            return None
        if m % 2 == 0 and n % 2 == 0:
            sh, sw = 2, 2
        else:
            sh, sw = 1, next(c for c in (4, 3, 2, 1) if n % c == 0)
        return ttnn.MinimalMatmulConfig(
            M_block_size=m,
            K_block_size=8,
            N_block_size=n,
            subblock_h=sh,
            subblock_w=sw,
            compute_with_storage_grid_size=grid,
        )

    def _rope_groups(self, shape):
        """B' of the (B', B * H / B') view of the (B, H, S, hd) heads the fused RoPE runs on. Its prefill split gives
        min(B', cores) batch slices and the remaining cores sequence tiles, each core every head of its slice, so take
        the view leaving the busiest core the fewest head rows (8 images x 16 heads of 8 sequence tiles: 32 x 4, 12
        rows). The outer-dim reshape is a view and leaves every row's position (its cos / sin) unchanged."""
        b, h, s = shape[:3]
        grid = self.device.compute_with_storage_grid_size()
        cores, st, n = grid.x * grid.y, -(-s // ttnn.TILE_SIZE), b * h

        def rows(g):
            bp = min(g, cores)
            sp = min(cores // bp, st)
            return -(-g // bp) * -(-st // sp) * (n // g)

        return min((g for g in range(1, n + 1) if n % g == 0), key=lambda g: (rows(g), g != b))

    def _rope(self, x, cos, sin):
        if self.meta_rope:
            shape = tuple(int(d) for d in x.shape)
            g = self._rope_groups(shape)
            if g != shape[0]:
                x = ttnn.reshape(x, (g, shape[0] * shape[1] // g) + shape[2:])
            x = ttnn.experimental.rotary_embedding_llama(
                x,
                cos,
                sin,
                self.trans_mat,
                is_decode_mode=False,
                compute_kernel_config=self.rope_cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            return x if g == shape[0] else ttnn.reshape(x, shape)
        # rotate_half as x @ R (R is a +-1 permutation, so exact), height-parallel: a 1D multicast matmul hands every
        # core of the full grid its own run of the (images x heads x patches) row tiles against the multicast 2 x 2-tile
        # R. (minimal_matmul splits a 2-tile N over the grid's 10 rows, leaving 8 of them idle: 102 us; ttnn.matmul's
        # auto config ran the batch on 16 cores: 252 us.)
        grid = self.device.compute_with_storage_grid_size()
        shape = tuple(int(d) for d in x.shape)
        rows = 1
        for d in shape[:-1]:
            rows *= d
        nt = shape[-1] // ttnn.TILE_SIZE
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=nt,
            out_subblock_h=1,
            out_subblock_w=nt,
            per_core_M=-(-(rows // ttnn.TILE_SIZE) // (grid.x * grid.y)),
            per_core_N=nt,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )
        rot = ttnn.matmul(
            x,
            self.rot,
            program_config=pc,
            compute_kernel_config=self.compute_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(rot, sin))

    def __call__(self, hidden_states, attention_mask=None, position_embeddings=None, **kwargs):
        x = hidden_states
        in_shape = tuple(x.shape)
        if len(in_shape) == 3:  # HF layout (B, S, D)
            x = ttnn.reshape(x, (in_shape[0], 1, in_shape[1], in_shape[2]))
        # Batch is the leading axis (one image per row); take it from the tensor, never assume 1.
        b, s = x.shape[0], x.shape[2]
        hd = self.head_dim

        cos, sin = position_embeddings
        if self.meta_rope:  # host tables come in HF order; device tables already in the interleaved one
            cos = cos[..., self._perm] if isinstance(cos, torch.Tensor) else cos
            sin = sin[..., self._perm] if isinstance(sin, torch.Tensor) else sin
        cos = self._to_tt(cos.reshape(1, 1, s, hd)) if isinstance(cos, torch.Tensor) else cos
        sin = self._to_tt(sin.reshape(1, 1, s, hd)) if isinstance(sin, torch.Tensor) else sin
        mask = attention_mask
        if isinstance(mask, torch.Tensor):
            mask = self._to_tt(mask.reshape(1, 1, s, s).float())

        # minimal_matmul folds the images into its rows over the full grid (ttnn.linear ran this batch of 32 x 256
        # rows as a batched matmul on 64 cores: 816 us).
        grid = self.device.compute_with_storage_grid_size()
        qkv_l1 = 2 * int(b) * int(s) * int(self.wqkv.shape[-1]) // (grid.x * grid.y) <= self.QKV_OUT_L1_BYTES
        qkv = ttnn.experimental.minimal_matmul(
            x,
            self.wqkv,
            config=self._narrow_n_cfg(x, self.wqkv),
            compute_kernel_config=self.compute_cfg,
            memory_config=ttnn.L1_MEMORY_CONFIG if qkv_l1 else ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,  # q / k / v in bf16 also when x is the fp32 norm output
        )
        if len(in_shape) == 4 and x.memory_config().buffer_type == ttnn.BufferType.L1:
            ttnn.deallocate(x)  # the caller's L1 norm output, read only here (see TtAttention)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.local_heads,
            num_kv_heads=self.local_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if qkv_l1:
            ttnn.deallocate(qkv)
        q = self._rope(q, cos, sin)
        k = self._rope(k, cos, sin)
        if q.dtype != ttnn.bfloat16:  # fp32 activations upstream; SDPA reads bf16
            q, k, v = (ttnn.typecast(t, ttnn.bfloat16) for t in (q, k, v))
        # Widest key chunk the sequence allows: SDPA's per-key-chunk rescale rounding biases its output level
        # upward (+0.23% at k_chunk 32, +0.02% at 512 on the text model's attention).
        k_chunk = next(c for c in (512, 256, 128, 64, 32) if s % c == 0)
        q_chunk = 64 if s % 64 == 0 else 32
        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            attn_mask=mask,
            scale=self.scale,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
                q_chunk_size=q_chunk,
                k_chunk_size=k_chunk,
                exp_approx_mode=False,
            ),
            compute_kernel_config=self.compute_cfg,
        )
        attn = ttnn.experimental.nlp_concat_heads(attn)
        # fp32 output: it is added straight into the vision layer's fp32 residual stream. minimal_matmul folds the
        # images into its rows over the full grid (ttnn.linear ran the 32 x 256-row batch on 88 cores: 360 us), with
        # N blocks no wider than each core's N range (the default 8-tile block would be half padding at 4 tiles).
        out = ttnn.experimental.minimal_matmul(
            attn,
            self.wo,
            config=self._narrow_n_cfg(attn, self.wo),
            compute_kernel_config=self.compute_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.float32,
        )
        if self.tp > 1:
            out = ttnn.all_reduce(out, topology=ttnn.Topology.Linear)
        if len(in_shape) == 3:
            return ttnn.reshape(out, (b, s, out.shape[-1]))
        return out


def build(device, torch_module=None):
    return TtPixtralAttention.build(device, torch_module)


def pixtral_attention(device, torch_module=None):
    return TtPixtralAttention.build(device, torch_module)

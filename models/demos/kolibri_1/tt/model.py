# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kolibri-1 forward on the TP mesh, composed from the graduated stubs in ../_stubs.

Every class here IS a graduated stub class (a subclass of it); what each adds is wiring the stubs
need to run as one model rather than one PCC'd module:

  token_embed       Embedding      the prompt ids (prefill) and each sampled token (decode)
  decoder_layer     DecoderLayer   the stub's sandwich-norm forward, unchanged, over the two below
  attention         Attention      the stub's fused qkv / q,k-norm / RoPE / sliding SDPA, plus resident
                                   K,V caches (filled in prefill, extended in decode) and a decode path
  f_p8_linear       FP8Linear      the attention's o_proj (column-parallel + all_gather) on the
                                   all-gathered heads
  sparse_moe_block  MoE            the routed experts (expert-parallel, the stub's two wide matmuls)
  router            Router         the fp32 router logits the MoE selects its top-6 from
  m_l_p             SharedExpert   the ungated shared expert (TP-split SwiGLU + all_reduce)
  r_m_s_norm        FinalNorm      the model's final norm
  decoder_head      LMHead         the LM head, fp32 logits (the reference runs the head in fp32)

INVOCATIONS counts the calls each graduated module receives during a forward (the e2e test's
evidence that every graduated module runs inside the real forward path).
"""
from __future__ import annotations

import contextlib
import struct
from collections import Counter
from pathlib import Path

import torch

import ttnn
from models.demos.kolibri_1._stubs.attention import TtKolibri1Attention
from models.demos.kolibri_1._stubs.decoder_head import TtKolibri1LMHead
from models.demos.kolibri_1._stubs.decoder_layer import _NORMS, TtKolibri1DecoderLayer
from models.demos.kolibri_1._stubs.f_p8_linear import TtKolibri1FP8Linear
from models.demos.kolibri_1._stubs.m_l_p import TtKolibri1MLP
from models.demos.kolibri_1._stubs.r_m_s_norm import TtKolibri1RMSNorm
from models.demos.kolibri_1._stubs.router import TtKolibri1Router
from models.demos.kolibri_1._stubs.sparse_moe_block import TtKolibri1SparseMoeBlock
from models.demos.kolibri_1._stubs.token_embed import TtKolibri1Embedding

GRADUATED = (
    "token_embed",
    "decoder_layer",
    "attention",
    "f_p8_linear",
    "sparse_moe_block",
    "router",
    "m_l_p",
    "r_m_s_norm",
    "decoder_head",
)
INVOCATIONS = Counter()

DRAM = ttnn.DRAM_MEMORY_CONFIG


@contextlib.contextmanager
def host_side_uploads():
    """While the model builds, every ttnn.from_torch(..., device=...) converts and lays out its tensor on the HOST
    and then copies it. Given an fp32 torch tensor and a bf16 / tile-layout target, ttnn uploads first and
    converts on the device (tilize, typecast, untilize): ~19 ms of device time per build here, 14 ms of it the
    128000 x 2560 embedding table. fp32 -> bf16 rounds to nearest even on either side."""
    orig = ttnn.from_torch

    def from_torch(tensor, *args, device=None, memory_config=None, **kwargs):
        if device is None:
            return orig(tensor, *args, memory_config=memory_config, **kwargs)
        if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32 and kwargs.get("dtype") == ttnn.bfloat16:
            tensor = tensor.to(torch.bfloat16)
        host = orig(tensor, *args, **kwargs)
        return ttnn.to_device(host, device, memory_config=memory_config or DRAM)

    ttnn.from_torch = from_torch
    try:
        yield
    finally:
        ttnn.from_torch = orig


class StepState:
    """Per-forward inputs shared by every layer: the mode, the RoPE rows for the positions being
    processed, and (decode) each user's current position."""

    def __init__(self, mode, cos=None, sin=None, cur_pos=None):
        self.mode = mode
        self.cos = cos
        self.sin = sin
        self.cur_pos = cur_pos


# ------------------------------------------------------------------------------------------- small stubs
class Embedding(TtKolibri1Embedding):
    def __call__(self, input_ids, **kwargs):
        INVOCATIONS["token_embed"] += 1
        return super().__call__(input_ids, **kwargs)


class FP8Linear(TtKolibri1FP8Linear):
    def __call__(self, x, **kwargs):
        """The stub's column-parallel projection; at prefill sizes the matmul runs as a full-grid minimal_matmul
        (ttnn.linear's default config used 100 of the 110 cores for [4096, 6144] x [6144, 640])."""
        INVOCATIONS["f_p8_linear"] += 1
        shape = list(x.shape)
        x = ttnn.reshape(x, [1, 1, -1, shape[-1]])
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
        rows = x.shape[-2]
        if rows < 2048:
            out = ttnn.linear(x, self.w, compute_kernel_config=self.mm_cfg)
        else:
            g = self.device.compute_with_storage_grid_size()
            m_tiles, n_tiles = rows // ttnn.TILE_SIZE, self.w.shape[-1] // ttnn.TILE_SIZE
            m_cores, n_cores = (g.x, g.y) if m_tiles > n_tiles else (g.y, g.x)
            n_block = -(-n_tiles // n_cores)
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=-(-m_tiles // m_cores),
                K_block_size=8,
                N_block_size=n_block,
                subblock_h=1,
                subblock_w=next(w for w in (4, 3, 2, 1) if n_block % w == 0),
                compute_with_storage_grid_size=ttnn.CoreCoord(g.x, g.y),
            )
            out = ttnn.experimental.minimal_matmul(
                x, self.w, config=cfg, compute_kernel_config=self.mm_cfg, dtype=ttnn.bfloat16
            )
        if self.tp > 1:
            out = ttnn.all_gather(out, dim=3, cluster_axis=self.tp_axis, topology=ttnn.Topology.Linear)
        return ttnn.reshape(out, shape[:-1] + [out.shape[-1]])


class Router(TtKolibri1Router):
    def __call__(self, x, **kwargs):
        INVOCATIONS["router"] += 1
        return super().__call__(x, **kwargs)


class SharedExpert(TtKolibri1MLP):
    def __call__(self, x, **kwargs):
        INVOCATIONS["m_l_p"] += 1
        return super().__call__(x, **kwargs)

    def partial(self, x):
        """The stub's forward without its all_reduce: x [1, 1, N, hidden] bf16 -> this chip's partial output, which
        the MoE block adds to its routed partial so ONE all_reduce sums both."""
        INVOCATIONS["m_l_p"] += 1
        g = ttnn.linear(x, self.w_gate, compute_kernel_config=self.mm_cfg)
        u = ttnn.linear(x, self.w_up, compute_kernel_config=self.mm_cfg)
        act = ttnn.multiply(ttnn.silu(g), u)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        out = ttnn.linear(act, self.w_down, compute_kernel_config=self.mm_cfg)
        ttnn.deallocate(act)
        return out


class FinalNorm(TtKolibri1RMSNorm):
    def __call__(self, x, **kwargs):
        INVOCATIONS["r_m_s_norm"] += 1
        return super().__call__(x, **kwargs)


class LMHead(TtKolibri1LMHead):
    """The graduated head with fp32 logits: the reference computes the head in fp32 (head_dtype), and
    bf16 logits would quantise them by up to 0.125 at |logit| >= 16, which the sampler would see."""

    def __call__(self, x, **kwargs):
        INVOCATIONS["decoder_head"] += 1
        shape = list(x.shape)
        x = ttnn.reshape(x, [1, 1, -1, shape[-1]])
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
        out = ttnn.linear(x, self.w, dtype=ttnn.float32, compute_kernel_config=self.mm_cfg)
        if self.tp > 1:
            out = ttnn.all_gather(out, dim=3, cluster_axis=self.tp_axis, topology=ttnn.Topology.Linear)
        return ttnn.reshape(out, shape[:-1] + [out.shape[-1]])


# ----------------------------------------------------------------------------------------------- attention
class Attention(TtKolibri1Attention):
    """The graduated attention with resident K,V caches.

    Caches are [B, kv_local, C, head_dim] per chip (each chip holds its own kv head). Prefill writes all
    B users' K,V for positions [0, T) in one fill (the cache viewed as one user of B*kv heads); decode
    appends one position per user (paged_update_cache at each user's current position) and attends with
    SDPA-decode, the same sliding window on sliding layers. o_proj is the graduated f_p8_linear stub."""

    def __init__(self, device, torch_module, batch: int, capacity: int) -> None:
        super().__init__(device, torch_module)
        self.o_proj = FP8Linear(device, torch_module.o_proj)
        ttnn.deallocate(self.wo)  # replaced by o_proj above
        self.wo = None
        self.batch, self.capacity = batch, capacity
        shape = [batch, self.n_local_kv, capacity, self.head_dim]
        self.k_cache = ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.v_cache = ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        flat = [1, batch * self.n_local_kv, capacity, self.head_dim]
        self._k_fill = ttnn.reshape(self.k_cache, flat)
        self._v_fill = ttnn.reshape(self.v_cache, flat)
        self.heads_mem = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=ttnn.CoreGrid(y=batch // 8, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.sdpa_decode_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.grid, q_chunk_size=0, k_chunk_size=0, exp_approx_mode=False
        )
        # fp32 accumulation in SDPA (the stub's prefill config accumulates in bf16): the reference
        # attention runs in fp32, and 50 layers of chaotic top-6 routing amplify every rounding.
        self.sdpa_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def __call__(self, hidden_states, position_ids=None, attention_mask=None, past_key_value=None, **kwargs):
        INVOCATIONS["attention"] += 1
        st = past_key_value
        if st is None:
            return super().__call__(hidden_states, position_ids, attention_mask, None, **kwargs)
        if st.mode == "prefill":
            return self._prefill(hidden_states, st)
        return self._decode(hidden_states, st)

    def _rope(self, x, cos, sin):
        """The stub's RoPE, with rotate-half run as ONE [rows, D] x [D, D] matmul over the full grid (as a batch of
        B * heads [T, D] products ttnn ran it on 16 cores, ~1.1 ms for Q at 4096 tokens). rot is 0 / +-1, so each
        output element is one input element: exact either way."""
        shape = list(x.shape)
        d = shape[-1]
        rows = x.volume() // d
        if rows >= 2048 and len(shape) == 4:
            # Prefill: every user's positions are 0 .. T-1 (the pipeline writes arange(Tp) for all), so one
            # [1, 1, T, D] cos / sin row set serves all of them, and ttnn's fused HF rotary op computes
            # x * cos + rotate_half(x) * sin in one pass (was the rotate-half matmul and three binary ops).
            T = shape[-2]
            cos1 = ttnn.slice(cos, [0, 0, 0, 0], [1, 1, T, d])
            sin1 = ttnn.slice(sin, [0, 0, 0, 0], [1, 1, T, d])
            out = ttnn.experimental.rotary_embedding(x, cos1, sin1)
            ttnn.deallocate(cos1)
            ttnn.deallocate(sin1)
            return out
        if rows >= 2048:
            g = self.device.compute_with_storage_grid_size()
            pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(g.x, g.y),
                in0_block_w=d // ttnn.TILE_SIZE,
                out_subblock_h=1,
                out_subblock_w=min(4, d // ttnn.TILE_SIZE),
                per_core_M=-(-(rows // ttnn.TILE_SIZE) // (g.x * g.y)),
                per_core_N=d // ttnn.TILE_SIZE,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=False,
            )
            rotated = ttnn.matmul(
                ttnn.reshape(x, [1, 1, rows, d]), self.rot, program_config=pc, compute_kernel_config=self.mm_cfg
            )
            rotated = ttnn.reshape(rotated, shape)
        else:
            rotated = ttnn.matmul(x, self.rot, compute_kernel_config=self.mm_cfg)
        return ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(rotated, sin))

    def _qk(self, q, k, st):
        q = ttnn.rms_norm(q, epsilon=self.eps, weight=self.q_norm_w, compute_kernel_config=self.mm_cfg)
        k = ttnn.rms_norm(k, epsilon=self.eps, weight=self.k_norm_w, compute_kernel_config=self.mm_cfg)
        if self.is_sliding:
            q = self._rope(q, st.cos, st.sin)
            k = self._rope(k, st.cos, st.sin)
        return q, k

    def _out(self, attn_heads, shape):
        if self.tp > 1:
            attn_heads = ttnn.all_gather(attn_heads, dim=3, cluster_axis=self.tp_axis, topology=ttnn.Topology.Linear)
        out = self.o_proj(attn_heads)
        return ttnn.reshape(out, shape)

    def _qkv_prefill_mm(self, m_tiles, n_tiles):
        """Full-grid blocks for the prefill QKV projection: minimal_matmul lays M along x when M > N (else along y);
        each axis gets an even ceil(tiles / cores) block in one round."""
        g = self.device.compute_with_storage_grid_size()
        m_cores, n_cores = (g.x, g.y) if m_tiles > n_tiles else (g.y, g.x)
        n_block = -(-n_tiles // n_cores)
        return ttnn.MinimalMatmulConfig(
            M_block_size=-(-m_tiles // m_cores),
            K_block_size=8,
            N_block_size=n_block,
            subblock_h=1,
            subblock_w=next(w for w in (4, 3, 2, 1) if n_block % w == 0),
            compute_with_storage_grid_size=ttnn.CoreCoord(g.x, g.y),
        )

    def _prefill(self, hidden_states, st):
        shape = list(hidden_states.shape)
        B, T, H = shape[0], shape[-2], shape[-1]
        x = ttnn.reshape(hidden_states, [B, 1, T, H])
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
        # One [B*T, H] x [H, qkv] minimal_matmul over the full grid (as a batch of B [T, H] products ttnn ran 56
        # cores, ~1.9 ms per layer at B*T = 4096).
        n = self.wqkv.shape[-1]
        xqkv = ttnn.experimental.minimal_matmul(
            ttnn.reshape(x, [1, 1, B * T, H]),
            self.wqkv,
            config=self._qkv_prefill_mm(B * T // ttnn.TILE_SIZE, n // ttnn.TILE_SIZE),
            compute_kernel_config=self.mm_cfg,
            dtype=ttnn.bfloat16,
        )
        xqkv = ttnn.reshape(xqkv, [B, 1, T, n])
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv,
            transpose_k_heads=False,
            memory_config=DRAM,
        )
        ttnn.deallocate(xqkv)
        q, k = self._qk(q, k, st)
        flat = [1, B * self.n_local_kv, T, self.head_dim]
        ttnn.fill_cache(self._k_fill, ttnn.reshape(k, flat), batch_idx=0)
        ttnn.fill_cache(self._v_fill, ttnn.reshape(v, flat), batch_idx=0)
        chunk = next(c for c in (256, 128, 64, 32) if T % c == 0 or c == 32)
        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            scale=self.scale,
            sliding_window_size=self.sliding_window,
            compute_kernel_config=self.sdpa_cfg,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.grid, q_chunk_size=chunk, k_chunk_size=chunk, exp_approx_mode=False
            ),
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=DRAM)
        return self._out(attn, shape)

    def _decode(self, hidden_states, st):
        shape = list(hidden_states.shape)  # [1, 1, B, H]
        x = hidden_states
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
        xqkv = ttnn.linear(x, self.wqkv, compute_kernel_config=self.mm_cfg)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv, num_heads=self.n_local_heads, num_kv_heads=self.n_local_kv, memory_config=self.heads_mem
        )
        ttnn.deallocate(xqkv)
        q = ttnn.to_memory_config(q, DRAM)
        k = ttnn.to_memory_config(k, DRAM)
        q, k = self._qk(q, k, st)
        k = ttnn.to_memory_config(k, self.heads_mem)
        ttnn.experimental.paged_update_cache(self.k_cache, k, update_idxs_tensor=st.cur_pos)
        ttnn.experimental.paged_update_cache(self.v_cache, v, update_idxs_tensor=st.cur_pos)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            self.k_cache,
            self.v_cache,
            cur_pos_tensor=st.cur_pos,
            scale=self.scale,
            sliding_window_size=self.sliding_window,
            program_config=self.sdpa_decode_cfg,
            compute_kernel_config=self.sdpa_cfg,
            memory_config=DRAM,
        )
        ttnn.deallocate(q)
        attn = ttnn.to_memory_config(attn, self.heads_mem)
        attn = ttnn.experimental.nlp_concat_heads_decode(attn, num_heads=self.n_local_heads)  # [1, 1, B, nq*D]
        attn = ttnn.to_memory_config(attn, DRAM)
        return self._out(attn, shape)


# ----------------------------------------------------------------------------------------------------- MoE
_KERNELS = Path(__file__).resolve().parent / "kernels"
_ROUTED_OUT = {}
_PLANES = {}  # (device, rows, hidden) -> the grouped experts' persistent fp32 partial planes


class RoutedGateUp:
    """Decode gate/up over the ROUTED experts only (tt/kernels/routed_gate_up_*.cpp, one generic_op).

    One decode step routes its 32 tokens to ~40% of a chip's 96 experts, yet the dense [32, 2560] x
    [2560, 96*512] gate and up matmuls stream every expert's columns. The kernel reads a [1, 96] mask of the
    experts any token chose, deals their weight columns round-robin over all cores, and computes only
    those, each column's K accumulated in fp32 dest at HiFi4 (the dense op's math). Outputs land in
    persistent [32, 96*512] buffers shared by every layer: a column it skips keeps whatever finite value an
    earlier call left, and its routing weight is exactly 0, so it contributes nothing downstream."""

    def __init__(self, device, w_gate, w_up, n_experts: int, inter: int, interleaved: bool = False) -> None:
        """interleaved: w_gate is w_up, one tile-pair-interleaved [K, 2*N] weight (gate column c at 2c, up at
        2c + 1); else separate [K, N] gate and up tensors."""
        self.w_gate, self.w_up = w_gate, w_up
        self.col_stride, self.up_offset = (2, 1) if interleaved else (1, 0)
        g = device.compute_with_storage_grid_size()
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
        self.core_xy = [(x, y) for y in range(g.y) for x in range(g.x)]
        self.kt, self.row_tiles = w_gate.shape[-2] // ttnn.TILE_SIZE, w_gate.shape[-1] // ttnn.TILE_SIZE
        self.n_experts, self.cpe = n_experts, inter // ttnn.TILE_SIZE
        self.nt = n_experts * self.cpe  # output columns
        self.kb = next(b for b in (16, 10, 8, 5, 4, 2, 1) if self.kt % b == 0)
        self.max_slots = -(-self.nt // len(self.core_xy))
        key = (id(device), self.nt)
        if key not in _ROUTED_OUT:
            shape = [1, 1, ttnn.TILE_SIZE, self.nt * ttnn.TILE_SIZE]
            _ROUTED_OUT[key] = tuple(
                ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) for _ in range(2)
            )
        self.g_out, self.u_out = _ROUTED_OUT[key]

    @staticmethod
    def _cb(index, fmt, page, pages, cores):
        return ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
        )

    def __call__(self, x, mask):
        """x [1, 1, 32, K] bf16 tile, mask [1, 1, 1, n_experts] fp32 row-major (non-zero = routed) ->
        the persistent (gate, up) [1, 1, 32, N] bf16 outputs, routed columns written."""
        x_tile, w_tile = 2048, 1088  # bf16 / bfp8_b tile bytes
        cores = self.cores
        cbs = [
            self._cb(0, ttnn.bfloat16, x_tile, self.kt, cores),
            self._cb(1, ttnn.bfloat8_b, w_tile, 2 * self.kb, cores),
            self._cb(2, ttnn.bfloat8_b, w_tile, 2 * self.kb, cores),
            self._cb(3, ttnn.float32, 512, 1, cores),
            self._cb(4, ttnn.uint32, 16, 1, cores),
            self._cb(5, ttnn.uint32, 16 * (-(-4 * (1 + self.max_slots) // 16)), 1, cores),
            self._cb(16, ttnn.bfloat16, x_tile, 2, cores),
            self._cb(17, ttnn.bfloat16, x_tile, 2, cores),
        ]
        acc = lambda *ts: [a for t in ts for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        addrs = [t.buffer_address() for t in (x, self.w_gate, self.w_up, mask)]
        for i, (cx, cy) in enumerate(self.core_xy):
            reader_rt[cx][cy] = addrs + [i]
            writer_rt[cx][cy] = [self.g_out.buffer_address(), self.u_out.buffer_address()]
        file = ttnn.KernelDescriptor.SourceType.FILE_PATH
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(_KERNELS / "routed_gate_up_reader.cpp"),
                source_type=file,
                core_ranges=cores,
                compile_time_args=[
                    self.kt,
                    self.row_tiles,
                    self.n_experts,
                    self.cpe,
                    len(self.core_xy),
                    self.kb,
                    self.max_slots,
                    x_tile,
                    w_tile,
                    self.col_stride,
                    self.up_offset,
                ]
                + acc(x, self.w_gate, self.w_up, mask),
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(_KERNELS / "routed_gate_up_writer.cpp"),
                source_type=file,
                core_ranges=cores,
                compile_time_args=acc(self.g_out, self.u_out),
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(_KERNELS / "routed_gate_up_compute.cpp"),
                source_type=file,
                core_ranges=cores,
                compile_time_args=[self.kt, self.kb],
                runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
                ),
            ),
        ]
        ttnn.generic_op(
            [x, self.w_gate, self.w_up, mask, self.g_out, self.u_out],
            ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs),
        )
        return self.g_out, self.u_out


class GroupedExperts:
    """Prefill routed experts over the tokens each expert actually got (tt/kernels/moe_grouped_*.cpp, then
    tt/kernels/moe_combine_*.cpp: two generic_ops).

    The dense prefill path ran all of a chip's 96 experts on every token -- [M, 2560] x [2560, 96*1024] gate/up
    and [M, 96*512] x [96*512, 2560] down -- while a token uses 6 of 384 experts, ~1.5 of a chip's 96. Here the
    work is cut into units of 128 tokens routed to one expert, dealt round-robin over all cores (routing is
    skewed: one expert can get half the tokens). A unit gathers its tokens' rows and runs gate/up (fp32 dest,
    HiFi4), silu(gate) * up * the fp32 routing weight (all in fp32 dest) and the down projection on those rows
    only, streaming the expert's weights once. Each output row lands in plane k of an fp32 [K, M, hidden]
    buffer, k = the expert's rank among the token's routed local experts; the combine sums a token's planes in
    fp32 and packs bf16 once. Nothing between x and the output is rounded to bf16 (the dense path rounded the
    SwiGLU output, the scaled activation and the routing weight)."""

    MAX_ROWS = 4096  # tokens per pass: bounds the fp32 plane buffer (top_k x MAX_ROWS x hidden)

    def __init__(self, device, w_gu, w_down, n_experts: int, inter: int, top_k: int) -> None:
        """w_gu: tile-pair-interleaved [hidden, 2 * n_experts * inter + pad] gate/up (gate column c at 2c, up at
        2c + 1; the row stride is the tensor's width); w_down: [n_experts * inter, hidden]."""
        self.device, self.w_gu, self.w_down = device, w_gu, w_down
        g = device.compute_with_storage_grid_size()
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
        self.core_xy = [(x, y) for y in range(g.y) for x in range(g.x)]
        self.n_experts, self.cpe = n_experts, inter // ttnn.TILE_SIZE
        self.kt = w_down.shape[-1] // ttnn.TILE_SIZE  # hidden tiles
        self.nt_gu = w_gu.shape[-1] // ttnn.TILE_SIZE
        self.kmax = min(top_k, n_experts)
        self.kb = next(b for b in (8, 5, 4, 2, 1) if self.kt % b == 0)
        self.rb = 4  # 32-token row blocks per unit: 128 tokens per pass over an expert's weights
        # 2 * rb fp32 dest tiles (gate, up per row block) need full-sync dest.
        self.mm_cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[3] = ttnn.UnpackToDestMode.UnpackToDestFp32  # the routing weights, multiplied in fp32 dest
        self.mm_cfg.unpack_to_dest_mode = modes
        self.sum_cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
        )
        self.count_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
        )
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[0] = ttnn.UnpackToDestMode.UnpackToDestFp32  # the fp32 partial planes, summed exactly
        modes[5] = ttnn.UnpackToDestMode.UnpackToDestFp32  # their 0/1 row masks
        self.sum_cfg.unpack_to_dest_mode = modes

    def __call__(self, x, w):
        """x [1, 1, M, hidden] bf16 tile, w [1, 1, M, n_experts] fp32 tile (this chip's routing weights, 0 where
        unrouted) -> this chip's routed-expert sum [1, 1, M, hidden] bf16."""
        m, hidden = x.shape[-2], x.shape[-1]
        if m > self.MAX_ROWS:
            parts = []
            for a in range(0, m, self.MAX_ROWS):
                b = min(a + self.MAX_ROWS, m)
                xs = ttnn.slice(x, [0, 0, a, 0], [1, 1, b, hidden])
                ws = ttnn.slice(w, [0, 0, a, 0], [1, 1, b, w.shape[-1]])
                parts.append(self(xs, ws))
                ttnn.deallocate(xs)
                ttnn.deallocate(ws)
            out = ttnn.concat(parts, dim=2)
            for p in parts:
                ttnn.deallocate(p)
            return out
        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        wt = ttnn.transpose(w, 2, 3)
        wt_rm = ttnn.to_layout(wt, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(wt)
        w_rm = ttnn.to_layout(w, ttnn.ROW_MAJOR_LAYOUT)
        # Tokens routed to each expert (exact: 0/1 summed in fp32 on the SFPU path); the reader deals units by it.
        n_e = ttnn.sum(ttnn.sign(w), dim=2, keepdim=True, compute_kernel_config=self.count_cfg)
        counts = ttnn.to_layout(n_e, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(n_e)
        # The fp32 partial planes persist across calls, zeroed once: a row a plane does not receive this call keeps
        # an earlier call's finite value, which the combine multiplies by its 0 mask.
        key = (id(self.device), m, hidden)
        if key not in _PLANES:
            _PLANES[key] = ttnn.zeros(
                [1, self.kmax, m, hidden], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device
            )
        y = _PLANES[key]
        self._experts(x_rm, wt_rm, w_rm, counts, y, m)
        out = ttnn.empty(
            [1, 1, m, hidden], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=DRAM
        )
        self._combine(y, w_rm, out, m)
        for t in (x_rm, wt_rm, w_rm, counts):
            ttnn.deallocate(t)
        return out

    @staticmethod
    def _acc(*ts):
        return [a for t in ts for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]

    def _kernel(self, name, ct, rt, config):
        return ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.cores,
            compile_time_args=ct,
            runtime_args=rt,
            config=config,
        )

    def _experts(self, x_rm, wt_rm, w_rm, counts, y, m):
        cb, c = RoutedGateUp._cb, self.cores
        bf, w_tile, f32 = 2048, 1088, 4096
        kt, cpe, kb = self.kt, self.cpe, self.kb
        row_bytes, wrow = x_rm.shape[-1] * 2, w_rm.shape[-1] * 4
        r32 = lambda n: -(-n // 32) * 32
        nc = len(self.core_xy)
        rb, unit = self.rb, 32 * self.rb
        max_units = -(-(-(-m * self.kmax // unit) + self.n_experts) // nc)  # all units, worst case, dealt over nc
        cbs = [
            cb(0, ttnn.bfloat16, bf, kt, c),  # 32 gathered row-major x rows (tilize input)
            cb(1, ttnn.bfloat8_b, w_tile, 2 * kb, c),  # gate weight tiles
            cb(2, ttnn.bfloat8_b, w_tile, 2 * kb, c),  # up weight tiles
            cb(3, ttnn.float32, f32, rb, c),  # routing-weight tiles (row j = token j's weight)
            cb(4, ttnn.uint32, 16, 1, c),  # unit count -> compute
            cb(5, ttnn.uint32, r32(4 * (2 + unit)), 3, c),  # unit count, then each unit's [e, n, tokens] -> writer
            cb(6, ttnn.float32, r32(4 * max(m, self.n_experts)), 1, c),  # counts, then an expert's weight row
            cb(7, ttnn.bfloat8_b, w_tile, 2 * cpe, c),  # down weight tiles
            cb(8, ttnn.bfloat16, bf, rb * kt, c),  # tilized x, rb row blocks
            cb(9, ttnn.float32, wrow * unit, 1, c),  # weight rows of a unit's tokens (writer: ranks)
            cb(11, ttnn.uint32, 16, 2, c),  # row blocks holding tokens, per unit -> compute
            cb(17, ttnn.float32, f32, rb * cpe, c),  # act = silu(gate) * up * routing weight
            cb(18, ttnn.float32, f32, 2 * rb, c),  # expert output tiles
        ]
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        addrs = [t.buffer_address() for t in (x_rm, self.w_gu, self.w_down, wt_rm, counts)]
        for i, (cx, cy) in enumerate(self.core_xy):
            reader_rt[cx][cy] = addrs + [i]
            writer_rt[cx][cy] = [w_rm.buffer_address(), y.buffer_address(), i]
        kernels = [
            self._kernel(
                "moe_grouped_reader.cpp",
                [kt, self.nt_gu, cpe, m, kb, self.n_experts, w_tile, row_bytes, nc, max_units, rb]
                + self._acc(x_rm, self.w_gu, self.w_down, wt_rm, counts),
                reader_rt,
                ttnn.ReaderConfigDescriptor(),
            ),
            self._kernel(
                "moe_grouped_writer.cpp",
                [kt, m, wrow, rb] + self._acc(w_rm, y) + [nc],
                writer_rt,
                ttnn.WriterConfigDescriptor(),
            ),
            self._kernel("moe_grouped_compute.cpp", [kt, kb, cpe, rb], [], self.mm_cfg),
        ]
        ttnn.generic_op(
            [x_rm, self.w_gu, self.w_down, wt_rm, counts, w_rm, y],
            ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs),
        )

    def _combine(self, y, w_rm, out, m):
        cb, c = RoutedGateUp._cb, self.cores
        kt, mt, nc, wrow = self.kt, m // ttnn.TILE_SIZE, len(self.core_xy), w_rm.shape[-1] * 4
        cw = next(w for w in (8, 5, 4, 2, 1) if kt % w == 0)  # output tiles per (row, column-chunk) unit
        cbs = [
            cb(0, ttnn.float32, 4096, 2 * self.kmax * cw, c),  # partial planes of a unit's cw output tiles
            cb(4, ttnn.uint32, 16, 2, c),  # planes and all-rows-valid bits per unit -> compute
            cb(5, ttnn.float32, 4096, 2 * self.kmax, c),  # a 0/1 row mask per plane
            cb(9, ttnn.float32, wrow * 32, 1, c),  # weight rows of the tile row's 32 tokens
            cb(16, ttnn.bfloat16, 2048, 2 * cw, c),  # summed output tiles
        ]
        reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, (cx, cy) in enumerate(self.core_xy):
            reader_rt[cx][cy] = [w_rm.buffer_address(), y.buffer_address(), i]
            compute_rt[cx][cy] = [i]
            writer_rt[cx][cy] = [out.buffer_address(), i]
        kernels = [
            self._kernel(
                "moe_combine_reader.cpp",
                [kt, mt, self.n_experts, nc, wrow, self.kmax, cw] + self._acc(w_rm, y),
                reader_rt,
                ttnn.ReaderConfigDescriptor(),
            ),
            self._kernel(
                "moe_combine_writer.cpp", [kt, mt, nc, cw] + self._acc(out), writer_rt, ttnn.WriterConfigDescriptor()
            ),
            self._kernel("moe_combine_compute.cpp", [kt, mt, nc, self.kmax, cw], compute_rt, self.sum_cfg),
        ]
        ttnn.generic_op([y, w_rm, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))


class MoE(TtKolibri1SparseMoeBlock):
    """The graduated sparse MoE block with its router logits from the graduated router stub and its
    ungated shared expert as the graduated m_l_p stub (the routed experts are the block's own)."""

    def __init__(self, device, moe) -> None:
        self._held_gate = None
        super().__init__(device, moe)
        # The stub's _upload calls (see _upload below) left gate and up as ONE tile-pair-interleaved weight.
        self.w_gu, self.w_up = self.w_up, None
        # Expert matmuls at HiFi4 (the stub uses HiFi2, which drops activation mantissa bits).
        self.hifi2 = self.hifi4
        self.router = Router(device, moe.gate)
        self.shared = SharedExpert(device, moe.shared_experts)
        for name in ("ws_gate", "ws_up", "ws_down", "w_router"):  # replaced by the two stubs above
            ttnn.deallocate(getattr(self, name))
            setattr(self, name, None)
        # Down projection, decode: one tile row of tokens against [n_local*inter, hidden] is a pure weight
        # stream split over N (80 tiles -> 80 cores, each streaming one weight column, which sits in a
        # single DRAM bank). ttnn's default walks K in blocks of 2 tiles, so the time goes to 768
        # multicast/sync rounds rather than to bytes; a wide K block streams the same column in few rounds.
        g = device.compute_with_storage_grid_size()
        k_tiles = self.w_down.shape[-2] // ttnn.TILE_SIZE
        self.down_decode_pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(g.x, g.y),
            in0_block_w=next(w for w in (16, 8, 4, 2, 1) if k_tiles % w == 0),
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        self.routed_gate_up = RoutedGateUp(device, self.w_gu, self.w_gu, self.n_local, self.inter, interleaved=True)
        self.grouped = GroupedExperts(device, self.w_gu, self.w_down, self.n_local, self.inter, self.top_k)

    def _upload(self, t, shard_dim, dtype):
        """The stub uploads the routed gate, then up, each [hidden, n_exp * inter]. Hold the gate and upload
        both as one weight whose 32-column tiles alternate gate, up ([g0 u0 g1 u1 ...]): the layout
        minimal_matmul_split's fused SwiGLU reads, and still one contiguous expert range per TP shard."""
        routed = shard_dim == -1 and dtype != ttnn.bfloat16 and t.shape[-1] == self.n_local * self.tp * self.inter
        if not routed:
            return super()._upload(t, shard_dim, dtype)
        if self._held_gate is None:
            self._held_gate = t
            return None
        g, u, self._held_gate = self._held_gate, t, None
        k, n = g.shape
        gu = torch.stack([g.reshape(k, n // 32, 32), u.reshape(k, n // 32, 32)], dim=2).reshape(k, 2 * n)
        # One zero tile column after each chip's shard. A shard row of 2 * n_local * inter / 32 tiles is a
        # multiple of the 8 DRAM banks, so all K tiles of a weight column sat in one bank, and every core
        # streaming the same column position (of any expert) queued on the same two banks; with the pad,
        # consecutive K tiles of a column fall in consecutive banks. The kernels take the row stride from the
        # tensor's width and never read the pad.
        gu = torch.nn.functional.pad(gu.reshape(k, self.tp, -1), (0, ttnn.TILE_SIZE)).reshape(k, -1)
        return super()._upload(gu, shard_dim, dtype)

    def _routed_mask(self, weights):
        """[1, 1, 32, n_exp] routing weights (replicated) -> this chip's [1, 1, 1, n_local] row-major mask, non-zero
        where any token routed to the expert."""
        w = ttnn.mesh_partition(weights, dim=3, cluster_axis=self.tp_axis) if self.tp > 1 else weights
        return ttnn.to_layout(ttnn.max(w, dim=2, keepdim=True), ttnn.ROW_MAJOR_LAYOUT)

    def routing_weights(self, x):
        logits = self.router(x)
        scores = ttnn.add(logits, self.router_bias)
        top_vals, top_ids = ttnn.topk(scores, k=self.top_k, dim=-1, largest=True, sorted=True)
        ttnn.deallocate(top_ids)
        n = scores.shape[-2]
        kth = ttnn.slice(top_vals, [0, 0, 0, self.top_k - 1], [1, 1, n, self.top_k])
        chosen = ttnn.ge(scores, kth)
        weights = ttnn.multiply(ttnn.sigmoid(logits), chosen)
        if self.norm_topk_prob:
            weights = ttnn.divide(weights, ttnn.add(ttnn.sum(weights, dim=-1, keepdim=True), 1e-20))
        return weights

    # Route on the fp32 normed hidden state rather than its bf16 rounding: the top-6-of-384 choice flips on
    # near-ties, and measured on the card prompt (50 layers) this cut the last-position sampling-distribution
    # gap to the fp32 reference from TV 0.012 to 0.002.
    router_fp32_input = True

    def __call__(self, hidden_states, **kwargs):
        INVOCATIONS["sparse_moe_block"] += 1
        shape = list(hidden_states.shape)
        x_in = ttnn.reshape(hidden_states, [1, 1, -1, shape[-1]])
        x = x_in if x_in.dtype == ttnn.bfloat16 else ttnn.typecast(x_in, ttnn.bfloat16)
        weights = self.routing_weights(x_in if self.router_fp32_input else x)
        if x.shape[-2] == ttnn.TILE_SIZE:  # decode: only the routed experts' columns (persistent outputs, not freed)
            col_scale = ttnn.linear(weights, self.expand, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi4)
            mask = self._routed_mask(weights)
            g, u = self.routed_gate_up(x, mask)
            ttnn.deallocate(mask)
            act = ttnn.multiply(ttnn.multiply(ttnn.silu(g), u), col_scale)
            ttnn.deallocate(col_scale)
            out = ttnn.linear(act, self.w_down, program_config=self.down_decode_pc, compute_kernel_config=self.hifi2)
            ttnn.deallocate(act)
        else:  # prefill: each expert over the tokens routed to it
            w = ttnn.mesh_partition(weights, dim=3, cluster_axis=self.tp_axis) if self.tp > 1 else weights
            out = self.grouped(x, w)
            if w is not weights:
                ttnn.deallocate(w)
        ttnn.deallocate(weights)
        # The routed and the shared expert are both row-parallel over the chips: add their partial sums here and
        # reduce once (the shared stub's own all_reduce made it two collectives per layer).
        shared = self.shared.partial(x)
        total = ttnn.add(out, shared, dtype=ttnn.bfloat16)
        ttnn.deallocate(out)
        ttnn.deallocate(shared)
        if self.tp > 1:
            total = ttnn.all_reduce(total, cluster_axis=self.tp_axis, topology=ttnn.Topology.Linear)
        return ttnn.reshape(total, shape)


# --------------------------------------------------------------------------------------------- the layer
class DecoderLayer(TtKolibri1DecoderLayer):
    """The graduated decoder layer (its __call__ -- sandwich norms and residual -- is used unchanged)
    over the cache-carrying Attention and the router/shared-expert MoE above. Same fields as the stub's
    own __init__, which would build its sub-blocks from the plain stub classes instead."""

    def __init__(self, device, torch_module, batch: int, capacity: int) -> None:
        self.device = device
        self.attn = Attention(device, torch_module.self_attn, batch, capacity)
        self.moe = MoE(device, torch_module.mlp)
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

    def __call__(self, hidden_states, position_ids=None, attention_mask=None, past_key_value=None, **kwargs):
        """The stub's sandwich-norm residual, except the two post-sublayer norms run on the sublayers' bf16
        outputs and add into the fp32 residual (the stub cast each output to fp32 first: a typecast plus twice
        the norm's traffic, for the same input values; only the norm's output is now bf16)."""
        INVOCATIONS["decoder_layer"] += 1
        h = hidden_states
        # The attention reads its input as bf16: normalize a bf16 copy of the residual (typecast + bf16 norm, 105 MB
        # at 4096 rows) instead of an fp32 norm whose output the attention then typecasts (147 MB).
        L1 = ttnn.L1_MEMORY_CONFIG  # the residual adds read their bf16 operand from L1 (~190 KB a core at 4096 rows)
        hb = ttnn.typecast(h, ttnn.bfloat16, memory_config=L1)  # the norm's bf16 input, too, only lives across it
        x = ttnn.rms_norm(
            hb, epsilon=self.eps, weight=self.norm_w["input_layernorm"], memory_config=DRAM, compute_kernel_config=self.norm_cfg
        )
        ttnn.deallocate(hb)
        a = self.attn(x, position_ids, attention_mask, past_key_value)
        n = ttnn.rms_norm(
            a, epsilon=self.eps, weight=self.norm_w["post_attn_norm"], memory_config=L1, compute_kernel_config=self.norm_cfg
        )
        ttnn.deallocate(a)
        h = ttnn.add(h, n, dtype=h.dtype, memory_config=DRAM)
        ttnn.deallocate(n)
        m = self.moe(self._norm(h, "post_attention_layernorm"))
        n = ttnn.rms_norm(
            m, epsilon=self.eps, weight=self.norm_w["post_ffn_norm"], memory_config=L1, compute_kernel_config=self.norm_cfg
        )
        ttnn.deallocate(m)
        out = ttnn.add(h, n, dtype=h.dtype, memory_config=DRAM)
        ttnn.deallocate(n)
        ttnn.deallocate(h)
        return out


# ----------------------------------------------------------------------------------------------- sampler
def _topk_tree(n_tiles: int, k_tiles: int, cores: int):
    """Group counts, stage by stage, of a top-k tree over a row n_tiles tiles wide (None: no tree).

    ttnn's top-k for k > 64 sorts each tile row on one core, so one [B, V] row of logits is ~17 us per
    tile on a single core (69 ms at V = 128000). Each stage splits the row into g groups of w tiles,
    one tile row per group (g <= cores, so every group gets its own core); g * k_tiles tiles survive
    into the next stage. A row of at most 20 tiles is left to one final top-k."""
    plan, cap = [], 40
    while n_tiles > 20:
        w = next((w for w in range(min(cap, n_tiles - 1), 2 * k_tiles - 1, -1) if n_tiles % w == 0), None)
        if w is None or n_tiles // w > cores:
            return None
        plan.append(n_tiles // w)
        n_tiles, cap = (n_tiles // w) * k_tiles, 10
    return plan or None


class Sampler:
    """generation_config's rule on device, fp32 throughout: top_k on logits/T, top_p over those k
    (exclusive cumulative probability < p keeps a token), then inverse CDF in TOKEN-ID order over the
    kept set against the step's uniform. Mirrors tests/e2e/golden.py::host_sample op for op.

    The top-k runs as a tree (_topk_tree). A [B, N] tile tensor and a [B*g, N/g] one have the same DRAM
    pages in the same order (page = tile row * row tiles + tile col, and B is whole tile rows), so each
    regrouping is a metadata view -- no data moves -- and every stage carries the token ids along as
    top-k labels. The inverse CDF then needs only the <= k kept tokens: sorted by token id and
    cumulated in that order, which is the full-vocabulary cumsum with its zero terms dropped (the same
    non-zero terms, added in the same order)."""

    def __init__(self, device, batch: int, top_k: int, top_p: float, temperature: float, vocab: int = 0) -> None:
        self.device = device
        self.top_k, self.top_p, self.temperature = int(top_k), float(top_p), float(temperature)
        self.k_pad = -(-self.top_k // 32) * 32
        mapper = ttnn.ReplicateTensorToMesh(device) if isinstance(device, ttnn.MeshDevice) else None
        self.ranks = ttnn.from_torch(
            torch.arange(self.top_k, dtype=torch.float32).expand(1, 1, batch, -1).contiguous(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=mapper,
        )
        # top_p count kernel (tt/kernels/topp_count.cpp): one core per user row.
        g = device.compute_with_storage_grid_size()
        self.count_xy = [(i % g.x, i // g.x) for i in range(batch)]
        self.count_cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in self.count_xy]
        )
        self.top_p_bits = struct.unpack("<I", struct.pack("<f", self.top_p))[0]
        grid = device.compute_with_storage_grid_size()
        tree_ok = vocab > 0 and batch % 32 == 0 and vocab % 32 == 0
        self.plan = _topk_tree(vocab // 32, self.k_pad // 32, grid.x * grid.y) if tree_ok else None
        self.token_index = None
        if self.plan:
            vocab_index = torch.arange(vocab, dtype=torch.int32).expand(1, 1, batch, -1).contiguous()
            self.token_index = ttnn.from_torch(
                vocab_index, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mapper
            )

    def _merge(self, x, labels, groups):
        """x, labels [1, 1, B, groups * k] (each group's top k, largest first) -> the row's top k as a groups-way
        merge, one core per user row (tt/kernels/topk_merge.cpp)."""
        B, k = x.shape[-2], self.top_k
        vals = ttnn.empty([1, 1, B, k], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        ids = ttnn.empty([1, 1, B, k], dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=self.device)
        rt = ttnn.RuntimeArgs()
        for i, (cx, cy) in enumerate(self.count_xy):
            rt[cx][cy] = [t.buffer_address() for t in (x, labels, vals, ids)] + [i]
        kernel = ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "topk_merge.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.count_cores,
            compile_time_args=[groups, k]
            + [a for t in (x, labels, vals, ids) for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()],
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        scratch = 64 + 2 * groups * k * 4 + 2 * k * 4 + 2 * (-(-groups * 4 // 64) * 64)
        cb = RoutedGateUp._cb(0, ttnn.uint32, -(-scratch // 32) * 32, 1, self.count_cores)
        ttnn.generic_op([x, labels, vals, ids], ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb]))
        return vals, ids

    def _top_k(self, x):
        """x [1, 1, B, V] -> (values [1, 1, B, k] largest first, their token ids [1, 1, B, k] uint32)."""
        if not self.plan:
            return ttnn.topk(x, k=self.top_k, dim=-1, largest=True, sorted=True)
        B, n = x.shape[-2], x.shape[-1]
        labels = self.token_index
        if self.k_pad == self.top_k:  # first stage on ttnn, then one merge of its sorted group lists
            g = self.plan[0]
            grouped = [1, 1, B * g, n // g]
            x, labels = ttnn.topk(
                ttnn.experimental.view(x, grouped),
                k=self.k_pad,
                dim=-1,
                largest=True,
                sorted=True,
                indices_tensor=ttnn.experimental.view(labels, grouped),
            )
            flat = [1, 1, B, g * self.k_pad]
            return self._merge(ttnn.experimental.view(x, flat), ttnn.experimental.view(labels, flat), g)
        for g in self.plan:
            grouped = [1, 1, B * g, n // g]
            x, labels = ttnn.topk(
                ttnn.experimental.view(x, grouped),
                k=self.k_pad,
                dim=-1,
                largest=True,
                sorted=True,
                indices_tensor=ttnn.experimental.view(labels, grouped),
            )
            n = g * self.k_pad
            x, labels = ttnn.experimental.view(x, [1, 1, B, n]), ttnn.experimental.view(labels, [1, 1, B, n])
        return ttnn.topk(x, k=self.top_k, dim=-1, largest=True, sorted=True, indices_tensor=labels)

    def _n_keep(self, p):
        """p [1, 1, B, k] fp32 (descending) -> n_keep [1, 1, B, 1] fp32: how many leading candidates have an
        exclusive cumulative probability below top_p, the sums taken one by one in fp32 (tt/kernels/topp_count.cpp)."""
        B = p.shape[-2]
        out = ttnn.empty([1, 1, B, 1], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        rt = ttnn.RuntimeArgs()
        for i, (cx, cy) in enumerate(self.count_xy):
            rt[cx][cy] = [p.buffer_address(), out.buffer_address(), i]
        kernel = ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "topp_count.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.count_cores,
            compile_time_args=[self.top_k // ttnn.TILE_SIZE, self.top_p_bits]
            + [a for t in (p, out) for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()],
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        cb = RoutedGateUp._cb(0, ttnn.float32, 64 + (self.top_k // ttnn.TILE_SIZE) * 128 + 64, 1, self.count_cores)
        ttnn.generic_op([p, out], ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb]))
        return out

    def _at(self, q, u):
        """q [1, 1, B, k] fp32 (kept weights in token-id order), u [1, 1, B, 1] -> at [1, 1, B, 1] fp32: how many
        positions have an inclusive cumulative sum <= u * z, z the total, the sums taken one by one in fp32
        (tt/kernels/sample_at.cpp; host_sample's cumsum is fp64, and the triangle-matmul scan this replaces was off
        by up to 2.4e-4, which put u on the wrong side of a CDF edge on some trajectories)."""
        B = q.shape[-2]
        out = ttnn.empty([1, 1, B, 1], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        rt = ttnn.RuntimeArgs()
        for i, (cx, cy) in enumerate(self.count_xy):
            rt[cx][cy] = [q.buffer_address(), u.buffer_address(), out.buffer_address(), i]
        kernel = ttnn.KernelDescriptor(
            kernel_source=str(_KERNELS / "sample_at.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.count_cores,
            compile_time_args=[self.top_k // ttnn.TILE_SIZE]
            + [a for t in (q, u, out) for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()],
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        row_bytes = (self.top_k // ttnn.TILE_SIZE) * 128  # the row's two 64 B face rows per tile
        cb = RoutedGateUp._cb(0, ttnn.float32, 64 + row_bytes + 64 + 64, 1, self.count_cores)
        ttnn.generic_op([q, u, out], ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb]))
        return out

    def __call__(self, logits, u):
        """logits [1, 1, B, V] fp32, u [1, 1, B, 1] fp32 -> token ids [1, 1, B, 1] fp32 (exact integers)."""
        x = logits if self.temperature == 1.0 else ttnn.multiply(logits, 1.0 / self.temperature)
        vals, tok_ids = self._top_k(x)
        B = vals.shape[-2]
        vmax = ttnn.slice(vals, [0, 0, 0, 0], [1, 1, B, 1])
        e = ttnn.exp(ttnn.subtract(vals, vmax))
        p = ttnn.divide(e, ttnn.sum(e, dim=-1, keepdim=True))
        # The top_p cut from exact exclusive sums (the triangle-matmul scan was off by 3.4e-4 median / 8.6e-4 max here,
        # enough to flip the 0.97 cut against the host rule: e2e device sampler != host sampler at sample 4, step 100).
        n_keep = self._n_keep(p)
        last = ttnn.eq(self.ranks, ttnn.subtract(ttnn.maximum(n_keep, 1.0), 1.0))
        cut = ttnn.sum(ttnn.multiply(vals, last), dim=-1, keepdim=True)
        q = ttnn.multiply(e, ttnn.ge(vals, cut))
        # The k candidates in token-id order: largest-first on -id is ascending id; q follows by gather.
        neg_id, order = ttnn.topk(
            ttnn.neg(ttnn.typecast(tok_ids, ttnn.float32)), k=self.top_k, dim=-1, largest=True, sorted=True
        )
        # Positions whose CDF is <= u*z precede the sampled token (the CDF only rises at kept tokens).
        at = self._at(ttnn.gather(q, -1, order), u)
        return ttnn.neg(ttnn.sum(ttnn.multiply(neg_id, ttnn.eq(self.ranks, at)), dim=-1, keepdim=True))

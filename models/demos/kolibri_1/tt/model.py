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

from collections import Counter

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
        INVOCATIONS["f_p8_linear"] += 1
        return super().__call__(x, **kwargs)


class Router(TtKolibri1Router):
    def __call__(self, x, **kwargs):
        INVOCATIONS["router"] += 1
        return super().__call__(x, **kwargs)


class SharedExpert(TtKolibri1MLP):
    def __call__(self, x, **kwargs):
        INVOCATIONS["m_l_p"] += 1
        return super().__call__(x, **kwargs)


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

    def _prefill(self, hidden_states, st):
        shape = list(hidden_states.shape)
        B, T, H = shape[0], shape[-2], shape[-1]
        x = ttnn.reshape(hidden_states, [B, 1, T, H])
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
        xqkv = ttnn.linear(x, self.wqkv, compute_kernel_config=self.mm_cfg)
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
class MoE(TtKolibri1SparseMoeBlock):
    """The graduated sparse MoE block with its router logits from the graduated router stub and its
    ungated shared expert as the graduated m_l_p stub (the routed experts are the block's own)."""

    def __init__(self, device, moe) -> None:
        super().__init__(device, moe)
        # Expert matmuls at HiFi4 (the stub uses HiFi2, which drops activation mantissa bits).
        self.hifi2 = self.hifi4
        self.router = Router(device, moe.gate)
        self.shared = SharedExpert(device, moe.shared_experts)
        for name in ("ws_gate", "ws_up", "ws_down", "w_router"):  # replaced by the two stubs above
            ttnn.deallocate(getattr(self, name))
            setattr(self, name, None)

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
        col_scale = ttnn.linear(weights, self.expand, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi4)
        ttnn.deallocate(weights)
        g = ttnn.linear(x, self.w_gate, compute_kernel_config=self.hifi2)
        u = ttnn.linear(x, self.w_up, compute_kernel_config=self.hifi2)
        act = ttnn.multiply(ttnn.multiply(ttnn.silu(g), u), col_scale)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        ttnn.deallocate(col_scale)
        out = ttnn.linear(act, self.w_down, compute_kernel_config=self.hifi2)
        ttnn.deallocate(act)
        if self.tp > 1:
            out = ttnn.all_reduce(out, cluster_axis=self.tp_axis, topology=ttnn.Topology.Linear)
        out = ttnn.add(out, self.shared(x))
        return ttnn.reshape(out, shape)


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
        INVOCATIONS["decoder_layer"] += 1
        return super().__call__(hidden_states, position_ids, attention_mask, past_key_value)


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
        n_tiles, cap = (n_tiles // w) * k_tiles, 20
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
        grid = device.compute_with_storage_grid_size()
        tree_ok = vocab > 0 and batch % 32 == 0 and vocab % 32 == 0
        self.plan = _topk_tree(vocab // 32, self.k_pad // 32, grid.x * grid.y) if tree_ok else None
        self.token_index = None
        if self.plan:
            vocab_index = torch.arange(vocab, dtype=torch.int32).expand(1, 1, batch, -1).contiguous()
            self.token_index = ttnn.from_torch(
                vocab_index, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mapper
            )

    def _top_k(self, x):
        """x [1, 1, B, V] -> (values [1, 1, B, k] largest first, their token ids [1, 1, B, k] uint32)."""
        if not self.plan:
            return ttnn.topk(x, k=self.top_k, dim=-1, largest=True, sorted=True)
        B, n = x.shape[-2], x.shape[-1]
        labels = self.token_index
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

    def __call__(self, logits, u):
        """logits [1, 1, B, V] fp32, u [1, 1, B, 1] fp32 -> token ids [1, 1, B, 1] fp32 (exact integers)."""
        x = logits if self.temperature == 1.0 else ttnn.multiply(logits, 1.0 / self.temperature)
        vals, tok_ids = self._top_k(x)
        B = vals.shape[-2]
        vmax = ttnn.slice(vals, [0, 0, 0, 0], [1, 1, B, 1])
        e = ttnn.exp(ttnn.subtract(vals, vmax))
        p = ttnn.divide(e, ttnn.sum(e, dim=-1, keepdim=True))
        excl = ttnn.subtract(ttnn.cumsum(p, dim=-1), p)
        n_keep = ttnn.sum(ttnn.lt(excl, self.top_p), dim=-1, keepdim=True)
        last = ttnn.eq(self.ranks, ttnn.subtract(ttnn.maximum(n_keep, 1.0), 1.0))
        cut = ttnn.sum(ttnn.multiply(vals, last), dim=-1, keepdim=True)
        q = ttnn.multiply(e, ttnn.ge(vals, cut))
        # The k candidates in token-id order: largest-first on -id is ascending id; q follows by gather.
        neg_id, order = ttnn.topk(
            ttnn.neg(ttnn.typecast(tok_ids, ttnn.float32)), k=self.top_k, dim=-1, largest=True, sorted=True
        )
        cdf = ttnn.cumsum(ttnn.gather(q, -1, order), dim=-1)
        z = ttnn.slice(cdf, [0, 0, 0, self.top_k - 1], [1, 1, B, self.top_k])
        # Positions whose CDF is <= u*z precede the sampled token (the CDF only rises at kept tokens).
        at = ttnn.sum(ttnn.le(cdf, ttnn.multiply(u, z)), dim=-1, keepdim=True)
        return ttnn.neg(ttnn.sum(ttnn.multiply(neg_id, ttnn.eq(self.ranks, at)), dim=-1, keepdim=True))

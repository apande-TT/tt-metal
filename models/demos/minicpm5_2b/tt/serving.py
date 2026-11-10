# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serving hooks over the optimized MiniCPM5Pipeline: one device row per request, for the vLLM generator publish-hf
renders into vllm_bundle/ from build_serving_pipeline, SERVING_ROWS, SERVING_KV_CAPACITY, SERVING_DEVICE_PARAMS and
SERVING_SERVE_ARGS.

The pipeline's own run is one 32-row batch of equal-length prompts in lockstep. Serving needs each request in its
own device row: ServingPipeline keeps the optimized decode step and sampler kernels as they are and adds

- prefill_rows(prompts, rows): the prompts of one batch (any lengths, padded to a common length) run through the same
  layers; each prompt's K / V go into its own cache row only (fill_cache), attention is ttnn's causal flash SDPA;
  returns each prompt's last-position logits.
- set_decode_rows(tokens, positions): every row's input token and position before a decode step (the step itself
  advances both on device, so steady decode writes nothing).
- set_sampling(inv_temp, top_k, top_p, seeds): per-row rules for the device sampler. The pick kernel keeps a
  candidate while its tail mass exceeds the build's 1 - top_p; a row's own top_p is that row's tail mass shifted by
  (top_p_row - top_p), and ranks past the row's top_k are shifted out. The Gumbel noise is read from a fixed
  on-device table, row (offset + position) for each row, the offset drawn from the request's seed, so no step uploads
  noise; a row whose seed is None decodes greedily (its noise is multiplied by 0).
- decode_tokens() / decode_logits(): one traceable decode step for every row.

The fused decode attention keeps a row's whole key range in L1: its circular buffers grow 384 B per cache position,
so 3456 positions is the last capacity that fits; SERVING_KV_CAPACITY stays on the measured 3072.
"""
from __future__ import annotations

import torch

import ttnn
from models.demos.minicpm5_2b._stubs import attention as attention_stub
from models.demos.minicpm5_2b._stubs import m_l_p as m_l_p_stub
from models.demos.minicpm5_2b._stubs import mlp as mlp_stub
from models.demos.minicpm5_2b.tt import pipeline as _pipeline

# Read by publish-hf without importing this module, so plain literals.
SERVING_ROWS = 32
SERVING_KV_CAPACITY = 3072
SERVING_DEVICE_PARAMS = {"l1_small_size": 24576, "trace_region_size": 209715200}  # as tests/e2e open the device
# With chunked prefill off, vLLM's TT platform caps a prefill step at max_model_len (3072) tokens: a burst of
# SERVING_ROWS prompts of PREFILL_PAD tokens then takes two steps, the second stalling the rows already decoding.
SERVING_SERVE_ARGS = ["--max-num-batched-tokens", "4096"]
PREFILL_PAD = 128
PREFILL_TOKENS = 4096
# A serving prefill pass costs about _PASS_MS whatever it holds (its ~1,400 ops dispatched one by one) plus _ROW_MS per
# prompt per PREFILL_PAD padded tokens (measured on p150 / Blackhole).
_PASS_MS, _ROW_MS = 22.0, 7.3
NOISE_TABLE_ROWS = 1 << 18
NOISE_WIDTH = 64
_SDPA_CHUNK = 128
# Serving prefill on the bring-up prefill's matmul programs (False: plain ttnn linears); a switch so the two can be
# compared on one pipeline.
SERVE_PREFILL_FAST = {"mm": True}
# L1 the 2-D multicast matmul's circular buffers may take beside the prefill's L1 tensors (its in0 / in1 blocks
# double-buffered, the bf16 output and the fp32 partials); a deeper K block is used only while they fit.
_MM_CB_BYTES = 960 * 1024


def _grid_linear(device, x, w, compute_cfg, memory_config):
    """x [.., M, K] @ w [K, N] on the full-grid 2-D multicast matmul (the bring-up prefill projections' program)."""
    if not SERVE_PREFILL_FAST["mm"]:
        return ttnn.linear(x, w, compute_kernel_config=compute_cfg, memory_config=memory_config)
    grid = device.compute_with_storage_grid_size()
    m_t, k_t, n_t = x.padded_shape[-2] // 32, w.padded_shape[-2] // 32, w.padded_shape[-1] // 32
    per_m, per_n = -(-m_t // grid.y), -(-n_t // grid.x)
    tile = 2048
    k_blk = next(
        (k for k in (8, 4, 2) if k_t % k == 0 and (2 * k * (per_m + per_n) + 3 * per_m * per_n) * tile <= _MM_CB_BYTES),
        1,
    )
    cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=k_blk,
        out_subblock_h=1,
        out_subblock_w=next(s for s in (4, 3, 2, 1) if per_n % s == 0),
        per_core_M=per_m,
        per_core_N=per_n,
        transpose_mcast=False,
        fused_activation=None,
    )
    return ttnn.linear(x, w, compute_kernel_config=compute_cfg, memory_config=memory_config, program_config=cfg)


def _serve_prefill_attention(att, x, kv_cache, ctx, b, seq):
    """[b, S, hidden] -> [b, S, hidden]: QKV, rope, K / V into the cache rows ctx.rows, causal flash SDPA, o_proj."""
    dram = ttnn.DRAM_MEMORY_CONFIG
    x2 = ttnn.reshape(x, (1, b * seq, x.shape[-1]))
    qkv = _grid_linear(att.device, x2, att.wqkv, att.compute_cfg, dram)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        ttnn.reshape(qkv, (b, 1, seq, qkv.shape[-1])),
        num_heads=att.n_heads,
        num_kv_heads=att.n_kv_heads,
        transpose_k_heads=False,
        memory_config=dram,
    )
    ttnn.deallocate(qkv)
    cos, sin_signed = ctx.rope
    q_in, k_in = q, k
    q = att._rope_kernel(q_in, cos, sin_signed, dram)
    k = att._rope_kernel(k_in, cos, sin_signed, dram)
    ttnn.deallocate(q_in)
    ttnn.deallocate(k_in)
    kv, d = att.n_kv_heads, att.head_dim
    for i, row in enumerate(ctx.rows):
        for new, cache in ((k, kv_cache.k), (v, kv_cache.v)):
            one = new if b == 1 else ttnn.slice(new, (i, 0, 0, 0), (i + 1, kv, seq, d))
            ttnn.fill_cache(cache, one, int(row))
            if b != 1:
                ttnn.deallocate(one)
    grid = att.device.compute_with_storage_grid_size()
    cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        q_chunk_size=_SDPA_CHUNK,
        k_chunk_size=_SDPA_CHUNK,
        exp_approx_mode=False,
    )
    attn = ttnn.transformer.scaled_dot_product_attention(
        q, k, v, is_causal=True, scale=att.scale, program_config=cfg, compute_kernel_config=att.compute_cfg
    )
    for t in (q, k, v):
        ttnn.deallocate(t)
    attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=dram)
    attn = ttnn.reshape(attn, (1, b * seq, att.n_heads * att.head_dim))
    out = _grid_linear(att.device, attn, att.wo, att.compute_cfg, dram)
    ttnn.deallocate(attn)
    return ttnn.reshape(out, (b, seq, out.shape[-1]))


_optimized_cached = attention_stub.TtAttention._cached


def _cached(self, x, cos, sin, kv_cache, ctx, b, seq):
    if ctx.mode == "serve_prefill":
        return _serve_prefill_attention(self, x, kv_cache, ctx, b, seq)
    return _optimized_cached(self, x, cos, sin, kv_cache, ctx, b, seq)


attention_stub.TtAttention._cached = _cached

# The MLPs' many-row path runs kernels sized for the 32 x 128 benchmark prefill (mm_swiglu hangs at 2 x 128 rows), so
# the serving prefill takes their bf16 weights through general matmul programs instead; decode keeps the optimized path.
_SERVE_PREFILL = {"on": False}


def _plain_mlp_call(optimized_call):
    def call(self, x, *args, prenorm=None, residual=None, **kwargs):
        if not _SERVE_PREFILL["on"]:
            return optimized_call(self, x, *args, prenorm=prenorm, residual=residual, **kwargs)
        self.last_residual_added = False
        dram = ttnn.DRAM_MEMORY_CONFIG
        if prenorm is not None:
            x = prenorm(x, memory_config=dram)
        shape = tuple(x.shape)
        x = ttnn.reshape(x, (1, 1, -1, shape[-1]))
        if SERVE_PREFILL_FAST["mm"]:
            # gate / up on the bring-up prefill's minimal_matmul (small streaming CBs, any row count: the 2-D
            # multicast's per-core output block for N = 6144 does not fit L1 at 32 x 128 rows), SiLU inside the multiply.
            gate = self._mm(x, self.w_gate)
            up = self._mm(x, self.w_up)
            h = self._silu_mul(gate, up)
        else:
            gate = ttnn.linear(x, self.w_gate, compute_kernel_config=self.compute_cfg, memory_config=dram)
            up = ttnn.linear(x, self.w_up, compute_kernel_config=self.compute_cfg, memory_config=dram)
            h = ttnn.multiply(ttnn.silu(gate), up, memory_config=dram)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = _grid_linear(self.device, h, self.w_down, self.compute_cfg, dram)
        ttnn.deallocate(h)
        return ttnn.reshape(out, shape)

    return call


for _cls in (mlp_stub.TtMlp, m_l_p_stub.TtMLP):
    _cls.__call__ = _plain_mlp_call(_cls.__call__)


def _host(t, dtype, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, dtype=dtype, layout=layout)


class ServingPipeline(_pipeline.MiniCPM5Pipeline):
    """The optimized pipeline with one device row per request (see the module docstring)."""

    def __init__(self, device, hf_model, capacity=SERVING_KV_CAPACITY, layers=None):
        cap = self.padded_len(capacity)  # a prompt that fits is padded to PREFILL_PAD and must still fit
        if cap > SERVING_KV_CAPACITY:
            raise ValueError(f"KV capacity {cap} exceeds the fused decode attention's L1 limit ({SERVING_KV_CAPACITY})")
        super().__init__(
            device,
            hf_model,
            layers=layers,
            batch=SERVING_ROWS,
            prompt_len=PREFILL_PAD,
            max_new_tokens=cap - PREFILL_PAD,
        )
        if self._logit_segments() is None:
            raise RuntimeError("serving needs the segmented-logits sampler (temperature 1.0, top_k <= 64, 32 rows)")
        B, k = self.batch, self.top_k
        self.inv_temp = self._tt(torch.ones(1, 1, B, k), ttnn.float32)
        self.tail_adj = self._tt(torch.zeros(1, 1, B, k), ttnn.float32)
        # Gumbel noise from a fixed table: row (offset_b + position_b) for row b, so no per-step upload; a request's
        # offset comes from its seed, and greedy rows multiply the noise by 0.
        g = torch.Generator().manual_seed(0)
        u = torch.rand(NOISE_TABLE_ROWS, NOISE_WIDTH, generator=g).clamp_(1e-10, 1.0 - 1e-7)
        self.noise_table = ttnn.from_torch(
            -torch.log(-torch.log(u)), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        self.noise_off = self._tt(torch.zeros(1, B, dtype=torch.int32), ttnn.int32)
        self.noise_on = self._tt(torch.zeros(1, 1, B, k), ttnn.float32)
        self.serve_sampler = True

    # ------------------------------------------------------------------ prefill into rows
    @staticmethod
    def padded_len(n_tokens):
        return max(PREFILL_PAD, -(-int(n_tokens) // PREFILL_PAD) * PREFILL_PAD)

    @classmethod
    def prefill_groups(cls, lens):
        """Prompt indices in prefill passes, the cheapest plan by the pass cost model: sorted by length, each pass
        padded to a power-of-two count of prompts (the shapes warm-up compiles) of at most PREFILL_TOKENS padded
        tokens (the 32 x 128 the layers' L1 intermediates were sized for)."""
        order = sorted(range(len(lens)), key=lambda i: lens[i])
        n = len(order)
        best = [(0.0, n)] * (n + 1)  # best[i]: (cost of the passes over order[i:], end of the first of them)
        for i in range(n - 1, -1, -1):
            best[i] = (float("inf"), n)
            for j in range(i + 1, n + 1):
                rows, S = 1 << (j - i - 1).bit_length(), cls.padded_len(lens[order[j - 1]])
                if rows * S > PREFILL_TOKENS and j > i + 1:
                    break
                cost = _PASS_MS + _ROW_MS * rows * S / PREFILL_PAD + best[j][0]
                if cost < best[i][0]:
                    best[i] = (cost, j)
        groups, i = [], 0
        while i < n:
            groups.append(order[i : best[i][1]])
            i = best[i][1]
        return groups

    def prefill_rows(self, prompts, rows):
        """Prompt token lists into device rows `rows` (one per prompt): fills those rows' caches, returns host fp32
        logits [n, vocab] of each prompt's last position."""
        n = len(prompts)
        if not 1 <= n <= self.batch or len(rows) != n:
            raise ValueError(f"prefill of {n} prompts into {len(rows)} rows")
        out = [None] * n
        for group in self.prefill_groups([len(p) for p in prompts]):
            logits = self._prefill_pass([prompts[i] for i in group], [rows[i] for i in group])
            for j, i in enumerate(group):
                out[i] = logits[j]
        return torch.stack(out)

    def _prefill_pass(self, prompts, rows):
        """One pass over `prompts`, padded with pad-token prompts to a power-of-two count; only the real prompts'
        K / V reach the cache (ctx.rows lists their rows alone)."""
        n = len(prompts)
        lens = [len(p) for p in prompts]
        S = self.padded_len(max(lens))
        if S > self.capacity:
            raise ValueError(f"prompt of {max(lens)} tokens exceeds the KV capacity {self.capacity}")
        g = 1 << (n - 1).bit_length()
        ids = torch.full((g, S), self.pad_id, dtype=torch.int32)
        for i, p in enumerate(prompts):
            ids[i, : lens[i]] = torch.as_tensor(p, dtype=torch.int32)
        H, D = self.hidden, self.head_dim
        x = self.token_embed(self._tt(ids, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT))
        pos = self._tt(torch.arange(S, dtype=torch.float32).repeat(g, 1), ttnn.float32)
        cos_sin = self.rotary_embedding(x, position_ids=pos)
        cos, sin = (ttnn.reshape(t, (g, 1, t.shape[-2], D)) for t in cos_sin)
        ctx = attention_stub.AttnContext("serve_prefill", None, rope=(cos, ttnn.multiply(sin, self.rot_sign)))
        ctx.rows = [int(r) for r in rows]
        _SERVE_PREFILL["on"] = True
        try:
            h = self.encoder_stack(x, position_embeddings=cos_sin, kv_caches=self.kv_caches, attn_ctx=ctx)
        finally:
            _SERVE_PREFILL["on"] = False
        h = self.final_norm(h)
        # every row of the pass (the padding rows at their last position), so the gather's shapes depend on g alone
        ends = lens + [S] * (g - n)
        last = [ttnn.slice(h, (i, ends[i] - 1, 0), (i + 1, ends[i], H)) for i in range(g)]
        if g < self.batch:
            last.append(self._tt(torch.zeros(self.batch - g, 1, H), ttnn.bfloat16))
        last = ttnn.reshape(ttnn.concat(last, dim=0), (1, self.batch, H))
        logits = self.decoder_head(last)
        out = ttnn.to_torch(logits).reshape(self.batch, -1)[:n, : self.vocab].float()
        for t in (x, pos, h, last, logits, cos, sin):
            ttnn.deallocate(t)
        return out

    # ------------------------------------------------------------------ decode inputs and rules
    def set_decode_rows(self, tokens=None, positions=None):
        """tokens / positions: int [rows] (None keeps what the device holds, e.g. the last step's own advance)."""
        B = self.batch
        if tokens is not None:
            t = torch.as_tensor(tokens, dtype=torch.int32).reshape(1, B)
            ttnn.copy_host_to_device_tensor(_host(t, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT), self.tok_buf)
        if positions is not None:
            p = torch.as_tensor(positions, dtype=torch.int32).reshape(B)
            ttnn.copy_host_to_device_tensor(_host(p.float().reshape(B, 1), ttnn.float32), self.pos_buf)
            ttnn.copy_host_to_device_tensor(_host(p.reshape(1, B), ttnn.int32), self.pos_i32)
            ttnn.copy_host_to_device_tensor(_host(p, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT), self.pos_rm)

    @staticmethod
    def noise_offset(seed, capacity):
        """A request's row offset into the noise table, from its seed."""
        g = torch.Generator().manual_seed(int(seed))
        return int(torch.randint(0, NOISE_TABLE_ROWS - capacity, (1,), generator=g))

    def set_sampling(self, inv_temp, top_k, top_p, seeds):
        """Per-row rule, [rows] each: 1 / temperature, candidates kept (1..self.top_k), nucleus mass (0, 1] and the
        request's seed (None: the row decodes greedily)."""
        B, K = self.batch, self.top_k
        sampled = [float(s is not None) for s in seeds]
        offsets = [self.noise_offset(s, self.capacity) if s is not None else 0 for s in seeds]
        inv = torch.as_tensor(inv_temp, dtype=torch.float32).reshape(B, 1).expand(B, K)
        rank = torch.arange(K).reshape(1, K)
        kk = torch.as_tensor(top_k, dtype=torch.int64).clamp(1, K).reshape(B, 1)
        shift = (torch.as_tensor(top_p, dtype=torch.float32).clamp(1e-6, 1.0) - self.top_p).reshape(B, 1)
        adj = torch.where(rank < kk, shift.expand(B, K), torch.full((B, K), -4.0))
        on = torch.as_tensor(sampled, dtype=torch.float32).reshape(B, 1).expand(B, K)
        off = torch.as_tensor(offsets, dtype=torch.int32).reshape(1, B)
        ttnn.copy_host_to_device_tensor(_host(inv.reshape(1, 1, B, K).contiguous(), ttnn.float32), self.inv_temp)
        ttnn.copy_host_to_device_tensor(_host(adj.reshape(1, 1, B, K).contiguous(), ttnn.float32), self.tail_adj)
        ttnn.copy_host_to_device_tensor(_host(on.reshape(1, 1, B, K).contiguous(), ttnn.float32), self.noise_on)
        ttnn.copy_host_to_device_tensor(_host(off, ttnn.int32), self.noise_off)

    def _row_noise(self):
        """Each row's Gumbel noise for this step, [1, 1, rows, top_k] fp32: table row offset_b + position_b."""
        B, K = self.batch, self.top_k
        l1 = ttnn.L1_MEMORY_CONFIG
        idx = ttnn.typecast(ttnn.add(self.pos_i32, self.noise_off, memory_config=l1), ttnn.uint32, memory_config=l1)
        idx = ttnn.to_layout(idx, ttnn.ROW_MAJOR_LAYOUT, memory_config=l1)
        nz = ttnn.embedding(idx, self.noise_table, layout=ttnn.TILE_LAYOUT, memory_config=l1)
        nz = ttnn.slice(ttnn.reshape(nz, (1, 1, B, NOISE_WIDTH)), (0, 0, 0, 0), (1, 1, B, K), memory_config=l1)
        return ttnn.multiply(ttnn.typecast(nz, ttnn.float32, memory_config=l1), self.noise_on, memory_config=l1)

    def _sample(self, logits):
        """The optimized sampler with each row's own rule: top-k candidates of the segmented logits, scaled by the
        row's 1 / temperature, the row's nucleus via its tail-mass shift, Gumbel-max with the row's noise; the pick
        kernel writes tok_buf."""
        if not self.serve_sampler:
            return super()._sample(logits)
        top_vals, top_idx = self._topk_segmented(logits, self.top_k)
        l1 = ttnn.L1_MEMORY_CONFIG
        scaled = ttnn.multiply(top_vals, self.inv_temp, memory_config=l1)
        probs = ttnn.softmax(scaled, dim=-1, memory_config=l1)
        incl = ttnn.matmul(probs, self.cumsum_upper, memory_config=l1)
        tail = ttnn.add(
            ttnn.add(ttnn.subtract(probs, incl, memory_config=l1), 1.0, memory_config=l1),
            self.tail_adj,
            memory_config=l1,
        )
        scored = ttnn.add(scaled, self._row_noise(), memory_config=l1)
        self._pick_kernel(tail, scaled, scored, top_idx)
        return None

    # ------------------------------------------------------------------ decode steps (traceable)
    def decode_tokens(self):
        """One decode step for every row; the sampled ids land in tok_buf [1, rows] (the next step's input)."""
        self._decode_forward()
        return self.tok_buf

    def decode_logits(self):
        """One decode step for every row; returns its logits (segmented, see logits_to_torch)."""
        return self._decode_forward()


def build_serving_pipeline(device, capacity=None, layers=None, model=None):
    """The serving pipeline on `device`, `capacity` KV positions per row (None: SERVING_KV_CAPACITY)."""
    hf_model = model if model is not None else _pipeline.load_hf_model()
    return ServingPipeline(device, hf_model, capacity=capacity or SERVING_KV_CAPACITY, layers=layers)

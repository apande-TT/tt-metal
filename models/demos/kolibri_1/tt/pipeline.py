# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The ONE Kolibri-1 text-generation pipeline on TT: the demo and the e2e test both call it.

    ids  -> token_embed -> 50 x decoder_layer(attention[o_proj = f_p8_linear], sparse_moe_block[router,
            m_l_p]) -> r_m_s_norm -> decoder_head -> on-device sampler -> next id -> token_embed -> ...

Stages (Kolibri1ForCausalLM is a causal LM): prefill = the prompt all B users share, run once, filling
every user's resident K,V cache rows and sampling each user's first token; decode = one token per user per step, reading
and extending those caches. Each step's output token is written into a resident buffer that the next
decode step reads, so the generation loop never moves data host->device; the host only reads the B
sampled ids back to apply the model's stop rule.
"""
from __future__ import annotations

import os
import time

import torch

import ttnn
from models.demos.kolibri_1.tt import inputs as kin
from models.demos.kolibri_1.tt.checkpoint import Checkpoint, layer_indices
from models.demos.kolibri_1.tt.model import (
    DecoderLayer,
    Embedding,
    FinalNorm,
    LMHead,
    Sampler,
    StepState,
    host_side_uploads,
)

PIPELINE_STAGES = ["prefill", "decode"]

# K,V positions held per user on device. Registered QB2 figures (TP=4, a CCL axis in play): 28.6 GB
# usable per chip; the weights take ~22 GB/chip (bf8 routed experts), and 50 layers x 2 x B=32 x 1 kv
# head x 2048 x 128 x 2 B = 1.7 GB/chip of cache fits the rest. Generation stops at this many
# positions (or max_position_embeddings, whichever is smaller) if no stop token came first.
KV_CAPACITY = 2048
OSL_ENV = "TT_PERF_OSL_TOKENS"


def _mapper(device):
    return ttnn.ReplicateTensorToMesh(device) if isinstance(device, ttnn.MeshDevice) else None


def _first_device(t):
    return ttnn.get_device_tensors(t)[0] if t.storage_type() == ttnn.StorageType.DEVICE else t


def read_host(t) -> torch.Tensor:
    """One chip's copy of a replicated device tensor, on host (test / demo side only)."""
    return ttnn.to_torch(_first_device(t))


class KolibriPipeline:
    def __init__(self, device, layers=None, batch=None, capacity: int = KV_CAPACITY, log=print) -> None:
        self.device = device
        self.log = log
        ck = Checkpoint()
        self.config = ck.config
        self.hf = ck.shell()  # full-depth reference skeleton (meta device): the model's section structure
        self.layer_ids = layer_indices(self.config, layers)
        self.batch = int(batch or kin.batch_size())
        self.capacity = min(int(capacity), int(self.config.max_position_embeddings))
        self.settings = kin.sampling_settings()
        self.hidden = int(self.config.hidden_size)
        self.head_dim = int(self.config.head_dim)

        t0 = time.time()
        with host_side_uploads():  # weights converted / tiled on the host, not by device ops at every build
            self.embed = Embedding(device, ck.embed_tokens())
            self.layers = []
            for n, i in enumerate(self.layer_ids):
                self.layers.append(DecoderLayer(device, ck.decoder_layer(i), self.batch, self.capacity))
                log(f"[build] layer {i} ({n + 1}/{len(self.layer_ids)}) {time.time() - t0:.0f}s", flush=True)
            self.norm = FinalNorm(device, ck.final_norm())
            self.head = LMHead(device, ck.lm_head())
            self.sampler = Sampler(
                device,
                self.batch,
                self.settings.top_k,
                self.settings.top_p,
                self.settings.temperature,
                vocab=int(self.config.vocab_size),
            )
        self.rope = self._share_rope_tables()
        self.positions = self._up(
            torch.arange(self.capacity, dtype=torch.float32).expand(1, 1, self.batch, -1).contiguous(), ttnn.float32
        )
        self.prompt_len = None
        self._prefill_hn = None
        log(
            f"[build] {len(self.layers)} layers, batch {self.batch}, KV capacity {self.capacity} ({time.time() - t0:.0f}s)"
        )

    # ------------------------------------------------------------------------------------------ setup
    def _up(self, t, dtype, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            t.contiguous(), dtype=dtype, layout=layout, device=self.device, mesh_mapper=_mapper(self.device)
        )

    def _write(self, name, t, dtype, layout=ttnn.TILE_LAYOUT):
        """Host data into the resident buffer `name`, IN PLACE when it already exists with this shape (a
        captured trace keeps reading the same addresses), else allocate it."""
        host = ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, mesh_mapper=_mapper(self.device))
        cur = getattr(self, name, None)
        if cur is not None and list(cur.shape) == list(host.shape) and cur.dtype == host.dtype:
            ttnn.copy_host_to_device_tensor(host, cur)
        else:
            setattr(self, name, ttnn.to_device(host, self.device))

    def _share_rope_tables(self):
        """Every sliding layer's stub uploads the same RoPE tables; keep one copy."""
        sliding = [layer.attn for layer in self.layers if layer.attn.is_sliding]
        if not sliding:
            return None
        keep = sliding[0]
        for attn in sliding[1:]:
            ttnn.deallocate(attn.cos_table)
            ttnn.deallocate(attn.sin_table)
            attn.cos_table, attn.sin_table = keep.cos_table, keep.sin_table
        return keep

    def load_request(self, prompt_ids, seeds) -> None:
        """Input encoding -> resident device buffers: the padded prompt, its positions, the per-user
        sampling uniforms and the decode-state counters. Host work, done before any forward runs."""
        B, T = self.batch, len(prompt_ids)
        if len(seeds) != B:
            raise ValueError(f"need {B} seeds, got {len(seeds)}")
        if T >= self.capacity:
            raise ValueError(f"prompt of {T} tokens exceeds the KV capacity {self.capacity}")
        Tp = max(32, -(-T // 32) * 32)
        pad = self.settings.pad_id
        rm, u32 = ttnn.ROW_MAJOR_LAYOUT, ttnn.uint32
        self.prompt_len, self.prefill_len = T, Tp
        # One prompt for all B users (B samples of it, seed b on row b): the prefill runs it once, at batch 1.
        self._write("prefill_ids", torch.tensor([list(prompt_ids) + [pad] * (Tp - T)], dtype=torch.int32), u32, rm)
        self._write("prefill_pos", torch.arange(Tp, dtype=torch.int32).reshape(1, Tp), u32, rm)
        uni = kin.sampling_uniforms(seeds, kin.uniforms_length())[: self.capacity]  # [C, B]
        self._write("u_table", uni.t().reshape(1, 1, B, self.capacity), ttnn.float32)
        self._write("last_col", torch.full((1, 1, B, 1), float(T - 1)), ttnn.float32)
        self._write("cur_pos", torch.full((B,), T, dtype=torch.int32), ttnn.int32, rm)
        self._write("pos_u32", torch.full((1, B), T, dtype=torch.int32), u32, rm)
        self._write("pos_col", torch.full((1, 1, B, 1), float(T)), ttnn.float32)
        self._write("tok_u32", torch.zeros((1, B), dtype=torch.int32), u32, rm)

    # --------------------------------------------------------------------------------- the forward
    def _rope_rows(self, pos_u32, shape):
        if self.rope is None:
            return None, None
        cos = ttnn.embedding(pos_u32, self.rope.cos_table, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(pos_u32, self.rope.sin_table, layout=ttnn.TILE_LAYOUT)
        return ttnn.reshape(cos, shape), ttnn.reshape(sin, shape)

    def _uniform(self, pos_col):
        """Each user's uniform for the position it samples at: an exact one-hot select from the table."""
        return ttnn.sum(ttnn.multiply(ttnn.eq(self.positions, pos_col), self.u_table), dim=-1, keepdim=True)

    def _commit(self, tok):
        """Sampled ids [1, 1, B, 1] -> the resident token buffer the next decode step embeds."""
        t = ttnn.typecast(tok, ttnn.uint32)
        t = ttnn.reshape(ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT), [1, self.batch])
        ttnn.copy(t, self.tok_u32)

    def _stack(self, h, st):
        for layer in self.layers:
            h = layer(h, None, None, st)
        return h

    def prefill_forward(self):
        """Prompt for all B users -> K,V caches filled, first token sampled. Returns (logits [1,1,B,V] at the
        last prompt position, token ids [1,1,B,1]); keeps the final-norm hidden state (see prefill_hidden).

        Every user has the same prompt (load_request takes one), so the layers run it once, at batch 1: each
        attention layer writes its K,V into all B users' cache rows, and the last position's state is repeated
        to the B rows the head and the sampler (one uniform per user) read."""
        B, T, Tp, H = self.batch, self.prompt_len, self.prefill_len, self.hidden
        cos, sin = self._rope_rows(self.prefill_pos, [1, 1, Tp, self.head_dim])
        st = StepState("prefill", cos, sin)
        h = ttnn.typecast(ttnn.reshape(self.embed(self.prefill_ids), [1, 1, Tp, H]), ttnn.float32)
        hn = self.norm(self._stack(h, st))
        # Position T-1: first the tile-aligned 32-row block that holds it (no data reordering), then the row, so the
        # untilize behind a non-aligned slice touches 32 rows instead of all Tp.
        tr = (T - 1) // 32 * 32
        block = ttnn.slice(hn, [0, 0, tr, 0], [1, 1, tr + 32, H])
        row = ttnn.slice(block, [0, 0, T - 1 - tr, 0], [1, 1, T - tr, H])
        ttnn.deallocate(block)
        last = ttnn.repeat(row, [1, 1, B, 1])
        ttnn.deallocate(row)
        logits = self.head(last)
        tok = self.sampler(logits, self._uniform(self.last_col))
        self._commit(tok)
        self._prefill_hn = hn
        return logits, tok

    @property
    def prefill_hidden(self):
        """Every user's final-norm prefill state [B, 1, Tp, H] (the shared prompt's, one copy per user)."""
        return ttnn.repeat(self._prefill_hn, [self.batch, 1, 1, 1])

    def decode_forward(self):
        """One token per user at its current position; advances the resident position counters."""
        B, H = self.batch, self.hidden
        cos, sin = self._rope_rows(self.pos_u32, [1, B, 1, self.head_dim])
        st = StepState("decode", cos, sin, self.cur_pos)
        h = ttnn.typecast(ttnn.reshape(self.embed(self.tok_u32), [1, 1, B, H]), ttnn.float32)
        logits = self.head(self.norm(self._stack(h, st)))
        tok = self.sampler(logits, self._uniform(self.pos_col))
        self._commit(tok)
        ttnn.plus_one(self.cur_pos)
        ttnn.plus_one(self.pos_u32)
        ttnn.copy(ttnn.add(self.pos_col, 1.0), self.pos_col)
        return logits, tok

    # ------------------------------------------------------------- decode contract (perf adapter)
    def decode_prefill(self, input_ids=None, seeds=None) -> dict:
        """Seed the resident K,V caches from a prompt (1-D ids, or [B, T] of one prompt per user; the card
        prompt when None) and sample each user's first token. Returns the decode state."""
        if input_ids is None:
            input_ids = self.default_prompt()
        ids = torch.as_tensor(input_ids)
        prompt = ids.reshape(-1, ids.shape[-1])[0].tolist()
        self.load_request(prompt, list(seeds) if seeds is not None else list(range(self.batch)))
        logits, tok = self.prefill_forward()
        return {"logits": logits, "tokens": tok}

    def decode_step(self, state=None) -> dict:
        """ONE decode token for all B users from the resident buffers (no host reads); the sampled ids are
        already written back where the next step reads them."""
        logits, tok = self.decode_forward()
        return {"logits": logits, "tokens": tok}

    # ------------------------------------------------------------------------- trace contract
    def default_prompt(self) -> list:
        return kin.encode_prompt(kin.load_tokenizer())

    def _stage_inputs(self) -> dict:
        return {"prompt_ids": self.default_prompt(), "seeds": list(range(self.batch))}

    def prefill_trace_inputs(self) -> dict:
        """The e2e test's / demo's inputs: the model-card prompt for every user, seed b on row b."""
        return self._stage_inputs()

    def prefill_trace_setup(self, inputs) -> None:
        """Pin the sequence axis to the padded prompt length (tile multiple; padding sits after the prompt,
        so causal attention keeps it out of every real position) and pre-upload the ids, positions,
        uniforms and decode counters into resident buffers."""
        self.load_request(inputs["prompt_ids"], inputs["seeds"])

    def prefill_trace_step(self):
        logits, _ = self.prefill_forward()
        return logits

    def prefill_trace_items(self) -> int:
        """Tokens one prefill call runs through the 50 decoder layers: the padded prompt, once (all B users
        share it; see prefill_forward)."""
        if self.prompt_len is None:
            self.prefill_trace_setup(self.prefill_trace_inputs())
        return self.prefill_len

    def decode_trace_inputs(self) -> dict:
        return self._stage_inputs()

    def decode_trace_setup(self, inputs) -> None:
        """Upload the request and run its prefill eagerly so the K,V caches and the first token are resident;
        the KV capacity (the decode stage's pinned sequence axis) is fixed at build."""
        self.load_request(inputs["prompt_ids"], inputs["seeds"])
        self.prefill_forward()

    def decode_trace_step(self):
        logits, _ = self.decode_forward()
        return logits

    def decode_trace_items(self) -> int:
        """One token per user per step."""
        return self.batch

    def decode_trace_repeats(self) -> int:
        """Decode steps one request runs at most: every position left in the KV capacity after the prompt
        and the token prefill samples (the schedule's cap; a request stops earlier on its stop token)."""
        if self.prompt_len is None:
            self.prefill_trace_setup(self.prefill_trace_inputs())
        return max(1, self.capacity - self.prompt_len - 1)

    # ------------------------------------------------------------------------------ generation loop
    def max_new_tokens(self, limit=None) -> int:
        """Safety cap: the positions left in the KV capacity (itself <= max_position_embeddings), unless the
        harness pins the horizon for profiling ($TT_PERF_OSL_TOKENS) or the caller passes a lower limit."""
        cap = self.capacity - self.prompt_len
        osl = os.environ.get(OSL_ENV)
        for lim in (int(osl) if osl else None, limit):
            if lim:
                cap = min(cap, int(lim))
        return cap

    def generate(self, prompt_ids, seeds, on_step=None, max_new_tokens=None) -> dict:
        """Prefill + decode until every user has produced one of the model's stop tokens (generation_config
        eos_token_id) or the cap is reached. on_step(step, logits, ids) sees each step's device logits and
        the host ids (the test reads logits back through it)."""
        stop = set(self.settings.stop_ids)
        t0 = time.time()
        state = self.decode_prefill([list(prompt_ids)], seeds)
        cap = self.max_new_tokens(max_new_tokens)
        ids = [int(v) for v in read_host(state["tokens"]).reshape(-1)[: self.batch]]
        if on_step:
            on_step(0, state["logits"], ids)
        seqs = [[i] for i in ids]
        done = [i in stop for i in ids]
        self.log(f"[prefill] {self.prompt_len} tokens x {self.batch} users: {time.time() - t0:.2f}s", flush=True)
        step = 1
        while not all(done) and step < cap:
            t1 = time.time()
            state = self.decode_step(state)
            ids = [int(v) for v in read_host(state["tokens"]).reshape(-1)[: self.batch]]  # syncs the device
            if on_step:
                on_step(step, state["logits"], ids)
            for b, i in enumerate(ids):
                if not done[b]:
                    seqs[b].append(i)
                    done[b] = i in stop
            step += 1
            self.log(
                f"[decode] step {step}: {self.batch - sum(done)} users active ({time.time() - t1:.3f}s)", flush=True
            )
        return {"tokens": seqs, "ended_on_stop": done, "steps": step, "cap": cap}


def build_pipeline(device, model=None, layers=None, **kwargs):
    """The resident pipeline object (PIPELINE_STAGES + per-stage trace hooks + the decode contract).

    `layers` caps the decoder depth (None = all 50; a cap keeps both layer kinds, see layer_indices).
    `model` and demo kwargs (prompt, text, ...) are accepted and ignored: the weights are read from the
    checkpoint layer by layer and the shapes come from the config."""
    return KolibriPipeline(device, layers=layers, batch=kwargs.get("batch"), log=kwargs.get("log", print))


# Depth of the zero-argument selftests' own build (the probes call them with no device): the per-stage
# op sequence is the same for every layer, and a full 50-layer build alone takes several minutes.
SELFTEST_LAYERS_ENV = "KOLIBRI_SELFTEST_LAYERS"


def _selftest_layers(layers):
    if layers is not None:
        return layers
    v = os.environ.get(SELFTEST_LAYERS_ENV) or os.environ.get("TT_PERF_LAYERS")
    return int(v) if v else 2


def _pcc(a, b) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def trace_capture_selftest(device=None, layers=None) -> bool:
    """Per stage in PIPELINE_STAGES: set the stage up, run its step eagerly (the reference output), set it
    up again, capture ONE step between begin/end_trace_capture, replay it, compare the replayed logits to
    the eager ones, and release the trace before the next stage. True only if every stage captured
    (a host op inside the step makes the capture raise) and replayed to the same output."""
    own = device is None
    if own:
        from models.demos.kolibri_1.demo.mesh import open_mesh

        device = open_mesh()
    try:
        pipe = build_pipeline(device, layers=_selftest_layers(layers))
        ok = True
        for stage in PIPELINE_STAGES:
            setup = getattr(pipe, f"{stage}_trace_setup")
            step = getattr(pipe, f"{stage}_trace_step")
            inputs = getattr(pipe, f"{stage}_trace_inputs")()
            setup(inputs)
            eager = read_host(step()).float()
            setup(inputs)
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            out = step()
            ttnn.end_trace_capture(device, tid, cq_id=0)
            setup(inputs)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            traced = read_host(out).float()
            ttnn.release_trace(device, tid)
            match = _pcc(traced, eager)
            print(
                f"[trace] {stage}: captured and replayed ({len(pipe.layers)} layers); replay vs eager logits pcc {match:.6f}",
                flush=True,
            )
            ok = ok and match >= 0.9999
        return ok
    finally:
        if own:
            from models.demos.kolibri_1.demo.mesh import close_mesh

            close_mesh(device)


def host_op_selftest(device=None, layers=None) -> dict:
    """The forward (prefill + one decode step, every op from the prompt ids to the sampled ids) under the
    host-op observer; tokenizing, weight build and the request upload happen before it starts."""
    from scripts.tt_hw_planner.host_op_observer import observe_host_ops, verdict

    own = device is None
    if own:
        from models.demos.kolibri_1.demo.mesh import open_mesh

        device = open_mesh()
    try:
        pipe = build_pipeline(device, layers=_selftest_layers(layers))
        inputs = pipe.prefill_trace_inputs()
        pipe.load_request(inputs["prompt_ids"], inputs["seeds"])
        with observe_host_ops() as ops:
            pipe.prefill_forward()
            pipe.decode_forward()
        ttnn.synchronize_device(device)
        v = verdict(ops)
        print(f"[host-ops] {v['reason']}", flush=True)
        return v
    finally:
        if own:
            from models.demos.kolibri_1.demo.mesh import close_mesh

            close_mesh(device)

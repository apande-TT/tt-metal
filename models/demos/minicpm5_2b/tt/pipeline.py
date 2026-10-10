# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The ONE chained TTNN pipeline for openbmb/MiniCPM5-2B text generation (LlamaForCausalLM).

Both demo/demo_text_generation.py and tests/e2e/ call this file; nothing else holds the wiring.

Forward (every op on device, every graduated stub inside it):
  ids -> token_embed -> encoder_stack[ layer i: decoder_layer (even i) | layer (odd i), each holding
         r_m_s_norm -> attention (resident KV) -> r_m_s_norm -> mlp (even i) | m_l_p (odd i) ]
      -> r_m_s_norm (final) -> decoder_head -> on-device sampler -> next token
  rotary_embedding turns the device position ids into the cos/sin every attention reads.

Sampling is generation_config's chain (top_k -> top_p -> temperature 1.0), drawn by Gumbel-max with
per-sample seeded noise (tt/inputs.py) so a seed gives the same sample on TT and in the HF golden.
"""
from __future__ import annotations

import collections
import contextlib
import os
import sys
import time

import torch

import ttnn

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 4)))
if _REPO not in sys.path:  # tt/ is also imported as a top-level package by the perf tooling
    sys.path.insert(0, _REPO)

from models.demos.minicpm5_2b._stubs import attention as attention_stub  # noqa: E402
from models.demos.minicpm5_2b._stubs import decoder_head as decoder_head_stub  # noqa: E402
from models.demos.minicpm5_2b._stubs import decoder_layer as decoder_layer_stub  # noqa: E402
from models.demos.minicpm5_2b._stubs import encoder_stack as encoder_stack_stub  # noqa: E402
from models.demos.minicpm5_2b._stubs import layer as layer_stub  # noqa: E402
from models.demos.minicpm5_2b._stubs import m_l_p as m_l_p_stub  # noqa: E402
from models.demos.minicpm5_2b._stubs import mlp as mlp_stub  # noqa: E402
from models.demos.minicpm5_2b._stubs import r_m_s_norm as r_m_s_norm_stub  # noqa: E402
from models.demos.minicpm5_2b._stubs import rotary_embedding as rotary_embedding_stub  # noqa: E402
from models.demos.minicpm5_2b._stubs import token_embed as token_embed_stub  # noqa: E402
from models.demos.minicpm5_2b.tt import inputs as tt_inputs  # noqa: E402

PIPELINE_STAGES = ["prefill", "decode"]

# Every graduated module of the bring-up (bringup_status.json + _stubs/*.py.last_good_native).
GRADUATED_STUBS = (
    "token_embed",
    "rotary_embedding",
    "encoder_stack",
    "decoder_layer",
    "layer",
    "attention",
    "mlp",
    "m_l_p",
    "r_m_s_norm",
    "decoder_head",
)

_TILE = 32
_NEG = -1e9


def _round_up(n, m=_TILE):
    return ((int(n) + m - 1) // m) * m


class CountedStub:
    """A graduated stub plus its Gate-2 invocation counter; calls go straight through to the stub."""

    def __init__(self, name, stub, counts):
        self.name, self.stub, self.counts = name, stub, counts

    def __call__(self, *args, **kwargs):
        self.counts[self.name] += 1
        return self.stub(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self.__dict__["stub"], item)


def sampling_params(hf_model):
    """(top_k, top_p, temperature) generate() applies for this checkpoint's generation_config.

    top_k is unset in generation_config.json, so generate() fills transformers' global default.
    """
    gc = hf_model.generation_config
    defaults = gc._get_default_generation_params()
    top_k = gc.top_k if gc.top_k is not None else defaults["top_k"]
    top_p = gc.top_p if gc.top_p is not None else defaults["top_p"]
    temperature = gc.temperature if gc.temperature is not None else defaults["temperature"]
    return int(top_k), float(top_p), float(temperature)


def eos_token_ids(hf_model):
    eos = hf_model.generation_config.eos_token_id
    return [int(e) for e in (eos if isinstance(eos, (list, tuple)) else [eos])]


class MiniCPM5Pipeline:
    """Resident TT text-generation pipeline: weights, KV caches and every per-step buffer on device."""

    def __init__(self, device, hf_model, layers=None, batch=None, prompt_len=None, max_new_tokens=None):
        cfg = hf_model.config
        self.device = device
        self.hf_model = hf_model  # HF reference: ground truth for section structure and goldens
        self.config = cfg
        n_total = cfg.num_hidden_layers
        self.n_layers = n_total if layers is None else max(1, min(int(layers), n_total))
        self.batch = int(batch) if batch is not None else tt_inputs.batch_size()
        self.prompt_len = int(prompt_len) if prompt_len is not None else int(tt_inputs.example_input_ids().shape[-1])
        self.max_new_tokens = int(max_new_tokens) if max_new_tokens is not None else tt_inputs.EXAMPLE_MAX_NEW_TOKENS
        self.vocab = cfg.vocab_size
        self.hidden = cfg.hidden_size
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        # Fixed capacities: prefill length padded to a tile, KV slots for prompt + every new token.
        self.prefill_len = _round_up(self.prompt_len)
        self.capacity = _round_up(self.prompt_len + self.max_new_tokens)
        assert self.capacity <= cfg.max_position_embeddings, "KV capacity exceeds max_position_embeddings"
        self.top_k, self.top_p, self.temperature = sampling_params(hf_model)
        self.eos_ids = eos_token_ids(hf_model)
        self.pad_id = int(hf_model.generation_config.pad_token_id)

        self.invocations = collections.Counter()
        hf = hf_model.model

        def counted(name, stub):
            return CountedStub(name, stub, self.invocations)

        self.token_embed = counted("token_embed", token_embed_stub.build(device, hf.embed_tokens))
        self.rotary_embedding = counted("rotary_embedding", rotary_embedding_stub.build(device, hf.rotary_emb))
        blocks = []
        for i in range(self.n_layers):
            hl = hf.layers[i]
            even = i % 2 == 0
            parts = {
                "input_layernorm": counted("r_m_s_norm", r_m_s_norm_stub.build(device, hl.input_layernorm)),
                "self_attn": counted("attention", attention_stub.build(device, hl.self_attn)),
                "post_attention_layernorm": counted(
                    "r_m_s_norm", r_m_s_norm_stub.build(device, hl.post_attention_layernorm)
                ),
                "mlp": counted("mlp", mlp_stub.build(device, hl.mlp))
                if even
                else counted("m_l_p", m_l_p_stub.build(device, hl.mlp)),
            }
            if even:
                blocks.append(counted("decoder_layer", decoder_layer_stub.build(device, hl, parts=parts)))
            else:
                blocks.append(counted("layer", layer_stub.build(device, hl, parts=parts)))
        self.encoder_stack = counted("encoder_stack", encoder_stack_stub.build(device, hf.layers, blocks=blocks))
        self.final_norm = counted("r_m_s_norm", r_m_s_norm_stub.build(device, hf.norm))
        self.decoder_head = counted("decoder_head", decoder_head_stub.build(device, hf_model.lm_head))

        self.kv_caches = [
            attention_stub.KVCache(device, self.batch, self.n_kv_heads, self.capacity, self.head_dim)
            for _ in range(self.n_layers)
        ]
        self._alloc_buffers()

    # ------------------------------------------------------------------ persistent device buffers
    def _tt(self, t, dtype, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=self.device)

    def _alloc_buffers(self):
        B, Sp, C = self.batch, self.prefill_len, self.capacity
        causal = torch.full((Sp, Sp), _NEG).triu(diagonal=1).reshape(1, 1, Sp, Sp)  # HF causal mask, additive
        self.prefill_mask = self._tt(causal, ttnn.bfloat16)
        self.prefill_pos = self._tt(torch.arange(Sp, dtype=torch.float32).repeat(B, 1), ttnn.float32)
        self.key_index_row = self._tt(torch.arange(C, dtype=torch.float32).reshape(1, 1, 1, C), ttnn.float32)
        self.key_index_col = self._tt(torch.arange(C, dtype=torch.float32).reshape(1, 1, C, 1), ttnn.float32)
        k = self.top_k
        self.cumsum_upper = self._tt(torch.triu(torch.ones(k, k)), ttnn.float32)  # p @ U = inclusive cumsum
        self.ids_buf = self._tt(torch.full((B, Sp), self.pad_id, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.tok_buf = self._tt(torch.zeros(B, 1, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.pos_buf = self._tt(torch.zeros(B, 1), ttnn.float32)
        self.noise_buf = self._tt(torch.zeros(1, 1, B, self.vocab), ttnn.bfloat16)
        self.noise_steps = []
        self.real_len = self.prompt_len

    def load_request(self, input_ids, noise):
        """Upload one request: prompt ids [B, S] and the per-step sampling noise [T, B, V] (bf16).

        Host->device input encoding only; the model math runs from these resident buffers.
        """
        B, S = input_ids.shape
        assert B == self.batch, f"request batch {B} != pipeline batch {self.batch}"
        assert S <= self.prefill_len, f"prompt {S} longer than the prefill capacity {self.prefill_len}"
        ids = torch.full((B, self.prefill_len), self.pad_id, dtype=torch.int32)
        ids[:, :S] = input_ids.to(torch.int32)
        ttnn.copy(self._tt(ids, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT), self.ids_buf)
        ttnn.copy(self._tt(torch.full((B, 1), float(S)), ttnn.float32), self.pos_buf)
        self.real_len = S
        for t in self.noise_steps:
            ttnn.deallocate(t)
        self.noise_steps = [
            self._tt(noise[t].reshape(1, 1, B, self.vocab), ttnn.bfloat16) for t in range(noise.shape[0])
        ]
        ttnn.copy(self.noise_steps[0], self.noise_buf)

    # ------------------------------------------------------------------ on-device sampler
    def _sample(self, logits):
        """generate()'s sampler on device: top_k -> top_p filter, Gumbel-max with the resident noise.

        logits [B, 1, V] bf16 -> token ids [1, 1, B] uint32. Mirrors HF's TopK/TopP warpers: keep the
        top_k logits (ties kept), then drop the low tail whose ascending cumulative mass <= 1 - top_p.
        """
        B, V, k = self.batch, self.vocab, self.top_k
        scores = ttnn.typecast(ttnn.reshape(logits, (1, 1, B, V)), ttnn.float32)
        if self.temperature != 1.0:
            scores = ttnn.multiply(scores, 1.0 / self.temperature)
        top_vals, _ = ttnn.topk(scores, k=k, dim=-1)  # [1, 1, B, k], descending
        probs = ttnn.softmax(top_vals, dim=-1)
        incl = ttnn.matmul(probs, self.cumsum_upper)  # mass of ranks <= j
        tail_mass = ttnn.add(ttnn.subtract(probs, incl), 1.0)  # mass of ranks >= j (HF's ascending cumsum)
        keep = ttnn.gt(tail_mass, 1.0 - self.top_p)
        kept_vals = ttnn.add(top_vals, ttnn.multiply(ttnn.subtract(keep, 1.0), _NEG))  # dropped -> +1e9
        threshold = ttnn.min(kept_vals, dim=-1, keepdim=True)  # smallest kept logit, [1, 1, B, 1]
        allowed = ttnn.ge(scores, threshold)
        perturbed = ttnn.add(scores, ttnn.typecast(self.noise_buf, ttnn.float32))
        perturbed = ttnn.add(perturbed, ttnn.multiply(ttnn.subtract(allowed, 1.0), -_NEG))
        return ttnn.argmax(perturbed, dim=-1)

    def _emit(self, logits):
        tok = self._sample(logits)
        ttnn.copy(ttnn.reshape(tok, (self.batch, 1)), self.tok_buf)

    # ------------------------------------------------------------------ the chained forward
    def _prefill_forward(self):
        """Prompt pass over the resident ids: seeds every layer's KV cache, emits the first token."""
        B, H = self.batch, self.hidden
        x = self.token_embed(self.ids_buf)
        cos_sin = self.rotary_embedding(x, position_ids=self.prefill_pos)
        ctx = attention_stub.AttnContext("prefill", self.prefill_mask)
        h = self.encoder_stack(x, position_embeddings=cos_sin, kv_caches=self.kv_caches, attn_ctx=ctx)
        self.prefill_hidden = self.final_norm(h)  # [B, Sp, H]: the state prefill hands every decode step
        last = ttnn.slice(self.prefill_hidden, (0, self.real_len - 1, 0), (B, self.real_len, H))
        logits = self.decoder_head(last)
        self._emit(logits)
        return logits

    def _decode_forward(self):
        """One token per row from the resident token/position: reads + writes the KV cache, emits the next."""
        B = self.batch
        x = self.token_embed(self.tok_buf)
        cos_sin = self.rotary_embedding(x, position_ids=self.pos_buf)
        pos = ttnn.reshape(self.pos_buf, (B, 1, 1, 1))
        write = ttnn.typecast(ttnn.eq(self.key_index_col, pos), ttnn.bfloat16)  # [B, 1, C, 1]
        mask = ttnn.typecast(ttnn.multiply(ttnn.gt(self.key_index_row, pos), _NEG), ttnn.bfloat16)  # [B, 1, 1, C]
        ctx = attention_stub.AttnContext("decode", mask, write)
        h = self.encoder_stack(x, position_embeddings=cos_sin, kv_caches=self.kv_caches, attn_ctx=ctx)
        logits = self.decoder_head(self.final_norm(h))
        self._emit(logits)
        ttnn.add(self.pos_buf, 1.0, output_tensor=self.pos_buf)
        return logits

    def prefill_step(self):
        return self._prefill_forward()

    def decode_step(self, step):
        """Decode new token `step` (1-based after the prefill's token 0) with that step's noise."""
        ttnn.copy(self.noise_steps[step], self.noise_buf)
        return self._decode_forward()

    # AR decode contract: decode_prefill seeds the resident self-attn KV; decode_step reads it.
    def decode_prefill(self, input_ids, noise):
        self.load_request(input_ids, noise)
        return self.prefill_step()

    def read_tokens(self):
        return ttnn.to_torch(self.tok_buf).reshape(self.batch).to(torch.int64)

    def run_text_generation(self, input_ids, noise, max_new_tokens=None, collect_logits=False, log=print):
        """Generate until every row hit an eos id or max_new_tokens (generate()'s stop rule).

        Returns dict(tokens [B, T] (pad after a row's eos), lengths [B], stop_reason, steps, logits
        (list of [B, V] float per step, when collect_logits), prefill_hidden [B, S, H]).
        """
        max_new_tokens = int(max_new_tokens or self.max_new_tokens)
        safety_cap = self.config.max_position_embeddings - input_ids.shape[-1]
        assert input_ids.shape[-1] + max_new_tokens <= self.capacity, "KV capacity below prompt + max_new_tokens"
        B = self.batch
        out_tokens, step_logits = [], []
        finished = torch.zeros(B, dtype=torch.bool)
        lengths = torch.full((B,), max_new_tokens, dtype=torch.int64)
        stop_reason = "max_new_tokens"
        prefill_logits = self.decode_prefill(input_ids, noise)
        prefill_hidden = ttnn.to_torch(self.prefill_hidden)[:, : self.real_len].float()
        t0 = time.time()
        for step in range(max_new_tokens):
            logits = self.decode_step(step) if step > 0 else prefill_logits
            ttnn.synchronize_device(self.device)
            tok = self.read_tokens()
            if collect_logits:
                step_logits.append(ttnn.to_torch(logits).reshape(B, self.vocab).float())
            tok = torch.where(finished, torch.full_like(tok, self.pad_id), tok)
            out_tokens.append(tok)
            newly = (~finished) & torch.isin(tok, torch.tensor(self.eos_ids))
            lengths[newly] = step + 1
            finished |= newly
            log(
                f"[minicpm5_2b] token {step + 1}/{max_new_tokens} done "
                f"({int(finished.sum())}/{B} rows finished, {time.time() - t0:.1f}s)"
            )
            if bool(finished.all()):
                stop_reason = "eos"
                break
            assert step + 1 < safety_cap, "decode reached the max_position_embeddings safety cap"
        return {
            "tokens": torch.stack(out_tokens, dim=1),
            "lengths": lengths,
            "stop_reason": stop_reason,
            "steps": len(out_tokens),
            "logits": step_logits,
            "prefill_hidden": prefill_hidden,
        }

    # ------------------------------------------------------------------ trace contract
    def _stored_request(self):
        return tt_inputs.stored_request(self.batch)

    def prefill_trace_inputs(self):
        ids, seeds = self._stored_request()
        return {"input_ids": ids, "noise": tt_inputs.gumbel_noise(seeds, 1, self.vocab)}

    def prefill_trace_setup(self, inputs):
        self.load_request(inputs["input_ids"], inputs["noise"])

    def prefill_trace_step(self):
        return self._prefill_forward()

    def prefill_trace_items(self):
        return self.batch * self.prefill_len

    def decode_trace_inputs(self):
        ids, seeds = self._stored_request()
        return {"input_ids": ids, "noise": tt_inputs.gumbel_noise(seeds, 2, self.vocab)}

    def decode_trace_setup(self, inputs):
        self.load_request(inputs["input_ids"], inputs["noise"])
        self._prefill_forward()  # seeds the resident KV + first token outside the trace
        ttnn.copy(self.noise_steps[1], self.noise_buf)

    def decode_trace_step(self):
        return self._decode_forward()

    def decode_trace_items(self):
        return self.batch

    def decode_trace_repeats(self):
        return self.max_new_tokens - 1  # the prefill emits token 1; decode emits the rest

    def trace_region_bytes(self):
        """Trace buffer for the largest stage (prefill at Sp x layers), with headroom."""
        per_layer = 2 * 1024 * 1024
        return int(64 * 1024 * 1024 + per_layer * self.n_layers)

    def trace_capture_selftest(self, pcc_min=0.99):
        return trace_capture_selftest(self.device, pipeline=self, pcc_min=pcc_min)

    def host_op_selftest(self):
        return host_op_selftest(self.device, pipeline=self)

    # ------------------------------------------------------------------ HF reference helpers (setup only)
    def hf_reference_logits(self, sequences):
        """HF logits of `sequences` [B, L] with this build's depth (all layers unless capped)."""
        hf = self.hf_model
        full = hf.model.layers
        n_cfg = hf.config.num_hidden_layers
        try:
            if self.n_layers != n_cfg:
                hf.model.layers = torch.nn.ModuleList(list(full)[: self.n_layers])
                hf.config.num_hidden_layers = self.n_layers
            with torch.no_grad():
                return hf(input_ids=sequences, use_cache=False).logits.float()
        finally:
            hf.model.layers = full
            hf.config.num_hidden_layers = n_cfg


def load_hf_model():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(tt_inputs.MODEL_ID, torch_dtype=torch.float32)
    return model.eval()


def build_pipeline(device, model=None, layers=None, prefill_layers=None, decode_layers=None, **kwargs):
    """Construct and return the resident pipeline object (does NOT run it).

    `layers` caps the depth of the model's one repeated stack (model.layers); None = all 42. Prefill and
    decode run the SAME stack, so prefill_layers / decode_layers are accepted as aliases of it (the
    first one given wins). Demo kwargs (prompt, text, ...) are accepted and ignored; `batch`,
    `prompt_len` and `max_new_tokens` size the resident buffers.
    """
    depth = next((d for d in (prefill_layers, decode_layers, layers) if d is not None), None)
    if depth is None and os.environ.get("TT_PERF_LAYERS"):
        depth = int(os.environ["TT_PERF_LAYERS"])
    hf_model = model if model is not None else load_hf_model()
    return MiniCPM5Pipeline(
        device,
        hf_model,
        layers=depth,
        batch=kwargs.get("batch"),
        prompt_len=kwargs.get("prompt_len"),
        max_new_tokens=kwargs.get("max_new_tokens"),
    )


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def _stage_reference(pipe, stage, inputs, traced_tokens_before):
    """HF logits the stage's step should produce (computed from the stage's own inputs)."""
    ids = inputs["input_ids"]
    if stage == "prefill":
        return pipe.hf_reference_logits(ids[:, :])[:, -1, :]
    seq = torch.empty(ids.shape[0], ids.shape[1] + 1, dtype=ids.dtype)
    seq[:, :-1], seq[:, -1] = ids, traced_tokens_before.reshape(-1).to(ids.dtype)  # prompt + fed token
    return pipe.hf_reference_logits(seq)[:, -1, :]


@contextlib.contextmanager
def _device_or_default(device):
    """Yield `device`; a bare call (the gate's probes) borrows the demo's device open instead."""
    if device is not None:
        yield device
        return
    from models.demos.minicpm5_2b.demo.device import opened_device

    with opened_device() as dev:
        yield dev


def trace_capture_selftest(device=None, pipeline=None, pcc_min=0.99, layers=None):
    """Capture ONE step per stage in a trace, replay it, release it, and compare to the HF reference.

    True only if every stage captured (host-free) and its replayed logits match HF (PCC, every row).
    """
    if pipeline is None:
        with _device_or_default(device) as dev:
            return _trace_capture_selftest(dev, build_pipeline(dev, layers=layers), pcc_min)
    return _trace_capture_selftest(device, pipeline, pcc_min)


def _trace_capture_selftest(device, pipe, pcc_min):
    ok = True
    for stage in PIPELINE_STAGES:
        setup = getattr(pipe, f"{stage}_trace_setup")
        step = getattr(pipe, f"{stage}_trace_step")
        inputs = getattr(pipe, f"{stage}_trace_inputs")()
        setup(inputs)
        step()  # compile outside the trace
        setup(inputs)
        fed = pipe.read_tokens() if stage == "decode" else None
        try:
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            out = step()
            ttnn.end_trace_capture(device, tid, cq_id=0)
        except Exception as exc:  # noqa: BLE001
            print(f"TRACE_CAPTURE_FAILED[{stage}]: {exc}; capacity C={pipe.capacity} cannot be traced")
            return False
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        got = ttnn.to_torch(out).reshape(pipe.batch, pipe.vocab).float()
        ttnn.release_trace(device, tid)
        ref = _stage_reference(pipe, stage, inputs, fed)
        worst = min(_pcc(got[b], ref[b]) for b in range(pipe.batch))
        print(f"trace selftest stage={stage} captured+replayed, min-row PCC vs HF={worst:.5f}")
        ok = ok and worst >= pcc_min
    return ok


def host_op_selftest(device=None, pipeline=None, layers=None):
    """The authoritative on-device check: no host aten op may fire inside the model math.

    Tokenisation, noise generation, upload and weight build happen OUTSIDE the observed region;
    prefill + decode steps (embedding through sampling) run INSIDE it.
    """
    if pipeline is None:
        with _device_or_default(device) as dev:
            return _host_op_selftest(build_pipeline(dev, layers=layers))
    return _host_op_selftest(pipeline)


def _host_op_selftest(pipe):
    from scripts.tt_hw_planner import host_op_observer

    ids, seeds = tt_inputs.batch_inputs(pipe.batch)
    pipe.load_request(ids, tt_inputs.gumbel_noise(seeds, 3, pipe.vocab))
    ttnn.synchronize_device(pipe.device)
    with host_op_observer.observe_host_ops() as ops:
        pipe.prefill_step()
        pipe.decode_step(1)
        pipe.decode_step(2)
        ttnn.synchronize_device(pipe.device)
    return host_op_observer.verdict(list(ops))

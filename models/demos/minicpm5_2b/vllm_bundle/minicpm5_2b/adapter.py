# SPDX-License-Identifier: Apache-2.0
"""Rendered by tt_hw_planner publish-hf from the demo's serving hooks (models.demos.minicpm5_2b.tt.serving); edit those, not this file.

vLLM generator for minicpm5_2b (LlamaForCausalLM) on Tenstorrent: the demo's own optimized pipeline, one device row per request.
The tt-metal vLLM plugin registers this class as TTLlamaForCausalLM via vllm_metadata.json (a bundle registers ahead of the
plugin's built-in generators) and drives it through the generator contract (initialize_vllm_model ->
allocate_kv_cache_per_layer -> warmups -> prefill_forward / decode_forward).

Serve profile: up to 32 concurrent sequences (--max-num-seqs 32, the pipeline's device rows), prompt and
generated tokens bounded by the resident KV capacity (--max-model-len 3072).

Sampling: with --additional-config '{"tt": {"sample_on_device_mode": "decode_only"}}' every decode token is sampled
on device by the pipeline's own sampler, each row by its request's temperature / top_k / top_p (greedy is
temperature 0 or top_k 1), among the row's top pipe.top_k candidates, from the request's seed (a random one when
unset); only the token ids are read back. vLLM samples on the host whenever a request needs what the device does not
do (logprobs, min_p, logit bias, ...): that step returns the logits. A request with a repetition / presence /
frequency penalty is sampled on the host here, from the logits and its token history. Prefill always returns the
logits (vLLM samples the first token).

Each request owns one device row for its whole life: the plugin hands prefill the row (`empty_slots`), the prompt's
K, V go into that row only, and the row is remembered by the request's first KV block id (unique per live request;
block 0 is vLLM's null block, so padding rows never match). Decode writes every row's input token before the step,
and the positions only when they change (a request joined or left). Rows with no request repeat a served row's token
and position; whatever they write into their own cache rows is rewritten by that row's next prefill -- a preempted
request is re-prefilled on resume.

Decode is traced whenever vLLM asks for it (enable_trace): one step of each kind (sampled ids, logits) is captured
and replayed for every token, the per-row inputs written in place before each replay. Under vLLM's async scheduling
a sampled step's ids are read back without waiting and the next step is queued before they arrive (its input ids are
the ones the last step left on device), so the chip runs while vLLM handles the last step. Prefill stays eager (its
length follows the prompt). A prefill that compiled programs the device had not run before may have placed their
buffers over a captured trace's, so the traces are released then and captured again at the next decode.
"""
from __future__ import annotations

import logging
import secrets
from types import SimpleNamespace

import torch

try:
    import models.demos.minicpm5_2b.tt.serving as _serving_module
    import ttnn  # noqa: F401

    _build_serving_pipeline = _serving_module.build_serving_pipeline
    _import_error = None
except Exception as e:  # noqa: BLE001  (available only inside the serving image)
    _build_serving_pipeline = None
    _import_error = e

log = logging.getLogger(__name__)

_SLUG = "minicpm5_2b"
_SERVING_MODULE = "models.demos.minicpm5_2b.tt.serving"
_KV_CAPACITY = 3072
# Concurrent prompts of this many tokens are compiled at startup, in every power-of-two count up to --max-num-seqs,
# rather than under a client's first requests.
_WARM_PREFILL_LEN = 128
_GREEDY = (1.0, 1, 1.0, None)  # a row's device rule: (1 / temperature, top_k, top_p, seed)
# Decode step kinds in recording order. Every recording records all of them in this order: one kind recorded
# beside the other's held trace lays L1 out differently, and its circular buffers can then clash with that
# trace's L1 outputs.
_KINDS = ("tokens", "logits")


def _request_seed(seed):
    return int(seed) if seed is not None and int(seed) >= 0 else secrets.randbits(31)


class _DeviceIds:
    """A sampled decode step's ids on device (`dev`), each request's device row (`rows`, vLLM's batch order) and,
    once a read is queued, its host destination."""

    __slots__ = ("dev", "rows", "host")

    def __init__(self, dev, rows):
        self.dev, self.rows, self.host = dev, rows, None


class TTLlamaForCausalLM:
    """Generator contract over the demo's serving pipeline (models.demos.minicpm5_2b.tt.serving)."""

    # Read by the plugin off the CLASS before the model exists.
    model_capabilities = {"supports_sample_on_device": True, "supports_async_decode": True}

    def __init__(self, pipe, max_batch_size, max_seq_len, mesh_device=None):
        self.pipe = pipe
        self.mesh_device = mesh_device
        self.max_batch_size = int(max_batch_size)
        self.max_seq_len = int(max_seq_len)
        self.vocab_size = int(getattr(pipe.config, "vocab_size", 0)) or None
        self.rows = int(pipe.batch)
        self.pad_id = int(pipe.pad_id)
        self._row_of_block = {}  # a live request's first KV block id -> its device row
        self._dev_pos = None  # every row's position on device now (None: unknown, write them all)
        self._dev_rule = None  # every row's rule on device now (None: never written)
        self._seeds = [None] * self.rows  # each row's request seed
        self._fresh = set()  # rows prefilled since their request's seed was taken
        self._host_gens = {}  # row -> the host sampler's generator for its request
        self.already_warmed_up_prefill = False
        self._traceable = mesh_device is not None
        self._traces = {}  # "tokens" / "logits" -> (trace id, its output tensor)
        self._untraceable = set()
        self._compiled = set()
        self._last_kind = None  # the last decode step's kind; "tokens": every row's next input is on device

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len=None, n_layers=None, **_ignored):
        msl = int(max_seq_len or _KV_CAPACITY)
        if msl > _KV_CAPACITY:
            raise ValueError(
                "%s: resident KV capacity is %d tokens; use --max-model-len %d or less (got %d)."
                % (cls.__name__, _KV_CAPACITY, _KV_CAPACITY, msl)
            )
        if _build_serving_pipeline is None:
            raise RuntimeError("serving module %s not importable: %s" % (_SERVING_MODULE, _import_error))
        pipe = _build_serving_pipeline(mesh_device, capacity=msl, layers=n_layers)
        if not 1 <= int(max_batch_size) <= int(pipe.batch):
            raise ValueError(
                "%s serves up to %d concurrent sequences (the pipeline's device rows); launch vLLM with "
                "--max-num-seqs %d or less (got %s)." % (cls.__name__, pipe.batch, pipe.batch, max_batch_size)
            )
        log.info(
            "built %s serving pipeline (device rows=%s, KV=%s, max seqs=%s)", _SLUG, pipe.batch, msl, max_batch_size
        )
        return cls(pipe, max_batch_size, msl, mesh_device)

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        return [None for _ in per_layer_specs]  # the pipeline owns its cache

    def allocate_kv_cache(self, kv_cache_shape=None, dtype=None, num_layers=None):
        return [None for _ in range(int(num_layers or 0))]

    def _vocab(self, logits):
        return logits[..., : self.vocab_size or logits.shape[-1]].float()

    def _programs(self):
        """Programs the device has compiled (None when the device does not say)."""
        count = getattr(self.mesh_device, "num_program_cache_entries", None)
        try:
            return int(count()) if callable(count) else None
        except Exception:  # noqa: BLE001
            return None

    def _run(self, kind):
        """One decode step on device: "tokens" -> the sampled ids, "logits" -> the logits."""
        return self.pipe.decode_tokens() if kind == "tokens" else self.pipe.decode_logits()

    def _capture(self, kind):
        """Record one decode step of this kind (recording runs nothing on the device); None when it cannot be."""
        tid = None
        try:
            tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            out = self._run(kind)
            ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
        except Exception as e:  # noqa: BLE001
            if tid is not None:
                try:
                    ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
                except Exception:  # noqa: BLE001
                    pass
                try:
                    ttnn.release_trace(self.mesh_device, tid)
                except Exception:  # noqa: BLE001
                    pass
            self._untraceable.add(kind)
            log.warning(
                "%s: %s decode trace capture failed (%s); those steps decode eagerly, much slower", _SLUG, kind, e
            )
            return None
        self._traces[kind] = (tid, out)
        log.info("%s: %s decode trace captured; every such decode token replays it", _SLUG, kind)
        return self._traces[kind]

    def _release_traces(self):
        for tid, _ in self._traces.values():
            try:
                ttnn.release_trace(self.mesh_device, tid)
            except Exception:  # noqa: BLE001
                pass
        self._traces = {}

    def _capture_all(self):
        """Record every compiled step kind afresh, in _KINDS order; a kind that cannot be recorded decodes eagerly."""
        self._release_traces()
        for kind in _KINDS:
            if kind in self._compiled and kind not in self._untraceable:
                self._capture(kind)

    def _step(self, kind, enable_trace):
        # Capturing before the step's programs have run once would compile them mid-capture.
        if enable_trace and self._traceable and kind in self._compiled and kind not in self._untraceable:
            if kind not in self._traces:
                self._capture_all()
            trace = self._traces.get(kind)
            if trace is not None:
                ttnn.execute_trace(self.mesh_device, trace[0], cq_id=0, blocking=False)
                return trace[1]
        # An eager step beside a held trace can clash with its L1 outputs, and a first run compiles programs whose
        # buffers may overlap it.
        self._release_traces()
        out = self._run(kind)
        self._compiled.add(kind)
        return out

    @staticmethod
    def _generator(seed):
        g = torch.Generator()
        g.manual_seed(_request_seed(seed))
        return g

    @staticmethod
    def _penalised(sp, n):
        return any(
            float(sp.presence_penalty[i]) != 0.0
            or float(sp.frequency_penalty[i]) != 0.0
            or float(sp.repetition_penalty[i]) != 1.0
            for i in range(n)
        )

    def _load_rule(self, sp, served):
        """Each served row's request rule -> the device sampler (written only on change); a row new to its request
        takes the request's seed (a random one when unset)."""
        K = int(self.pipe.top_k)
        seeds = getattr(sp, "seed", None)
        rule = list(self._dev_rule or [_GREEDY] * self.rows)
        for i, row in enumerate(served):
            if row in self._fresh:
                self._seeds[row] = _request_seed(seeds[i] if seeds is not None else None)
                self._fresh.discard(row)
            temp, top_k, top_p = float(sp.temperature[i]), int(sp.top_k[i]), float(sp.top_p[i])
            if temp <= 0.0 or top_k == 1:
                rule[row] = _GREEDY
            else:
                rule[row] = (1.0 / temp, min(top_k, K) if top_k > 0 else K, min(top_p, 1.0), self._seeds[row])
        if rule != self._dev_rule:
            inv_t, k, p, row_seeds = (list(c) for c in zip(*rule))
            self.pipe.set_sampling(inv_t, k, p, row_seeds)
            self._dev_rule = rule

    def _host_sample(self, logits, sp, served, prompt_tokens, output_tokens):
        """vLLM's rule on the host for a step whose requests carry penalties: logits [n, V] -> ids [n] int32."""
        ids = torch.empty((logits.shape[0],), dtype=torch.int32)
        seeds = getattr(sp, "seed", None)
        for i, row in enumerate(served):
            x = logits[i].clone()
            out = output_tokens[i][output_tokens[i] >= 0].long() if output_tokens is not None else x.new_empty(0).long()
            rep, pres, freq = (
                float(sp.repetition_penalty[i]),
                float(sp.presence_penalty[i]),
                float(sp.frequency_penalty[i]),
            )
            if rep != 1.0:
                seen = (
                    out if prompt_tokens is None else torch.cat([out, prompt_tokens[i][prompt_tokens[i] >= 0].long()])
                )
                seen = seen[seen < x.shape[0]].unique()
                x[seen] = torch.where(x[seen] > 0, x[seen] / rep, x[seen] * rep)
            if pres != 0.0 or freq != 0.0:
                count = torch.bincount(out[out < x.shape[0]], minlength=x.shape[0]).float()
                x -= freq * count + pres * (count > 0).float()
            temp, top_k, top_p = float(sp.temperature[i]), int(sp.top_k[i]), float(sp.top_p[i])
            if temp <= 0.0 or top_k == 1:
                ids[i] = int(torch.argmax(x))
                continue
            x = x / temp
            if 0 < top_k < x.shape[0]:
                x[x < torch.topk(x, top_k).values[-1]] = float("-inf")
            if top_p < 1.0:
                order = torch.argsort(x, descending=True)
                probs = torch.softmax(x[order], dim=-1)
                drop = torch.cumsum(probs, dim=-1) - probs >= top_p  # exclusive cumulative probability
                x[order[drop]] = float("-inf")
            if row not in self._host_gens:
                self._host_gens[row] = self._generator(seeds[i] if seeds is not None else None)
            ids[i] = int(torch.multinomial(torch.softmax(x, dim=-1), 1, generator=self._host_gens[row]))
        return ids

    def prefill_forward(
        self, tokens, page_table=None, *, prompt_lens=None, start_pos=None, empty_slots=None, **_compat
    ):
        n = int(tokens.shape[0])
        if start_pos is not None and int(torch.as_tensor(start_pos).max()) != 0:
            raise ValueError("chunked / prefix-cached prefill is not supported by this generator")
        rows = [int(s) for s in empty_slots] if empty_slots is not None else list(range(n))
        if len(rows) != n:
            raise ValueError("prefill of %d sequences got %d slots" % (n, len(rows)))
        lens = [int(prompt_lens[i]) if prompt_lens is not None else int(tokens.shape[1]) for i in range(n)]
        for T in lens:
            if T > self.max_seq_len:
                raise ValueError("prompt of %d tokens exceeds max_model_len=%d" % (T, self.max_seq_len))
        before = self._programs()
        logits = self.pipe.prefill_rows([tokens[i, : lens[i]].to(torch.int64).tolist() for i in range(n)], rows)
        if before is None or self._programs() != before:
            self._release_traces()
        for i, row in enumerate(rows):
            if page_table is not None:
                self._row_of_block[int(page_table[i, 0])] = row
            self._fresh.add(row)
            self._host_gens.pop(row, None)
        self._last_kind = None
        return self._vocab(torch.as_tensor(logits)).reshape(n, 1, -1)

    def decode_forward(
        self,
        tokens,
        start_pos=None,
        page_table=None,
        *,
        enable_trace=False,
        sampling_params=None,
        prompt_tokens=None,
        output_tokens=None,
        reset_batch=None,
        read_from_device=True,
        **_compat,
    ):
        tokens = torch.as_tensor(tokens).reshape(-1)
        n = int(tokens.shape[0])
        positions = (
            torch.as_tensor(start_pos).reshape(-1) if start_pos is not None else torch.zeros(n, dtype=torch.int64)
        )
        dev = self._dev_pos
        tok = torch.full((self.rows,), self.pad_id, dtype=torch.int32)
        pos = dev.clone() if dev is not None else torch.zeros((self.rows,), dtype=torch.int32)
        idle = torch.ones((self.rows,), dtype=torch.bool)
        served = []  # device row of each request, in vLLM's batch order (padding rows trail)
        for i in range(n):
            p = int(positions[i])
            if p < 0:
                break  # vLLM's padding rows
            row = self._row_of_block.get(int(page_table[i, 0])) if page_table is not None else i
            if row is None:
                raise RuntimeError("decode row %d (first KV block %d) was never prefilled" % (i, int(page_table[i, 0])))
            tok[row], pos[row], idle[row] = int(tokens[i]), p, False
            served.append(row)
        if served:
            tok[idle], pos[idle] = tok[served[0]], pos[served[0]]
        # vLLM's async decode submits a step before the last one's ids are back and applied, so its tokens are
        # placeholders and its positions one step behind: after a sampled step over the same rows (reset_batch
        # False) each row's next input id and position are the ones that step left on device.
        if reset_batch is False and self._last_kind == "tokens" and dev is not None:
            tok, pos = None, dev
        self.pipe.set_decode_rows(tok, None if dev is not None and torch.equal(pos, dev) else pos)
        self._dev_pos = None
        sp = sampling_params
        on_device = sp is not None and not self._penalised(sp, len(served))
        if on_device:
            self._load_rule(sp, served)
        kind = "tokens" if on_device else "logits"
        out = self._step(kind, enable_trace)
        self._last_kind = kind
        self._dev_pos = pos + 1
        rows = torch.tensor(served, dtype=torch.long)
        if on_device:
            ids = _DeviceIds(out, rows)
            return ids if not read_from_device else self.process_decode_output_host(ids)
        logits = self._vocab(self.pipe.logits_to_torch(out)).index_select(0, rows)
        if sp is not None:  # vLLM wants ids, but a penalty needs the token history the device does not keep
            return self._host_sample(logits, sp, served, prompt_tokens, output_tokens)
        return logits.unsqueeze(1)

    def read_decode_output(self, tt_out, async_read=False, **_compat):
        """vLLM's async decode: start reading a sampled step's ids back without waiting (the copy is queued
        behind the step); returns (the pending output, [its event]). Host outputs pass through."""
        if not isinstance(tt_out, _DeviceIds):
            return (tt_out, []) if async_read else tt_out
        if not async_read:
            return self.process_decode_output_host(tt_out)
        tt_out.host = tt_out.dev.cpu(blocking=False)
        return tt_out, [ttnn.record_event(self.mesh_device, 0)]

    def process_decode_output_host(self, tt_out, is_tokens=True, **_compat):
        """A sampled step's ids (read, or still on device) -> int32 [n] in vLLM's batch order."""
        if not isinstance(tt_out, _DeviceIds):
            return tt_out
        host = tt_out.host if tt_out.host is not None else tt_out.dev.cpu(blocking=True)
        ids = ttnn.to_torch(ttnn.get_device_tensors(host)[0]).reshape(-1)[: self.rows].to(torch.int64)  # may be uint32
        return ids.index_select(0, tt_out.rows).to(torch.int32)

    def warmup_model_prefill(self, *, kv_cache=None, can_sample_on_device=False, enable_trace=False, **_compat):
        if enable_trace:
            return  # prefill stays eager: its length follows the prompt
        bos = getattr(self.pipe.config, "bos_token_id", None)
        bos = int(bos) if bos is not None else self.pad_id
        T = min(_WARM_PREFILL_LEN, self.max_seq_len - 2)
        n = 1
        while n <= self.max_batch_size:
            self.prefill_forward(
                torch.full((n, T), bos, dtype=torch.int64), None, prompt_lens=[T] * n, empty_slots=list(range(n))
            )
            n *= 2
        self.already_warmed_up_prefill = True

    def warmup_model_decode(
        self,
        *,
        kv_cache=None,
        max_batch_size=1,
        num_blocks=0,
        can_sample_on_device=False,
        enable_trace=False,
        **_compat,
    ):
        if enable_trace:
            if not self._traceable:
                log.warning(
                    "%s: vLLM asked for traced decode without a mesh device; decoding eagerly, which is much slower",
                    _SLUG,
                )
                return
            self._capture_all()
            return
        if not self.already_warmed_up_prefill:
            return
        T = min(_WARM_PREFILL_LEN, self.max_seq_len - 2)
        self.decode_forward(torch.zeros((1, 1), dtype=torch.int64), torch.tensor([T]), None)
        greedy = SimpleNamespace(
            temperature=[0.0],
            top_k=[1],
            top_p=[1.0],
            presence_penalty=[0.0],
            frequency_penalty=[0.0],
            repetition_penalty=[1.0],
            seed=[None],
        )
        self.decode_forward(torch.zeros((1, 1), dtype=torch.int64), torch.tensor([T + 1]), None, sampling_params=greedy)

    def cleanup(self):
        self._release_traces()

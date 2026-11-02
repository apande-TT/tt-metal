# SPDX-License-Identifier: Apache-2.0
"""vLLM generator for nvidia_nemotron_3_5_lightning_30b_a3b_bf16 (NemotronHForCausalLM) on Tenstorrent.

The tt-metal vLLM plugin registers this class as TTNemotronHForCausalLM via vllm_metadata.json and drives it through
the generator contract (initialize_vllm_model -> allocate_kv_cache_per_layer -> warmups ->
prefill_forward / decode_forward). It wraps the demo's serving pipeline (tt/serve.py).

Serve profile: up to 32 concurrent sequences (--max-num-seqs 32, the pipeline's 32 device rows), prompt and
generated tokens bounded by the pipeline KV capacity (--max-model-len 2048), and --max-num-batched-tokens 4096 so 32
128-token prompts that arrive together are read in one prefill rather than two, the second stalling the first's decode.

Sampling: with --additional-config '{"tt": {"sample_on_device_mode": "decode_only"}}' every decode token is sampled
on device by the pipeline's own sampler, each row by its request's temperature / top_k / top_p (greedy is top_k 1),
among the row's 128 most likely tokens (generation_config sets top_p 0.95 and no top_k), from a uniform per position drawn from the request's seed
(a random one when unset); only the token ids are read back. vLLM samples on the host whenever a request needs
what the device does not do (logprobs, min_p, logit bias, ...): that step returns the logits. A request with a
repetition / presence / frequency penalty is sampled on the host here, from the logits and its token history.
Prefill always returns the logits (vLLM samples the first token).

Each request owns one device row for its whole life: the plugin hands prefill the row (`empty_slots`), the
prompt's Mamba state and K,V go into that row only, and the row is remembered by the request's first KV block id (unique per
live request; block 0 is vLLM's null block, so padding rows never match). Decode writes every row's input token
before the step, and the positions and the served-row mask only when they change (a request joined or left).
Rows with no request repeat a served row's position; whatever they write into their own state and cache rows is
rewritten by that row's next prefill -- a preempted request is re-prefilled on resume.

The prompts of one prefill call run together: sorted by length, in batches of a power-of-two count of prompts
(few distinct programs) of at most _PREFILL_TOKENS padded tokens, each batch one pass through the layers.

Decode is traced whenever vLLM asks for it (enable_trace): one step of each kind (sampled ids, logits) is captured
and replayed for every token, the per-row inputs written in place before each replay. Under vLLM's async scheduling
a sampled step's ids are read back without waiting and the next step is queued before they arrive (its input ids
are the ones the last step left on device), so the chip runs while vLLM handles the last step. Prefill stays eager (its
length follows the prompt). A prefill batch shape not seen before compiles new programs whose buffers may overlap
the captured traces, so they are released then and captured again at the next decode.
"""
from __future__ import annotations

import logging
import secrets
from types import SimpleNamespace

import torch

try:
    import ttnn  # noqa: F401
    from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16.tt.serve import KV_CAPACITY as _KV_CAPACITY
    from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16.tt.serve import build_pipeline as _build_pipeline
    from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16.tt.serve import read_host as _read_host
except Exception:  # available only inside the serving image
    ttnn = None
    _build_pipeline = None
    _read_host = None
    _KV_CAPACITY = 2048

log = logging.getLogger(__name__)

# Padded prompt tokens one prefill batch runs: the 32-user, 128-token batch the pipeline's prefill was sized for.
_PREFILL_TOKENS = 4096
_WARM_PREFILL_LEN = 128
_WARM_PREFILL_BATCHES = (1, 2, 4, 8, 16, 32)
# The mesh the pipeline runs on: tensor parallel over all four chips.
MESH_ROWS, MESH_COLS = 1, 4


def _padded(T):
    return max(32, -(-int(T) // 32) * 32)


class _DeviceIds:
    """A sampled decode step's ids [1, 1, rows, 1] on device (`dev`, the step's output), each request's device
    row (`rows`, vLLM's batch order) and, once a read is queued, its host destination."""

    __slots__ = ("dev", "rows", "host")

    def __init__(self, dev, rows):
        self.dev, self.rows, self.host = dev, rows, None


class TTNemotronHForCausalLM:
    """Generator contract over the demo serving pipeline (tt/serve.py)."""

    # Read by the plugin off the CLASS before the model exists.
    model_capabilities = {
        "supports_sample_on_device": True,
        "supports_async_decode": True,
        "fabric_config": {"config": ttnn.FabricConfig.FABRIC_1D} if ttnn is not None else None,
    }

    def __init__(self, pipe, max_batch_size, max_seq_len, mesh_device=None):
        self.pipe = pipe
        self.mesh_device = mesh_device
        self.max_batch_size = int(max_batch_size)
        self.max_seq_len = int(max_seq_len)
        self.vocab_size = int(getattr(pipe.config, "vocab_size", 0)) or None
        self.rows = int(pipe.batch)
        self.pad_id = int(pipe.settings.pad_id)
        self._row_of_block = {}  # a live request's first KV block id -> its device row
        self._dev_pos = None  # every row's position on device now (None: unknown, write them all)
        pipe.serve_rows()
        self._dev_on = torch.ones((self.rows,), dtype=torch.float32)  # rows serving a request now
        s = pipe.settings
        full = lambda v: torch.full((self.rows,), float(v))  # noqa: E731
        self._dev_rule = (full(s.top_k), full(s.top_p), full(1.0 / s.temperature))  # each row's rule on device
        self._uniforms = None  # host copy of the device u_table [rows, C]
        self._fresh = set()  # rows prefilled since their request's uniforms were written
        self._host_gens = {}  # row -> the host sampler's generator for its request
        self.already_warmed_up_prefill = False
        self._traceable = mesh_device is not None
        self._traces = {}  # "tokens" / "logits" -> (trace id, its output tensor)
        self._untraceable = set()
        self._compiled = set()
        self._prefill_shapes = set()
        self._last_kind = None  # the last decode step's kind; "tokens": every row's next input is on device

    @classmethod
    def get_max_tokens_all_users(
        cls, model_name="", num_devices=1, tt_data_parallel=1, max_model_len=0, max_num_seqs=1, **_ignored
    ):
        """KV token budget vLLM sizes its block pool from: every row's full capacity."""
        return int(max_num_seqs) * int(min(int(max_model_len or _KV_CAPACITY), _KV_CAPACITY))

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len=None, n_layers=None, **_ignored):
        msl = int(max_seq_len or _KV_CAPACITY)
        if msl > _KV_CAPACITY:
            raise ValueError(
                "%s: resident KV capacity is %d tokens; use --max-model-len %d or less (got %d)."
                % (cls.__name__, _KV_CAPACITY, _KV_CAPACITY, msl)
            )
        if _build_pipeline is None:
            raise RuntimeError("tt/serve.py not importable; run inside the serving image")
        shape = tuple(int(v) for v in mesh_device.shape)
        if shape != (MESH_ROWS, MESH_COLS):
            if int(mesh_device.get_num_devices()) != MESH_ROWS * MESH_COLS:
                raise ValueError("%s needs %d chips; got mesh %s" % (cls.__name__, MESH_ROWS * MESH_COLS, shape))
            log.info("reshaping mesh %s -> (%d, %d)", shape, MESH_ROWS, MESH_COLS)
            mesh_device.reshape(ttnn.MeshShape(MESH_ROWS, MESH_COLS))
        pipe = _build_pipeline(mesh_device, layers=n_layers)
        if not 1 <= int(max_batch_size) <= int(pipe.batch):
            raise ValueError(
                "%s serves up to %d concurrent sequences (the pipeline's device rows); launch vLLM with "
                "--max-num-seqs %d or less (got %s)." % (cls.__name__, pipe.batch, pipe.batch, max_batch_size)
            )
        log.info(
            "built nemotron pipeline (device rows=%s, KV=%s, max seqs=%s)", pipe.batch, _KV_CAPACITY, max_batch_size
        )
        return cls(pipe, max_batch_size, msl, mesh_device)

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        return [None for _ in per_layer_specs]  # the pipeline owns its cache

    def allocate_kv_cache(self, kv_cache_shape=None, dtype=None, num_layers=None):
        return [None for _ in range(int(num_layers or 0))]

    def _host_logits(self, logits_dev):
        """Device logits [1, 1, rows, V(+pad)] -> host fp32 [rows, V]."""
        host = _read_host(logits_dev).reshape(self.rows, -1)
        v = self.vocab_size or host.shape[-1]
        return host[:, :v].float()

    def _run(self, kind):
        """One decode step on device: "tokens" -> sampled ids [1, 1, rows, 1], "logits" -> the gathered logits."""
        return self.pipe.decode_tokens() if kind == "tokens" else self.pipe.decode_step()["logits"]

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
                "nemotron: %s decode trace capture failed (%s); those steps decode eagerly, much slower", kind, e
            )
            return None
        self._traces[kind] = (tid, out)
        log.info("nemotron: %s decode trace captured; every such decode token replays it", kind)
        return self._traces[kind]

    def _release_traces(self):
        for tid, _ in self._traces.values():
            try:
                ttnn.release_trace(self.mesh_device, tid)
            except Exception:  # noqa: BLE001
                pass
        self._traces = {}

    def _step(self, kind, enable_trace):
        # Capturing before the step's programs have run once would compile them mid-capture.
        if enable_trace and self._traceable and kind in self._compiled and kind not in self._untraceable:
            trace = self._traces.get(kind) or self._capture(kind)
            if trace is not None:
                ttnn.execute_trace(self.mesh_device, trace[0], cq_id=0, blocking=False)
                return trace[1]
        if kind not in self._compiled:
            self._release_traces()  # the first run compiles programs, whose buffers may overlap a captured trace
        out = self._run(kind)
        self._compiled.add(kind)
        return out

    @staticmethod
    def _generator(seed):
        g = torch.Generator()
        g.manual_seed(int(seed) if seed is not None and int(seed) >= 0 else secrets.randbits(63))
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
        """Each served row's request rule -> the device's per-row sampling buffers (written only on change), and
        uniforms drawn from the seed of every request new to its row."""
        K = int(self.pipe.sampler.top_k)
        k, p, inv_t = (t.clone() for t in self._dev_rule)
        for i, row in enumerate(served):
            temp, top_k, top_p = float(sp.temperature[i]), int(sp.top_k[i]), float(sp.top_p[i])
            if temp <= 0.0 or top_k == 1:
                k[row], p[row], inv_t[row] = 1.0, 1.0, 1.0
            else:
                k[row], p[row], inv_t[row] = (
                    float(min(top_k, K)) if top_k > 0 else float(K),
                    min(top_p, 1.0),
                    1.0 / temp,
                )
        if not all(torch.equal(a, b) for a, b in zip((k, p, inv_t), self._dev_rule)):
            self.pipe.set_sampling(k, p, inv_t)
            self._dev_rule = (k, p, inv_t)
        fresh = [(i, row) for i, row in enumerate(served) if row in self._fresh]
        if fresh:
            if self._uniforms is None:
                self._uniforms = torch.rand((self.rows, int(self.pipe.u_table.shape[-1])))
            for i, row in fresh:
                self._uniforms[row] = torch.rand(self._uniforms.shape[1], generator=self._generator(sp.seed[i]))
                self._fresh.discard(row)
            self.pipe.set_uniforms(self._uniforms)

    def _host_sample(self, logits, sp, served, prompt_tokens, output_tokens):
        """vLLM's rule on the host for a step whose requests carry penalties: logits [n, V] -> ids [n] int32."""
        ids = torch.empty((logits.shape[0],), dtype=torch.int32)
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
                self._host_gens[row] = self._generator(sp.seed[i])
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
        out = [None] * n
        for group in self._prefill_groups(lens):
            shape = (len(group), _padded(max(lens[i] for i in group)))
            if shape not in self._prefill_shapes:
                self._prefill_shapes.add(shape)
                self._release_traces()
            logits = self.pipe.prefill_rows(
                [tokens[i, : lens[i]].to(torch.int64).tolist() for i in group], [rows[i] for i in group]
            )
            host = self._host_logits(logits)
            for j, i in enumerate(group):
                out[i] = host[j]
        for i, row in enumerate(rows):
            if page_table is not None:
                self._row_of_block[int(page_table[i, 0])] = row
            self._fresh.add(row)
            self._host_gens.pop(row, None)
        self._last_kind = None
        return torch.stack(out).reshape(n, 1, -1)

    @staticmethod
    def _prefill_groups(lens):
        """Prompt indices in prefill batches: by length (a batch pads little), each a power-of-two count of
        prompts of at most _PREFILL_TOKENS padded tokens."""
        order = sorted(range(len(lens)), key=lambda i: lens[i])
        groups = []
        while order:
            g = 1 << (len(order).bit_length() - 1)
            while g > 1 and g * _padded(lens[order[g - 1]]) > _PREFILL_TOKENS:
                g //= 2
            groups.append(order[:g])
            order = order[g:]
        return groups

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
        # Rows with no request repeat a served row's position, so attention over them costs what that row's does.
        # What they write into their own state and cache rows is rewritten by the row's next prefill.
        if served:
            tok[idle], pos[idle] = tok[served[0]], pos[served[0]]
        on = (~idle).float()
        new_on = None if torch.equal(on, self._dev_on) else on
        # vLLM's async decode submits a step before the last one's ids are back and applied, so its tokens are
        # placeholders and its positions one step behind: after a sampled step over the same rows (reset_batch
        # False) each row's next input id and position are the ones that step left on device.
        if reset_batch is False and self._last_kind == "tokens" and dev is not None:
            tok, pos = None, dev
        if dev is not None and torch.equal(pos, dev):
            self.pipe.set_decode_rows(tok, active=new_on)  # steady decode: the device advanced every row itself
        else:
            self.pipe.set_decode_rows(tok, pos, active=new_on)
        self._dev_on = on
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
        logits = self._host_logits(out).index_select(0, rows)
        if sp is not None:  # vLLM wants ids, but a penalty needs the token history the device does not keep
            return self._host_sample(logits, sp, served, prompt_tokens, output_tokens)
        return logits.unsqueeze(1)

    def read_decode_output(self, tt_out, async_read=False):
        """vLLM's async decode: start reading a sampled step's ids back without waiting (the copy is queued
        behind the step); returns (the pending output, [its event]). Host outputs pass through."""
        if not isinstance(tt_out, _DeviceIds):
            return (tt_out, []) if async_read else tt_out
        if not async_read:
            return self.process_decode_output_host(tt_out)
        tt_out.host = tt_out.dev.cpu(blocking=False)
        return tt_out, [ttnn.record_event(self.mesh_device, 0)]

    def process_decode_output_host(self, tt_out, is_tokens=True):
        """A sampled step's ids (read, or still on device) -> int32 [n] in vLLM's batch order."""
        if not isinstance(tt_out, _DeviceIds):
            return tt_out
        host = tt_out.host if tt_out.host is not None else tt_out.dev.cpu(blocking=True)
        ids = ttnn.to_torch(ttnn.get_device_tensors(host)[0])
        return ids.reshape(-1)[: self.rows].index_select(0, tt_out.rows).to(torch.int32)

    def warmup_model_prefill(self, *, kv_cache=None, can_sample_on_device=False, enable_trace=False):
        if enable_trace:
            return  # prefill stays eager: its length follows the prompt
        bos = int(getattr(self.pipe.config, "bos_token_id", 1) or 1)
        self.pipe.decode_prefill([[bos] * 8])  # sizes every resident buffer the decode step reads
        self._dev_pos = None
        self.prefill_forward(torch.full((1, 8), bos, dtype=torch.int64), None, prompt_lens=[8], empty_slots=[0])
        # The batch shapes concurrent 128-token prompts take (with --max-num-batched-tokens 4096: up to 32 at once),
        # compiled now rather than under a client's first requests. Varied tokens, as real prompts: the MoE's
        # per-expert buffers are sized by how the tokens route.
        g = torch.Generator().manual_seed(0)
        for n in _WARM_PREFILL_BATCHES:
            ids = torch.randint(1000, int(self.vocab_size or 32000), (n, _WARM_PREFILL_LEN), generator=g)
            self.prefill_forward(ids, None, prompt_lens=[_WARM_PREFILL_LEN] * n, empty_slots=list(range(n)))
        self.already_warmed_up_prefill = True

    def warmup_model_decode(
        self, *, kv_cache=None, max_batch_size=1, num_blocks=0, can_sample_on_device=False, enable_trace=False
    ):
        if enable_trace:
            if not self._traceable:
                log.warning(
                    "nemotron: vLLM asked for traced decode without a mesh device; decoding eagerly, which is much slower"
                )
                return
            for kind in ("tokens", "logits"):
                if kind in self._compiled and kind not in self._traces and kind not in self._untraceable:
                    self._capture(kind)
            return
        if not self.already_warmed_up_prefill:
            return
        self.decode_forward(torch.zeros((1, 1), dtype=torch.int64), torch.tensor([8]), None)
        greedy = SimpleNamespace(
            temperature=[0.0],
            top_k=[1],
            top_p=[1.0],
            presence_penalty=[0.0],
            frequency_penalty=[0.0],
            repetition_penalty=[1.0],
            seed=[None],
        )
        self.decode_forward(torch.zeros((1, 1), dtype=torch.int64), torch.tensor([9]), None, sampling_params=greedy)

    def cleanup(self):
        self._release_traces()

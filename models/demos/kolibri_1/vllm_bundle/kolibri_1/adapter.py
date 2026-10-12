# SPDX-License-Identifier: Apache-2.0
"""vLLM generator for kolibri_1 (Kolibri1ForCausalLM) on Tenstorrent.

The tt-metal vLLM plugin registers this class as TTKolibri1ForCausalLM via vllm_metadata.json and drives it through
the generator contract (initialize_vllm_model -> allocate_kv_cache_per_layer -> warmups ->
prefill_forward / decode_forward). Kolibri1ForCausalLM is not a plugin built-in, so this wraps the demo's own
resident pipeline (models.demos.kolibri_1.tt.pipeline) rather than a stock generator.

Starting serve profile (matches the validated demo): ONE sequence at a time (--max-num-seqs 1), prompt
and generated tokens bounded by the pipeline KV capacity (--max-model-len 2048), host-side sampling
(logits are returned to vLLM). Concurrent users / long context / on-device sampling are model work; see
the TODOs below.
"""
from __future__ import annotations

import logging

import torch

try:
    import ttnn  # noqa: F401
    from models.demos.kolibri_1.tt.pipeline import build_pipeline as _build_pipeline, read_host as _read_host
    try:
        from models.demos.kolibri_1.tt.pipeline import KV_CAPACITY as _KV_CAPACITY
    except Exception:  # noqa: BLE001
        _KV_CAPACITY = 2048
except Exception:  # available only inside the serving image
    _build_pipeline = None
    _read_host = None
    _KV_CAPACITY = 2048

log = logging.getLogger(__name__)


class TTKolibri1ForCausalLM:
    """Generator contract over the demo resident pipeline (models.demos.kolibri_1.tt.pipeline)."""

    # Read by the plugin off the CLASS before the model exists.
    model_capabilities = {"supports_sample_on_device": False}

    def __init__(self, pipe, max_batch_size, max_seq_len):
        self.pipe = pipe
        self.max_batch_size = int(max_batch_size)
        self.max_seq_len = int(max_seq_len)
        self.vocab_size = int(getattr(pipe.config, "vocab_size", 0)) or None
        self._state = None
        self._active = False
        self.already_warmed_up_prefill = False

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len=None, n_layers=None, **_ignored):
        if int(max_batch_size) != 1:
            raise ValueError(
                "%s serves one sequence at a time; launch vLLM with --max-num-seqs 1 (got %s)."
                % (cls.__name__, max_batch_size)
            )
        msl = int(max_seq_len or _KV_CAPACITY)
        if msl > _KV_CAPACITY:
            raise ValueError(
                "%s: resident KV capacity is %d tokens; use --max-model-len %d or less (got %d)."
                % (cls.__name__, _KV_CAPACITY, _KV_CAPACITY, msl)
            )
        if _build_pipeline is None:
            raise RuntimeError("pipeline module models.demos.kolibri_1.tt.pipeline not importable; run inside the serving image")
        pipe = _build_pipeline(mesh_device, layers=n_layers)
        log.info("built kolibri_1 pipeline (device rows=%s, KV=%s)", getattr(pipe, "batch", "?"), _KV_CAPACITY)
        return cls(pipe, max_batch_size, msl)

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        return [None for _ in per_layer_specs]  # the pipeline owns its cache

    def allocate_kv_cache(self, kv_cache_shape=None, dtype=None, num_layers=None):
        return [None for _ in range(int(num_layers or 0))]

    def _host_logits_row0(self, logits_dev):
        host = _read_host(logits_dev).reshape(-1)
        v = self.vocab_size or host.numel()
        return host[:v].float().reshape(1, 1, v)

    def prefill_forward(self, tokens, page_table=None, *, prompt_lens=None, start_pos=None, **_compat):
        if tokens.shape[0] != 1:
            raise ValueError("one sequence at a time: got a prefill batch of %d" % tokens.shape[0])
        if start_pos is not None and int(torch.as_tensor(start_pos).max()) != 0:
            raise ValueError("chunked / prefix-cached prefill is not supported by this generator")
        T = int(prompt_lens[0]) if prompt_lens is not None else int(tokens.shape[1])
        if T > self.max_seq_len:
            raise ValueError("prompt of %d tokens exceeds max_model_len=%d" % (T, self.max_seq_len))
        ids = tokens[0, :T].to(torch.int64).tolist()
        self._state = self.pipe.decode_prefill([ids])  # seeds the resident caches for every device row
        self._active = True
        return self._host_logits_row0(self._state["logits"])

    def decode_forward(self, tokens, start_pos=None, page_table=None, **_compat):
        if not self._active:
            raise RuntimeError("decode_forward before prefill_forward")
        # TODO(author): vLLM sampled `tokens` on the host. The demo pipeline commits its OWN
        #   device-sampled token, so to honor vLLM's token exactly (non-greedy sampling, penalties)
        #   write that id into the pipeline resident next-token buffer before stepping. Greedy decode
        #   matches without this; wire it for full sampler fidelity.
        self._state = self.pipe.decode_step(self._state)
        return self._host_logits_row0(self._state["logits"])

    def warmup_model_prefill(self, *, kv_cache=None, can_sample_on_device=False, enable_trace=False):
        if enable_trace:
            return  # eager generator; trace warmup is a no-op
        bos = int(getattr(self.pipe.config, "bos_token_id", 1) or 1)
        self.prefill_forward(torch.full((1, 8), bos, dtype=torch.int64), None, prompt_lens=[8])
        self.already_warmed_up_prefill = True

    def warmup_model_decode(self, *, kv_cache=None, max_batch_size=1, num_blocks=0, can_sample_on_device=False, enable_trace=False):
        if enable_trace or not self._active:
            return
        self.decode_forward(torch.zeros((1, 1), dtype=torch.int64), None, None)
        self._active = False  # warmup sequence done; next prefill reseeds

    def cleanup(self):
        pass

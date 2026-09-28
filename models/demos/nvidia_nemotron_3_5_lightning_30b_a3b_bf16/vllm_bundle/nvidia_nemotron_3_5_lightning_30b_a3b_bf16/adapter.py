# SPDX-License-Identifier: Apache-2.0
"""vLLM generator for nvidia_nemotron_3_5_lightning_30b_a3b_bf16 (NemotronHForCausalLM) on Tenstorrent.

The tt-metal vLLM plugin (vllm-tt-plugin) registers this class as ``TTNemotronHForCausalLM`` through the
``vllm_metadata.json`` beside it, and drives it through the generator contract its ``TTModelLoader`` /
``TTModelRunner`` expect: ``initialize_vllm_model`` -> ``allocate_kv_cache_per_layer`` -> warmups ->
``prefill_forward`` / ``decode_forward``.

What this generator is, and is not
----------------------------------
It wraps the demo's resident pipeline (``tt/pipeline.py``): the optimized 52-block hybrid decoder on the
TP=2 x DP=2 mesh with its own Mamba conv/SSM state and attention K/V cache. That pipeline advances EVERY
batch row at one shared position and its attention cache is ``_ssm_cache.KV_CAPACITY`` tokens deep, so:

* it serves ONE sequence at a time (``--max-num-seqs 1``): the whole 32-row device batch runs the same
  sequence, row 0 is read back;
* prompt + generated tokens are bounded by ``KV_CAPACITY`` (``--max-model-len`` at most that);
* sampling happens on the host in vLLM (logits are returned), so every sampler feature works, including
  the repetition penalty this model wants under greedy decoding;
* no chunked prefill, no prefix caching, no trace yet (eager decode).

Concurrent users and long context need per-slot state fill and per-row attention positions in the model,
which is model work, not adapter work. Until then the manifest's serve profile pins the limits above.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import torch

import ttnn
from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16._stubs import _ssm_cache
from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16.tt import pipeline as P

log = logging.getLogger(__name__)

KV_CAPACITY = int(_ssm_cache.KV_CAPACITY)
MESH_ROWS, MESH_COLS = 2, 2  # DP x TP the pipeline is built for


def _mesh_shape(mesh_device) -> tuple[int, int] | None:
    try:
        s = mesh_device.shape
        return int(s[0]), int(s[1])
    except Exception:  # noqa: BLE001
        try:
            return int(mesh_device.shape.num_rows), int(mesh_device.shape.num_cols)
        except Exception:  # noqa: BLE001
            return None


class TTNemotronHForCausalLM:
    """Generator contract implementation over ``P.NemotronHPipeline``."""

    # Read by TTPlatform / TTModelRunner off the CLASS before the model exists.
    model_capabilities = {
        "supports_sample_on_device": False,  # logits back to the host; vLLM samples
        "supports_async_decode": False,  # decode_forward returns host tensors synchronously
        "supports_chunked_prefill": False,
        "supports_prefix_caching": False,
        "supports_device_penalties": False,
        "fabric_config": {"config": ttnn.FabricConfig.FABRIC_1D},
    }

    def __init__(self, pipe, max_batch_size: int, max_seq_len: int):
        self.pipe = pipe
        self.max_batch_size = int(max_batch_size)
        self.max_seq_len = int(max_seq_len)
        self.vocab_size = int(pipe.vocab_size)
        self.already_warmed_up_prefill = False
        self._pos = 0  # tokens resident in the device state for the active sequence
        self._active = False

    # ------------------------------------------------------------------ #
    #  class-level contract
    # ------------------------------------------------------------------ #
    @classmethod
    def get_max_tokens_all_users(
        cls,
        model_name: str = "",
        num_devices: int = 1,
        tt_data_parallel: int = 1,
        max_model_len: int = 0,
        max_num_seqs: int = 1,
    ) -> int:
        """KV token budget vLLM sizes its block pool from: one sequence of at most KV_CAPACITY tokens."""
        cap = int(max_model_len) if max_model_len else KV_CAPACITY
        return int(min(cap, KV_CAPACITY))

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations=None,
        **_ignored: Any,
    ):
        if int(max_batch_size) != 1:
            raise ValueError(
                f"{cls.__name__} serves one sequence at a time; launch vLLM with --max-num-seqs 1 "
                f"(got max_num_seqs={max_batch_size})."
            )
        if int(max_seq_len) > KV_CAPACITY:
            raise ValueError(
                f"{cls.__name__}: the resident attention cache holds {KV_CAPACITY} tokens; launch vLLM with "
                f"--max-model-len {KV_CAPACITY} or less (got {max_seq_len})."
            )
        n_dev = int(mesh_device.get_num_devices())
        shape = _mesh_shape(mesh_device)
        if n_dev == MESH_ROWS * MESH_COLS and shape is not None and shape != (MESH_ROWS, MESH_COLS):
            log.info(
                "reshaping mesh %s -> (%d, %d) for the TP=%d x DP=%d pipeline",
                shape,
                MESH_ROWS,
                MESH_COLS,
                MESH_COLS,
                MESH_ROWS,
            )
            mesh_device.reshape(ttnn.MeshShape(MESH_ROWS, MESH_COLS))
        elif n_dev != MESH_ROWS * MESH_COLS:
            raise ValueError(
                f"{cls.__name__} needs a {MESH_ROWS}x{MESH_COLS} mesh (4 chips); got {n_dev} device(s) {shape}"
            )
        # Full depth unless the caller caps it; the pipeline loads its own weights (bf16 reference on the host)
        # from the HF id the demo is bound to, via the mounted HF cache.
        pipe = P.build_pipeline(mesh_device, layers=n_layers, batch=P.BATCH)
        log.info("built depth=%d batch=%d mesh=%s", pipe.n_layers, pipe.batch, pipe.describe().get("mesh_shape"))
        return cls(pipe, max_batch_size, max_seq_len)

    # ------------------------------------------------------------------ #
    #  cache: owned by the pipeline, vLLM's block tables are bookkeeping only
    # ------------------------------------------------------------------ #
    def allocate_kv_cache_per_layer(self, per_layer_specs):
        return [None for _ in per_layer_specs]

    def allocate_kv_cache(self, kv_cache_shape=None, dtype=None, num_layers=None):
        return [None for _ in range(int(num_layers or 0))]

    # ------------------------------------------------------------------ #
    #  forwards
    # ------------------------------------------------------------------ #
    def _host_logits_row0(self, logits_dev) -> torch.Tensor:
        """Device (1,B,V)/(B,1,V) logits -> host float32 (1, 1, V) for the active row."""
        host = P._first_shard(logits_dev).reshape(self.pipe.batch, -1)[:, : self.vocab_size].float()
        ttnn.deallocate(logits_dev)
        return host[0:1].unsqueeze(1)

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        page_table: torch.Tensor | None = None,
        *,
        enable_trace: bool = False,
        prompt_lens: Sequence[int] | torch.Tensor | None = None,
        start_pos: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,
        kv_cache: Any = None,
        sampling_params: Any = None,
        **_compat: Any,
    ) -> torch.Tensor:
        if tokens.shape[0] != 1:
            raise ValueError(f"one sequence at a time: got a prefill batch of {tokens.shape[0]}")
        if start_pos is not None and int(torch.as_tensor(start_pos).max()) != 0:
            raise ValueError("chunked / prefix-cached prefill is not supported by this generator")
        T = int(prompt_lens[0]) if prompt_lens is not None else int(tokens.shape[1])
        if T > self.max_seq_len:
            raise ValueError(f"prompt of {T} tokens exceeds max_model_len={self.max_seq_len}")
        ids = tokens[0, :T].to(torch.int64).reshape(1, T).repeat(self.pipe.batch, 1)
        logits = self.pipe.prefill_fill(ids)  # seeds Mamba/attention state for all rows, dec_ids <- argmax
        self._pos = T
        self._active = True
        return self._host_logits_row0(logits)

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor | None = None,
        page_table: torch.Tensor | None = None,
        *,
        enable_trace: bool = False,
        kv_cache: Any = None,
        sampling_params: Any = None,
        reset_batch: bool = False,
        read_from_device: bool = True,
        **_compat: Any,
    ) -> torch.Tensor:
        if not self._active:
            raise RuntimeError("decode_forward before prefill_forward")
        if self._pos >= KV_CAPACITY:
            raise RuntimeError(f"sequence reached the resident cache capacity ({KV_CAPACITY} tokens)")
        tok = int(tokens.reshape(-1)[0])
        # vLLM sampled this token on the host; make it the token the device consumes next
        # (the pipeline had written its own greedy argmax into dec_ids).
        new = self.pipe._ids_to_device(torch.full((self.pipe.batch, 1), tok, dtype=torch.int64))
        ttnn.copy(new, self.pipe._persistent["dec_ids"])
        ttnn.deallocate(new)
        logits = self.pipe._decode_token()
        self._pos += 1
        return self._host_logits_row0(logits)

    # ------------------------------------------------------------------ #
    #  warmup: compile every kernel once so the first request is not the JIT
    # ------------------------------------------------------------------ #
    def warmup_model_prefill(
        self, *, kv_cache: Any = None, can_sample_on_device: bool = False, enable_trace: bool = False
    ) -> None:
        if enable_trace:
            return  # eager-only generator; trace warmup is a no-op
        bos = int(getattr(self.pipe.config, "bos_token_id", 1) or 1)
        ids = torch.full((1, 8), bos, dtype=torch.int64)
        self.prefill_forward(ids, None, enable_trace=False, prompt_lens=[8])
        self.already_warmed_up_prefill = True

    def warmup_model_decode(
        self,
        *,
        kv_cache: Any = None,
        max_batch_size: int = 1,
        num_blocks: int = 0,
        can_sample_on_device: bool = False,
        enable_trace: bool = False,
    ) -> None:
        if enable_trace or not self._active:
            return
        self.decode_forward(torch.zeros((1, 1), dtype=torch.int64), None, None, enable_trace=False)
        self._active = False  # the warmup sequence is done; the next prefill reseeds the state

    def cleanup(self) -> None:
        pass

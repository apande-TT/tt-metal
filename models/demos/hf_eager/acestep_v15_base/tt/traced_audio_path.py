# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Trace + 2-CQ execution for ACE-Step audio tokenizer → detokenizer (Call B + D).

Chains the two subsystems in one device trace so ``quantized`` stays on-device
between them (no D2H→H2D round trip through host). Host-side ``tokenize_preprocess``
and ``assemble_context_latents`` remain outside the trace.

Per ``generate()`` call: ``x_patched`` is streamed in (optional 2-CQ overlap).
"""
from __future__ import annotations

import torch

import ttnn
from models.tt_dit.utils.tracing import Tracer

from .common import from_torch, to_torch
from .subsystem_audio_tokenizer import AudioTokenizerTT
from .subsystem_detokenizer import DetokenizerTT
from .traced_decoder import _device_supports_2cq


class TracedAudioPath2CQ:
    """Trace-captured audio tokenizer + detokenizer with optional 2-CQ input overlap."""

    CQ_OPS = 0
    CQ_IO = 1

    def __init__(
        self,
        audio_tokenizer: AudioTokenizerTT,
        detokenizer: DetokenizerTT,
        *,
        use_2cq: bool | None = None,
    ):
        self.audio_tokenizer = audio_tokenizer
        self.detokenizer = detokenizer
        self.device = audio_tokenizer.device
        self.tokenizer_mod = audio_tokenizer.mod
        self.detokenizer_mod = detokenizer.mod
        self.use_2cq = use_2cq if use_2cq is not None else _device_supports_2cq(self.device)

        self._x_patched_dram: ttnn.Tensor | None = None
        self._op_event = None
        self._write_event = None
        self._captured = False
        self._tracer = Tracer(
            self._device_forward,
            device=self.device,
            prep_run=True,
            clone_prep_inputs=False,
        )

    @property
    def is_captured(self) -> bool:
        return self._captured

    def capture(self, *, x_patched: torch.Tensor) -> None:
        """Warm up, then capture a trace of tokenizer → detokenizer."""
        if self._captured:
            return

        x_tt = from_torch(x_patched, self.device)

        if self.use_2cq:
            self._op_event = ttnn.record_event(self.device, self.CQ_OPS)

        self._tracer(
            x_tt,
            traced=True,
            tracer_cq_id=self.CQ_OPS,
            tracer_blocking_execution=True,
        )

        if self.use_2cq:
            inputs = self._tracer.inputs
            self._x_patched_dram = ttnn.allocate_tensor_on_device(
                inputs[0].shape,
                inputs[0].dtype,
                inputs[0].layout,
                self.device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            self._op_event = ttnn.record_event(self.device, self.CQ_OPS)

        self._captured = True

    def prefetch_input(self, x_patched: torch.Tensor) -> None:
        """Start H2D prefetch on CQ_IO (may overlap with another stage on CQ_OPS)."""
        if not self.use_2cq:
            return
        self._prefetch_input(x_patched)

    def run_trace_and_read(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Complete audio-path trace after prefetch and read outputs to host."""
        if not self.use_2cq:
            raise RuntimeError("run_trace_and_read requires use_2cq; use __call__ for single-CQ")

        inputs = self._tracer.inputs
        ttnn.wait_for_event(self.CQ_OPS, self._write_event)
        ttnn.copy(self._x_patched_dram, inputs[0])
        self._op_event = ttnn.record_event(self.device, self.CQ_OPS)
        lm_hints_tt, quantized_tt = self._tracer(
            inputs[0],
            traced=True,
            tracer_cq_id=self.CQ_OPS,
            tracer_blocking_execution=False,
        )

        ttnn.synchronize_device(self.device)
        self._op_event = ttnn.record_event(self.device, self.CQ_OPS)
        return to_torch(lm_hints_tt, self.device), to_torch(quantized_tt, self.device)

    def __call__(self, x_patched: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._captured:
            raise RuntimeError("TracedAudioPath2CQ.capture() must run before inference")

        if self.use_2cq:
            self._prefetch_input(x_patched)
            inputs = self._tracer.inputs
            ttnn.wait_for_event(self.CQ_OPS, self._write_event)
            ttnn.copy(self._x_patched_dram, inputs[0])
            self._op_event = ttnn.record_event(self.device, self.CQ_OPS)
            lm_hints_tt, quantized_tt = self._tracer(
                inputs[0],
                traced=True,
                tracer_cq_id=self.CQ_OPS,
                tracer_blocking_execution=False,
            )
        else:
            x_tt = from_torch(x_patched.to(torch.bfloat16), self.device)
            lm_hints_tt, quantized_tt = self._tracer(
                x_tt,
                traced=True,
                tracer_cq_id=self.CQ_OPS,
                tracer_blocking_execution=True,
            )

        ttnn.synchronize_device(self.device)
        if self.use_2cq:
            self._op_event = ttnn.record_event(self.device, self.CQ_OPS)
        return to_torch(lm_hints_tt, self.device), to_torch(quantized_tt, self.device)

    def release(self) -> None:
        self._tracer.release_trace()
        self._captured = False

    def _device_forward(self, x_patched_tt: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        quantized, _indices = self.tokenizer_mod(x_patched_tt)
        lm_hints = self.detokenizer_mod(quantized)
        if isinstance(lm_hints, (tuple, list)):
            lm_hints = lm_hints[0]
        return lm_hints, quantized

    def _prefetch_input(self, x_patched: torch.Tensor) -> None:
        ttnn.wait_for_event(self.CQ_IO, self._op_event)
        ttnn.copy_host_to_device_tensor(
            self._to_host_tt(x_patched),
            self._x_patched_dram,
            cq_id=self.CQ_IO,
        )
        self._write_event = ttnn.record_event(self.device, self.CQ_IO)

    def _to_host_tt(self, tensor: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            tensor.to(torch.bfloat16),
            device=None,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

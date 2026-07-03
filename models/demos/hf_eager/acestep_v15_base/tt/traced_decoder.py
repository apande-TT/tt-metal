# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Trace + 2-CQ execution for the ACE-Step DiT decoder (ODE denoise hot path).

Follows the tt-metal generation-loop pattern (GUIDELINES §11 trace, §12 2-CQ):
  - CQ 0: capture + replay decoder compute via ``execute_trace``
  - CQ 1: host→device input streaming (when 2 command queues are available)

Constant per ``generate()`` call: encoder_hidden_states, context_latents.
Per ODE step: hidden_states (xt) and timestep tensors.
"""
from __future__ import annotations

import torch

import ttnn
from models.tt_dit.utils.tracing import Tracer

from .common import from_torch, is_mesh_device, to_torch
from .subsystem_decoder import DecoderTT


class TracedDecoder2CQ:
    """Trace-captured ACE-Step DiT decoder with optional 2-CQ input overlap."""

    CQ_OPS = 0
    CQ_IO = 1

    def __init__(self, decoder: DecoderTT, *, use_2cq: bool | None = None):
        self.decoder = decoder
        self.device = decoder.device
        self.mod = decoder.mod
        self.use_2cq = use_2cq if use_2cq is not None else _device_supports_2cq(self.device)

        self._enc_tt: ttnn.Tensor | None = None
        self._ctx_tt: ttnn.Tensor | None = None
        self._hidden_dram: ttnn.Tensor | None = None
        self._timestep_dram: ttnn.Tensor | None = None
        self._timestep_r_dram: ttnn.Tensor | None = None
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

    def capture(
        self,
        *,
        encoder_hidden_states: torch.Tensor,
        context_latents: torch.Tensor,
        sample_hidden_states: torch.Tensor,
        sample_timestep: torch.Tensor,
    ) -> None:
        """Warm up, then capture a trace of one DiT decoder forward."""
        if self._captured:
            return

        self._enc_tt = from_torch(encoder_hidden_states, self.device)
        self._ctx_tt = from_torch(context_latents, self.device)

        sample_t = sample_timestep.reshape(-1, 1).to(torch.bfloat16)
        hidden_tt = from_torch(sample_hidden_states.to(torch.bfloat16), self.device)
        timestep_tt = from_torch(sample_t, self.device)
        timestep_r_tt = from_torch(sample_t, self.device)

        if self.use_2cq:
            self._op_event = ttnn.record_event(self.device, self.CQ_OPS)

        self._tracer(
            hidden_tt,
            timestep_tt,
            timestep_r_tt,
            traced=True,
            tracer_cq_id=self.CQ_OPS,
            tracer_blocking_execution=True,
        )

        if self.use_2cq:
            inputs = self._tracer.inputs
            self._hidden_dram = ttnn.allocate_tensor_on_device(
                inputs[0].shape,
                inputs[0].dtype,
                inputs[0].layout,
                self.device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            self._timestep_dram = ttnn.allocate_tensor_on_device(
                inputs[1].shape,
                inputs[1].dtype,
                inputs[1].layout,
                self.device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            self._timestep_r_dram = ttnn.allocate_tensor_on_device(
                inputs[2].shape,
                inputs[2].dtype,
                inputs[2].layout,
                self.device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            self._op_event = ttnn.record_event(self.device, self.CQ_OPS)

        self._captured = True

    def __call__(
        self,
        hidden_states: torch.Tensor,
        *,
        timestep: torch.Tensor,
        timestep_r: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self._captured:
            raise RuntimeError("TracedDecoder2CQ.capture() must run before inference")

        timestep_r = timestep if timestep_r is None else timestep_r

        if self.use_2cq:
            self._prefetch_inputs(hidden_states, timestep, timestep_r)
            inputs = self._tracer.inputs
            ttnn.wait_for_event(self.CQ_OPS, self._write_event)
            ttnn.copy(self._hidden_dram, inputs[0])
            ttnn.copy(self._timestep_dram, inputs[1])
            ttnn.copy(self._timestep_r_dram, inputs[2])
            self._op_event = ttnn.record_event(self.device, self.CQ_OPS)
            out_tt = self._tracer(
                inputs[0],
                inputs[1],
                inputs[2],
                traced=True,
                tracer_cq_id=self.CQ_OPS,
                tracer_blocking_execution=False,
            )
        else:
            host_hidden = self._to_host_tt(hidden_states)
            host_timestep = self._to_host_tt(timestep.reshape(-1, 1))
            host_timestep_r = self._to_host_tt((timestep_r or timestep).reshape(-1, 1))
            out_tt = self._tracer(
                host_hidden,
                host_timestep,
                host_timestep_r,
                traced=True,
                tracer_cq_id=self.CQ_OPS,
                tracer_blocking_execution=True,
            )

        ttnn.synchronize_device(self.device)
        return to_torch(out_tt, self.device).to(torch.float32)

    def release(self) -> None:
        self._tracer.release_trace()
        self._captured = False

    def _device_forward(
        self,
        hidden_states_tt: ttnn.Tensor,
        timestep_tt: ttnn.Tensor,
        timestep_r_tt: ttnn.Tensor,
    ) -> ttnn.Tensor:
        out, _ = self.mod(
            hidden_states=hidden_states_tt,
            timestep=timestep_tt,
            timestep_r=timestep_r_tt,
            attention_mask=None,
            encoder_hidden_states=self._enc_tt,
            encoder_attention_mask=None,
            context_latents=self._ctx_tt,
            use_cache=False,
        )
        return out

    def _prefetch_inputs(
        self,
        host_hidden: torch.Tensor,
        timestep: torch.Tensor,
        timestep_r: torch.Tensor,
    ) -> None:
        ttnn.wait_for_event(self.CQ_IO, self._op_event)
        ttnn.copy_host_to_device_tensor(self._to_host_tt(host_hidden), self._hidden_dram, cq_id=self.CQ_IO)
        ttnn.copy_host_to_device_tensor(
            self._to_host_tt(timestep.reshape(-1, 1)),
            self._timestep_dram,
            cq_id=self.CQ_IO,
        )
        ttnn.copy_host_to_device_tensor(
            self._to_host_tt((timestep_r or timestep).reshape(-1, 1)),
            self._timestep_r_dram,
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


_2CQ_SUPPORT_CACHE: dict[int, bool] = {}


def _device_supports_2cq(device) -> bool:
    try:
        device_id = device.id()
    except (AttributeError, TypeError):
        device_id = id(device)

    cached = _2CQ_SUPPORT_CACHE.get(device_id)
    if cached is not None:
        return cached

    try:
        n = device.num_command_queues()
        result = n is not None and n >= 2
        _2CQ_SUPPORT_CACHE[device_id] = result
        return result
    except (AttributeError, TypeError):
        pass

    if is_mesh_device(device):
        try:
            ttnn.record_event(device, 1)
            _2CQ_SUPPORT_CACHE[device_id] = True
            return True
        except Exception:
            _2CQ_SUPPORT_CACHE[device_id] = False
            return False

    _2CQ_SUPPORT_CACHE[device_id] = False
    return False

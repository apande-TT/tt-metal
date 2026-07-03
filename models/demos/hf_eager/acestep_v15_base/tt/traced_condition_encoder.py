# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Trace + 2-CQ execution for ACE-Step condition encoder (Call A).

Host-side sequence-packing permutations and the final attention mask are computed
outside the trace; the captured region is pure device compute (text projector,
lyric encoder, timbre encoder, two pack matmuls).

Per ``generate()`` call: text/lyric/refer hidden states are streamed in (optional
2-CQ overlap). Permutation matrices derived from attention/order masks are also
uploaded each call.
"""
from __future__ import annotations

import torch

import ttnn
from models.tt_dit.utils.tracing import Tracer

from .common import from_torch, to_torch
from .subsystem_condition_encoder import ConditionEncoderTT
from .traced_decoder import _device_supports_2cq


class TracedConditionEncoder2CQ:
    """Trace-captured condition encoder with optional 2-CQ input overlap."""

    CQ_OPS = 0
    CQ_IO = 1

    def __init__(self, condition_encoder: ConditionEncoderTT, *, use_2cq: bool | None = None):
        self.condition_encoder = condition_encoder
        self.device = condition_encoder.device
        self.mod = condition_encoder.mod
        self.use_2cq = use_2cq if use_2cq is not None else _device_supports_2cq(self.device)

        self._text_dram: ttnn.Tensor | None = None
        self._lyric_dram: ttnn.Tensor | None = None
        self._refer_dram: ttnn.Tensor | None = None
        self._perm1_dram: ttnn.Tensor | None = None
        self._perm2_dram: ttnn.Tensor | None = None
        self._enc_out_dram: ttnn.Tensor | None = None
        self._last_out_tt: ttnn.Tensor | None = None
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
        text_hidden_states: torch.Tensor,
        text_attention_mask: torch.Tensor,
        lyric_hidden_states: torch.Tensor,
        lyric_attention_mask: torch.Tensor,
        refer_audio_acoustic_hidden_states_packed: torch.Tensor,
        refer_audio_order_mask: torch.Tensor,
    ) -> None:
        """Warm up, then capture a trace of one condition-encoder forward."""
        if self._captured:
            return

        perm1, perm2, _ = self._compute_perms_and_mask(
            text_attention_mask,
            lyric_attention_mask,
            refer_audio_order_mask,
            refer_audio_acoustic_hidden_states_packed.shape[0],
        )

        text_tt = from_torch(text_hidden_states, self.device)
        lyric_tt = from_torch(lyric_hidden_states, self.device)
        refer_tt = from_torch(refer_audio_acoustic_hidden_states_packed, self.device)
        perm1_tt = from_torch(perm1, self.device)
        perm2_tt = from_torch(perm2, self.device)

        if self.use_2cq:
            self._op_event = ttnn.record_event(self.device, self.CQ_OPS)

        out_tt = self._tracer(
            text_tt,
            lyric_tt,
            refer_tt,
            perm1_tt,
            perm2_tt,
            traced=True,
            tracer_cq_id=self.CQ_OPS,
            tracer_blocking_execution=True,
        )

        if self.use_2cq:
            inputs = self._tracer.inputs
            self._text_dram = ttnn.allocate_tensor_on_device(
                inputs[0].shape,
                inputs[0].dtype,
                inputs[0].layout,
                self.device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            self._lyric_dram = ttnn.allocate_tensor_on_device(
                inputs[1].shape,
                inputs[1].dtype,
                inputs[1].layout,
                self.device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            self._refer_dram = ttnn.allocate_tensor_on_device(
                inputs[2].shape,
                inputs[2].dtype,
                inputs[2].layout,
                self.device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            self._perm1_dram = ttnn.allocate_tensor_on_device(
                inputs[3].shape,
                inputs[3].dtype,
                inputs[3].layout,
                self.device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            self._perm2_dram = ttnn.allocate_tensor_on_device(
                inputs[4].shape,
                inputs[4].dtype,
                inputs[4].layout,
                self.device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            self._op_event = ttnn.record_event(self.device, self.CQ_OPS)

        self._enc_out_dram = ttnn.allocate_tensor_on_device(
            out_tt.shape,
            out_tt.dtype,
            out_tt.layout,
            self.device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        self._captured = True

    def run_prefill_and_trace(
        self,
        *,
        text_hidden_states: torch.Tensor,
        text_attention_mask: torch.Tensor,
        lyric_hidden_states: torch.Tensor,
        lyric_attention_mask: torch.Tensor,
        refer_audio_acoustic_hidden_states_packed: torch.Tensor,
        refer_audio_order_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Launch encoder trace (non-blocking when 2-CQ enabled). Returns CPU attention mask."""
        if not self._captured:
            raise RuntimeError("TracedConditionEncoder2CQ.capture() must run before inference")

        perm1, perm2, encoder_attention_mask = self._compute_perms_and_mask(
            text_attention_mask,
            lyric_attention_mask,
            refer_audio_order_mask,
            refer_audio_acoustic_hidden_states_packed.shape[0],
        )

        if self.use_2cq:
            self._prefetch_inputs(
                text_hidden_states,
                lyric_hidden_states,
                refer_audio_acoustic_hidden_states_packed,
                perm1,
                perm2,
            )
            inputs = self._tracer.inputs
            ttnn.wait_for_event(self.CQ_OPS, self._write_event)
            ttnn.copy(self._text_dram, inputs[0])
            ttnn.copy(self._lyric_dram, inputs[1])
            ttnn.copy(self._refer_dram, inputs[2])
            ttnn.copy(self._perm1_dram, inputs[3])
            ttnn.copy(self._perm2_dram, inputs[4])
            self._op_event = ttnn.record_event(self.device, self.CQ_OPS)
            self._last_out_tt = self._tracer(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                traced=True,
                tracer_cq_id=self.CQ_OPS,
                tracer_blocking_execution=False,
            )
        else:
            self._last_out_tt = self._tracer(
                from_torch(text_hidden_states, self.device),
                from_torch(lyric_hidden_states, self.device),
                from_torch(refer_audio_acoustic_hidden_states_packed, self.device),
                from_torch(perm1, self.device),
                from_torch(perm2, self.device),
                traced=True,
                tracer_cq_id=self.CQ_OPS,
                tracer_blocking_execution=True,
            )

        return encoder_attention_mask.bool()

    def finish_trace_to_device(self) -> ttnn.Tensor:
        """Wait for encoder trace and copy output into the device-resident staging buffer."""
        ttnn.synchronize_device(self.device)
        ttnn.copy(self._last_out_tt, self._enc_out_dram)
        return self._enc_out_dram

    def __call__(
        self,
        text_hidden_states: torch.Tensor,
        text_attention_mask: torch.Tensor,
        lyric_hidden_states: torch.Tensor,
        lyric_attention_mask: torch.Tensor,
        refer_audio_acoustic_hidden_states_packed: torch.Tensor,
        refer_audio_order_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._captured:
            raise RuntimeError("TracedConditionEncoder2CQ.capture() must run before inference")

        perm1, perm2, encoder_attention_mask = self._compute_perms_and_mask(
            text_attention_mask,
            lyric_attention_mask,
            refer_audio_order_mask,
            refer_audio_acoustic_hidden_states_packed.shape[0],
        )

        if self.use_2cq:
            self._prefetch_inputs(
                text_hidden_states,
                lyric_hidden_states,
                refer_audio_acoustic_hidden_states_packed,
                perm1,
                perm2,
            )
            inputs = self._tracer.inputs
            ttnn.wait_for_event(self.CQ_OPS, self._write_event)
            ttnn.copy(self._text_dram, inputs[0])
            ttnn.copy(self._lyric_dram, inputs[1])
            ttnn.copy(self._refer_dram, inputs[2])
            ttnn.copy(self._perm1_dram, inputs[3])
            ttnn.copy(self._perm2_dram, inputs[4])
            self._op_event = ttnn.record_event(self.device, self.CQ_OPS)
            out_tt = self._tracer(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                traced=True,
                tracer_cq_id=self.CQ_OPS,
                tracer_blocking_execution=False,
            )
        else:
            out_tt = self._tracer(
                from_torch(text_hidden_states, self.device),
                from_torch(lyric_hidden_states, self.device),
                from_torch(refer_audio_acoustic_hidden_states_packed, self.device),
                from_torch(perm1, self.device),
                from_torch(perm2, self.device),
                traced=True,
                tracer_cq_id=self.CQ_OPS,
                tracer_blocking_execution=True,
            )

        ttnn.synchronize_device(self.device)
        if self.use_2cq:
            self._op_event = ttnn.record_event(self.device, self.CQ_OPS)
        encoder_hidden_states = to_torch(out_tt, self.device).to(torch.float32)
        return encoder_hidden_states, encoder_attention_mask.bool()

    def release(self) -> None:
        self._tracer.release_trace()
        self._captured = False

    def _device_forward(
        self,
        text_tt: ttnn.Tensor,
        lyric_tt: ttnn.Tensor,
        refer_tt: ttnn.Tensor,
        perm1_tt: ttnn.Tensor,
        perm2_tt: ttnn.Tensor,
    ) -> ttnn.Tensor:
        return self.mod.forward_traced(text_tt, lyric_tt, refer_tt, perm1_tt, perm2_tt)

    def _compute_perms_and_mask(
        self,
        text_attention_mask: torch.Tensor,
        lyric_attention_mask: torch.Tensor,
        refer_audio_order_mask: torch.Tensor,
        refer_batch_rows: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lyric_mask = lyric_attention_mask.to(torch.float32)
        text_mask = text_attention_mask.to(torch.float32)
        timbre_mask = self.mod._timbre_mask_from_order(refer_audio_order_mask, refer_batch_rows).to(torch.float32)
        perm1, mask1 = self.mod._compute_pack_perm(lyric_mask, timbre_mask)
        perm2, mask2 = self.mod._compute_pack_perm(mask1.to(torch.float32), text_mask)
        return perm1, perm2, mask2

    def _prefetch_inputs(
        self,
        text_hidden_states: torch.Tensor,
        lyric_hidden_states: torch.Tensor,
        refer_audio_acoustic_hidden_states_packed: torch.Tensor,
        perm1: torch.Tensor,
        perm2: torch.Tensor,
    ) -> None:
        ttnn.wait_for_event(self.CQ_IO, self._op_event)
        ttnn.copy_host_to_device_tensor(
            self._to_host_tt(text_hidden_states),
            self._text_dram,
            cq_id=self.CQ_IO,
        )
        ttnn.copy_host_to_device_tensor(
            self._to_host_tt(lyric_hidden_states),
            self._lyric_dram,
            cq_id=self.CQ_IO,
        )
        ttnn.copy_host_to_device_tensor(
            self._to_host_tt(refer_audio_acoustic_hidden_states_packed),
            self._refer_dram,
            cq_id=self.CQ_IO,
        )
        ttnn.copy_host_to_device_tensor(
            self._to_host_tt(perm1),
            self._perm1_dram,
            cq_id=self.CQ_IO,
        )
        ttnn.copy_host_to_device_tensor(
            self._to_host_tt(perm2),
            self._perm2_dram,
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

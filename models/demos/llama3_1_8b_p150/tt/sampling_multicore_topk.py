# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Power-of-two chunked multi-step TopK so ``ttnn.topk`` takes the MULTI-CORE path.

``TopKDeviceOperation::select_program_factory`` only picks ``TopKMultiCoreProgramFactory``
when *all* of these hold for the reduced width W:

    W >= 8192 (multi_core_min_width)   W < 65535 (multi-core indices are UInt16)
    is_power_of_two(W)                 k <= 64
    verify_multi_core_cost(...) -> width % split_size == 0 for some power-of-2 split

The stock single-device path (``multi_step_reduction``, mesh == [1, 1]) splits the
128256-wide logits into two 64128-wide halves. 64128 is not a power of two, so *both*
TopK calls fall back to ``TopKSingleCoreProgramFactory`` -- the whole vocabulary is
reduced on ONE Tensix core, which dominated device time on this pipeline.

Fix: reduce in ``n`` chunks whose width is a power of two <= 32768 (the largest
power of two still under the UInt16 index limit). For a 128256 vocab that is
n=4 chunks of 32768, so the logits are padded 128256 -> 131072 with -inf first.
``n`` itself must be a power of two too, because the downstream ``ttnn.sampling``
op requires its candidate width (n * max_top_k) to yield a power-of-2 tile count.

Indices are left to the kernel (``GENERATE_INDICES``): with no ``indices_tensor``
the reader synthesises the index tile from the width-tile position, i.e. the index
is already chunk-local, so the per-chunk global offset is simply ``chunk * width``.
That also drops the DRAM read of a full-width index tensor per TopK call.
"""
from __future__ import annotations

import sys

import torch
from loguru import logger

import ttnn
from models.common.sampling import generator as _common_sampling_generator
from models.common.sampling.generator import SamplingGenerator as _BaseSamplingGenerator
from models.common.sampling.tt_sampling import TTSampling

# Largest chunk width the multi-core TopK factory accepts: a power of two that is
# still strictly below the UInt16 index limit (65535), and >= multi_core_min_width.
_MAX_MULTICORE_TOPK_WIDTH = 32768
_MIN_MULTICORE_TOPK_WIDTH = 8192


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def plan_topk_chunking(vocab_width: int, max_chunks: int = 64):
    """Pick (num_chunks, chunk_width) so every TopK call hits the multi-core factory.

    Returns ``None`` when no legal chunking exists (caller keeps the stock path).
    Prefers the WIDEST legal chunk, i.e. the fewest TopK invocations.
    """
    chunk_width = _MAX_MULTICORE_TOPK_WIDTH
    while chunk_width >= _MIN_MULTICORE_TOPK_WIDTH:
        num_chunks = -(-vocab_width // chunk_width)  # ceil
        # `ttnn.sampling` needs (num_chunks * max_top_k) / 32 to be a power of two;
        # with max_top_k == 32 that reduces to num_chunks being a power of two.
        if num_chunks >= 2 and _is_power_of_two(num_chunks) and num_chunks <= max_chunks:
            return num_chunks, chunk_width
        chunk_width //= 2
    return None


class MultiCoreTopKSampling(TTSampling):
    """``TTSampling`` whose single-device multi-step TopK runs multi-core."""

    def __init__(self, mesh_device, tt_ccl, args, k=None, p=None, temp=None):
        # Resolved before super().__init__ because _create_indices_tensors() (called
        # from there) needs the chunking to build the matching offsets tensor.
        self._topk_plan = None
        if list(mesh_device.shape) == [1, 1]:
            padded_vocab_size = getattr(args, "padded_vocab_size", None) or args.vocab_size
            self._topk_plan = plan_topk_chunking(padded_vocab_size)
        super().__init__(mesh_device=mesh_device, tt_ccl=tt_ccl, args=args, k=k, p=p, temp=temp)

    @property
    def _multicore_topk_active(self) -> bool:
        return self.multi_step_reduction and self._topk_plan is not None

    def _create_indices_tensors(self):
        if not self._multicore_topk_active:
            return super()._create_indices_tensors()

        num_chunks, chunk_width = self._topk_plan
        logger.info(
            f"MultiCoreTopKSampling: reducing {self.padded_vocab_size} logits as "
            f"{num_chunks} x {chunk_width} (power-of-2 -> multi-core TopK)"
        )

        # Chunk c covers padded-vocab columns [c * chunk_width, (c + 1) * chunk_width),
        # and the kernel-generated indices are local to that chunk.
        offsets = torch.zeros(1, 1, self.max_batch_size, self.max_top_k * num_chunks, dtype=torch.int64)
        for chunk in range(num_chunks):
            offsets[:, :, :, chunk * self.max_top_k : (chunk + 1) * self.max_top_k] = chunk * chunk_width
        self.tt_indices_device_offsets = ttnn.from_torch(
            offsets,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, None), mesh_shape=self.cluster_shape),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # GENERATE_INDICES path: no precomputed index tensor is read from DRAM.
        self.tt_indices_tensor = None

    def _multi_step_topk(self, x_bf16):
        """Chunked, power-of-two-wide TopK. Returns (values, indices) concatenated."""
        num_chunks, chunk_width = self._topk_plan
        padded_width = num_chunks * chunk_width

        if x_bf16.shape[-1] < padded_width:
            x_bf16 = ttnn.pad(
                x_bf16,
                [(0, 0), (0, 0), (0, 0), (0, padded_width - x_bf16.shape[-1])],
                value=-sys.float_info.max,
                sub_core_grids=self.sub_core_grids,
            )

        chunks = ttnn.split(x_bf16, chunk_width, dim=3)
        topk_values_list = []
        topk_indices_list = []
        for chunk in chunks:
            values, indices = ttnn.topk(
                chunk,
                k=self.max_top_k,
                dim=-1,
                sub_core_grids=self.sub_core_grid_topk,
            )
            topk_values_list.append(values)
            topk_indices_list.append(indices)
            chunk.deallocate()

        topk_values = ttnn.concat(topk_values_list, dim=3)
        topk_indices = ttnn.concat(topk_indices_list, dim=3)
        for values, indices in zip(topk_values_list, topk_indices_list):
            ttnn.deallocate(values)
            ttnn.deallocate(indices)
        return topk_values, topk_indices

    def forward(self, x: ttnn.Tensor, tt_out_tok: ttnn.Tensor = None):
        # Force-argmax and the multi-device path are untouched by this lever.
        if not self._multicore_topk_active or self._force_argmax_sampling:
            return super().forward(x, tt_out_tok=tt_out_tok)

        # Convert to bfloat16 for top-k operations (typecast is no-op if already bfloat16)
        x_bf16 = ttnn.typecast(x, dtype=ttnn.bfloat16, sub_core_grids=self.sub_core_grids)
        x_bf16 = self._mask_invalid_vocab_logits(x_bf16)

        topk_values_gathered_bf16_interleaved, topk_indices_gathered = self._multi_step_topk(x_bf16)

        # --- shared tail: global indices -> ttnn.sampling (mirrors TTSampling.forward) ---
        topk_indices_gathered_int32 = ttnn.typecast(
            topk_indices_gathered, dtype=ttnn.int32, sub_core_grids=self.sub_core_grids
        )
        ttnn.deallocate(topk_indices_gathered)

        if self.sampling_memory_config != ttnn.DRAM_MEMORY_CONFIG:
            topk_indices_gathered_int32_sharded = ttnn.to_memory_config(
                topk_indices_gathered_int32, self.sampling_memory_config
            )
            ttnn.deallocate(topk_indices_gathered_int32)
        else:
            topk_indices_gathered_int32_sharded = topk_indices_gathered_int32

        topk_global_indices = ttnn.add(
            self.tt_indices_device_offsets,
            topk_indices_gathered_int32_sharded,
            dtype=ttnn.uint32,
            memory_config=self.sampling_memory_config,
        )
        ttnn.deallocate(topk_indices_gathered_int32_sharded)

        topk_global_indices_interleaved = ttnn.to_memory_config(topk_global_indices, ttnn.DRAM_MEMORY_CONFIG)
        topk_global_indices_interleaved_untilised = ttnn.untilize(
            topk_global_indices_interleaved, use_multicore=True, sub_core_grids=self.sub_core_grids
        )
        ttnn.manual_seed(
            seeds=self.seeds_tt_tensor,
            user_ids=self.user_ids_tt_tensor,
            sub_core_grids=self._sampling_sub_core_grids,
        )
        tt_out_tok = ttnn.sampling(
            topk_values_gathered_bf16_interleaved,
            topk_global_indices_interleaved_untilised,
            k=self.k_tensor,
            p=self.p_tensor,
            temp=self.temp_tensor,
            sub_core_grids=self._sampling_sub_core_grids,
            output_tensor=tt_out_tok,
        )

        if self.log_probs_calculator.enable_log_probs and self.log_probs_calculator._use_topk_logprobs:
            self.tt_log_probs = self.log_probs_calculator.calculate_topk_log_probs(
                logits_tensor=x,
                topk_values=topk_values_gathered_bf16_interleaved,
                topk_global_indices=topk_global_indices_interleaved,
                sub_core_grid_topk=self.sub_core_grid_topk,
            )
        elif self.log_probs_calculator.enable_log_probs:
            self.tt_log_probs = self.log_probs_calculator.calculate_log_probs(x, tt_out_tok)
        else:
            self.tt_log_probs = None

        ttnn.deallocate(topk_values_gathered_bf16_interleaved)
        ttnn.deallocate(topk_global_indices_interleaved)
        ttnn.deallocate(topk_global_indices_interleaved_untilised)

        return tt_out_tok, self.tt_log_probs


class SamplingGenerator(_BaseSamplingGenerator):
    """``SamplingGenerator`` bound to :class:`MultiCoreTopKSampling`.

    The base class instantiates ``TTSampling`` by module-global name, so the swap is
    scoped to this constructor call rather than patched globally.
    """

    def __init__(self, *args, **kwargs):
        previous = _common_sampling_generator.TTSampling
        _common_sampling_generator.TTSampling = MultiCoreTopKSampling
        try:
            super().__init__(*args, **kwargs)
        finally:
            _common_sampling_generator.TTSampling = previous

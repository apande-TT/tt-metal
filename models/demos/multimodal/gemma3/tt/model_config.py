# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import gc
import inspect
import math
import os
from functools import lru_cache

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.multimodal.gemma3.tt.load_checkpoints import convert_vision_hf_to_meta, convert_vision_meta_to_hf
from models.tt_transformers.tt.common import (
    Mode,
    calculate_prefill_warmup_seq_lens,
    cap_seq_lens_to_max_prefill_chunk_size,
)
from models.tt_transformers.tt.load_checkpoints import convert_hf_to_meta, convert_meta_to_hf, standardize_hf_keys
from models.tt_transformers.tt.model_config import (
    HfAttentionWrapper,
    HfDecoderWrapper,
    HfModelWrapper,
    MathFidelitySetting,
)
from models.tt_transformers.tt.model_config import ModelArgs as TTModelArgs
from models.tt_transformers.tt.model_config import ModelOptimizations, OpGroup, PrecisionSetting, TensorGroup
from models.tt_transformers.tt.prefetcher import Prefetcher

# file names for performance and accuracy mode override files
PERFORMANCE_DECODER_CONFIG_FILENAME = "performance_decoder_config.json"
ACCURACY_DECODER_CONFIG_FILENAME = "accuracy_decoder_config.json"

# A/B: trade decode matmul cores for a bigger per-core K block (see find_grid_k_n).
_FEWER_CORES_FOR_BIGGER_K_BLOCK = True

# knob:grid on the DECODE FF2 (32 x 15360 x 3840) is DONE -- 60 cores is the end of that road.
# find_grid_k_n's K-block rule already settles this shape on 60. The only wider core count that
# divides both K and N *and* factors onto an 11x10 grid is 80 (8x10); 96 divides both but every
# factor pair needs a 12-wide or 12-tall grid. Measured 80 on device: 381.30 vs 381.26 ms, i.e. dead
# flat, at unchanged PCC. That matches what the short-prefill FF2 and the decode FF1/FF3 already
# showed -- this op is on a DRAM-bandwidth floor reading its 33MB weight, not short of cores, so the
# grid=partial tag on it is not an occupancy bug. 0 disables the override entirely.
_GEMMA3_FF2_DECODE_CORES = int(os.environ.get("GEMMA3_FF2_DECODE_CORES", "0"))

# knob:shard on the DECODE FF2 OUTPUT: MEASURED AND BLOCKED. Stock keeps it L1 width-sharded, and
# that was the one remaining shard axis on this op (weight already DRAM-bank-sharded, input already
# L1 width-sharded). Moving it to DRAM would also have freed L1 that the decode trace region
# competes for -- but it CRASHES the run: the consumers downstream of FF2 are built around a
# width-sharded input and do not accept an interleaved one. Left off.
_GEMMA3_FF2_DECODE_OUT_DRAM = os.environ.get("GEMMA3_FF2_DECODE_OUT_DRAM", "0") == "1"

# --- FF1/FF3 OUTPUT dtype (knob:dtype for the FF1/FF3 up-projections) ---------------------------
# These matmuls are memory-bound with their WEIGHTS already at the bfloat4_b floor, so the only bytes
# left to cut are on the OUTPUT side: every one of them writes a [M, 15360] bf16 result that the very
# next op (the SiLU-fused ttnn.mul that gates FF1 with FF3) consumes and re-emits as bfloat8_b
# ANYWAY. Carrying that intermediate at bf16 buys nothing downstream while doubling both the write
# and the mul's read.
#
# This is deliberately NARROWER than the MLP-wide activation walk already measured on this model,
# which regressed +60% (it moved the FF1/FF3 INPUT and FF2's input too, forcing typecasts through the
# whole block). Here only the two up-projections' OUTPUT dtype changes; inputs stay bf16.
#
# The text-path MLP lives in models/tt_transformers, outside this model dir, so its ttnn.linear
# `dtype=` argument is not editable here -- hence the wrapper. Keying on the FF1/FF3 (K, N) role
# rather than a layer index is what makes this reach all 48 layers x 2 projections.
_GEMMA3_FF13_BFP8_OUT = os.environ.get("GEMMA3_FF13_BFP8_OUT", "1") == "1"
_FF13_K_N = (3840, 15360)

# knob:dtype for the FF1/FF3 up-projections. Their WEIGHT is already at bfloat4_b, the lowest format
# TTNN has, so the only bytes left on this memory-bound op are on the OUTPUT side -- and there the
# step from bfloat8_b to bfloat4_b is worth ~4MB of the M=512 call's ~45MB, counted twice because
# the SiLU-gated mul that consumes it reads it straight back.
_FF13_OUT_DTYPE = getattr(ttnn, os.environ.get("GEMMA3_FF13_OUT_DTYPE", "bfloat4_b"))

# knob:dtype for the DECODE FF2 (32 x 15360 x 3840) is MEASURED OUT. Its WEIGHT is already
# PrecisionSetting.BFP4, the lowest format TTNN has, and at M=32 that weight is ~33MB of the call's
# ~34MB -- so the output was the only dtype axis left, and it is worth only ~0.25MB. Both steps were
# run: bfloat8_b measured 381.29 vs 381.25 (flat, i.e. below noise, and PCC 0.9648 -> 0.9635 for
# nothing), and bfloat4_b CRASHED the run. Disabled -- set the env var to re-enable.
_FF2_K_N = (15360, 3840)
_FF2_DECODE_OUT_DTYPE = getattr(ttnn, os.environ.get("GEMMA3_FF2_DECODE_OUT_DTYPE", ""), None)


def _install_ff13_out_dtype_seam():
    if not _GEMMA3_FF13_BFP8_OUT or getattr(ttnn.linear, "_gemma3_ff13_dtype_seam", False):
        return

    stock_linear = ttnn.linear

    def _linear(input_tensor_a, input_tensor_b, *args, **kwargs):
        if kwargs.get("dtype") == ttnn.bfloat16:
            try:
                k = int(input_tensor_a.shape[-1])
                n = int(input_tensor_b.shape[-1])
            except Exception:
                k = n = -1
            if (k, n) == _FF13_K_N:
                # PREFILL ONLY. bfloat4_b on every FF1/FF3 output measures -1.3% device_ms but takes
                # PCC to 0.9434, under the 0.95 floor. The bytes that lever is worth all live in the
                # prefill calls (the M=512 output is ~8MB of a ~45MB call; the M=32 decode output is
                # ~0.5MB of a ~33MB call), so spend the format where it pays and leave the decode
                # token -- the tensor the accuracy gate actually reads -- at bfloat8_b.
                try:
                    m = int(input_tensor_a.shape[-2])
                except Exception:
                    m = 0
                kwargs["dtype"] = _FF13_OUT_DTYPE if m >= 128 else ttnn.bfloat8_b
            elif _FF2_DECODE_OUT_DTYPE is not None and (k, n) == _FF2_K_N:
                try:
                    m = int(input_tensor_a.shape[-2])
                except Exception:
                    m = 0
                if m < 128:
                    kwargs["dtype"] = _FF2_DECODE_OUT_DTYPE
        return stock_linear(input_tensor_a, input_tensor_b, *args, **kwargs)

    _linear._gemma3_ff13_dtype_seam = True
    _linear._stock_linear = stock_linear
    ttnn.linear = _linear
    logger.info("gemma3: FF1/FF3 output dtype seam installed (bf16 -> bfloat8_b)")


_install_ff13_out_dtype_seam()

# --- PREFILL RMSNorm: shard the WIDTH so the norm stops running on 4 cores (knob:grid) -----------
# ttnn.rms_norm with no program_config parallelises over TILE ROWS only. A prefill norm over
# [1, 1, 128, 3840] therefore has just 4 rows to hand out and lands on FOUR cores of ~110, where it
# measures 61us for ~2MB of traffic -- about 32 GB/s. There are 192 such norms per prefill pass
# (4 per layer x 48 layers), so at ISL 128 this one shape is ~23.5ms of device time and the largest
# LayerNorm bucket in the profile.
#
# The DECODE path already solves this: it hands rms_norm a LayerNormShardedMultiCoreProgramConfig
# with the activation L1 WIDTH-sharded, which parallelises over the 3840-wide dimension instead of
# the 32-row one, and measures 6.1us on 30 cores. Prefill never takes that path because
# models/common/rmsnorm.py gates the sharded config on ``in_sharded=(mode == Mode.DECODE)``.
#
# That gate is in models/common, outside this model dir, so apply the same treatment at the ttnn
# boundary: width-shard the activation, run the sharded norm, hand the result back interleaved.
# Keying on the hidden WIDTH (not a layer index) is what makes this reach all 48 layers and all four
# norms in each -- and it deliberately does NOT touch gemma3's q_norm/k_norm, whose width is
# head_dim, nor the decode norms, which are sharded already.
_GEMMA3_SHARDED_PREFILL_NORM = os.environ.get("GEMMA3_SHARDED_PREFILL_NORM", "1") == "1"
_NORM_WIDTH = 3840
# Above this M the default row-parallel norm already has enough rows to fill the grid.
_NORM_MAX_M = 512
_NORM_CORES = 40
_NORM_PLANS = {}


def _prefill_norm_plan(m, width, grid):
    """(sharded_memory_config, program_config) for an [M, width] prefill norm, or None."""
    key = (m, width, grid.x, grid.y)
    if key in _NORM_PLANS:
        return _NORM_PLANS[key]
    plan = None
    w_tiles = width // ttnn.TILE_SIZE
    cores = _NORM_CORES
    core_grid = None
    for y in range(1, int(grid.y) + 1):
        if cores % y == 0 and cores // y <= int(grid.x):
            core_grid = ttnn.CoreGrid(y=y, x=cores // y)
            break
    # Only worth it when the width shard beats the rows the default would have parallelised over.
    if core_grid is not None and w_tiles % cores == 0 and cores > m // ttnn.TILE_SIZE:
        block_w = w_tiles // cores
        subblock_w = max(s for s in range(1, 5) if block_w % s == 0)
        plan = (
            ttnn.create_sharded_memory_config(
                (m, width // cores),
                core_grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            ),
            ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[core_grid.x, core_grid.y],
                subblock_w=subblock_w,
                block_h=m // ttnn.TILE_SIZE,
                block_w=block_w,
                inplace=False,
            ),
        )
    _NORM_PLANS[key] = plan
    return plan


def _install_prefill_norm_shard_seam():
    if not _GEMMA3_SHARDED_PREFILL_NORM or getattr(ttnn.rms_norm, "_gemma3_norm_shard_seam", False):
        return

    stock_rms_norm = ttnn.rms_norm

    def _rms_norm(input_tensor, **kwargs):
        plan = None
        if kwargs.get("program_config") is None and kwargs.get("residual_input_tensor") is None:
            try:
                shape = list(input_tensor.shape)
                m, width = int(shape[-2]), int(shape[-1])
                ok = (
                    width == _NORM_WIDTH
                    and m % ttnn.TILE_SIZE == 0
                    and m <= _NORM_MAX_M
                    and not input_tensor.memory_config().is_sharded()
                    and all(int(d) == 1 for d in shape[:-2])
                )
                if ok:
                    plan = _prefill_norm_plan(m, width, input_tensor.device().compute_with_storage_grid_size())
            except Exception:
                plan = None
        if plan is None:
            return stock_rms_norm(input_tensor, **kwargs)
        shard_mc, prg_cfg = plan
        kwargs = dict(kwargs)
        kwargs["program_config"] = prg_cfg
        kwargs["memory_config"] = shard_mc
        x = ttnn.to_memory_config(input_tensor, shard_mc)
        out = stock_rms_norm(x, **kwargs)
        ttnn.deallocate(x)
        result = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out)
        return result

    _rms_norm._gemma3_norm_shard_seam = True
    _rms_norm._stock_rms_norm = stock_rms_norm
    ttnn.rms_norm = _rms_norm
    logger.info("gemma3: prefill RMSNorm width-shard seam installed")


_install_prefill_norm_shard_seam()

# --- LM HEAD precision: bfp4 weights + the matching LoFi fidelity (knob:dtype) -------------------
# The LM head is the biggest DECODE op left. It runs 16 splits of [32, 3840] @ [3840, 16032] plus a
# 5888-wide remainder (16*16032 + 5888 = 262400 = padded_vocab_size) on EVERY token, and measures
# 191us per split -- ~3.1ms per token out of a ~37ms decode step. Its weight is 3840x16032 bfloat8_b
# = 65.4MB per split, so at 342 GB/s the op is essentially a weight read; halving the weight is the
# whole lever, and GUIDELINES 01 sec.12 pairs a bfp4 weight with LoFi rather than HiFi2.
#
# Neither is reachable through ModelArgs: lm_head.py takes the model-wide ``dtype`` ctor argument
# (text_demo passes bfloat8_b for every weight, and lowering THAT would hit attention too) and
# hard-codes its own HiFi2 compute_kernel_config. So pin both on the LMHead class itself, which
# keeps the change to this one module instead of the whole model.
_GEMMA3_LM_HEAD_BFP4 = os.environ.get("GEMMA3_LM_HEAD_BFP4", "1") == "1"


def _install_lm_head_precision_seam():
    from models.tt_transformers.tt import lm_head as lm_head_mod

    cls = lm_head_mod.LMHead
    if not _GEMMA3_LM_HEAD_BFP4 or getattr(cls, "_gemma3_lm_head_precision_seam", False):
        return

    stock_init = cls.__init__

    def __init__(self, *args, **kwargs):
        kwargs["dtype"] = ttnn.bfloat4_b
        stock_init(self, *args, **kwargs)
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    cls.__init__ = __init__
    cls._gemma3_lm_head_precision_seam = True
    logger.info("gemma3: LM head precision seam installed (bfp4 weights + LoFi)")


_install_lm_head_precision_seam()

# --- PREFILL create-heads: land the Q/K/V slices in L1, not DRAM (knob:grid) ---------------------
# ``nlp_create_qkv_heads`` is pure data movement -- it slices a [1, 1, S, 4096] fused QKV into
# q[1, 16, S, 256] and k/v[1, 4, S, 256]. Both the call sites this model reaches (the text
# attention in models/tt_transformers and the vision attention next door) hard-code
# ``memory_config=ttnn.DRAM_MEMORY_CONFIG``, so the op reads DRAM and writes DRAM for a slice that
# does no math at all. That round trip is what the profiler's ``grid=tiny`` tag on
# NlpCreateHeadsDeviceOperation is measuring: with an interleaved DRAM destination the op hands out
# one work unit per (batch, seq_tile) and the write, not the core count, sets the time.
#
# GUIDELINES 04 sec.4 pairs a DRAM create-heads with DRAM-staged SDPA precisely because at ViT
# high-res Q+K+V is ~14MB and cannot share L1 with SDPA's flash buffers. That reasoning does not
# transfer here: this prefill's Q+K+V is ~1MB, so the slices fit L1 alongside the flash chunks and
# SDPA reads them at L1 bandwidth instead of pulling them back over the NoC.
#
# Nothing about the math changes -- only where the slices land -- so PCC is bit-identical. The
# decode path is untouched: it goes through the separate ``nlp_create_qkv_heads_decode`` op, which
# this wrapper never sees. Keying on the DRAM memory_config rather than a call site is what makes
# it reach every layer of both towers.
_GEMMA3_CREATE_HEADS_L1 = os.environ.get("GEMMA3_CREATE_HEADS_L1", "1") == "1"

# tt-lang rung: replace the head split with hand-authored ttl tile-gather kernels
# (models/demos/multimodal/gemma3/tt/ttl_create_heads.py). Off by default -- see that module and
# the recorded attempt for why.
_GEMMA3_CREATE_HEADS_TTL = os.environ.get("GEMMA3_CREATE_HEADS_TTL", "0") == "1"

# cpp rung: replace the head split with a ttnn.generic_op tile gather that parallelises over the
# OUTPUT tile space instead of over seq tiles (models/demos/multimodal/gemma3/tt/cpp_create_heads.py).
_GEMMA3_CREATE_HEADS_CPP = os.environ.get("GEMMA3_CREATE_HEADS_CPP", "1") == "1"


def _create_heads_cpp_probe(msg):
    """A can_run guard that never fires would fake parity with the stock op; prove it fired."""
    path = os.environ.get("GEMMA3_CREATE_HEADS_CPP_PROBE", "/tmp/gemma3_create_heads_cpp_probe.log")
    try:
        with open(path, "a") as fh:
            fh.write("{}\n".format(msg))
    except Exception:
        pass


# knob:shard MEASURED AND BLOCKED (2026-08-02). Going from L1 INTERLEAVED to L1 SHARDED means
# taking the op's ``Sharded{}`` program factory instead of ``Interleaved{}``, and that factory is
# decode-shaped, not prefill-shaped (nlp_create_qkv_heads_device_operation.cpp):
#   * the input shard width is hard-checked as (num_q_heads/num_kv_heads + 2)*head_dim, which for a
#     fused width of (q + 2kv)*head_dim forces the core count to be EXACTLY num_kv_heads -- 8 of
#     ~110 here. The shard rung cannot widen this op; it can only narrow it.
#   * compute_output_specs pins the OUTPUT shard to {TILE_HEIGHT, head_dim} -- ONE tile row per
#     core -- so the sharded path only holds a seq_len of 32. This prefill runs S=128 and S=512.
# Tried it anyway on the real device, twice. (1) Full-length: the input shard resolved cleanly
# (w=8192, rows=128, shard=(128,1024), 8 cores) and the first sharded call crashed the run, because
# 8 cores x 32 rows cannot hold q's 16 heads x 128 rows. (2) Chunked into 32-row pieces so the
# output spec is legal: a BLOCK_SHARDED output throws "Shard grid must be one full rectangular
# grid" (the op derives that grid from num_cores_to_corerangeset(num_q_heads=16), and 16 cores is
# not a rectangle on this 11-wide grid), but a HEIGHT_SHARDED output RUNS on all 48 layers and
# measures 387.81 ms -- 6.2 ms FASTER than the interleaved path below, even paying 4x the ops.
#
# It is still wrong (PCC 0.189), and the reason is a LAYOUT contract, not the kernel: at
# nlp_create_qkv_heads_program_factory.cpp:391, k_base_addr = q_base_addr +
# per_core_in_q_heads*head_size, so the sharded path reads each input core's 1024-wide slab as
# GROUP-INTERLEAVED [2 q heads | 1 k head | 1 v head], whereas the fused QKV this model produces is
# [all 16 q | all 8 k | all 8 v]. Claiming that 6.2 ms means permuting the wqkv OUTPUT COLUMNS into
# group-interleaved order at load time -- a structural change, not a knob, and an all-or-nothing
# one: once the weight is permuted the interleaved fallback below becomes incorrect, so it would
# have to cover the vision tower and the S=512 prefill too. Left on the interleaved-L1 path above.


def _install_create_heads_l1_seam():
    stock_create_heads = ttnn.experimental.nlp_create_qkv_heads
    if not _GEMMA3_CREATE_HEADS_L1 or getattr(stock_create_heads, "_gemma3_create_heads_l1_seam", False):
        return

    def _nlp_create_qkv_heads(*args, **kwargs):
        if kwargs.get("memory_config") == ttnn.DRAM_MEMORY_CONFIG:
            kwargs = {**kwargs, "memory_config": ttnn.L1_MEMORY_CONFIG}
        if _GEMMA3_CREATE_HEADS_TTL and args and not kwargs.get("transpose_k_heads", True):
            from models.demos.multimodal.gemma3.tt import ttl_create_heads

            return ttl_create_heads.create_qkv_heads(
                args[0],
                int(kwargs["num_heads"]),
                int(kwargs["num_kv_heads"]),
                kwargs.get("memory_config", ttnn.L1_MEMORY_CONFIG),
            )
        if _GEMMA3_CREATE_HEADS_CPP and args and not kwargs.get("transpose_k_heads", True):
            from models.demos.multimodal.gemma3.tt import cpp_create_heads

            nq, nkv = int(kwargs["num_heads"]), int(kwargs["num_kv_heads"])
            if cpp_create_heads.can_run(args[0], nq, nkv):
                _create_heads_cpp_probe("FIRE {}".format(tuple(args[0].shape)))
                return cpp_create_heads.create_qkv_heads(
                    args[0], nq, nkv, kwargs.get("memory_config", ttnn.L1_MEMORY_CONFIG)
                )
            _create_heads_cpp_probe("SKIP {}".format(tuple(args[0].shape)))
        return stock_create_heads(*args, **kwargs)

    _nlp_create_qkv_heads._gemma3_create_heads_l1_seam = True
    _nlp_create_qkv_heads._stock_nlp_create_qkv_heads = stock_create_heads
    ttnn.experimental.nlp_create_qkv_heads = _nlp_create_qkv_heads
    logger.info("gemma3: create-heads L1 seam installed (DRAM -> L1 Q/K/V slices)")


_install_create_heads_l1_seam()

# --- PREFILL concat-heads: parallelise over HEADS instead of seq tiles (knob:grid) ---------------
# ``nlp_concat_heads`` is the mirror image of the head split above: it folds q[1, H, S, 256] back
# into [1, 1, S, H*256] and, like the split, does no math at all. Both call sites this model reaches
# hard-code ``memory_config=ttnn.DRAM_MEMORY_CONFIG``, so it is a DRAM->DRAM gather.
#
# Why the profiler tags it ``grid=tiny``: nlp_concat_heads_program_factory.cpp:49 computes
#   num_blocks = ashape[0] * ashape[2] / TILE_HEIGHT      # batch * seq / 32
# and hands THAT to split_work_to_cores. The unit of work is a seq tile-row, and the head axis is
# walked serially inside each core. This prefill runs S=128 (4 work units -> 4 cores of 110) and
# S=1024 (32 cores). Measured on the current tree: 96 x 51.4us at 4 cores, 48 x 215.1us at 32 cores.
#
# The op has a SECOND program factory, taken when the input is sharded, and it splits on a different
# axis entirely: line 57 sets num_blocks_per_core = shard_height / seq_len, i.e. one core per HEAD.
# That is 16 cores here regardless of S, and it turns the gather into an L1->L1 shuffle instead of a
# strided DRAM read (the current reader fetches 16 scattered head slabs per seq tile, which is why
# it only achieves ~20 GB/s of a much larger DRAM budget).
#
# So the grid knob is the coordinated shard of GUIDELINES 01 sec.10/11, not a program_config kwarg
# (this op takes none): height-shard the input into L1 one head per core, let the sharded factory
# run, and hand the width-sharded result back to the caller in the memory config it asked for. The
# two extra reshards read and write the same DRAM bytes the stock op already moved, but contiguously.
#
# Contract checked against nlp_concat_heads_device_operation.cpp validate(): shard width must equal
# the padded last dim (head_dim), shard height must be a multiple of seq_len, num_heads must be
# divisible by heads-per-shard, and the OUTPUT must be sharded-but-not-height-sharded (the sharded
# kernel writes CB 16, which only exists when out_sharded) -- hence WIDTH_SHARDED L1 out.
_GEMMA3_CONCAT_HEADS_MODE = os.environ.get("GEMMA3_CONCAT_HEADS_MODE", "shard")  # off | l1 | shard

# Sharding buys cores only while the seq-tile split has fewer work units than there are heads;
# above that the interleaved factory already spreads wider. Set to 0 to shard at every length.
_GEMMA3_CONCAT_HEADS_SHARD_MAX_SEQ = int(os.environ.get("GEMMA3_CONCAT_HEADS_SHARD_MAX_SEQ", "0"))


def _concat_heads_probe(msg):
    """A guard that silently declines would fake parity with the stock op; prove which path ran."""
    path = os.environ.get("GEMMA3_CONCAT_HEADS_PROBE", "/tmp/gemma3_concat_heads_probe.log")
    try:
        with open(path, "a") as fh:
            fh.write("{}\n".format(msg))
    except Exception:
        pass


def _concat_heads_shard_input(x):
    """Height-shard [1, H, S, D] into L1 as one head per core, or None if the contract does not hold."""
    shape = tuple(x.shape)
    if len(shape) != 4 or x.is_sharded():
        return None
    batch, heads, seq, head_dim = shape
    if batch != 1 or seq % 32 or head_dim % 32:
        return None
    if _GEMMA3_CONCAT_HEADS_SHARD_MAX_SEQ and seq > _GEMMA3_CONCAT_HEADS_SHARD_MAX_SEQ:
        return None
    grid = x.device().compute_with_storage_grid_size()
    if heads > grid.x * grid.y:
        return None
    shard_spec = ttnn.ShardSpec(
        ttnn.num_cores_to_corerangeset(heads, grid, True),
        (seq, head_dim),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.to_memory_config(
        x, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    )


def _install_concat_heads_seam():
    stock_concat_heads = ttnn.experimental.nlp_concat_heads
    if _GEMMA3_CONCAT_HEADS_MODE == "off" or getattr(stock_concat_heads, "_gemma3_concat_heads_seam", False):
        return

    def _nlp_concat_heads(*args, **kwargs):
        out_mem = kwargs.get("memory_config", ttnn.DRAM_MEMORY_CONFIG)
        if _GEMMA3_CONCAT_HEADS_MODE == "shard" and args:
            sharded_in = _concat_heads_shard_input(args[0])
            if sharded_in is not None:
                _concat_heads_probe("FIRE {}".format(tuple(args[0].shape)))
                sharded_out = stock_concat_heads(
                    sharded_in,
                    memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1),
                )
                ttnn.deallocate(sharded_in)
                out = ttnn.to_memory_config(sharded_out, out_mem)
                ttnn.deallocate(sharded_out)
                return out
            _concat_heads_probe("SKIP {}".format(tuple(args[0].shape)))
        if out_mem == ttnn.DRAM_MEMORY_CONFIG:
            kwargs = {**kwargs, "memory_config": ttnn.L1_MEMORY_CONFIG}
        return stock_concat_heads(*args, **kwargs)

    _nlp_concat_heads._gemma3_concat_heads_seam = True
    _nlp_concat_heads._stock_nlp_concat_heads = stock_concat_heads
    ttnn.experimental.nlp_concat_heads = _nlp_concat_heads
    logger.info("gemma3: concat-heads seam installed (mode=%s)", _GEMMA3_CONCAT_HEADS_MODE)


_install_concat_heads_seam()

# SDPA decode k_chunk when force_fixed_decode_k_chunk is True (paged text path; see text_demo).
_GEMMA3_SDPA_DECODE_K_CHUNK_DEFAULT = 256
# Under program trace, use the smallest valid k_chunk (pow2, multiple of 32) to reduce L1 vs static CB limits.
_GEMMA3_SDPA_DECODE_K_CHUNK_PROGRAM_TRACE = 32


class ModelArgs(TTModelArgs):
    OP_KEYS = (
        # Embedding
        "EMB_WEIGHTS",
        # Feed forward
        "MLP_WEIGHTS",
        "FF1_OUTPUT",
        "FF3_OUTPUT",
        "FF2_OUTPUT",
        "MLP_W_LAYOUT",
        # Attention
        "ATTN_WEIGHTS",
        "XQKV_MM_OUTPUT",
        "QKV_HEADS_OUTPUT",
        "QV_ROT_EMB_OUTPUT",
        "KV_UNPAD_OUTPUT",
        "QK_MM_OUTPUT",
        "QKV_MM_OUTPUT",
        "CONCAT_HEADS_OUTPUT",
        "ATTN_OUTPUT",
        "ATTN_W_LAYOUT",
        # Decoder
        "DECODE_RESIDUAL",
        "OUTPUT_MM",
    )

    MAX_QKV_MM_SEQ_LEN = 2048

    def __init__(
        self,
        mesh_device,
        instruct=False,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=1024 * 128,
        optimizations=None,
        cache_hf=False,  # Set to False to reduce memory usage by not caching HF model
        enable_program_trace: bool = False,
    ):
        # Resolve HF_MODEL to a local snapshot path before super().__init__() so that
        # all HF calls (AutoConfig, tokenizer, weights) skip the refs/main lookup,
        # which is absent on some CI machines.  Left in env so sub-tests in the same
        # pytest session (e.g. siglip/test_attention.py) also get the absolute path.
        hf_model = os.environ.get("HF_MODEL", "")
        if hf_model and not os.path.isabs(hf_model):
            snapshot = ModelArgs._resolve_hf_snapshot(hf_model)
            if snapshot:
                logger.info(f"[Gemma3] Resolved HF model '{hf_model}' to snapshot: {snapshot}")
                os.environ["HF_MODEL"] = str(snapshot)
        self._enable_program_trace = enable_program_trace
        # Trace path needs fixed k_chunk and flags before super().__init__: base __init__ may consult attention config.
        if enable_program_trace:
            self.force_fixed_decode_k_chunk = True
            self._gemma3_sdpa_decode_k_chunk_override = _GEMMA3_SDPA_DECODE_K_CHUNK_PROGRAM_TRACE

        super().__init__(
            mesh_device,
            instruct=instruct,
            dummy_weights=dummy_weights,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            optimizations=optimizations,
            cache_hf=cache_hf,
        )

        # For Gemma3 we still need a real tokenizer even when using dummy_weights,
        # because prompt encoding relies on HF chat templates, not on checkpoint weights.
        if dummy_weights and self.tokenizer is None:
            self.tokenizer = self.create_tokenizer()

        # Fuse the decode Q/K rotary and the paged K/V cache update. The base class turns this off for
        # every multimodal model (`use_qk_fused = not is_multimodal and not use_hf_rope`) and gemma3
        # then re-pinned it False -- but both are blanket policy gates, not contracts. The fusion
        # touches only the TEXT decode path, and gemma3 leaves `use_hf_rope` False, so the one real
        # incompatibility (fused QK is unimplemented for HF rope) never applies here.
        #
        # It gates two fusions in models/tt_transformers/tt/attention.py: fused QK rotary (~line 158)
        # and `paged_fused_update_cache` (~line 687). The second is the one that pays -- it replaces
        # 576 single-core PagedUpdateCache launches with 288 fused ones, and collapses the three
        # RotaryEmbeddingLlama lines into one. Net launch count is roughly unchanged (it trades 288
        # 'other' launches for 288 datamove ones), which is why this is worth a couple of ms rather
        # than the ~0.5ms/token a pure launch saving would predict.
        self.use_qk_fused = True
        self.model_config["LM_HEAD_OUTPUT_MEMCFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.padded_vocab_size = 262400
        # Raise the per-device cap so on-device sampling is enabled for Gemma3's 131200-wide shard.
        self.device_sampling_max_per_device_vocab = 192 * 1024

        if enable_program_trace:
            self._relax_attention_ops_for_program_trace()

        # HiFi2 NA fixes single-device decode token drift. It increases SDPA decode L1 usage and can
        # overlap Metal's static circular-buffer region used for program tracing (or multi-device
        # layouts), causing TT_THROW in validate_circular_buffer_region. Skip in those cases.
        if not enable_program_trace:
            self._force_sdpa_decode_hifi2_na()

        if self.num_devices == 1:
            # Turn off fp32_dest_acc_en to not trigger L1 OOM
            self._force_sdpa_prefill_hifi4_fp16()

        # NOTE: raising prefill_len_cutoff to 1024 was tried here and REVERTED. It does merge the
        # batch-2 prefill into one M=1024 matmul so the weight is multicast once instead of twice,
        # but doubling per_core_M (2 -> 4) doubles the per-core output CB to ~491KB and costs more
        # than it saves: the op went 43.0 -> 50.7ms, DRAM% 16 -> 25, FLOPs% 76 -> 65.

        # GUIDELINES 01 §12: a bf8b matmul should hold at LoFi, which is ~2x the math rate. FF2
        # (bfp8 x bfp8, FLOP-bound at ~85% of peak) proved it -- and raised PCC while doing it.
        # The QKV and attention-output projections are the same case: bfp8 weights still on HiFi2.
        # These run AFTER _relax_attention_ops_for_program_trace and override its HIFI2_FP16, which
        # is safe on L1: LoFi and HIFI2_FP16 are identical apart from math_fidelity (both
        # fp32_dest_acc_en=False, packer_l1_acc=True), and fp32_dest_acc_en is what that relaxation
        # was reaching for. SDPA is deliberately left alone -- gemma3 needs HIFI2_NA there for
        # decode correctness.
        # FF2's WEIGHT is the last one in the MLP still above the floor: FF1/FF3 are already bfloat4_b
        # while w2 -- the single biggest weight in the block at 15360x3840 -- is still bfloat8_b. On a
        # memory-bound down-projection the weight read IS the cost, so halving it is the dtype rung
        # here. gemma3's own PERF.md documents "bfp4 MLP weights" as the intended performance
        # configuration, and FF2 already runs LoFi (below), which is the matching fidelity for bfp4.
        # WQKV and WO are the last weights above the floor once FF1/FF3 and FF2 are bfp4. Both are
        # pure weight reads on the decode path -- 3840x8192 and 4096x3840 read in full for every
        # token -- so halving them is the only lever those projections have left, and unlike the
        # prefill grid work it moves the per-token metric. This IS a step past the usual convention:
        # the stock accuracy preset keeps "BFP8 attention" even where it takes bfp4 MLPs, so the PCC
        # gate decided it rather than precedent (0.9793 -> 0.9744 against a 0.95 floor).
        #
        # Measured: decode QKV 60.0 -> 53.0us, decode attn-out 32.4 -> 27.7us, prefill QKV
        # 158.3 -> 116.4us, prefill attn-out 76.9 -> 60.0us; model 424.6 -> 415.2ms.
        #
        # NOTE the first measurement of this lever read 699.7ms and was NOT a lever effect: that
        # profile showed a uniform ~1.7x slowdown across ops it cannot touch (LayerNorm 9.7 ->
        # 19.3us), the degraded-device signature that also hit the trace gate. Re-measured clean.
        self._set_tensor_dtype(
            {
                TensorGroup.FF2: PrecisionSetting.BFP4,
                TensorGroup.WQKV: PrecisionSetting.BFP4,
                TensorGroup.WO: PrecisionSetting.BFP4,
            }
        )

        self._set_op_fidelity(
            {
                OpGroup.LI_FF2: MathFidelitySetting.LOFI,
                OpGroup.LI_QKV_DECODE: MathFidelitySetting.LOFI,
                OpGroup.LI_QKV_PREFILL: MathFidelitySetting.LOFI,
                OpGroup.LI_O_DECODE: MathFidelitySetting.LOFI,
                OpGroup.LI_O_PREFILL: MathFidelitySetting.LOFI,
            }
        )

    def _short_prefill_ff1_3_prg_config(self, seq_len: int):
        """Full-grid 2D-mcast config for a prefill FF1/FF3 whose M is SHORTER than the grid is tall.

        ``mlp1_3_grid`` resolves through ``find_prefill_grid(prefill_rows=8, dim_tiles)``, which is
        hard-capped at ``max_rows = max_cols = 8`` behind a "TODO Improve configuration for BH"
        comment, so on this 11x10 Blackhole it returns (8, 8) = 64 cores. Two separate things then go
        to waste at ISL 128:

        * ``per_core_M = ceil(128 / (32 * 8)) = 1``, so the 8 grid ROWS can only cover 8 M-tiles
          while the op has just 4 — HALF THE ROWS DO NO WORK, leaving 32 cores of ~110 active.
        * grid_x is capped at 8, so N is split 8 ways when the device has more columns to give.

        Both are reachable. The wasted ROWS came first: dropping the grid to the M-tiles that
        actually exist keeps every core busy AND buys a bigger per-core K block, because
        ``matmul_config`` derives ``in0_block_w = find_largest_divisor(k_tiles // grid_rows)``:
        120//8 = 15 -> block 5, versus 120//4 = 30 -> block 6.

        The COLUMNS are NOT reachable on THIS path, and the reason is narrower than this file used to
        state. It is not that unpinning ``per_core_N`` from ``dram_shard_grid_width`` is unsafe in
        general: the long-prefill sibling (``_long_prefill_ff1_3_prg_config``, per_core_M=2) runs the
        SAME weight at grid_x=10 (per_core_N 48) and grid_x=11 (per_core_N 44) at PCC 0.9742,
        unchanged. It is specific to this per_core_M=1 config, and it is not a ragged-column effect
        either — both column counts were measured here:

            grid_x=11, per_core_N=44 (ragged, 480 = 10*44 + 40) -> PCC 0.2157
            grid_x=10, per_core_N=48 (exact, 480 = 10*48)       -> PCC 0.1520

        Two different column counts, one of them dividing 480 exactly, both garbage, while the same
        widths are clean one path over. So the pin stays here and 32 cores is the floor for this
        shape until the per_core_M=1 case itself is understood; do not re-derive it from the general
        claim, which is false.

        Returns None when it cannot improve on stock, so the caller falls through to it.
        """
        m = min(seq_len, self.prefill_len_cutoff)
        k = self.dim // self.cluster_shape[0]
        n = self.hidden_dim // self.cluster_shape[1]
        m_tiles, k_tiles, n_tiles = m // ttnn.TILE_SIZE, k // ttnn.TILE_SIZE, n // ttnn.TILE_SIZE
        if m_tiles >= self.prefill_rows:
            return None  # stock already fills the rows; leave that path alone

        # COLUMNS: pinned to the weight's DRAM shard width. Both wider widths measure garbage PCC
        # on this per_core_M=1 path (see the docstring); the pin is load-bearing HERE specifically.
        cols = self.dram_shard_grid_width
        per_core_N = math.ceil(n / (ttnn.TILE_SIZE * cols))
        # ROWS: the M-tiles that actually exist, not the fixed prefill_rows. Must also divide k_tiles,
        # which keeps matmul_config's k-divisibility assert and its in0_block_w derivation valid.
        rows = max((y for y in range(1, m_tiles + 1) if m_tiles % y == 0 and k_tiles % y == 0), default=1)
        if rows >= self.prefill_rows:
            return None
        return self.matmul_config(
            m=m,
            k=k,
            n=n,
            grid_size=(cols, rows),
            per_core_M=m_tiles // rows,
            per_core_N=per_core_N,
        )

    def _long_prefill_ff2_minimal_config(self, seq_len: int):
        """Give the LONG-prefill FF2 the whole grid. Unlike every other MLP config here, this one is
        NOT pinned by the weight's DRAM shard width, because it is a ``minimal_matmul``.

        Stock hands it ``mlp2_grid`` = ``find_prefill_grid(8, hidden_dim_tiles)`` = (8, 8), the same
        "TODO Improve configuration for BH" cap, so it runs on 64 of 110 cores at FLOPs% 54.7.

        ``minimal_matmul`` does not take a ``per_core_N``: the program factory splits M over ONE grid
        axis and N over the other and every core in the grid gets a slice
        (``M_tiles_per_core = round_up(M_tiles, axis)/axis``), so there is no shard-width pin to
        violate and no PCC cliff to fall off -- widening the grid just makes each core's slice
        smaller. It transposes the axes when M > N; here M=512 < N=3840, so M rides grid.y and N
        rides grid.x.

        M=512: M_tiles=16 over y and N_tiles=120 over x. y=8 is the smallest height that reaches the
        minimum 2 M-tiles per core (9 and 10 also give 2, but only by padding M with phantom tiles),
        and x=11 -- prime, dividing nothing -- still beats x=10 because the split is a round_up:
        ceil(120/11)=11 against ceil(120/10)=12. So (8,8) -> (11,8) = 64 -> 88 cores and 30 -> 22
        output tiles per core. Block sizes stay at stock's 8/8/8 so the L1 CB budget is untouched.
        """
        m = min(seq_len, self.prefill_len_cutoff)
        n = self.dim
        if m % ttnn.TILE_SIZE:
            return None
        m_tiles, n_tiles = m // ttnn.TILE_SIZE, n // ttnn.TILE_SIZE
        if m >= n:
            return None  # the factory would transpose the axes; not the case we measured
        gx, gy = self.max_grid_size.x, self.max_grid_size.y

        def _smallest_best(tiles, cap):
            best = min(math.ceil(tiles / c) for c in range(1, cap + 1))
            return min(c for c in range(1, cap + 1) if math.ceil(tiles / c) == best)

        rows, cols = _smallest_best(m_tiles, gy), _smallest_best(n_tiles, gx)
        stock_rows, stock_cols = self.mlp2_grid(seq_len)
        if cols * rows <= stock_cols * stock_rows:
            return None
        return ttnn.MinimalMatmulConfig(
            M_block_size=8,
            K_block_size=8,
            N_block_size=8,
            compute_with_storage_grid_size=ttnn.CoreCoord(cols, rows),
        )

    @lru_cache(maxsize=None)
    def get_mlp_ff2_prg_config(self, mode: Mode, seq_len: int = 1, prefetcher: Prefetcher = None):
        """The same wasted-rows fix as ``_short_prefill_ff1_3_prg_config``, for the FF2 side.

        Stock builds the short-prefill FF2 on ``mlp2_grid`` = ``find_prefill_grid(prefill_rows=8,
        hidden_dim_tiles)`` = (8, 8), so ``per_core_M = ceil(128 / (32*8)) = 1`` and the 8 grid ROWS
        are asked to cover 8 M-tiles when the op has only 4 -- half the rows do no work. Dropping the
        grid to the M-tiles that actually exist keeps every core busy and, because ``matmul_config``
        derives ``in0_block_w = find_largest_divisor(k_tiles // rows)``, it also buys a bigger per-core
        K block on FF2's very wide K: 480//8 = 60 -> block 6, versus 480//4 = 120 -> block 8.

        COLUMNS stay at stock's ``dram_shard_grid_width``, because per_core_N is pinned to w2's
        8-bank DRAM shard width and mismatching it silently corrupts (measured PCC 0.31 elsewhere in
        this file). Returns to stock whenever it cannot improve on it.
        """
        if mode == Mode.PREFILL and seq_len > 128 and prefetcher is None and not self.is_galaxy:
            pc = self._long_prefill_ff2_minimal_config(seq_len)
            if pc is not None:
                return pc
        if mode == Mode.PREFILL and seq_len <= 128 and prefetcher is None and not self.is_galaxy:
            m = min(seq_len, self.prefill_len_cutoff)
            k = self.hidden_dim // (self.cluster_shape[1] if self.is_galaxy else 1)
            m_tiles, k_tiles = m // ttnn.TILE_SIZE, k // ttnn.TILE_SIZE
            cols = self.dram_shard_grid_width
            rows = max((y for y in range(1, m_tiles + 1) if m_tiles % y == 0 and k_tiles % y == 0), default=1)
            if m % ttnn.TILE_SIZE == 0 and rows < self.prefill_rows:
                return self.matmul_config(
                    m=m,
                    k=k,
                    n=self.dim,
                    grid_size=(cols, rows),
                    per_core_M=m_tiles // rows,
                    per_core_N=math.ceil(self.dim / (ttnn.TILE_SIZE * cols)),
                )
        return super().get_mlp_ff2_prg_config(mode, seq_len, prefetcher)

    def _long_prefill_ff1_3_prg_config(self, seq_len: int):
        """Widen the LONG-prefill FF1/FF3 (M >= prefill_rows tiles) past stock's 8x8 = 64 cores.

        ``find_prefill_grid`` is hard-capped at ``max_rows = max_cols = 8`` behind a "TODO Improve
        configuration for BH" comment, so the M=512 chunk runs on 64 of this 11x10 device's 110
        cores. The profiler tags that op ``bound=FLOP`` at 71.9% -- it is not waiting on DRAM, it is
        short of math engines, which is exactly the case where more cores pays (and the opposite of
        the SDPA / hand-matmul case where every extra core re-reads the same stream).

        COLUMNS are the reachable axis here, and unlike the short-prefill path this one can afford
        to move them: stock pins ``per_core_N`` to ``dram_shard_grid_width`` (8 -> 60 tiles) but
        480 N-tiles also divide EXACTLY by 10, so per_core_N=48 tiles the output with no remainder
        and no ragged last column. ROWS stay on a divisor of BOTH m_tiles and k_tiles, which keeps
        ``matmul_config``'s k-divisibility assert valid and leaves no row computing padding.

        M=512: (8,8)/per_core_N=60 -> (10,8)/per_core_N=48, i.e. 64 -> 80 cores and 120 -> 96
        output tiles per core. Returns None when it cannot beat stock's core count.
        """
        m = min(seq_len, self.prefill_len_cutoff)
        k = self.dim // self.cluster_shape[0]
        n = self.hidden_dim // self.cluster_shape[1]
        if m % ttnn.TILE_SIZE:
            return None
        m_tiles, k_tiles, n_tiles = m // ttnn.TILE_SIZE, k // ttnn.TILE_SIZE, n // ttnn.TILE_SIZE
        if m_tiles < self.prefill_rows:
            return None  # the short-prefill path owns this shape
        gx, gy = self.max_grid_size.x, self.max_grid_size.y
        cols = max((c for c in range(1, gx + 1) if n_tiles % c == 0), default=1)
        rows = max((r for r in range(1, gy + 1) if k_tiles % r == 0 and m_tiles % r == 0), default=1)
        stock_rows, stock_cols = self.find_prefill_grid(self.prefill_rows, self.dim // ttnn.TILE_SIZE)
        if cols * rows <= stock_cols * stock_rows:
            return None
        return self.matmul_config(
            m=m,
            k=k,
            n=n,
            grid_size=(cols, rows),
            per_core_M=m_tiles // rows,
            per_core_N=n_tiles // cols,
        )

    @lru_cache(maxsize=None)
    def get_mlp_ff1_3_prg_config(self, mode: Mode, seq_len: int = 1, prefetcher: Prefetcher = None):
        if mode == Mode.PREFILL and prefetcher is None and not self.is_galaxy:
            pc = self._short_prefill_ff1_3_prg_config(seq_len)
            if pc is not None:
                return pc
            pc = self._long_prefill_ff1_3_prg_config(seq_len)
            if pc is not None:
                return pc
        return super().get_mlp_ff1_3_prg_config(mode, seq_len, prefetcher)

    def _set_tensor_dtype(self, dtype_by_tensor):
        """Override the weight dtype for specific tensor groups across EVERY decoder.

        Same shape as ``_set_op_fidelity`` below, and deliberately applied to every decoder rather
        than a layer subset so the lever cannot land on only the profiled slice.
        """
        for decoder_id, conf in list(self.optimizations.decoder_optimizations.items()):
            tensor_precision = {key: value for key, value in conf.tensor_dtype_settings.items() if value is not None}
            tensor_precision.update(dtype_by_tensor)
            op_fidelity = dict(conf.op_fidelity_settings)
            fixed_conf = ModelOptimizations({"TensorPrecision": tensor_precision, "OpFidelity": op_fidelity})
            fixed_conf.__name__ = getattr(conf, "__name__", fixed_conf.__name__)
            self.optimizations.set_decoder_conf(decoder_id, fixed_conf)
        self.model_config["DECODERS_OPTIMIZATIONS"] = self.optimizations

    def _set_op_fidelity(self, fidelity_by_op):
        """Override math fidelity for specific op groups across every decoder."""
        for decoder_id, conf in list(self.optimizations.decoder_optimizations.items()):
            tensor_precision = {key: value for key, value in conf.tensor_dtype_settings.items() if value is not None}
            op_fidelity = dict(conf.op_fidelity_settings)
            op_fidelity.update(fidelity_by_op)
            fixed_conf = ModelOptimizations({"TensorPrecision": tensor_precision, "OpFidelity": op_fidelity})
            fixed_conf.__name__ = getattr(conf, "__name__", fixed_conf.__name__)
            self.optimizations.set_decoder_conf(decoder_id, fixed_conf)
        self.model_config["DECODERS_OPTIMIZATIONS"] = self.optimizations

    def _relax_attention_ops_for_program_trace(self):
        """Lower L1 for prefill+decode attention under program tracing (minimal_matmul / SDPA / linear)."""
        trace_groups = (
            OpGroup.LI_QKV_PREFILL,
            OpGroup.LI_O_PREFILL,
            OpGroup.SDPA_PREFILL,
            OpGroup.LI_QKV_DECODE,
            OpGroup.LI_O_DECODE,
            OpGroup.SDPA_DECODE,
        )
        for decoder_id, conf in list(self.optimizations.decoder_optimizations.items()):
            tensor_precision = {k: v for k, v in conf.tensor_dtype_settings.items() if v is not None}
            op_fidelity = dict(conf.op_fidelity_settings)
            for grp in trace_groups:
                if grp in op_fidelity:
                    op_fidelity[grp] = MathFidelitySetting.HIFI2_FP16
            fixed_conf = ModelOptimizations({"TensorPrecision": tensor_precision, "OpFidelity": op_fidelity})
            fixed_conf.__name__ = getattr(conf, "__name__", fixed_conf.__name__)
            self.optimizations.set_decoder_conf(decoder_id, fixed_conf)
        self.model_config["DECODERS_OPTIMIZATIONS"] = self.optimizations

    def _force_sdpa_decode_hifi2_na(self):
        """Gemma3 decode SDPA requires no-accumulation HiFi2 for correctness (single-device)."""
        for decoder_id, conf in list(self.optimizations.decoder_optimizations.items()):
            tensor_precision = {key: value for key, value in conf.tensor_dtype_settings.items() if value is not None}
            op_fidelity = dict(conf.op_fidelity_settings)
            op_fidelity[OpGroup.SDPA_DECODE] = MathFidelitySetting.HIFI2_NA
            fixed_conf = ModelOptimizations({"TensorPrecision": tensor_precision, "OpFidelity": op_fidelity})
            fixed_conf.__name__ = getattr(conf, "__name__", fixed_conf.__name__)
            self.optimizations.set_decoder_conf(decoder_id, fixed_conf)
        self.model_config["DECODERS_OPTIMIZATIONS"] = self.optimizations

    def _force_sdpa_prefill_hifi4_fp16(self):
        for decoder_id, conf in list(self.optimizations.decoder_optimizations.items()):
            tensor_precision = {key: value for key, value in conf.tensor_dtype_settings.items() if value is not None}
            op_fidelity = dict(conf.op_fidelity_settings)
            op_fidelity[OpGroup.SDPA_PREFILL] = MathFidelitySetting.HIFI4_FP16
            fixed_conf = ModelOptimizations({"TensorPrecision": tensor_precision, "OpFidelity": op_fidelity})
            fixed_conf.__name__ = getattr(conf, "__name__", fixed_conf.__name__)
            self.optimizations.set_decoder_conf(decoder_id, fixed_conf)
        self.model_config["DECODERS_OPTIMIZATIONS"] = self.optimizations

    @staticmethod
    def _resolve_hf_snapshot(hf_model_name):
        hf_cache = os.path.normpath(
            os.environ.get("HF_HUB_CACHE")
            or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
        )
        model_slug = "models--" + hf_model_name.replace("/", "--")
        snapshots_dir = os.path.normpath(os.path.join(hf_cache, model_slug, "snapshots"))
        # Prevent path traversal: ensure the resolved path stays within hf_cache.
        if not snapshots_dir.startswith(hf_cache + os.sep):
            return None
        if not os.path.isdir(snapshots_dir):
            return None
        snaps = [
            os.path.join(snapshots_dir, s)
            for s in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, s))
        ]
        return max(snaps, key=os.path.getmtime) if snaps else None

    def get_max_prefill_chunk_size(self):
        model_overrides = {
            "gemma-3-4b": {"P150": 128},
            "medgemma-4b": {"P150": 128},
            "gemma-3-27b": {"P150": 128},
            "medgemma-27b": {"P150": 128},
        }
        model_name = self.base_model_name
        device_name = self.device_name
        if model_name in model_overrides and device_name in model_overrides[model_name]:
            return model_overrides[model_name][device_name] * 1024
        return super().get_max_prefill_chunk_size()

    def get_mlp_ff2_mem_config(self, mode: Mode, prefetcher: Prefetcher = None):
        """knob:shard probe on the DECODE FF2 output: L1 width-sharded -> DRAM."""
        if _GEMMA3_FF2_DECODE_OUT_DRAM and mode == Mode.DECODE and prefetcher is None:
            return ttnn.DRAM_MEMORY_CONFIG
        return super().get_mlp_ff2_mem_config(mode, prefetcher)

    def find_grid_k_n(self, K: int, N: int):
        """Size the DRAM-sharded (decode) matmul grid against the REAL device grid, not a fixed 8x8.

        ``tt_transformers.find_grid_k_n`` hard-codes ``max_rows = max_cols = 8`` -- its sibling
        ``find_grid`` has a Blackhole branch, this one was never given it. On an 11x10 P150 that caps
        the decode MLP at 40 of 110 cores: 60 cores divide both 120 and 480 tiles exactly, but 60
        needs a 6x10 grid and 10 > the hard-coded 8, so the search falls through to 5x8.

        This is the top lever on the decode path. The FF1/FF3 and FF2 decode matmuls are the two
        costliest ops in the profile and measure ~50% DRAM / ~50% FLOP, so they are half
        compute-starved -- confirmed independently by the weight-dtype sweep, where doubling their
        bytes cost only +15%. Cores, not bytes, are what they are short of.

        Everything downstream (dram_matmul_config's in0_block_w/per_core_N, the L1 width-shard of
        the activation, the binary-mult shard, the sharded norm config) is derived from this grid,
        so widening it here keeps them all consistent.

        MORE CORES IS NOT UNCONDITIONALLY BETTER, and the measurement says so. Splitting K across
        more cores also shrinks each core's K block, and ``dram_matmul_config`` derives
        ``in0_block_w`` from exactly that. Taking 60 cores unconditionally moved FF2 the right way
        (K block 6 -> 8, 50.9 -> 49.6 ms, and its gate mul 18.6 -> 18.1) but moved FF1/FF3 the wrong
        way (K block 3 -> 2, 55.2 -> 60.0 ms) for a net loss. So widen only while the K block does
        not shrink: that keeps 40 cores for FF1/FF3 and takes 60 for FF2.
        """
        grid = getattr(self, "max_grid_size", None)
        if grid is None:
            return super().find_grid_k_n(K, N)
        max_rows, max_cols = int(grid.y), int(grid.x)
        if _GEMMA3_FF2_DECODE_CORES and (K, N) == (self.hidden_dim, self.dim):
            for rows in range(max_rows, 0, -1):
                if _GEMMA3_FF2_DECODE_CORES % rows == 0 and _GEMMA3_FF2_DECODE_CORES // rows <= max_cols:
                    return rows, _GEMMA3_FF2_DECODE_CORES // rows
        base_rows, base_cols = super().find_grid_k_n(K, N)
        base_cores = base_rows * base_cols
        # The K block dram_matmul_config will actually use for a given core count.
        k_block = lambda cores: self.find_largest_divisor(K // cores)
        best = (base_rows, base_cols)
        for cores in sorted(c for c in range(1, max_rows * max_cols + 1) if K % c == 0 and N % c == 0):
            if cores <= base_cores or k_block(cores) < k_block(base_cores):
                continue
            for rows in range(1, max_rows + 1):
                if cores % rows == 0 and cores // rows <= max_cols:
                    best = (rows, cores // rows)
                    break
        if best != (base_rows, base_cols) or not _FEWER_CORES_FOR_BIGGER_K_BLOCK:
            return best
        # Nothing wider keeps the K block (FF1/FF3: 120 K-tiles, so 40 cores gives block 3 and the
        # next core count up drops it to 2). If the K block is genuinely what these matmuls are
        # short of, then spending FEWER cores to raise it should pay. Measured, not assumed.
        #
        # Among those candidates, prefer one whose per-core N slice tiles the weight's DRAM BANK
        # shard exactly (knob:shard). ``create_dram_sharded_mem_config`` width-shards the weight over
        # dram_grid_size.x banks, so each bank holds N/banks columns; the DRAM-sharded matmul then
        # gives each core N/cores of them. When N/cores does not divide N/banks, a core's slice
        # straddles a bank boundary and its weight stream is split across two banks' read queues
        # instead of one. FF1/FF3 today lands on 30 cores for 8 banks (480 N-tiles: 60 per bank, 16
        # per core -- 60 % 16 != 0, so 14 of the 30 cores straddle). Requiring alignment picks 24
        # cores instead: 20 N-tiles per core, 3 cores per bank, no straddle -- and it happens to buy
        # a bigger K block too (5 vs 4). FF2 never reaches this branch (a wider grid already keeps
        # its K block), so this is scoped to the FF1/FF3 shape.
        dram_grid = getattr(self, "dram_grid_size", None)
        banks = int(dram_grid.x) if dram_grid is not None else 0
        aligned = lambda cores: banks > 0 and N % banks == 0 and (N // banks) % (N // cores) == 0
        candidates = [c for c in range(1, base_cores) if K % c == 0 and N % c == 0 and k_block(c) > k_block(base_cores)]
        for cores in sorted(candidates, key=lambda c: (aligned(c), c), reverse=True):
            for rows in range(1, max_rows + 1):
                if cores % rows == 0 and cores // rows <= max_cols:
                    return rows, cores // rows
        return best

    # NOTE -- an L1 handoff for the prefill FF1/FF3 output was tried here and REMOVED. Keeping
    # w1_out/w3_out resident in L1 does cut a real DRAM round-trip and is worth ~12ms of eager
    # device_ms even alongside the LoFi FF2 above, but the two compete for L1 against the trace
    # region's statically allocated circular buffers, and the production trace+1cq per-token metric
    # is what pays: with both, per-token diverged to 77.3/77.9ms; with LoFi FF2 alone it is
    # 41.6/42.2ms. The eager profile cannot see that cost, so the handoff looks free there.

    def _short_prefill_qkv_prg_config(self, seq_len: int):
        """The wasted-rows fix again, on the QKV projection -- plus the block sizes stock never set.

        Stock's short-prefill QKV is the config carrying its own "FIXME: optimize this config for
        prefill": grid (8, 10) with ``per_core_M = max(1, ceil(seq_len/32/8)) = 1``, ``in0_block_w=1``
        and ``out_subblock_h/w = 1``. Three things go to waste at ISL 128 and one edit fixes all
        three, because ``matmul_config`` derives the last two from the grid:

        * 10 grid ROWS at per_core_M=1 cover 10 M-tiles when the op has 4 -- six rows idle, 32 of 80
          cores active. Trimming rows to the M-tiles that exist makes every assigned core work.
        * ``in0_block_w=1`` walks K in 120 single-tile steps; at rows=4 the derived block is
          find_largest_divisor(120 // 4) = 6.
        * ``out_subblock_w=1`` against per_core_N=32 wastes the DST budget; get_out_subblock_w picks
          up to 8 at out_subblock_h=1.

        COLUMNS stay at ``dram_shard_grid_width`` -- per_core_N is pinned to the QKV weight's 8-bank
        DRAM shard, and mismatching it silently corrupts (measured PCC 0.31 elsewhere in this file).
        """
        k = self.dim // self.cluster_shape[0]
        n = self.qkv_size // self.cluster_shape[1]
        m_tiles, k_tiles = seq_len // ttnn.TILE_SIZE, k // ttnn.TILE_SIZE
        if seq_len % ttnn.TILE_SIZE or m_tiles >= self.prefill_rows:
            return None
        rows = max((y for y in range(1, m_tiles + 1) if m_tiles % y == 0 and k_tiles % y == 0), default=1)
        if rows >= self.prefill_rows:
            return None
        cols = self.dram_shard_grid_width
        return self.matmul_config(
            m=seq_len,
            k=k,
            n=n,
            grid_size=(cols, rows),
            fuse_batch=True,  # stock uses fuse_batch for seq_len <= MAX_QKV_MM_SEQ_LEN
            per_core_M=m_tiles // rows,
            per_core_N=math.ceil(n / (ttnn.TILE_SIZE * cols)),
        )

    # The traced long-prefill QKV grid. Both stock and the trace branch below hard-code
    # CoreCoord(8, 10) on Blackhole, and the 8 is the expensive half of that pair.
    _QKV_MINIMAL_STOCK_GRID = (8, 10)  # (x, y), i.e. N rides 8 columns and M rides 10 rows

    # Grid variant 2 (under measurement): let the search spend more than stock's 80 cores, and fund
    # the extra cores' circular buffers by halving the traced N block rather than out of the trace
    # region. Both halves move together -- more cores at the SAME per-core CB is the combination
    # already measured as a per-token regression.
    _QKV_MINIMAL_SPEND_CORES = os.environ.get("GEMMA3_QKV_MINIMAL_SPEND_CORES", "1") == "1"
    _QKV_MINIMAL_TRACE_N_BLOCK = int(os.environ.get("GEMMA3_QKV_MINIMAL_TRACE_N_BLOCK", "2"))

    @staticmethod
    def _minimal_matmul_subblock(
        m_block: int, n_block: int, n_ge_m: bool, fp32_dest_acc_en: bool = False, dst_full_sync_en: bool = False
    ):
        """The subblock ``MinimalMatmulConfig`` silently drops when you set the block sizes.

        knob:fidelity on the compute-bound prefill matmuls. ``subblock_h``/``subblock_w`` are the
        MAC-block shape handed to ``matmul_block_init`` as (rt_dim, ct_dim) -- how many output tiles
        the math thread accumulates in DST per unpack of in0/in1. minimal_matmul_program_factory.cpp
        picks them itself (``determine_default_block_sizes``: 2x4 when N >= M and fp32_dest_acc_en is
        off, i.e. EIGHT tiles) but ONLY when no config is supplied:

            subblock_h = config.has_value() ? config.value().subblock_h : default_subblock_h;

        Every call site here supplies a config to set the block sizes, and the nanobind signature
        defaults ``subblock_h=1, subblock_w=1``. So asking for a block size silently also asks for a
        1x1 MAC block -- one output tile per matmul_block call, an eighth of the DST the library
        would have used, on ops the roofline tags bound_by=compute.

        Pick the largest legal subblock instead of inheriting that 1x1. The validate() constraints
        are ``M_block % subblock_h == 0``, ``N_block % subblock_w == 0`` and
        ``subblock_h * subblock_w <= get_dest_reg_count(...)``, and the orientation preference
        follows the factory's: widen along N when N >= M, along M otherwise.
        """
        # get_dest_reg_count(): (DEST_REGISTER_FULL_SIZE * DATUMS_PER_ROW) / tile area = 16 tiles,
        # HALVED when dst_full_sync_en is off (it is, by default) and halved again for fp32 dest. So
        # the real cap on this model's lofi config is EIGHT, not 16 -- asking for 16 TT_FATALs.
        max_volume = 16
        if not dst_full_sync_en:
            max_volume //= 2
        if fp32_dest_acc_en:
            max_volume //= 2
        divisors = lambda b: [d for d in range(1, b + 1) if b % d == 0]  # noqa: E731
        candidates = [(h, w) for h in divisors(m_block) for w in divisors(n_block) if h * w <= max_volume]

        # Largest DST volume wins. Break ties toward the factory's own 1:2 orientation -- it widens
        # along N when N >= M and along M otherwise -- so an 8-tile budget lands on 2x4 / 4x2 rather
        # than a degenerate 1x8, which fills DST just as full but unpacks a single in0 row per block.
        def _rank(hw):
            h, w = hw
            skew = abs(w - 2 * h) if n_ge_m else abs(h - 2 * w)
            return (h * w, -skew)

        return max(candidates, key=_rank)

    def _long_prefill_qkv_minimal_grid(self, seq_len: int):
        """Pick the QKV minimal_matmul grid from the tile counts instead of hard-coding 8x10.

        Same reasoning as ``_long_prefill_ff2_minimal_config``, on the other big prefill matmul.
        ``minimal_matmul`` takes no ``per_core_N``, so there is no DRAM-shard-width pin to violate:
        minimal_matmul_program_factory.cpp sets ``transpose_core_grid = M > N`` (false here, M=1024 <
        N=8192), then ``M_tiles_per_core = round_up(M_tiles, grid.y)/grid.y`` and
        ``N_tiles_per_core = round_up(N_tiles, grid.x)/grid.x``. Widening only shrinks each slice.

        What the hard-coded (8, 10) costs at S=1024: M_tiles=32 over y=10 is 4 per core, but N_tiles=256
        over x=8 is 32 per core -- 128 output tiles each. N is the axis starved of cores and M is the
        one being padded, so the fix is to spend the SAME budget the other way round; see the cap
        below for why this deliberately stops short of the widest grid.

        Block sizes stay at the caller's -- the CB budget is per core, so moving only the grid leaves
        L1, and the trace region it competes with, untouched.
        """
        # Above MAX_QKV_MM_SEQ_LEN the caller reshapes the sequence before the matmul, so the M the
        # factory sees is no longer seq_len; leave those lengths on the stock grid.
        if not (128 < seq_len <= getattr(self, "MAX_QKV_MM_SEQ_LEN", 2048)):
            return None
        n = self.qkv_size // self.cluster_shape[1]
        if seq_len % ttnn.TILE_SIZE or n % ttnn.TILE_SIZE or seq_len >= n:
            return None  # M >= N would transpose the axes; not the case reasoned about here
        m_tiles, n_tiles = seq_len // ttnn.TILE_SIZE, n // ttnn.TILE_SIZE
        gx, gy = self.max_grid_size.x, self.max_grid_size.y

        def _smallest_best(tiles, cap):
            best = min(math.ceil(tiles / c) for c in range(1, cap + 1))
            return min(c for c in range(1, cap + 1) if math.ceil(tiles / c) == best)

        stock_cols, stock_rows = self._QKV_MINIMAL_STOCK_GRID

        # Rank by per-core OUTPUT TILES, not core count: that is what the factory hands each core,
        # and a wider grid that rounds up worse would be a regression dressed up as more cores.
        def _per_core_tiles(c, r):
            return math.ceil(m_tiles / r) * math.ceil(n_tiles / c)

        # ...and do NOT spend cores to get there. MEASURED 2026-08-02: the unconstrained pick is
        # (11, 8) = 88 cores at 96 tiles per core, and it is a real prefill win -- the op goes
        # 405.8 -> 325.2 us/call and device_ms 377.08 -> 373.31. But prefill and decode share one
        # trace region, the CB budget is per core, and 8 extra cores' worth of CBs came out of it:
        # trace+1cq per-token went 35.08 -> 35.48 (+1.15%), and that is the metric that ships.
        # So hold the core count at stock's and buy the win from the AXIS ASSIGNMENT alone, which is
        # free: stock spends its 80 cores as 8 columns x 10 rows, but here N_tiles=256 is the axis
        # starved for cores and M_tiles=32 is the one being padded (10 rows cover 32 tiles as 4 each,
        # wasting 8 phantom tiles). Transposing that same budget to 10 x 8 gives 4 x 26 = 104 tiles
        # per core against stock's 4 x 32 = 128, for identical L1.
        #
        # CANDIDATE (grid variant 2): the rejection above was an L1 verdict, not a core-count one --
        # 88 cores lost because 8 more cores each reserved a full CB set out of the shared trace
        # region. So take the 88 and PAY for them out of the block size: halving N_block_size on the
        # traced branch (see get_attn_qkv_program_config) shrinks in1 + out + intermediate by 8 tiles
        # each per core, which is more L1 than the 8 extra cores add. The effective work per core is
        # unchanged by that halving -- N_tiles_per_core = 24 at gx=11 divides by 2 and by 4 alike --
        # so this is strictly the same 96-tiles-per-core prefill win at LOWER per-core L1 than the
        # config that was measured as a per-token regression.
        budget = gx * gy if self._QKV_MINIMAL_SPEND_CORES else stock_cols * stock_rows
        best = min(
            (
                (_per_core_tiles(c, r), c * r, c, r)
                for c in range(1, gx + 1)
                for r in range(1, gy + 1)
                if c * r <= budget
            ),
            default=None,
        )
        if best is None or best[0] >= _per_core_tiles(stock_cols, stock_rows):
            return None
        return ttnn.CoreCoord(best[2], best[3])

    def get_attn_qkv_program_config(self, mode: Mode, seq_len: int = 1, prefetcher: Prefetcher = None):
        """Smaller MinimalMatmul blocks for traced long prefill (default 8³ overflows static CB vs L1)."""
        if (
            mode == Mode.PREFILL
            and prefetcher is None
            and not self.is_galaxy
            and not self.use_minimal_qkv_prefill_matmul(seq_len)
        ):
            pc = self._short_prefill_qkv_prg_config(seq_len)
            if pc is not None:
                return pc
        traced_long_prefill = self._enable_program_trace and mode == Mode.PREFILL and seq_len > 128
        # The widened grid has to be applied on BOTH paths. Stock's minimal branch and the traced
        # branch below hard-code the SAME CoreCoord(8, 10), so putting it only on the traced one made
        # the lever an invisible no-op under the (untraced) profiler -- measured: still 80 cores.
        # The two paths differ only in block size: 4 under trace, where 8**3 overflows the static CB.
        grid = None
        if (
            is_blackhole()
            and mode == Mode.PREFILL
            and prefetcher is None
            and not self.is_galaxy
            and (traced_long_prefill or self.use_minimal_qkv_prefill_matmul(seq_len))
        ):
            grid = self._long_prefill_qkv_minimal_grid(seq_len)
        if traced_long_prefill:
            # N_block_size is the CB axis that funds the wider grid: in1, out and intermediate are all
            # K_block x N_block / M_block x N_block, so halving it takes 8 tiles off each of the three
            # per core. The compute kernel clamps the last block anyway (compute.cpp sets
            # current_N_block_tiles = n_tile_end - n_tile), so a smaller block loses no math -- and at
            # gx=11 it divides N_tiles_per_core=24 exactly, where 4 does too.
            n_block = self._QKV_MINIMAL_TRACE_N_BLOCK if grid is not None else 4
            sub_h, sub_w = self._minimal_matmul_subblock(4, n_block, n_ge_m=True)
            return ttnn.MinimalMatmulConfig(
                M_block_size=4,
                K_block_size=4,
                N_block_size=n_block,
                subblock_h=sub_h,
                subblock_w=sub_w,
                compute_with_storage_grid_size=grid
                or (ttnn.CoreCoord(8, 10) if is_blackhole() else ttnn.CoreCoord(8, 8)),
            )
        if grid is not None:
            sub_h, sub_w = self._minimal_matmul_subblock(8, 8, n_ge_m=True)
            return ttnn.MinimalMatmulConfig(
                M_block_size=8,
                K_block_size=8,
                N_block_size=8,
                subblock_h=sub_h,
                subblock_w=sub_w,
                compute_with_storage_grid_size=grid,
            )
        return super().get_attn_qkv_program_config(mode, seq_len, prefetcher)

    # STRUCTURAL, MEASURED AND REJECTED (2026-08-02): landing the PREFILL QKV matmul output in L1
    # instead of DRAM removes the head-split's DRAM READ leg (the grid rung already moved its
    # write), and it is a real device_ms win -- 393.95 -> 390.55 (-3.4 ms) at unchanged PCC. It is
    # still the wrong trade: the prefill tensor it parks in L1 is [1, 1, S, 8192], and that
    # pressure lands on the region the decode trace shares, so trace+1cq per-token went 35.48-35.58
    # -> 35.75/36.02, about +1%. Eager device_ms cannot see trace-region pressure, so this class of
    # L1 lever has to be judged on the per-token metric, which is the one that ships.

    def get_attn_sdpa_prefill_program_config(self, seq_len: int = 1, chunk_start_idx: int = None):
        """Give the SHORT-prefill SDPA enough q-chunks to fill the grid it is already given.

        Stock hard-codes ``compute_with_storage_grid_size=(8, 8)`` and picks ``q_chunk = 64`` for any
        seq_len < 2048. SDPA parallelises over (batch x heads x q_chunks), so at ISL 128 that is
        1 x 16 x ceil(128/64) = 32 work units on 64 cores -- HALF THE GRID IDLE, which is the
        grid=partial tag on this op. It measures 105us per call for ~1.7MB of QKV traffic (~16 GB/s),
        i.e. it is occupancy-bound, not bandwidth-bound.

        Widening the grid cannot fix that: with only 32 units, cores past 32 have nothing to do. The
        fix is more UNITS -- halve q_chunk to 32 (one tile, the floor) and the same 128 tokens split
        into 4 chunks per head, giving 64 units for the 64 cores. q_chunk must divide
        chunk_start_idx, and 32 divides everything 64 divided, so the stock constraint still holds.

        Only fires when the default genuinely underfills; otherwise defer to stock.
        """
        cfg = super().get_attn_sdpa_prefill_program_config(seq_len, chunk_start_idx)
        grid = cfg.compute_with_storage_grid_size
        cores = int(grid.x) * int(grid.y)
        heads = self.n_heads // self.cluster_shape[1]
        q_chunk = int(cfg.q_chunk_size)
        if q_chunk <= ttnn.TILE_SIZE or heads * math.ceil(seq_len / q_chunk) >= cores:
            return cfg
        while q_chunk > ttnn.TILE_SIZE and heads * math.ceil(seq_len / q_chunk) < cores:
            q_chunk //= 2
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(int(grid.x), int(grid.y)),
            exp_approx_mode=False,
            q_chunk_size=q_chunk,
            k_chunk_size=int(cfg.k_chunk_size),
        )

    def get_attn_sdpa_decode_program_config(self, prefetcher: Prefetcher = None):
        force_fixed_k_chunk = getattr(self, "force_fixed_decode_k_chunk", False)
        if not force_fixed_k_chunk:
            return super().get_attn_sdpa_decode_program_config(prefetcher)

        override = getattr(self, "_gemma3_sdpa_decode_k_chunk_override", None)
        k_chunk_tokens = _GEMMA3_SDPA_DECODE_K_CHUNK_DEFAULT if override is None else int(override)
        if prefetcher is not None:
            sdpa_grid_size = (8, 8)
            start_core = ttnn.CoreCoord(1, 0)
            num_sdpa_cores = sdpa_grid_size[0] * sdpa_grid_size[1]
            return ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=sdpa_grid_size,
                sub_core_grids=ttnn.num_cores_to_corerangeset_in_subcoregrids(
                    start_core, num_sdpa_cores, prefetcher.all_worker_cores_range_set, row_wise=True
                ),
                exp_approx_mode=False,
                q_chunk_size=0,
                k_chunk_size=k_chunk_tokens,
            )

        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=k_chunk_tokens,
        )

    def get_warmup_prefill_supported_seq_lens(self):
        DEFAULT_VALUE = self.capped_warmup_seq_len

        # This dictionary is used to override the default ceil warmup prefill value
        # Longer seqlens take too much time to warmup, so CI times out
        model_specific_ceil_warmup_lengths = {
            "gemma-3-4b": 2048,
            "gemma-3-27b": 2048,
        }

        max_seq_len_to_warmup = model_specific_ceil_warmup_lengths.get(self.base_model_name, DEFAULT_VALUE)
        if max_seq_len_to_warmup > self.capped_warmup_seq_len:
            max_seq_len_to_warmup = self.capped_warmup_seq_len

        to_warmup_seq_lens = calculate_prefill_warmup_seq_lens(
            max_seq_len_to_warmup, self.trace_prefill_supported_seq_lens
        )

        to_warmup_seq_lens = self.filter_warmup_seq_lens(to_warmup_seq_lens)

        return to_warmup_seq_lens

    def filter_warmup_seq_lens(self, to_warmup_seq_lens):
        # TODO: Add more model-specific filtering here
        # This filtering is based on the current PR's (https://github.com/tenstorrent/tt-metal/pull/33143) sequence lengths that are used for warmup
        return to_warmup_seq_lens

    def get_trace_prefill_supported_seq_lens(self):
        # NOTE: opting P150 IN to traced prefill ("P150": [128]) was tried here and REVERTED.
        # ``Generator.can_enable_trace`` gates prefill tracing on membership of this list, so today
        # every prefill is dispatched op-by-op while decode replays a trace -- prefill is the only
        # stage still paying host dispatch, which is what the host_overhead bucket measures. But
        # capture EXECUTES the pass it records: the M=128 FF1/FF3 count went 192 -> 384 (two extra
        # prefill passes inside the profiled window) and device_ms 448.9 -> 515.1. In a long-lived
        # server that capture is amortised over many prompts; in this pipeline -- one prefill plus a
        # handful of decode tokens -- it never is, so tracing prefill costs more than the dispatch
        # gaps it removes.
        default_supported_seq_lens = {
            # for gemma we have different default supported seq lens than in tt_transformers
            # TODO: should be empty until https://github.com/tenstorrent/tt-metal/issues/33041 is fixed
            "N150": [],
            "N300": [],
            "T3K": [],
            "TG": [],
            "P150": [],
        }

        # TODO: If no specific sequence lengths are listed for a model and device, the default one will be used (from the default_supported_seq_lens dictionary)
        # TODO: should be empty until https://github.com/tenstorrent/tt-metal/issues/33041 is fixed
        model_specific_supported_seq_lens = {
            # EXAMPLE: "gemma-3-4b": {
            #     "N150": [128, 1024, 2048],
            # }
        }

        model_name = self.base_model_name
        device_name = self.device_name

        # If there is no entry for a model in model_specific_supported_seq_lens, use the entry in default_supported_seq_lens
        result = model_specific_supported_seq_lens.get(model_name, {}).get(
            device_name, default_supported_seq_lens.get(device_name)
        )

        if result is not None:
            return cap_seq_lens_to_max_prefill_chunk_size(result, self.capped_warmup_seq_len)
        else:
            return []

    def _set_model_specific_params(self):
        self.rms_norm_add_unit_offset = True
        self.embed_scale = self.dim**0.5

    # def _set_vision_params(self, vision_config):
    #     self.vision_dim = vision_config.get("hidden_size", 1280)
    #     self.vision_mlp_ratio = vision_config.get("intermediate_size", self.vision_dim * 4) // self.vision_dim
    #     self.vision_hidden_dim = vision_config.get("intermediate_size", self.vision_dim * self.vision_mlp_ratio)
    #     self.vision_attn_n_heads = vision_config.get("num_attention_heads", 16)
    #     self.vision_head_dim = self.vision_dim // self.vision_attn_n_heads
    #     self.vision_n_layers = vision_config.get("num_hidden_layers", 32)
    #     self.vision_patch_size = vision_config.get("patch_size", 14)
    #     self.vision_in_channels = vision_config.get("num_channels", 3)
    #     self.vision_act_layer = ttnn.UnaryOpType.GELU  # or read from config if variable
    #     self.vision_dropout = vision_config.get("attention_dropout", 0.0)
    #     self.vision_max_num_tiles = 4
    #     self.vision_n_global_layers = 8

    def _set_vision_params(self, vision_config):
        self.vision_chunk_size = vision_config.get("vision_chunk_size", 896)
        self.vision_max_num_chunks = vision_config.get("vision_max_num_chunks", 4)
        self.vision_num_cross_attention_layers = vision_config.get("vision_num_cross_attention_layers", 8)
        self.vision_dim = vision_config.get("hidden_size", 1152)

        intermediate_size = vision_config.get("intermediate_size", self.vision_dim * 4)
        self.vision_mlp_ratio = intermediate_size // self.vision_dim
        self.vision_hidden_dim = int(self.vision_dim * self.vision_mlp_ratio)
        self.vision_attn_n_heads = vision_config.get("num_attention_heads", 16)
        self.vision_head_dim = self.vision_dim // self.vision_attn_n_heads

        self.vision_n_layers = vision_config.get("num_hidden_layers", 27)
        self.vision_patch_size = vision_config.get("patch_size", 14)
        self.vision_in_channels = vision_config.get("num_channels", 3)

        self.vision_dropout = vision_config.get("attention_dropout", 0.0)
        self.mm_tokens_per_image = vision_config.get("mm_tokens_per_image", 256)

        # Optional vision activation layer, defaults to GELU
        act_layer = vision_config.get("act_layer", "gelu").lower()
        self.vision_act_layer = {
            "gelu": ttnn.UnaryOpType.GELU,
            "relu": ttnn.UnaryOpType.RELU,
            "silu": ttnn.UnaryOpType.SILU,
        }.get(act_layer, ttnn.UnaryOpType.GELU)

        self.vision_n_global_layers = vision_config.get("n_global_layers", 8)

    def _set_hf_params(self, checkpoint_dir):
        def merge_text_config(base_config):
            text_config = base_config.get("text_config", {})
            # Merge non-nested keys into text_config
            text_config.update({k: v for k, v in base_config.items() if k not in ["text_config", "vision_config"]})
            return text_config

        def merge_vision_config(base_config):
            vision_config = base_config.get("vision_config", {})
            # Merge non-nested keys into vision_config
            vision_config.update({k: v for k, v in base_config.items() if k not in ["text_config", "vision_config"]})
            return vision_config

        from transformers import AutoConfig

        # For dummy_weights we still load only the small HF config,
        # but we avoid loading checkpoint weights.
        self.hf_config = AutoConfig.from_pretrained(self.CKPT_DIR).to_dict()

        if "text_config" in self.hf_config or "vision_config" in self.hf_config:
            self._set_params_from_dict(self.hf_config)
            if "vision_config" in self.hf_config:
                merged_vision_config = merge_vision_config(self.hf_config)
                self._set_vision_params(merged_vision_config)
        else:
            self._set_params_from_dict(self.hf_config)

    def get_state_dict_prefix(self, module_name, layer_num, is_vision=False):
        if is_vision:
            text_prefix = "model.vision_tower.vision_model.encoder."
        else:
            text_prefix = ""

        layer_prefix = f"layers.{layer_num}." if layer_num is not None else ""

        module_map = {
            "MLP": "feed_forward",
            "Attention": "attention",
            "TransformerBlock": "",
            "": "",  # If no module is given, just get layer prefix
        }

        vision_module_map = {
            "MLP": "mlp.",
            "Attention": "self_attn.",
            "TransformerBlock": "",
            "": "",
        }

        module_map = vision_module_map if is_vision else module_map

        return text_prefix + layer_prefix + module_map[module_name]

    def _gemma_dummy_hf_model(self):
        """Build Gemma3 from HF config only (random init), matching tt_transformers ModelArgs dummy_weights flow.

        Uses from_config + layer truncation + bfloat16 to avoid fp32 OOM on host when allocating the full model.
        """
        from transformers import AutoConfig, Gemma3ForConditionalGeneration

        logger.info("Gemma3 ModelArgs: building HF dummy model from config (dummy_weights=True)")

        config = AutoConfig.from_pretrained(self.CKPT_DIR, trust_remote_code=self.trust_remote_code_hf)
        if hasattr(config, "text_config") and config.text_config is not None:
            config.text_config.num_layers = self.n_layers
            config.text_config.num_hidden_layers = self.n_layers
        else:
            if hasattr(config, "num_layers"):
                config.num_layers = self.n_layers
            if hasattr(config, "num_hidden_layers"):
                config.num_hidden_layers = self.n_layers

        model_cls = Gemma3ForConditionalGeneration
        from_config_exc = None
        try:
            try:
                model = model_cls.from_config(
                    config, torch_dtype=torch.bfloat16, trust_remote_code=self.trust_remote_code_hf
                )
            except TypeError:
                try:
                    model = model_cls.from_config(config, torch_dtype=torch.bfloat16)
                except TypeError:
                    try:
                        model = model_cls.from_config(config, trust_remote_code=self.trust_remote_code_hf)
                    except TypeError:
                        model = model_cls.from_config(config)
        except Exception as exc:
            from_config_exc = exc
            logger.info("Error loading dummy Gemma3 using .from_config. Error: {}", exc)
            if hasattr(model_cls, "_from_config"):
                try:
                    try:
                        model = model_cls._from_config(
                            config, torch_dtype=torch.bfloat16, trust_remote_code=self.trust_remote_code_hf
                        )
                    except TypeError:
                        model = model_cls._from_config(config, torch_dtype=torch.bfloat16)
                except Exception as fallback_exc:
                    logger.info("Error loading dummy Gemma3 using ._from_config. Error: {}", fallback_exc)
                    if from_config_exc is not None:
                        raise fallback_exc from from_config_exc
                    raise
            else:
                raise

        gc.collect()
        return model

    # TODO Update function for large models: For 1 layer tests we only want to load 1 checkpoint file, instead of all.
    def load_state_dict(self):
        from transformers import Gemma3ForConditionalGeneration

        if self.dummy_weights:
            logger.info("Gemma3 ModelArgs: using dummy_weights path; NOT loading checkpoints from HF_MODEL")
            model = self._gemma_dummy_hf_model()
            state_dict = model.state_dict()
            del model
            gc.collect()
        else:
            model = Gemma3ForConditionalGeneration.from_pretrained(
                self.CKPT_DIR,
                torch_dtype="auto",
            )
            if self.cache_hf_flag:
                self.cached_hf_model = model
            state_dict = model.state_dict()

        if self.is_multimodal:
            state_dict = convert_vision_hf_to_meta(state_dict, self.head_dim)
        else:
            state_dict = standardize_hf_keys(state_dict)
            state_dict = convert_hf_to_meta(state_dict, self.head_dim)

        keys_dict = list(state_dict.keys())[:]
        remv = [f"layers.{i}." for i in list(range(self.n_layers, self.full_model_n_layers))]
        for k in keys_dict:
            if any([r in k for r in remv]):
                state_dict.pop(k)

        return state_dict

    @staticmethod
    def _gemma3_multi_modal_projector(model):
        # transformers 5.x wraps the inner Gemma3Model as `model.model`, moving
        # multi_modal_projector off the top-level Gemma3ForConditionalGeneration.
        mmp = getattr(model, "multi_modal_projector", None)
        if mmp is None:
            mmp = model.model.multi_modal_projector
        return mmp

    @staticmethod
    def _gemma3_vision_tower(model):
        # transformers 5.x wraps the inner Gemma3Model as `model.model`, moving
        # vision_tower off the top-level Gemma3ForConditionalGeneration (same as
        # multi_modal_projector above).
        vt = getattr(model, "vision_tower", None)
        if vt is None:
            vt = model.model.vision_tower
        return vt

    @classmethod
    def _gemma3_vision_transformer(cls, model):
        # transformers 5.x flattened SiglipVisionModel (dropped the `.vision_model` /
        # SiglipVisionTransformer wrapper); embeddings/encoder/post_layernorm are now direct
        # attributes. Return that transformer level on <5 (`.vision_model`) and >=5 (the tower itself).
        vt = cls._gemma3_vision_tower(model)
        return vt.vision_model if hasattr(vt, "vision_model") else vt

    def reference_vision_multi_modal(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = self._gemma3_multi_modal_projector(model)
        return layer

    def reference_vision_rms_norm(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = self._gemma3_multi_modal_projector(model).mm_soft_emb_norm
        return layer

    def reference_rms_norm(self, i=0):
        model = self.reference_transformer(wrap=False)
        layer = model.model.layers[i].self_attn.q_norm
        layer._load_state_dict = layer.load_state_dict
        layer.load_state_dict = lambda x: layer._load_state_dict(convert_meta_to_hf(x, self.head_dim))
        return layer

    def reference_rms_norm_text(self):
        model = self.reference_transformer(wrap=False)
        layer = model.model.norm
        layer._load_state_dict = layer.load_state_dict
        layer.load_state_dict = lambda x: layer._load_state_dict(convert_meta_to_hf(x, self.head_dim))
        return layer

    def get_hf_model_cls(self):
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

        if not self.is_multimodal:
            return AutoModelForCausalLM

        # AutoModelForVision2Seq was removed in transformers 5.x; its model mapping
        # was folded into AutoModelForImageTextToText (available since 4.46).
        for model_cls in (AutoModelForImageTextToText,):
            if type(self.hf_config) == dict:
                return model_cls

        raise ValueError(f"Unknown model for config {type(self.hf_config)}")

    def reference_mlp(self):
        model = self.reference_transformer(wrap=False)
        layer = model.model.layers[0].mlp
        layer._load_state_dict = layer.load_state_dict
        layer.load_state_dict = lambda x: layer._load_state_dict(convert_meta_to_hf(x, self.head_dim))
        return layer

    def reference_vision_transformer(self, wrap=True, load_checkpoint=False):
        from transformers import Gemma3ForConditionalGeneration

        if self.dummy_weights and not load_checkpoint:
            model = self._gemma_dummy_hf_model()
        else:
            model = Gemma3ForConditionalGeneration.from_pretrained(self.CKPT_DIR)
        # transformers 5.x from_pretrained honors the checkpoint dtype (bf16); force float32 so the
        # golden reference matches float32 inputs (e.g. the multi_modal_projector matmul, which
        # otherwise raises "expected m1 and m2 to have the same dtype, but got: float != BFloat16").
        model = model.float()
        if wrap:
            wrapper = HfModelWrapper(model, self.head_dim)
            return wrapper
        else:
            return model

    def reference_gemma_model(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = model
        layer._load_state_dict = layer.load_state_dict
        layer.load_state_dict = lambda x: layer._load_state_dict(convert_vision_meta_to_hf(x, self.head_dim))
        return layer

    def reference_vision_model(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = self._gemma3_vision_transformer(model)
        return layer

    def reference_vision_mlp(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = self._gemma3_vision_transformer(model).encoder.layers[0].mlp
        return layer

    def reference_siglip_patch_embed(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = self._gemma3_vision_transformer(model).embeddings.patch_embedding
        return layer

    def reference_vision_pos_embedding(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = self._gemma3_vision_transformer(model).embeddings.position_embedding
        return layer

    def reference_vision_embedding(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = self._gemma3_vision_transformer(model).embeddings
        return layer

    def reference_vision_layernorm(self, layer_name="layer_norm1"):
        model = self.reference_vision_transformer(wrap=False)
        if layer_name == "layer_norm1":
            layer = self._gemma3_vision_transformer(model).encoder.layers[0].layer_norm1
        elif layer_name == "layer_norm2":
            layer = self._gemma3_vision_transformer(model).encoder.layers[0].layer_norm2
        else:
            layer = self._gemma3_vision_transformer(model).post_layernorm
        return layer

    def reference_vision_attention(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = self._gemma3_vision_transformer(model).encoder.layers[0].self_attn  # Common naming
        return layer

    def reference_vision_encoder_block(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = self._gemma3_vision_transformer(model).encoder.layers[0]
        return layer

    def reference_vision_encoder(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = self._gemma3_vision_transformer(model).encoder
        return layer

    def reference_decoder(self, i=0):
        model = self.reference_transformer(wrap=False)
        layer = model.model.layers[i]
        rotary_emb = model.model.rotary_emb

        rotary_emb_local = model.model.rotary_emb_local
        wrapper = HfGemmaDecoderWrapper(layer, self.head_dim, rotary_emb, rotary_emb_local)

        return wrapper

    def reference_decoder_text(self, i=0):
        model = self.reference_transformer(wrap=False)
        layer = model.model.layers[0]
        use_position_embeddings = layer.__class__.__name__ != "Phi3DecoderLayer" or self.base_model_name in ("phi-4",)
        if hasattr(model.model, "rotary_emb_local"):
            rotary_emb_local = model.model.rotary_emb_local
        else:
            rotary_emb_local = None
        wrapper = HfDecoderWrapper(
            layer, self.head_dim, model.model.rotary_emb if use_position_embeddings else None, rotary_emb_local
        )
        return wrapper

    def reference_attention(self, rope_embeddings="global"):
        model = self.reference_transformer(wrap=False)
        layer = model.model.layers[0].self_attn
        use_position_embeddings = layer.__class__.__name__ in ("Gemma3Attention",)
        rope_layer_type = None
        if "gemma-3" in self.model_name:
            if rope_embeddings == "local":
                rotary_emb = model.model.rotary_emb_local
                rope_layer_type = "sliding_attention"
            else:
                rotary_emb = model.model.rotary_emb
                rope_layer_type = "full_attention"
        else:
            rotary_emb = model.model.rotary_emb
        # transformers 5.x Gemma3 consolidated RoPE into one module that selects `{layer_type}_inv_freq`.
        # Layer 0 is a sliding (local) layer, so the attention's own layer_type would force LOCAL rope,
        # but this unit test compares against the explicitly requested rope module (global by default)
        # and the TT RotarySetup uses the global rope_theta. Pin the layer_type to the chosen module so
        # reference and TT use the same rope (matches the pre-5.x behavior).
        wrapper = HfAttentionWrapper(
            layer,
            self.head_dim,
            rotary_emb if use_position_embeddings else None,
            rope_layer_type=rope_layer_type,
        )
        return wrapper


class HfGemmaDecoderWrapper:
    def __init__(self, decoder, head_dim, rotary_emb, rotary_emb_local):
        from transformers import DynamicCache

        self.decoder = decoder
        self.head_dim = head_dim
        self.rotary_emb = rotary_emb
        self.rotary_emb_local = rotary_emb_local
        self.past_key_values = DynamicCache()

    def forward(self, x, start_pos, freqs_cis_i, mask=None):
        position_ids = torch.tensor([list(range(start_pos, start_pos + x.shape[1]))] * x.shape[0])
        # TODO: Generalize for other HF models

        # transformers 5.x consolidated Gemma3 RoPE into a module that selects `{layer_type}_inv_freq`
        # (layer_type=None -> AttributeError 'None_inv_freq'). Pass the matching layer_type when the
        # rotary forward accepts it; <5 rotaries don't take the kwarg.
        _takes_layer_type = "layer_type" in inspect.signature(self.rotary_emb.forward).parameters
        if _takes_layer_type:
            position_embeddings_global = self.rotary_emb(x, position_ids, layer_type="full_attention")
            position_embeddings_local = self.rotary_emb_local(x, position_ids, layer_type="sliding_attention")
        else:
            position_embeddings_global = self.rotary_emb(x, position_ids)
            position_embeddings_local = self.rotary_emb_local(x, position_ids)
        if mask is not None:
            while len(mask.shape) < 4:
                mask = mask.unsqueeze(0)
        # transformers 5.x renamed the decoder cache kwarg past_key_value -> past_key_values.
        cache_kw = (
            "past_key_values"
            if "past_key_values" in inspect.signature(self.decoder.forward).parameters
            else "past_key_value"
        )
        result = self.decoder.forward(
            x,
            position_embeddings_global=position_embeddings_global,
            position_embeddings_local=position_embeddings_local,
            use_cache=True,
            position_ids=position_ids,
            attention_mask=mask,
            **{cache_kw: self.past_key_values},
        )
        # transformers 5.x decoder layers return the hidden-states tensor directly instead of a
        # tuple; only unwrap [0] when it's actually a tuple (otherwise result[0] drops a leading dim).
        output = result[0] if isinstance(result, tuple) else result
        return output

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def load_state_dict(self, state_dict):
        return self.decoder.load_state_dict(convert_meta_to_hf(state_dict, self.head_dim))

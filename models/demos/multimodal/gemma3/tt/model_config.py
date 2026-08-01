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
                kwargs["dtype"] = ttnn.bfloat8_b
        return stock_linear(input_tensor_a, input_tensor_b, *args, **kwargs)

    _linear._gemma3_ff13_dtype_seam = True
    _linear._stock_linear = stock_linear
    ttnn.linear = _linear
    logger.info("gemma3: FF1/FF3 output dtype seam installed (bf16 -> bfloat8_b)")


_install_ff13_out_dtype_seam()

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

        self.use_qk_fused = False  # For Gemma 3, we do not use qk fused ops (rotary embedding + paged cache update)
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
        self._set_tensor_dtype({TensorGroup.FF2: PrecisionSetting.BFP4})

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

        Widening the COLUMNS was tried first and is not available: it necessarily changes
        ``per_core_N`` off ``dram_shard_grid_width``, and stock's comment that this "silently gives
        bad PCC" is load-bearing on P150 — measured PCC 0.31 at grid_x=10 (per_core_N 60 -> 48).
        ``per_core_N`` is therefore pinned to the weight's 8-bank DRAM shard width, which pins the
        column count too, so 32 active cores is a hard floor here unless the weight is re-sharded.

        What IS reachable is the wasted rows. Dropping the grid to the M-tiles that actually exist
        keeps every core busy AND buys a bigger per-core K block, because ``matmul_config`` derives
        ``in0_block_w = find_largest_divisor(k_tiles // grid_rows)``: 120//8 = 15 -> block 5, versus
        120//4 = 30 -> block 6. That is the same trade this run already banked on the decode
        matmuls (fewer cores, bigger K block) applied to the prefill grid.

        Returns None when it cannot improve on stock, so the caller falls through to it.
        """
        m = min(seq_len, self.prefill_len_cutoff)
        k = self.dim // self.cluster_shape[0]
        n = self.hidden_dim // self.cluster_shape[1]
        m_tiles, k_tiles, n_tiles = m // ttnn.TILE_SIZE, k // ttnn.TILE_SIZE, n // ttnn.TILE_SIZE
        if m_tiles >= self.prefill_rows:
            return None  # stock already fills the rows; leave that path alone

        # COLUMNS: pinned to stock's, so per_core_N keeps matching the weight's DRAM shard width.
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

    @lru_cache(maxsize=None)
    def get_mlp_ff1_3_prg_config(self, mode: Mode, seq_len: int = 1, prefetcher: Prefetcher = None):
        if mode == Mode.PREFILL and prefetcher is None and not self.is_galaxy:
            pc = self._short_prefill_ff1_3_prg_config(seq_len)
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
        for cores in sorted(
            (c for c in range(1, base_cores) if K % c == 0 and N % c == 0 and k_block(c) > k_block(base_cores)),
            reverse=True,
        ):
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

    def get_attn_qkv_program_config(self, mode: Mode, seq_len: int = 1, prefetcher: Prefetcher = None):
        """Smaller MinimalMatmul blocks for traced long prefill (default 8³ overflows static CB vs L1)."""
        if self._enable_program_trace and mode == Mode.PREFILL and seq_len > 128:
            return ttnn.MinimalMatmulConfig(
                M_block_size=4,
                K_block_size=4,
                N_block_size=4,
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 10) if is_blackhole() else ttnn.CoreCoord(8, 8),
            )
        return super().get_attn_qkv_program_config(mode, seq_len, prefetcher)

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

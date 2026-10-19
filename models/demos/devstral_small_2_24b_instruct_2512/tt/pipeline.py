# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The ONE chained TTNN pipeline for mistralai/Devstral-Small-2-24B-Instruct-2512 (Mistral3ForConditionalGeneration).

Both task heads run here, and both the demos (demo/) and the e2e tests (tests/e2e/) call these functions:

  text_generation       text -> text          run_text_generation(pipe, request)
  image_text_to_text    image + text -> text  run_image_text_to_text(pipe, request)

The forward is an explicit chain over the graduated bring-up stubs (models/tt_transformers/demo/
devstral_small_2_24b_instruct_2512/_stubs, all 22 of them):

  encode   pixel_values -> pixtral_vision_model [patchify+linear, pixtral_r_m_s_norm (ln_pre),
           pixtral_rotary_embedding, pixtral_transformer == encoder_stack over the 24 vision layers, which rotate
           pixtral_attention_layer / layer / attention, each = pixtral_r_m_s_norm + pixtral_attention +
           pixtral_r_m_s_norm + pixtral_m_l_p | mlp]
           -> mistral3_multi_modal_projector [mistral3_r_m_s_norm, mistral3_patch_merger, linear/gelu/linear]
  prefill  token_embed(prompt) (+ image features gathered into the [IMG] rows on device)
           -> ministral3_model [ministral3_rotary_embedding, 40 x ministral3_decoder_layer (ministral3_r_m_s_norm,
           ministral3_attention writing the resident KV cache, ministral3_r_m_s_norm, ministral3_m_l_p),
           ministral3_r_m_s_norm] -> decoder_head (lm_head) at the last prompt row -> on-device sampling
  decode   one token per batch row: token_embed -> ministral3_model (decode mode, KV append + SDPA-decode)
           -> decoder_head -> on-device sampling, the sampled id written back into the resident input buffer

Sampling is the checkpoint's own generation_config (temperature 0.15, eos 2, pad 11) made reproducible with
the Gumbel-max reparameterisation: token = argmax over the top-K logits of logits / T + g, with g a per-row
seeded Gumbel table resident on device. tests/e2e/hf_reference.py applies the identical rule inside HF
model.generate(), so both sides draw the same sample and stop at the same EOS.

Mesh: TP = 4 (1x4). The stubs shard their own weights (ShardTensorToMesh) and all_reduce / all_gather
internally; embeddings, norms and rotary tables are replicated. Batch B (default 32, $TT_PERF_BATCH) is a
separate leading axis -- never sharded.
"""
from __future__ import annotations

import functools
import json
import math
import os
import time
from pathlib import Path

import torch

import ttnn
from models.demos.devstral_small_2_24b_instruct_2512.tt.last_row import LastRow
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs import attention as _st_attention
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs import decoder_head as _st_decoder_head
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs import encoder_stack as _st_encoder_stack
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs import layer as _st_layer
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs import (
    ministral3_attention as _st_ministral3_attention,
)
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs import (
    ministral3_decoder_layer as _st_ministral3_decoder_layer,
)
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs import ministral3_m_l_p as _st_ministral3_mlp
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs import (
    ministral3_model as _st_ministral3_model,
)
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs import (
    ministral3_r_m_s_norm as _st_ministral3_norm,
)
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs import (
    ministral3_rotary_embedding as _st_ministral3_rope,
)
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs import (
    mistral3_multi_modal_projector as _st_projector,
)
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs import (
    mistral3_patch_merger as _st_patch_merger,
)
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs import (
    mistral3_r_m_s_norm as _st_mistral3_norm,
)
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs import mlp as _st_mlp
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs import (
    pixtral_attention as _st_pixtral_attention,
)
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs import (
    pixtral_attention_layer as _st_pixtral_attention_layer,
)
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs import pixtral_m_l_p as _st_pixtral_mlp
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs import pixtral_r_m_s_norm as _st_pixtral_norm
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs import (
    pixtral_rotary_embedding as _st_pixtral_rope,
)
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs import (
    pixtral_transformer as _st_pixtral_transformer,
)
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs import (
    pixtral_vision_model as _st_vision_model,
)
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs import token_embed as _st_token_embed

MODEL_ID = "mistralai/Devstral-Small-2-24B-Instruct-2512"
PIPELINE_STAGES = ["encode", "prefill", "decode"]
TASKS = ("text_generation", "image_text_to_text")

PACKAGE_DIR = Path(__file__).resolve().parents[1]

BATCH_ENV = "TT_PERF_BATCH"
LAYERS_ENV = "TT_PERF_LAYERS"
OSL_ENV = "TT_PERF_OSL_TOKENS"
DEFAULT_BATCH = 32
# Resident KV positions per row. The checkpoint's max_length (262144) cannot be resident: at B=32 the
# cache alone would need ~340 GB/chip. 4096 positions cost 40 layers x 2 x 32 x 2 local kv heads x 4096 x
# 128 x 2 B = 5.4 GB/chip, inside the 28.6 GB/chip usable with a CCL axis (QB2 registered figure).
DEFAULT_KV_CAPACITY = 4096
# Resident KV cache dtype: decode attention re-reads every cached position of every layer each token, so block-float8
# halves that traffic (and the cache's DRAM).
KV_CACHE_DTYPE = ttnn.bfloat8_b
PLANNED_TP = 4
# Performance levers applied inside _stubs/ (a symlink out of this directory, so they are listed here too):
#   ministral3_m_l_p: gate/up folded into one tile-interleaved weight per device; minimal_matmul(fuse_swiglu)
#                     emits silu(gate) * up in the same op (prefill/encode); decode keeps separate gate/up
#                     with the silu fused into the multiply and gate/up/product kept in L1 (no DRAM round trip);
#                     decode gate / up / down read block-float8 copies (prefill keeps bf16); the decode down
#                     projection runs a 1D multicast config with 8-tile K blocks (the auto config's 2-tile blocks
#                     made it multicast-latency-bound); layers 5-35 read block-float4 decode gate / up copies with
#                     8-tile K blocks at HiFi3 (exact on a bf4 weight; bf4 on every layer: PCC 0.9921 < 0.995). The post-attention decode norm
#                     unshards its bf16 output into L1, so gate / up fetch each K block of the row from L1, not DRAM.
#                     Decode gate / up / silu-multiply run as ONE C++ program (tt/gate_up_decode.py: the activation
#                     multicast once by a spare core, gate and up streamed by the two data-movement RISCs, fp32
#                     silu * up, one rounding). The decode down projection is ONE C++ program too
#                     (tt/down_decode.py: the product multicast once, a bank-major weight copy streamed one column per
#                     data-movement RISC on 8 x 10 cores, fp32 DEST over all K -- bit-identical to the 1D-multicast
#                     matmul); layers 16-24 read a block-float4 down copy at HiFi3 (bf4 down everywhere: PCC 0.989).
#                     Prefill MLP chunk 4096 -> 16384 rows (one call).
#                     Prefill/encode down projection on minimal_matmul too (full grid); the text prefill down
#                     computes at HiFi2 (fp32 accumulation) on a bfloat8_b weight.
#                     Fused gate/up computes at HiFi2 (LoFi on layers 16-24) (fp32 accumulation), 8 x 8 x 12-tile blocks; the text
#                     decoder's fused gate/up weight is bfloat8_b (the vision MLPs keep bf16).
#   ministral3_model decode embedding: ONE program (tt/embed_decode.py) gathers each core's column slice of the token
#                     rows, tilizes them to fp32 and writes the first layer's width-sharded residual directly (no
#                     one-core tiled ttnn.embedding, typecast or reshard).
#   decoder_head: keeps its pre-gather vocab shard (.local) for the per-device sampler top-k (DevstralPipeline._topk);
#                     lm_head weight in bfloat8_b (half the largest decode-step DRAM read). Decode skips the
#                     full-row logits all_gather (the sampler reads the shards; VocabShards joins them on host reads).
#   sample(): the candidates' noise is ONE data-movement program (NoiseLookup, kernels/noise_lookup.cpp) reading the
#                     step's row of the resident table directly (no embedding / tilize / typecast / bitwise and / gather);
#                     the draw is one top-1 over the noisy scores labelled with the candidate ids (no argmax + index gather).
#                     The per-device vocab-shard top-k is VocabTopK (generic_op, kernels/topk_mid.cpp): 64 local ->
#                     8 middle -> 1 final core instead of ttnn.topk's 64 -> 1 serial merge, 16-bit shard positions.
#                     The draw's pad select, eos update and next-token row are ONE data-movement program
#                     (SampleTail, kernels/sample_tail.cpp: integer select on the id / mask tiles; no fp32 chain,
#                     untilize or reshape); with 32 rows of 32 candidates the top-1 itself joins it (DrawTail,
#                     kernels/draw_tail.cpp: integer-keyed argmax per row; no 64-column pads, no ttnn.topk).
#   ministral3_r_m_s_norm.decode: decode norms run width-sharded over 40 cores and read the fp32 residual
#                     directly (no bf16 cast op, no single-core interleaved norm); the decode QKV reads the input
#                     norm's fp32 width shards as they are (20-tile K blocks of 5 shards; no unshard, cast, reshard;
#                     3 readers per DRAM bank);
#                     decode residual adds write straight into the norms' sharded layout (no re-shard per norm).
#                     The decode all_gathers of the reduced o_proj / down shards write into that layout too.
#                     The final decode norm's gamma multiplies its fp32 width shards as a full 32-row tile row (no row
#                     broadcast, no fp32 unshard through DRAM) into the DRAM bf16 row the LM head reads.
#   ministral3_attention.query_rope_tables: the llama-4 query scale is folded into q's cos/sin ONCE per forward
#                     (ministral3_model) and shared by all layers, not re-derived and multiplied in each layer.
#   KV cache in bfloat8_b (KV_CACHE_DTYPE): half the bytes decode SDPA re-reads every token. Prefill writes only the
#                     cast prompt rows: one fill_cache over (batch x kv_heads) folded into heads, the prompt
#                     height-sharded one (user, head) slab per core (no pad to the capacity, no full-capacity typecast).
#                     Decode SDPA reads q straight from its post-RoPE head shards (no copy to DRAM): its grid is the
#                     8-wide head grid, so each batch row's output core is the core holding that row's q. Its output
#                     goes L1-interleaved (GQA allows no sharded output) for the reshard into the concat's head shards;
#                     it computes at HiFi2 (fp32 accumulation) on the bf8 cache, in fixed 4-tile key chunks.
#   ministral3_attention: prefill QKV (HiFi2, bf8 weight) and o_proj projections on minimal_matmul (full grid); prefill o_proj is
#                     column-parallel over the all_gathered heads, emitting the residual's hidden slice (no reduce-scatter). Decode concatenates the SDPA heads
#                     with ONE C++ copy program (tt/concat_heads.py ConcatHeadsDecode: each core gathers one o_proj input
#                     tile's rows out of SDPA's L1-interleaved tiles; no reshard, nlp_concat_heads_decode, unshard).
#                     nlp_create_qkv_heads_decode reads the decode QKV's width shards in place (no unshard) and puts k
#                     on its own cores (overlap_qk_coregrid=False), so q / k RoPE is ONE rotary_embedding_llama_fused_qk
#                     (64-row cos / sin tables, q's rows then k's) and K / V are ONE paged_fused_update_cache. Those
#                     tables are gathered each step by ONE program (tt/rope_gather.py) from cos / sin tables of every
#                     position below the KV capacity, built once by the same device ops on one dense position column
#                     (32 positions a tile; ~0.1 ms at load instead of 2.65 ms) (no per-step cos / sin math,
#                     concat or reshard; the fp32 position column is then no longer advanced). Decode QKV
#                     is a DRAM-sharded matmul (3 readers per bank, HiFi2) over its own bank-sharded copy of [q | k | v]; decode
#                     QKV and o_proj read block-float8 copies (prefill keeps bf16). Decode o_proj is the MLP down's
#                     C++ streaming program (tt/down_decode.py: bank-major bf8 copy, 8 x 10 cores, ~9 KB pushes,
#                     bit-identical to the 1D-multicast matmul), its bf16 partials in L1 for the reduce-scatter.
#   prompts pad to a tile multiple only; prefill SDPA pads to whole chunks internally. Prefill QKV and o_proj run HiFi2 on bf8 weights.
#   ministral3_m_l_p short-prompt down projection: a grid of whole 16 x 16-tile output blocks per core.
#   prefill's last prompt rows: ONE data-movement program (tt/last_row.py) reads every user's row straight out of its
#                     tiles into the tiled LM-head input (no block slice, untilize or embedding gather).
#   ministral3_m_l_p fused gate/up N block: the per-core N range when it is under 12 tiles; M blocks no taller than the
#                     per-core M range (data-parallel vision MLP: 7 tiles); the
#                     vision MLP's SwiGLU product reaches the down projection through L1, which runs 4-tile N blocks.
#   ministral3_attention short-prompt QKV and o_proj: one output block per core (its whole M x N range, e.g. 12 x 5 tiles).
#   ministral3_attention prefill SDPA: one key chunk of the prompt (even tiles), 160-row q chunks (7 even rounds), no pad,
#                     HiFi2 matmuls.
#   ministral3_attention prefill RoPE: runs on a (batch, heads) regrouping of the heads (an outer-dim view) that balances
#                     rotary_embedding_llama's batch x sequence split over the grid (416 rows: 8 x 32, 128 rows: 256 x 1);
#                     cos / sin / the pair-swap tile are height-sharded into L1 around each op (one sequence tile per
#                     core, laid out once per forward by ministral3_attention.prefill_rope_tables), so the op stops
#                     re-reading them from DRAM per row (when that moves fewer bytes: 416 / 128 rows, q and k).
#                     nlp_create_qkv_heads splits q / k / v into L1 (<= 512 KB per core); the RoPEs read q / k from L1.
#                     The fused QKV output (<= 400 KB per core) also lands in L1 for that split.
#                     The prefill RoPE computes at HiFi2. The concatenated SDPA heads reach the all_gather through L1;
#                     short prompts hand the o_proj output to the residual add and SDPA's output to the concat
#                     through L1 (416 rows too).
#   mistral3_patch_merger: the merging layer runs on minimal_matmul over the full grid (images folded into rows).
#   pixtral_attention: the vision QKV (output to L1 for the head split) and o_proj run on minimal_matmul over the full
#                     grid too, with blocks of each core's M / N range (data-parallel QKV 7 x 9, o_proj 6 x 4 tiles); so
#                     do the projector linear_1
#                     (with its bias; 6 x 4-tile blocks) and linear_2. The RoPE rotate-half is a height-parallel 1D
#                     matmul (every core a run of row tiles).
#   token_embed: a hidden-sharded copy of the table; the text prefill gathers each device's residual slice from it
#                     (no full-width rows, no mesh_partition). The slice stays bf16 into the first layer, whose input
#                     norm gathers per-token sums of squares from a (vocab, 32) table (exact bf16 hi / mid / lo parts)
#                     and whose fused residual add (tt/add_stats.py) widens it to fp32: no cast, no reduce of it.
#   vision tower + projector: data-parallel over the mesh (full weights per device, each device its own images, one
#                     batch all_gather of the projector output) -- no per-layer all_reduces.
#   pixtral_vision_model: equal-size images are uploaded already patchified on the host (patchify_host; tile layout),
#                     so the device skips the reshape / permute chain over 14-pixel row-major sticks; the patch
#                     projection runs on minimal_matmul (one 19-tile K block).
#   pixtral_m_l_p: the vision fused gate/up weight is block-float8; it runs at LoFi on the L1 ffn norm output cast
#                  to bf16 in L1 (full 8-tile K blocks). The down projection runs at HiFi2.
#   attention (vision layer): each RMSNorm gamma is folded into the q/k/v and gate/up weights that read it (no separate
#                     fp32 gamma multiply); the QKV and the gate/up read their norm outputs from L1 after an L1 cast
#                     to bf16 (the gate/up's product fits in L1 too).
#   ministral3_decoder_layer prefill: the residual stays hidden-sharded over the TP mesh: the fp32 adds and the norms
#                     run on the reduce-scatter shard (sum-of-squares all_gather) and the bf16 norm output is all_gathered.
#   ministral3_decoder_layer prefill residual adds: C++ add + next-norm sum of squares in one pass (tt/add_stats.py);
#                     the last layer's final add emits the bf16 residual for the gather directly (no fp32 copy + cast).
#   text decoder all_reduces (o_proj, MLP down): reduce-scatter + bf16 all_gather of the reduced shard; prefill's
#                     and decode's partials are produced and reduce-scattered in bf16 (half the bytes, no cast),
#                     Ring over 2 links on QB2's ring fabric (mesh.py opens FABRIC_1D_RING there).
#   ministral3_decoder_layer: each RMSNorm's gamma is folded into the q/k/v and gate/up weights that read it
#                     (no gamma multiply after the norm; the decode norm unshards + casts to bf16 in one op).
#   ministral3_attention meta_rope: q/k weight columns, the rotary tables and the prefill rotation use interleaved-pair
#                     head_dim order, so decode's q and k RoPE each run as ONE fused rotary_embedding_llama on the
#                     height-sharded heads (no rotate-half matmul / mul / add chain, no unshard + reshard of k); q
#                     shares k's tables while the KV capacity is below the llama-4 scaling threshold (scale == 1).
#                     Prefill's q / k RoPE are one fused rotary_embedding_llama each too (interleaved, all users at once).


def kv_capacity_for(tp):
    """Resident KV positions per row on a TP-way mesh: the per-chip KV budget stays the planned 5.4 GB, so a
    narrower mesh (fewer chips reachable, e.g. 1x2 -> 4 local kv heads per chip) holds proportionally fewer
    positions. TP=2: 2048 positions, beside ~25 GB/chip of resident bf16 weights."""
    return max(512, DEFAULT_KV_CAPACITY * min(int(tp), PLANNED_TP) // PLANNED_TP)


# Gumbel-max over the top-K logits. At the checkpoint's temperature (0.15) the mass outside the top 32
# logits is negligible; both TT and the HF golden apply the same truncation.
SAMPLING_TOP_K = 32
# The Gumbel noise is indexed by TOKEN ID (bucket = id mod NOISE_BUCKETS), never by rank, so two near-equal
# logits that the two sides order differently still receive the same noise each.
NOISE_BUCKETS = 1024
TRACE_REGION_SIZE = 300_000_000
# Prompts are right-padded only to a tile multiple: prefill attention pads its own q/k/v to whole 512-wide SDPA key
# chunks (narrower key chunks bias attention's output level, see ministral3_attention.sdpa_len), so no other layer
# computes pad rows. The padded rows come after every real row, so causal attention leaves rows 0..L-1 unchanged.

_TOPK_KERNELS = "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels"
_TOPK_MID_KERNEL = str(Path(__file__).resolve().parent / "kernels" / "topk_mid.cpp")
_NOISE_LOOKUP_KERNEL = str(Path(__file__).resolve().parent / "kernels" / "noise_lookup.cpp")


class NoiseLookup:
    """The sampler's per-candidate Gumbel noise as ONE data-movement program (generic_op, kernels/noise_lookup.cpp):
    out[b, j] = noise_table[step, b * NB + ids[b, j] % NB] as a (1, 1, 32, 32) fp32 tile.

    Replaces the step-row embedding, its reshape / tilize / typecast to fp32, the bucket bitwise_and and the
    one-core ttnn.gather (~70 us of six ops): each RISC of CORES cores reads the step counter, then its ROWS rows
    of that step's bf16 noise row straight from the resident table, and looks its rows' candidates up with integer
    ops (bf16 -> fp32 is a 16-bit shift, so the values are bit-identical to the typecast + gather path)."""

    CORES = 4
    ROWS = 4  # rows per RISC: 32 rows = CORES x 2 RISCs x ROWS

    def __init__(self, device, batch, k, buckets):
        assert batch == 32 and k == 32 and buckets & (buckets - 1) == 0, (batch, k, buckets)
        assert self.CORES * 2 * self.ROWS == 32 and 16 % self.ROWS == 0
        self.device, self.buckets = device, buckets
        self.core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.CORES - 1, 0))])

    def _cb(self, index, size):
        fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.uint32, page_size=size)
        return ttnn.CBDescriptor(total_size=size, core_ranges=self.core, format_descriptors=[fmt])

    def __call__(self, step, table, ids):
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape((1, 1, 32, 32)), ttnn.float32, ttnn.TILE_LAYOUT, self.device, ttnn.DRAM_MEMORY_CONFIG
        )
        io = [step, table, ids, out]
        ct = [self.buckets, self.ROWS]
        for t in io:
            ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        addrs = [t.buffer_address() for t in io]
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=_NOISE_LOOKUP_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.core,
                compile_time_args=ct,
                runtime_args=[((c, 0), addrs + [(2 * c + h) * self.ROWS, h]) for c in range(self.CORES)],
                config=cfg,
            )
            for h, cfg in ((0, ttnn.ReaderConfigDescriptor()), (1, ttnn.WriterConfigDescriptor()))
        ]
        # per RISC: 64 B step scratch, ROWS noise rows, ROWS rows of ids and of the output (two face runs each)
        run = self.ROWS * 16 * 4
        cbs = [
            self._cb(0, 128),
            self._cb(1, 2 * self.ROWS * self.buckets * 2),
            self._cb(2, 2 * 2 * run),
            self._cb(3, 2 * 2 * run),
        ]
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
        return out


_SAMPLE_TAIL_KERNEL = str(Path(__file__).resolve().parent / "kernels" / "sample_tail.cpp")


class SampleTail:
    """The sampler's end as ONE data-movement program (generic_op, kernels/sample_tail.cpp) on one RISC: from the
    (1, 1, B, 1) uint32 draw and the fp32 0 / 1 unfinished mask, nxt = unfinished ? drawn : pad, the mask drops rows
    that emit eos (in place), and nxt lands in the (1, B) row-major uint32 next-token buffer (in place).

    Replaces the fp32 select chain (typecast, subtract, multiply, add), the eos update (ne, multiply, copy) and the
    id row's typecast / untilize / reshape / copy (~43 us of 11 ops): the ids are exact integers below 2^24, so the
    integer select gives the same values the fp32 chain did."""

    CB = 0

    def __init__(self, device, batch, pad_token_id, eos_token_id):
        assert 1 <= batch <= 32, batch
        self.device, self.batch = device, batch
        self.pad, self.eos = int(pad_token_id), int(eos_token_id)
        self.core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    def __call__(self, drawn, unfinished, tok):
        io = [drawn, unfinished, tok]
        ct = [self.batch, self.pad, self.eos, self.CB]
        for t in io:
            ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        size = 4 * 1024 + 128  # two faces of each tile and the id row
        fmt = ttnn.CBFormatDescriptor(buffer_index=self.CB, data_format=ttnn.uint32, page_size=size)
        cbs = [ttnn.CBDescriptor(total_size=size, core_ranges=self.core, format_descriptors=[fmt])]
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=_SAMPLE_TAIL_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.core,
                compile_time_args=ct,
                runtime_args=[((0, 0), [t.buffer_address() for t in io])],
                config=ttnn.ReaderConfigDescriptor(),
            )
        ]
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))


_DRAW_TAIL_KERNEL = str(Path(__file__).resolve().parent / "kernels" / "draw_tail.cpp")


class DrawTail:
    """The sampler's draw and end as ONE data-movement program (generic_op, kernels/draw_tail.cpp), one RISC per
    16 rows: the top-1 of the (1, 1, 32, K = 32) fp32 noisy scores labelled with its uint32 candidate id (what
    ttnn.topk(k=1, indices_tensor=ids) draws, after padding both tiles to 64 columns), then SampleTail's pad select,
    eos update and next-token row. Replaces the two pads, the top-1 and SampleTail (~16 us of 4 ops). Returns the
    (1, 1, B, 1) uint32 draw."""

    CB = 0

    def __init__(self, device, batch, pad_token_id, eos_token_id):
        assert batch == ttnn.TILE_SIZE, batch
        self.device, self.batch = device, batch
        self.pad, self.eos = int(pad_token_id), int(eos_token_id)
        self.core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    @staticmethod
    def supports(scores, ids):
        tile = (1, 1, ttnn.TILE_SIZE, ttnn.TILE_SIZE)
        return (
            tuple(int(d) for d in scores.shape) == tile
            and tuple(int(d) for d in ids.shape) == tile
            and scores.dtype == ttnn.float32
            and ids.dtype == ttnn.uint32
            and scores.layout == ttnn.TILE_LAYOUT
            and ids.layout == ttnn.TILE_LAYOUT
            and not scores.is_sharded()
            and not ids.is_sharded()
        )

    def __call__(self, scores, ids, unfinished, tok):
        drawn = ttnn.allocate_tensor_on_device(
            ttnn.Shape((1, 1, self.batch, 1)), ttnn.uint32, ttnn.TILE_LAYOUT, self.device, ttnn.DRAM_MEMORY_CONFIG
        )
        io = [scores, ids, unfinished, tok, drawn]
        acc = []
        for t in io:
            acc += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        size = 12 * 1024 + 128  # per RISC: scores / ids face pairs, the unfinished and drawn faces; the id row
        fmt = ttnn.CBFormatDescriptor(buffer_index=self.CB, data_format=ttnn.uint32, page_size=size)
        cbs = [ttnn.CBDescriptor(total_size=size, core_ranges=self.core, format_descriptors=[fmt])]
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=_DRAW_TAIL_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.core,
                compile_time_args=[half, self.pad, self.eos, self.CB] + acc,
                runtime_args=[((0, 0), [t.buffer_address() for t in io])],
                config=cfg,
            )
            for half, cfg in ((0, ttnn.ReaderConfigDescriptor()), (1, ttnn.WriterConfigDescriptor()))
        ]
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
        return drawn


class VocabTopK:
    """Top-k (largest, sorted) of each row of a (1, 1, B, W) bf16 vocab shard as ONE three-level program (generic_op).

    ttnn.topk's multi-core program sorts W/64 columns on each of 64 local cores and sends all 64 candidate tiles to
    a single final core, which merges them serially (63 tile merges: ~100 of its 156 us at 32 x 32768). Here the
    64 local cores (columns 0..7 x rows 0..7, the stock local kernels) report in groups of 8 to 8 middle cores
    (row 8, kernels/topk_mid.cpp), and the final core (0, 9) merges only the middle cores' 8 results (stock final
    kernels): 7 + 7 serial merges. Each level alternates its cores' output order, as the stock op does across its
    local cores, so every merge sees bitonic pairs.

    Ids are the 16-bit column positions within the shard (W <= 65536): the sort runs in 16-bit DEST and no id
    tensor is streamed in. The caller adds the shard's vocab offset (after the cross-device gather)."""

    GROUPS = 8  # middle cores
    LOCALS = 8  # local cores per middle core
    VAL, IDX = ttnn.bfloat16, ttnn.uint16
    TILE_BYTES = 2048  # bf16 and uint16 tiles

    def __init__(self, device, k):
        assert k == 32, "the stock kernels' K=32 layout (one candidate tile per core) is assumed"
        grid = device.compute_with_storage_grid_size()
        assert grid.x >= self.GROUPS and grid.y >= self.LOCALS + 2, f"grid {grid} too small for VocabTopK"
        self.device, self.k, self.kt = device, k, 1
        rng = lambda x0, y0, x1, y1: ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))
        g, n = self.GROUPS, self.LOCALS
        self.local_set = ttnn.CoreRangeSet([rng(0, 0, g - 1, n - 1)])
        self.mid_set = ttnn.CoreRangeSet([rng(0, n, g - 1, n)])
        self.final_core = (0, n + 1)
        self.final_set = ttnn.CoreRangeSet([rng(0, n + 1, 0, n + 1)])
        self.mid_final_set = self.mid_set.merge(self.final_set)
        self.all_set = self.local_set.merge(self.mid_final_set)
        self.phys = lambda x, y: device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))

    def _cb(self, index, tiles, dtype, cores):
        return ttnn.CBDescriptor(
            total_size=tiles * self.TILE_BYTES,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=self.TILE_BYTES)
            ],
        )

    def __call__(self, x):
        g, n, k, kt = self.GROUPS, self.LOCALS, self.k, self.kt
        rows, width = int(x.shape[-2]), int(x.shape[-1])
        ht, wt = rows // 32, width // 32
        assert rows % 32 == 0 and width <= 65536 and wt % (g * n) == 0, f"VocabTopK shape {tuple(x.shape)}"
        wt_local = wt // (g * n)
        assert wt_local >= 2 and wt_local & (wt_local - 1) == 0
        out_shape = ttnn.Shape((1, 1, rows, k))
        vals = ttnn.allocate_tensor_on_device(
            out_shape, self.VAL, ttnn.TILE_LAYOUT, self.device, ttnn.DRAM_MEMORY_CONFIG
        )
        ids = ttnn.allocate_tensor_on_device(
            out_shape, self.IDX, ttnn.TILE_LAYOUT, self.device, ttnn.DRAM_MEMORY_CONFIG
        )

        # CB ids as in the stock factory. The gather buffers (c_4/c_5) are written remotely at the sender's own
        # address of them, so they go first on every core (same L1 address everywhere). A middle core reads that
        # address after its own gather wrapped the buffer, so the buffer holds exactly one gather (g == n).
        assert g == n
        IN, IN_IDX, LT_VAL, LT_IDX, G_VAL, G_IDX, M_VAL, M_IDX, OUT_VAL, OUT_IDX = range(10)
        cbs = [
            self._cb(G_VAL, n * kt, self.VAL, self.all_set),
            self._cb(G_IDX, n * kt, self.IDX, self.all_set),
            self._cb(OUT_VAL, 2, self.VAL, self.all_set),
            self._cb(OUT_IDX, 2, self.IDX, self.all_set),
            self._cb(IN, 4, self.VAL, self.local_set),
            self._cb(IN_IDX, 4, self.IDX, self.local_set),
            self._cb(LT_VAL, wt_local, self.VAL, self.local_set),
            self._cb(LT_IDX, wt_local, self.IDX, self.local_set),
            self._cb(M_VAL, g * kt, self.VAL, self.mid_final_set),
            self._cb(M_IDX, g * kt, self.IDX, self.mid_final_set),
        ]
        # Semaphores 0/1: local -> middle hand-off; 2/3: middle -> final (a middle core runs both protocols).
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=self.all_set, initial_value=0) for i in range(4)]

        compute_cfg = ttnn.ComputeConfigDescriptor()
        compute_cfg.fp32_dest_acc_en = False
        logk = int(math.log2(k))
        sort_ct = lambda width_tiles, a, b, c, d: [a, b, c, d, OUT_VAL, OUT_IDX, ht, width_tiles, k, kt, logk]
        kern = lambda src, cores, ct, rt, cfg, **kw: ttnn.KernelDescriptor(
            kernel_source=src,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ct,
            runtime_args=rt,
            config=cfg,
            **kw,
        )
        locals_ = [(gx, ly) for gx in range(g) for ly in range(n)]  # global local index = gx * n + ly
        kernels = [
            kern(
                f"{_TOPK_KERNELS}/dataflow/reader_create_index_local_topk.cpp",
                self.local_set,
                [IN, IN_IDX, ht, wt_local, wt] + list(ttnn.TensorAccessorArgs(x).get_compile_time_args()),
                [((cx, cy), [x.buffer_address(), 0, i * wt_local, 0]) for i, (cx, cy) in enumerate(locals_)],
                ttnn.ReaderConfigDescriptor(),
                defines=[("GENERATE_INDICES", "1")],
            ),
            kern(
                f"{_TOPK_KERNELS}/compute/topk_local.cpp",
                self.local_set,
                sort_ct(wt_local, IN, IN_IDX, LT_VAL, LT_IDX) + [int(math.log2(wt_local)), 1, 1, 0, 0],
                [((cx, cy), [i % 2]) for i, (cx, cy) in enumerate(locals_)],
                compute_cfg,
            ),
            kern(
                _TOPK_MID_KERNEL,
                self.mid_set,
                sort_ct(n * kt, G_VAL, G_IDX, M_VAL, M_IDX) + [int(math.log2(n * kt)), 1],
                [((gx, n), [gx % 2]) for gx in range(g)],
                compute_cfg,
                compiler_include_paths=[f"{_TOPK_KERNELS}/compute"],
            ),
            kern(
                f"{_TOPK_KERNELS}/dataflow/writer_local_topk.cpp",
                self.mid_set,
                [2, 3, self.phys(*self.final_core).x, self.phys(*self.final_core).y, ht, k, kt]
                + [OUT_VAL, OUT_IDX, G_VAL, G_IDX],
                [((gx, n), [gx]) for gx in range(g)],
                ttnn.WriterConfigDescriptor(),
            ),
            kern(
                f"{_TOPK_KERNELS}/dataflow/reader_final_topk.cpp",
                self.final_set,
                [2, 3, self.phys(0, n).x, self.phys(0, n).y, self.phys(g - 1, n).x, self.phys(g - 1, n).y]
                + [ht, g * kt, g, G_VAL, G_IDX],
                [],
                ttnn.ReaderConfigDescriptor(),
            ),
            kern(
                f"{_TOPK_KERNELS}/compute/topk_final.cpp",
                self.final_set,
                sort_ct(g * kt, G_VAL, G_IDX, M_VAL, M_IDX) + [int(math.log2(g * kt)), 1, 1, 0, 0],
                [],
                compute_cfg,
            ),
            kern(
                f"{_TOPK_KERNELS}/dataflow/writer_final_topk.cpp",
                self.final_set,
                [OUT_VAL, OUT_IDX, ht, kt]
                + list(ttnn.TensorAccessorArgs(vals).get_compile_time_args())
                + list(ttnn.TensorAccessorArgs(ids).get_compile_time_args()),
                [(self.final_core, [vals.buffer_address(), ids.buffer_address()])],
                ttnn.WriterConfigDescriptor(),
            ),
        ]
        for gx in range(g):
            column = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(gx, 0), ttnn.CoreCoord(gx, n - 1))])
            mid, top, bottom = self.phys(gx, n), self.phys(gx, 0), self.phys(gx, n - 1)
            # This group's local cores send their candidates to middle core (gx, n) ...
            kernels.append(
                kern(
                    f"{_TOPK_KERNELS}/dataflow/writer_local_topk.cpp",
                    column,
                    [0, 1, mid.x, mid.y, ht, k, kt, OUT_VAL, OUT_IDX, G_VAL, G_IDX],
                    [((gx, ly), [ly]) for ly in range(n)],
                    ttnn.WriterConfigDescriptor(),
                )
            )
            # ... which releases them (multicast over the column) and collects n * kt tiles.
            kernels.append(
                kern(
                    f"{_TOPK_KERNELS}/dataflow/reader_final_topk.cpp",
                    ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(gx, n), ttnn.CoreCoord(gx, n))]),
                    [0, 1, top.x, top.y, bottom.x, bottom.y, ht, n * kt, n, G_VAL, G_IDX],
                    [],
                    ttnn.ReaderConfigDescriptor(),
                )
            )
        ttnn.generic_op([x, vals, ids], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems))
        return vals, ids


# Component name -> the stub class(es) that implement it. Every class here is invoked by the forward chain
# below; Gate 2 reads INVOKED (filled by the probe installed at build) against this table.
GRADUATED_STUBS = {
    "token_embed": _st_token_embed.TtTokenEmbed,
    "ministral3_rotary_embedding": _st_ministral3_rope.TtMinistral3RotaryEmbedding,
    "ministral3_model": _st_ministral3_model.TtMinistral3Model,
    "ministral3_decoder_layer": _st_ministral3_decoder_layer.TtMinistral3DecoderLayer,
    "ministral3_attention": _st_ministral3_attention.TtMinistral3Attention,
    "ministral3_m_l_p": _st_ministral3_mlp.TtMinistral3MLP,
    "ministral3_r_m_s_norm": _st_ministral3_norm.TtMinistral3RMSNorm,
    "decoder_head": _st_decoder_head.DecoderHead,
    "pixtral_vision_model": _st_vision_model.TtPixtralVisionModel,
    "pixtral_rotary_embedding": _st_pixtral_rope.TtPixtralRotaryEmbedding,
    "pixtral_r_m_s_norm": _st_pixtral_norm.TtPixtralRMSNorm,
    "pixtral_transformer": _st_pixtral_transformer.TtPixtralTransformer,
    "encoder_stack": _st_encoder_stack.TtEncoderStack,
    "pixtral_attention_layer": _st_pixtral_attention_layer.TtPixtralAttentionLayer,
    "layer": _st_layer.TtLayer,
    "attention": _st_attention.TtAttention,
    "pixtral_attention": _st_pixtral_attention.TtPixtralAttention,
    "pixtral_m_l_p": _st_pixtral_mlp.TtPixtralMLP,
    "mlp": _st_mlp.TtMLP,
    "mistral3_multi_modal_projector": _st_projector.TtMistral3MultiModalProjector,
    "mistral3_patch_merger": _st_patch_merger.TtMistral3PatchMerger,
    "mistral3_r_m_s_norm": _st_mistral3_norm.TtMistral3RMSNorm,
}
TEXT_STUBS = (
    "token_embed",
    "ministral3_rotary_embedding",
    "ministral3_model",
    "ministral3_decoder_layer",
    "ministral3_attention",
    "ministral3_m_l_p",
    "ministral3_r_m_s_norm",
    "decoder_head",
)
TASK_STUBS = {"text_generation": TEXT_STUBS, "image_text_to_text": tuple(GRADUATED_STUBS)}

# component -> number of forward calls that executed its stub code (its own class or a subclass of it).
INVOKED: dict = {}


def reset_invocations():
    INVOKED.clear()


def _install_invocation_probe():
    """Count, per graduated component, the calls that ran its stub's code.

    A call on an instance of class C counts for every graduated class in C's MRO: TtLayer runs attention.py's
    forward, TtPixtralTransformer runs encoder_stack.py's. Installed once; it only counts (no torch op), so it
    is invisible to the host-op observer and to trace capture."""
    owners = {cls: name for name, cls in GRADUATED_STUBS.items()}
    for cls in set(GRADUATED_STUBS.values()):
        for klass in cls.__mro__:
            fn = klass.__dict__.get("__call__")
            if fn is None or getattr(fn, "_devstral_probe", False) or klass not in owners:
                continue

            def _probe(self, *args, __fn=fn, **kwargs):
                for k in type(self).__mro__:
                    name = owners.get(k)
                    if name is not None:
                        INVOKED[name] = INVOKED.get(name, 0) + 1
                return __fn(self, *args, **kwargs)

            _probe._devstral_probe = True
            setattr(klass, "__call__", _probe)


# ----------------------------------------------------------------------------------------------- loading


@functools.lru_cache(maxsize=1)
def load_hf_model():
    """The HF reference: the FP8 checkpoint with weights bit-identical to transformers' bf16 dequantizing load,
    its text linears held as FP8 + scale in host memory (reference/checkpoint.py: ~48 GB -> ~22 GB)."""
    from models.demos.devstral_small_2_24b_instruct_2512.reference.checkpoint import load_reference_model

    return load_reference_model(MODEL_ID)


@functools.lru_cache(maxsize=1)
def load_processor():
    import transformers

    # transformers flags the checkpoint's pre-tokenizer regex as wrong and fixes it on request.
    return transformers.AutoProcessor.from_pretrained(MODEL_ID, fix_mistral_regex=True)


def requested_batch(default=DEFAULT_BATCH) -> int:
    raw = (os.environ.get(BATCH_ENV) or "").strip()
    return int(raw) if raw else int(default)


def _env_layers():
    raw = (os.environ.get(LAYERS_ENV) or "").strip()
    return int(raw) if raw and int(raw) > 0 else None


def log(msg):
    print(f"[devstral] {msg}", flush=True)


# ------------------------------------------------------------------------------------------ input encoding


def chat_messages(prompt, image=None):
    content = ([{"type": "image", "image": image}] if image is not None else []) + [{"type": "text", "text": prompt}]
    return [{"role": "user", "content": content}]


def encode_request(
    processor, prompt, image=None, batch=DEFAULT_BATCH, seeds=None, kv_capacity=DEFAULT_KV_CAPACITY, model=None
):
    # Pass the pipeline's own capacity (pipe.kv_capacity): the safety cap must be the one the device holds.
    """HOST encoding of one request for `batch` rows: chat template + tokenizer (+ image processor).

    Content is identical in every row; only the sampling seed differs (seed = row index by default). The prompt is
    right-padded with pad_token_id to padded_prompt_len(); causal attention keeps rows 0..L-1 independent of it.
    `model` supplies the config (default: the cached reference)."""
    model = model if model is not None else load_hf_model()
    cfg, gen = model.config, model.generation_config
    enc = processor.apply_chat_template(
        chat_messages(prompt, image), add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    prompt_tokens = enc["input_ids"][0]
    real_len = int(prompt_tokens.shape[0])
    padded = padded_prompt_len(real_len, kv_capacity)
    tokens = torch.full((batch, padded), int(gen.pad_token_id), dtype=torch.int64)
    tokens[:, :real_len] = prompt_tokens
    seeds = list(range(batch)) if seeds is None else list(seeds)
    req = {
        "prompt": prompt,
        "hf_inputs": {k: v for k, v in enc.items()},
        "tokens": tokens,
        "real_len": real_len,
        "batch": batch,
        "seeds": seeds,
        "max_new_tokens": kv_capacity - real_len,
        "image_token_id": int(cfg.image_token_id),
    }
    if image is not None:
        pv = enc["pixel_values"]
        req["pixel_values"] = pv.expand(batch, *pv.shape[1:]).contiguous()
        req["image_sizes"] = [tuple(int(v) for v in enc["image_sizes"][0])] * batch
    return req


def padded_prompt_len(real_len, kv_capacity):
    """Prefill length: real_len rounded up to a tile multiple (<= the resident KV capacity, itself a tile multiple)."""
    return int(math.ceil(real_len / 32) * 32)


def gumbel_table(seeds, steps, buckets=NOISE_BUCKETS):
    """[steps, B, buckets] Gumbel noise, one torch.Generator per row seeded with that row's seed, rounded to bf16
    (the precision the device table holds) so TT and the HF golden add bit-identical values. Token v at a step
    of row b gets noise[step, b, v % buckets]."""
    rows = []
    for s in seeds:
        gen = torch.Generator().manual_seed(int(s))
        u = torch.rand(steps, buckets, generator=gen, dtype=torch.float64).clamp_(1e-12, 1 - 1e-12)
        rows.append(-torch.log(-torch.log(u)))
    return torch.stack(rows, dim=1).to(torch.bfloat16).to(torch.float32)


def merge_rows(tokens, image_token_id, tokens_per_image):
    """HOST index for the on-device [IMG] merge: row r of the (B*S + B*M)-row table [text embeds ; image feats].

    Row b's j-th [IMG] position reads image feature b*M + j, exactly the order HF's masked_scatter fills."""
    b, s = tokens.shape
    flat = torch.arange(b * s, dtype=torch.int64).reshape(b, s)
    is_img = tokens == image_token_id
    counts = is_img.sum(dim=1)
    assert bool((counts == tokens_per_image).all()), f"[IMG] count {counts.tolist()} != {tokens_per_image} per image"
    img_rows = b * s + torch.arange(b * tokens_per_image, dtype=torch.int64).reshape(b, tokens_per_image)
    flat[is_img] = img_rows.reshape(-1)
    return flat.reshape(1, b * s)


# ---------------------------------------------------------------------------------------------- pipeline


class DevstralPipeline:
    """Resident TT pipeline: every weight is built once on the mesh, the KV cache is allocated once."""

    PIPELINE_STAGES = PIPELINE_STAGES
    # The vision tower and projector run data-parallel over the mesh (full weights on every device, each device its own
    # images, one all_gather of the projector output) instead of tensor-parallel (two fp32 all_reduces per layer).
    VISION_DATA_PARALLEL = True

    def __init__(
        self,
        device,
        hf_model,
        layers=None,
        encode_layers=None,
        prefill_layers=None,
        decode_layers=None,
        batch=None,
        kv_capacity=None,
        top_k=SAMPLING_TOP_K,
    ):
        _install_invocation_probe()
        self.device = device
        self.hf = hf_model  # the reference stays reachable: ground truth for section structure
        self.config = hf_model.config
        self.generation_config = hf_model.generation_config
        self.batch = int(batch or requested_batch())
        assert 1 <= self.batch <= 32, "decode packs one batch row per core of a 32-core shard grid"
        self.kv_capacity = int(kv_capacity if kv_capacity is not None else kv_capacity_for(self._tp()))
        self.top_k = int(top_k)
        self.temperature = float(self.generation_config.temperature or 1.0)
        self.eos_token_id = int(self.generation_config.eos_token_id)
        self.pad_token_id = int(self.generation_config.pad_token_id)
        self._repl = ttnn.ReplicateTensorToMesh(device) if isinstance(device, ttnn.MeshDevice) else None

        tcfg, vcfg = self.config.text_config, self.config.vision_config
        text_depth = decode_layers if decode_layers is not None else prefill_layers
        text_depth = text_depth if text_depth is not None else layers
        vision_depth = encode_layers if encode_layers is not None else layers
        # None = every layer; a cap never builds fewer than one layer per stack.
        self.text_layers = (
            tcfg.num_hidden_layers if text_depth is None else max(1, min(int(text_depth), tcfg.num_hidden_layers))
        )
        self.vision_layers = (
            vcfg.num_hidden_layers if vision_depth is None else max(1, min(int(vision_depth), vcfg.num_hidden_layers))
        )

        t0 = time.time()
        m = hf_model.model
        log(f"building vision tower ({self.vision_layers} layers) + projector")
        self.vision_dp = self.VISION_DATA_PARALLEL and self._tp() > 1
        _st_pixtral_attention.DATA_PARALLEL = self.vision_dp
        try:
            self.vision_tower = _st_vision_model.TtPixtralVisionModel(
                device, m.vision_tower, n_layers=self.vision_layers
            )
            self.multi_modal_projector = _st_projector.TtMistral3MultiModalProjector(device, m.multi_modal_projector)
        finally:
            _st_pixtral_attention.DATA_PARALLEL = False
        log(f"building language model ({self.text_layers} layers, TP={self._tp()})")
        self.language_model = _st_ministral3_model.TtMinistral3Model(
            device, m.language_model, n_layers=self.text_layers
        )
        self.lm_head = _st_decoder_head.DecoderHead(device, hf_model.lm_head)
        attn0 = self.language_model.layers[0].self_attn
        self.kv_shape = [self.batch, attn0.local_kv_heads, self.kv_capacity, attn0.head_dim]
        self.kv_caches = [
            (
                ttnn.zeros(self.kv_shape, dtype=KV_CACHE_DTYPE, layout=ttnn.TILE_LAYOUT, device=device),
                ttnn.zeros(self.kv_shape, dtype=KV_CACHE_DTYPE, layout=ttnn.TILE_LAYOUT, device=device),
            )
            for _ in self.language_model.layers
        ]
        # decode RoPE rows come from tables of every position below the KV capacity (built here, once)
        prepare_rope = getattr(self.language_model, "prepare_decode_rope", None)
        if prepare_rope is not None:
            prepare_rope(self.kv_capacity, self.batch)
        self._stage = {}
        self._decode_repeats = None
        # TP > 1: each device top-k's its own lm_head vocab shard (VocabTopK, shard positions); the gathered
        # (B, TP * K) candidate positions become vocab ids by adding their shard's first id (_shard_base), and one
        # more top-k over the TP * K candidates picks the winners.
        self._shard_base = None
        full = self.batch == 32 and self.top_k == 32
        self._noise_lookup = NoiseLookup(device, self.batch, self.top_k, NOISE_BUCKETS) if full else None
        self._sample_tail = SampleTail(device, self.batch, self.pad_token_id, self.eos_token_id)
        self._draw_tail = DrawTail(device, self.batch, self.pad_token_id, self.eos_token_id) if full else None
        self._last_row = LastRow(device) if self.batch == ttnn.TILE_SIZE else None
        tp = self._tp()
        if tp > 1:
            vocab = int(tcfg.vocab_size)
            self._shard_shape = (1, 1, self.batch, vocab // tp)
            self._vocab_topk = VocabTopK(device, self.top_k)
            base = (torch.arange(tp, dtype=torch.int32) * (vocab // tp)).repeat_interleave(self.top_k)
            self._shard_base = ttnn.from_torch(
                base.reshape(1, 1, 1, -1).expand(1, 1, self.batch, -1).contiguous(),
                dtype=ttnn.uint32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=self._repl,
            )
        log(f"pipeline resident in {time.time() - t0:.1f}s (KV cache {self.kv_shape} per layer, bf16)")

    def gathered_tables(self):
        """The tables a step reads BY INDEX (perf_automation's stage_seams.GATHERED_TABLES_ATTR): the token
        embedding's replicated table (decode's one-program gather, tt/embed_decode.py), its hidden-sharded copy and
        its per-row squared-sum table (prefill's per-device gathers), and the vision tower's rotary cos / sin tables
        (gathered by patch position). A unit of work reads rows of these, never the table, so the perf harness
        charges each at the rows that came out of it instead of at its whole size."""
        te = self.language_model.embed_tokens
        rope = self.vision_tower.patch_positional_embedding
        return [t for t in (te.weight, te.weight_shard, te.sq_table, rope.cos_table, rope.sin_table) if t is not None]

    def _tp(self):
        try:
            return int(self.device.get_num_devices())
        except Exception:
            return 1

    # ------------------------------------------------------------------ host -> device (outside any forward)

    def _put(self, t, dtype, layout):
        return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, device=self.device, mesh_mapper=self._repl)

    def _put_pixels(self, pixel_values, sizes):
        """Upload an image batch: equal-size images go up already patchified on the host (the vision tower's own
        (c, ky, kx) row order, tile layout), so the device skips its reshape / permute chain over 14-pixel row-major
        sticks; mixed sizes upload the raw pixels for the device patchify."""
        patches = self.vision_tower.patchify_host(pixel_values, sizes)
        if patches is None:
            return {"pixel_values": self._put(pixel_values.to(torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)}
        tp = self._tp()
        if self.vision_dp and patches.shape[0] % tp == 0 and patches.shape[0] // tp > 1:
            # data-parallel vision: each device its own run of images (VISION_DATA_PARALLEL); with replicated
            # pixels instead, every device computes every image (same result, no gather needed)
            pv = ttnn.from_torch(
                patches.to(torch.bfloat16).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=0),
            )
            return {"pixel_values": pv, "prepatched": True, "vision_split": True}
        return {
            "pixel_values": self._put(patches.to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT),
            "prepatched": True,
        }

    def upload(self, req):
        """Move one encoded request onto the device: prompt, positions, image, merge index, noise table, state."""
        b, s = req["tokens"].shape
        assert b == self.batch, f"request has {b} rows, pipeline was built for {self.batch}"
        assert req["real_len"] + 1 <= self.kv_capacity, "prompt exceeds the resident KV capacity"
        steps = self.kv_capacity - req["real_len"] + 1
        dev = {
            "real_len": req["real_len"],
            "seq": s,
            "tokens": self._put(req["tokens"].to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            "positions": self._put(
                torch.arange(s, dtype=torch.float32).reshape(1, 1, s, 1), ttnn.float32, ttnn.TILE_LAYOUT
            ),
            # forward_prefill's gather index of each user's last prompt row in the untilized 32-row block holding it
            "last_rows": self._put(
                torch.arange(b, dtype=torch.int32).reshape(1, b) * ttnn.TILE_SIZE
                + (req["real_len"] - 1) % ttnn.TILE_SIZE,
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
            ),
            # Cast on the host (exact: the table already holds bf16 values): an fp32 upload into a bf16 row-major
            # tensor is converted on device as tilize + typecast + untilize of the whole (steps, B * NB) table, ~5 ms.
            "noise": self._put(
                gumbel_table(req["seeds"], steps, NOISE_BUCKETS).reshape(steps, b * NOISE_BUCKETS).to(torch.bfloat16),
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
            ),
        }
        if "pixel_values" in req:
            sizes = req["image_sizes"]
            dev["image_sizes"] = sizes
            dev.update(self._put_pixels(req["pixel_values"], sizes))
            dev["patch_positions"] = self._put(
                self.vision_tower.patch_position_ids(sizes).reshape(1, -1).to(torch.int32),
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
            )
            m = self.tokens_per_image(sizes[0])
            dev["tokens_per_image"] = m
            dev["merge_rows"] = self._put(
                merge_rows(req["tokens"], req["image_token_id"], m).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
            )
        # decode state: resident buffers updated in place by every decode step
        dev["state"] = {
            "tok": self._put(torch.zeros(1, b, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            "cur_pos": self._put(
                torch.full((b,), req["real_len"], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
            ),
            "pos": self._put(torch.full((b, 1, 1, 1), float(req["real_len"])), ttnn.float32, ttnn.TILE_LAYOUT),
            "step": self._put(torch.zeros(1, 1, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            "unfinished": self._put(torch.ones(1, 1, b, 1), ttnn.float32, ttnn.TILE_LAYOUT),
            "noise": dev["noise"],
            "position": req["real_len"],
        }
        return dev

    def tokens_per_image(self, size):
        p = self.config.vision_config.patch_size * self.config.spatial_merge_size
        return (int(size[0]) // p) * (int(size[1]) // p)

    # ------------------------------------------------------------------------------ the forward chain (TT)

    def forward_encode(self, dev):
        """pixel_values -> vision tower -> projector: (B, 1, M, hidden) image embeddings."""
        feats = self.vision_tower(
            dev["pixel_values"],
            image_sizes=dev["image_sizes"],
            position_ids=dev["patch_positions"],
            prepatched=dev.get("prepatched", False),
        )
        out = self.multi_modal_projector(feats, image_sizes=dev["image_sizes"])
        if dev.get("vision_split"):  # each device encoded its own images: gather them in device (= image) order
            out = ttnn.all_gather(out, dim=0, num_links=1, topology=ttnn.Topology.Linear)
        b, d = self.batch, out.shape[-1]
        return ttnn.reshape(out, (b, 1, dev["tokens_per_image"], d))

    def embed_prompt(self, dev, image_embeds=None):
        """(B, 1, S, hidden) prompt embeddings; image features land on the [IMG] rows through an on-device gather."""
        if image_embeds is None:
            return self.language_model.prefill_embeds(dev["tokens"])  # (B, 1, S, D) or its hidden shard
        x = self.language_model.embed_tokens(dev["tokens"])  # (B, S, D)
        b, s, d = x.shape
        m = image_embeds.shape[-2]
        text_rows = ttnn.to_layout(ttnn.reshape(x, (1, 1, b * s, d)), ttnn.ROW_MAJOR_LAYOUT)
        img_rows = ttnn.to_layout(ttnn.reshape(image_embeds, (1, 1, b * m, d)), ttnn.ROW_MAJOR_LAYOUT)
        table = ttnn.reshape(ttnn.concat([text_rows, img_rows], dim=2), (b * s + b * m, d))
        merged = ttnn.embedding(dev["merge_rows"], table, layout=ttnn.TILE_LAYOUT)  # (1, B*S, D)
        return ttnn.reshape(merged, (b, 1, s, d))

    def forward_prefill(self, dev, embeds):
        """Prompt -> resident KV + normed hidden (B, S, D) + logits of the last prompt row (1, 1, B, V)."""
        hidden = self.language_model(
            inputs_embeds=embeds, position_ids=dev["positions"], kv_caches=self.kv_caches, mode="prefill"
        )
        b, _, d = hidden.shape
        # The last prompt row in two slices: first the tile-aligned 32-row block that holds it (a tiled slice), then
        # the row itself -- a direct unaligned slice untilized the whole (B, S, D) hidden state first.
        r = dev["real_len"] - 1
        if self._last_row is not None and LastRow.supports(hidden):
            # every user's row R straight out of its tiles into the tiled LM-head input (tt/last_row.py)
            return hidden, self.lm_head(self._last_row(hidden, r))
        r0 = r // ttnn.TILE_SIZE * ttnn.TILE_SIZE
        block = ttnn.slice(hidden, (0, r0, 0), (b, r0 + ttnn.TILE_SIZE, d))
        if "last_rows" in dev:
            # The rows themselves by one gather out of the untilized block (row b * 32 + r % 32 per user), emitted
            # tiled as the (B, D) LM-head input: the unaligned slice untilized, sliced, re-tilized padding each user's
            # row to a full tile (90 us) and then reshaped the (B, 1, D) result to (1, 1, B, D) through DRAM (48 us).
            rows = ttnn.reshape(ttnn.to_layout(block, ttnn.ROW_MAJOR_LAYOUT), (b * ttnn.TILE_SIZE, d))
            last = ttnn.embedding(dev["last_rows"], rows, layout=ttnn.TILE_LAYOUT)  # (1, B, D)
        else:
            last = ttnn.slice(block, (0, r - r0, 0), (b, r - r0 + 1, d))
        logits = self.lm_head(ttnn.reshape(last, (1, 1, b, d)))
        return hidden, logits

    def sample(self, logits, state):
        """On-device seeded sampling (Gumbel-max over the top-K of logits / T), EOS bookkeeping, write-back.

        Writes the next input ids into state["tok"] and the finished mask into state["unfinished"] IN PLACE and
        returns the (1, 1, B, 1) uint32 ids the model itself drew (before pad substitution)."""
        b, k, nb = self.batch, self.top_k, NOISE_BUCKETS
        vals, idx = self._topk(logits)
        scores = ttnn.multiply(ttnn.typecast(vals, ttnn.float32), 1.0 / self.temperature)
        if self._noise_lookup is not None:  # the candidates' noise in one program, straight from the table
            scores = ttnn.add(scores, self._noise_lookup(state["step"], state["noise"], idx))
        else:
            # The step's noise: one row-major lookup, a free row-major reshape to (B, NB), then a 32-row tilize
            # (the tile-layout lookup tilized one 32768-wide row padded to a full tile and reshaped it after the cast).
            noise = ttnn.embedding(state["step"], state["noise"], layout=ttnn.ROW_MAJOR_LAYOUT)  # (1, 1, B*NB) bf16
            noise = ttnn.typecast(ttnn.to_layout(ttnn.reshape(noise, (1, 1, b, nb)), ttnn.TILE_LAYOUT), ttnn.float32)
            bucket = ttnn.bitwise_and(idx, nb - 1)  # token v draws noise bucket v % NB (NB is a power of two)
            scores = ttnn.add(scores, ttnn.gather(noise, dim=-1, index=bucket))
        if self._draw_tail is not None and DrawTail.supports(scores, idx):
            # the top-1, the select, the eos update and the next-token row in one program (DrawTail)
            drawn = self._draw_tail(scores, idx, state["unfinished"], state["tok"])
            ttnn.plus_one(state["step"])
            return drawn
        # The best score and its vocab id in one op: the top-1 carries the candidates' ids as labels (no argmax +
        # index gather). Equal-score ties stay arbitrary, as the unstable candidate ordering already made them.
        _, drawn = ttnn.topk(scores, k=1, dim=-1, largest=True, sorted=True, indices_tensor=idx)  # (1, 1, B, 1)
        # generate(): finished rows emit pad; a row finishes on the step it emits eos.
        if self._sample_tail is not None and drawn.dtype == ttnn.uint32 and drawn.layout == ttnn.TILE_LAYOUT:
            # the select, the eos update and the next-token row in one program (SampleTail)
            self._sample_tail(drawn, state["unfinished"], state["tok"])
            ttnn.plus_one(state["step"])
            return drawn
        unf = state["unfinished"]
        nxt = ttnn.add(
            ttnn.multiply(ttnn.subtract(ttnn.typecast(drawn, ttnn.float32), float(self.pad_token_id)), unf),
            float(self.pad_token_id),
        )
        ttnn.copy(ttnn.multiply(unf, ttnn.ne(nxt, float(self.eos_token_id))), unf)
        ids = ttnn.reshape(ttnn.to_layout(ttnn.typecast(nxt, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), (1, b))
        ttnn.copy(ids, state["tok"])
        ttnn.plus_one(state["step"])
        return drawn

    def _topk(self, logits):
        """Top-k of the full-vocab logits row, values sorted, uint32 vocab ids.

        TP > 1: the top-k of the full row is the top-k of the union of each device's shard top-k, so each device
        sorts only its vocab/TP columns of the lm_head output (labelled with global ids), the TP short lists are
        all_gathered and one top-k over TP*k candidates picks the winners."""
        k = self.top_k
        local = getattr(self.lm_head, "local", None)
        if self._shard_base is None or local is None or tuple(local.shape) != self._shard_shape:
            return ttnn.topk(logits, k=k, dim=-1, largest=True, sorted=True)
        # Shard top-k (VocabTopK: 16-bit positions within the shard), gathered, then labelled with vocab ids: the
        # gathered block of device d holds positions in vocab [d * W, (d + 1) * W).
        vals, ids = self._vocab_topk(local)
        vals = ttnn.all_gather(vals, dim=3, num_links=1, topology=ttnn.Topology.Linear)
        ids = ttnn.all_gather(ids, dim=3, num_links=1, topology=ttnn.Topology.Linear)
        ids = ttnn.add(ttnn.typecast(ids, ttnn.uint32), self._shard_base)
        return ttnn.topk(vals, k=k, dim=-1, largest=True, sorted=True, indices_tensor=ids)

    def decode_prefill(self, request):
        """Decode contract: prefill the prompt once, seed the resident KV and the decode state, sample token 0.

        `request` is an encoded request (encode_request) or a token tensor [B, S] / [S] (perf harness)."""
        req = request if isinstance(request, dict) and "tokens" in request else self._token_request(request)
        dev = self.upload(req)
        image_embeds = self.forward_encode(dev) if "pixel_values" in dev else None
        hidden, logits = self.forward_prefill(dev, self.embed_prompt(dev, image_embeds))
        state = dev["state"]
        state["drawn"] = self.sample(logits, state)
        state["logits"], state["hidden"], state["image_embeds"] = logits, hidden, image_embeds
        return state

    def decode_step(self, state):
        """Decode contract: ONE token for every row from resident buffers only; samples on device and feeds the
        drawn id back into state["tok"] for the next step."""
        hidden = self.language_model(
            input_ids=state["tok"],
            position_ids=state["pos"],
            kv_caches=self.kv_caches,
            cur_pos=state["cur_pos"],
            mode="decode",
        )
        # The sampler reads each device's vocab shard (lm_head.local), so the token path skips the full-row
        # all_gather; host readers join the shards on the vocab dim (VocabShards / host_read).
        logits = self.lm_head(hidden, gather=False)
        ttnn.plus_one(state["cur_pos"])
        if getattr(self.language_model, "_rope_gather", None) is None:  # the gathered RoPE rows index by cur_pos
            ttnn.add(state["pos"], 1.0, output_tensor=state["pos"])
        state["drawn"] = self.sample(logits, state)
        state["logits"] = VocabShards(logits) if self._tp() > 1 else logits
        state["position"] = state["position"] + 1
        return state

    def _token_request(self, prompt_tokens):
        t = torch.as_tensor(prompt_tokens).to(torch.int64)
        t = t.reshape(1, -1) if t.dim() == 1 else t
        if t.shape[0] != self.batch:
            t = t[:1].expand(self.batch, t.shape[1])
        real_len = int(t.shape[1])
        padded = padded_prompt_len(real_len, self.kv_capacity)
        tokens = torch.full((self.batch, padded), self.pad_token_id, dtype=torch.int64)
        tokens[:, :real_len] = t
        return {
            "tokens": tokens,
            "real_len": real_len,
            "batch": self.batch,
            "seeds": list(range(self.batch)),
            "max_new_tokens": self.kv_capacity - real_len,
            "image_token_id": int(self.config.image_token_id),
        }

    # ---------------------------------------------------------------- per-stage trace contract (perf engine)
    #
    # The stage inputs are the golden inputs the e2e tests and the demos run (demo/inputs.py through the HF
    # processor), not the bring-up's _captured/*.pt: those tensors are gitignored and exist only on the box that
    # ran the capture (this checkout has manifests only, and no token_embed capture at all), and the captured
    # image was a 1540x1540 render the gated pipeline never sees.

    def _trace_request(self, task):
        """The encoded e2e / demo request of `task` for this pipeline's batch and KV capacity."""
        from models.demos.devstral_small_2_24b_instruct_2512.demo.inputs import IMAGE_PROMPT, TEXT_PROMPT, load_image

        image = load_image() if task == "image_text_to_text" else None
        return encode_request(
            load_processor(),
            IMAGE_PROMPT if image is not None else TEXT_PROMPT,
            image=image,
            batch=self.batch,
            kv_capacity=self.kv_capacity,
            model=self.hf,
        )

    def encode_trace_inputs(self):
        """The image_text_to_text request's image, one per batch row."""
        req = self._trace_request("image_text_to_text")
        return {"pixel_values": req["pixel_values"], "image_sizes": req["image_sizes"]}

    def encode_trace_setup(self, inputs):
        pv, sizes = inputs["pixel_values"], [tuple(int(v) for v in s) for s in inputs["image_sizes"]]
        h, w = sizes[0]
        pv = pv[..., :h, :w].contiguous()  # the processor's padding is cropped off before patchify (as HF does)
        self._stage["encode"] = {
            **self._put_pixels(pv, sizes),
            "image_sizes": sizes,
            "patch_positions": self._put(
                self.vision_tower.patch_position_ids(sizes).reshape(1, -1).to(torch.int32),
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
            ),
            "tokens_per_image": self.tokens_per_image(sizes[0]),
            "patches": (h // self.config.vision_config.patch_size) * (w // self.config.vision_config.patch_size),
        }

    def encode_trace_step(self):
        return self.forward_encode(self._stage["encode"])

    def encode_trace_items(self):
        """Patches the 24 vision layers process in one call: B images x (h/14 * w/14)."""
        st = self._stage.get("encode")
        if st is None:
            h, w = self.encode_trace_inputs()["image_sizes"][0]
            p = self.config.vision_config.patch_size
            st = {"patches": (h // p) * (w // p)}
        return self.batch * st["patches"]

    def prefill_trace_inputs(self):
        """The text_generation request's prompt tokens (unpadded), one copy per batch row."""
        req = self._trace_request("text_generation")
        return {"input_ids": req["tokens"][:, : req["real_len"]].contiguous()}

    def prefill_trace_setup(self, inputs):
        """Pin the prompt to a tile-multiple capacity, pre-upload tokens / positions / noise / state."""
        prompt = inputs["input_ids"] if isinstance(inputs, dict) else inputs
        self._stage["prefill"] = self.upload(self._token_request(prompt))

    def prefill_trace_step(self):
        dev = self._stage["prefill"]
        hidden, logits = self.forward_prefill(dev, self.embed_prompt(dev))
        self.sample(logits, dev["state"])
        return logits

    def prefill_trace_items(self):
        """Prompt rows the text layers process in one call: B x the padded prompt length."""
        dev = self._stage.get("prefill")
        if dev is None:
            real_len = int(self.prefill_trace_inputs()["input_ids"].shape[1])
            return self.batch * padded_prompt_len(real_len, self.kv_capacity)
        return self.batch * dev["seq"]

    def decode_trace_inputs(self):
        return self.prefill_trace_inputs()

    def decode_trace_setup(self, inputs):
        prompt = inputs["input_ids"] if isinstance(inputs, dict) else inputs
        self._stage["decode"] = self.decode_prefill(self._token_request(prompt))

    def decode_trace_step(self):
        logits = self.decode_step(self._stage["decode"])["logits"]
        return logits.tensor if isinstance(logits, VocabShards) else logits

    def decode_trace_items(self):
        """One token for every batch row."""
        return self.batch

    def decode_trace_repeats(self):
        """Decode steps one real request runs: the length the reference request generated (to EOS) when it has
        been measured, else the schedule's horizon (KV capacity minus the captured prompt)."""
        if self._decode_repeats:
            return int(self._decode_repeats)
        rec = PACKAGE_DIR / "tests" / "e2e" / "decode_horizon.json"
        if rec.is_file():
            try:
                return int(json.loads(rec.read_text())["text_generation"]["decode_steps"])
            except Exception:
                pass
        return self.kv_capacity - int(self.prefill_trace_inputs()["input_ids"].shape[1])


def build_pipeline(
    device,
    model=None,
    layers=None,
    encode_layers=None,
    prefill_layers=None,
    decode_layers=None,
    batch=None,
    kv_capacity=None,
    **kwargs,
):
    """THE build surface: construct and RETURN the resident pipeline object (it does not run it).

    `layers` caps the depth of EVERY repeated stack (None = every layer, never 0); `encode_layers` overrides the
    vision stack, `prefill_layers` / `decode_layers` the text stack (one stack serves both stages). Demo kwargs
    (prompt, image, ...) are accepted and ignored. Falls back to $TT_PERF_LAYERS when no cap is passed."""
    if layers is None and encode_layers is None and prefill_layers is None and decode_layers is None:
        layers = _env_layers()
    return DevstralPipeline(
        device,
        model if model is not None else load_hf_model(),
        layers=layers,
        encode_layers=encode_layers,
        prefill_layers=prefill_layers,
        decode_layers=decode_layers,
        batch=batch,
        kv_capacity=kv_capacity,
    )


# --------------------------------------------------------------------------------------- task heads (TT)


class VocabShards:
    """Decode logits kept as each device's (1, 1, B, vocab/TP) lm_head shard: the on-device sampler only needs
    the shards, so no all_gather runs on the token path; host_read joins them on the vocab dim."""

    __slots__ = ("tensor",)

    def __init__(self, tensor):
        self.tensor = tensor


def host_read(t, device):
    """Device -> host read of a replicated tensor (the first replica), or the full vocab row of VocabShards."""
    if isinstance(t, VocabShards):
        return ttnn.to_torch(t.tensor, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=-1))
    if isinstance(device, ttnn.MeshDevice):
        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))[: t.shape[0]]
    return ttnn.to_torch(t)


def _host_rows(t, rows, device):
    return host_read(t, device).reshape(rows, -1)


def _generate(pipe, req, teacher=None, on_step=None, image=False):
    """Prefill + decode to the model's own stop rule (EOS on every row) with the KV capacity as safety cap.

    teacher: optional [B, T] token ids to FEED instead of the drawn ones (teacher forcing); the drawn ids are
    still recorded. on_step(t, state) is called after every token (state holds the device logits)."""
    b = pipe.batch
    t0 = time.time()
    state = pipe.decode_prefill(req)
    ttnn.synchronize_device(pipe.device)
    log(f"prefill done: {b} rows x {req['real_len']} tokens in {time.time() - t0:.2f}s")
    cap = req["max_new_tokens"]
    osl = (os.environ.get(OSL_ENV) or "").strip()
    if osl:
        cap = min(cap, int(osl))
    if teacher is not None:
        cap = min(cap, int(teacher.shape[1]))
    drawn_steps, fed_steps, finished = [], [], [False] * b
    for t in range(cap):
        if t > 0:
            state = pipe.decode_step(state)
        drawn = _host_rows(state["drawn"], b, pipe.device)[:, 0].to(torch.int64)
        fed = _host_rows(state["tok"], b, pipe.device)[:, 0].to(torch.int64)
        if teacher is not None:
            fed = teacher[:, t].to(torch.int64)
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(fed.to(torch.int32).reshape(1, b), dtype=ttnn.uint32, mesh_mapper=pipe._repl),
                state["tok"],
            )
        drawn_steps.append(drawn)
        fed_steps.append(fed)
        if on_step is not None:
            on_step(t, state)
        finished = [f or int(v) == pipe.eos_token_id for f, v in zip(finished, fed.tolist())]
        n_done = sum(finished)
        log(f"decode step {t}: {n_done}/{b} rows finished ({time.time() - t0:.1f}s)")
        if n_done == b:
            break
    return {
        "drawn": torch.stack(drawn_steps, dim=1),
        "tokens": torch.stack(fed_steps, dim=1),
        "finished": finished,
        "steps": len(fed_steps),
        "hit_cap": not all(finished),
        "state": state,
    }


def run_text_generation(pipe, req, teacher=None, on_step=None):
    """Call 1, text -> text: chained TT prefill + decode over the graduated stubs."""
    assert "pixel_values" not in req
    return _generate(pipe, req, teacher=teacher, on_step=on_step)


def run_image_text_to_text(pipe, req, teacher=None, on_step=None):
    """Call 2, image + text -> text: TT vision encode -> projector -> merged prefill -> decode."""
    assert "pixel_values" in req
    return _generate(pipe, req, teacher=teacher, on_step=on_step, image=True)


def decode_text(processor, tokens, eos_token_id):
    """Generated ids [B, T] -> strings, cut at each row's EOS."""
    out = []
    for row in tokens.tolist():
        cut = row.index(eos_token_id) if eos_token_id in row else len(row)
        out.append(processor.tokenizer.decode(row[:cut], skip_special_tokens=True))
    return out


# --------------------------------------------------------------------------------------------- self-tests


def _open_default_mesh():
    from models.demos.devstral_small_2_24b_instruct_2512.mesh import open_mesh

    return open_mesh(trace_region_size=TRACE_REGION_SIZE)


def _pcc(a, b):
    a, b = a.flatten().to(torch.float64), b.flatten().to(torch.float64)
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def trace_capture_selftest(device=None):
    """Per stage: capture ONE step in a trace, replay it, compare to the eager step, RELEASE before the next.

    The eager run of the same step is the reference; it is itself PCC-gated against HF by tests/e2e."""
    own = device is None
    if own:
        device = _open_default_mesh()
    ok = True
    try:
        pipe = build_pipeline(device)
        for stage in PIPELINE_STAGES:
            setup, step = getattr(pipe, f"{stage}_trace_setup"), getattr(pipe, f"{stage}_trace_step")
            inputs = getattr(pipe, f"{stage}_trace_inputs")()
            setup(inputs)
            # A step that reads a sampled token must see the SAME token in the eager and traced run: a near-tie
            # in the prefill sample may draw differently on a re-setup, which is not a trace defect.
            state = pipe._stage.get(stage)
            tok = state.get("tok") if isinstance(state, dict) else None
            tok0 = host_read(tok, device) if tok is not None else None
            ref = host_read(step(), device).float()
            setup(inputs)
            if tok0 is not None:
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(tok0, dtype=tok.dtype, mesh_mapper=pipe._repl), pipe._stage[stage]["tok"]
                )
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            out = step()
            ttnn.end_trace_capture(device, tid, cq_id=0)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            got = host_read(out, device).float()
            ttnn.release_trace(device, tid)
            pcc = _pcc(ref, got)
            stage_ok = pcc >= 0.99
            ok = ok and stage_ok
            log(f"trace selftest stage={stage}: captured + replayed, trace-vs-eager PCC={pcc:.6f} ok={stage_ok}")
    finally:
        if own:
            from models.demos.devstral_small_2_24b_instruct_2512.mesh import close_mesh

            close_mesh(device)
    return ok


def host_op_selftest(device=None):
    """Run every task head's forward (encode, merged prefill, decode steps) under the host-op observer.

    Tokenization, image processing, the weight build and the host->device upload of the encoded request happen
    OUTSIDE the observed region; everything from encoded inputs to sampled tokens happens inside."""
    from scripts.tt_hw_planner.host_op_observer import observe_host_ops, verdict

    own = device is None
    if own:
        device = _open_default_mesh()
    try:
        pipe = build_pipeline(device)
        processor = load_processor()
        from models.demos.devstral_small_2_24b_instruct_2512.demo.inputs import IMAGE_PROMPT, TEXT_PROMPT, load_image

        all_ops = []
        for task, req in (
            ("text_generation", encode_request(processor, TEXT_PROMPT, batch=pipe.batch, kv_capacity=pipe.kv_capacity)),
            (
                "image_text_to_text",
                encode_request(
                    processor, IMAGE_PROMPT, image=load_image(), batch=pipe.batch, kv_capacity=pipe.kv_capacity
                ),
            ),
        ):
            dev = pipe.upload(req)
            with observe_host_ops() as ops:
                image_embeds = pipe.forward_encode(dev) if "pixel_values" in dev else None
                _, logits = pipe.forward_prefill(dev, pipe.embed_prompt(dev, image_embeds))
                state = dev["state"]
                state["drawn"] = pipe.sample(logits, state)
                for _ in range(2):
                    state = pipe.decode_step(state)
            ttnn.synchronize_device(device)
            log(f"host-op observer {task}: {len(ops)} aten op(s) inside the forward")
            all_ops += list(ops)
        return verdict(all_ops)
    finally:
        if own:
            from models.demos.devstral_small_2_24b_instruct_2512.mesh import close_mesh

            close_mesh(device)

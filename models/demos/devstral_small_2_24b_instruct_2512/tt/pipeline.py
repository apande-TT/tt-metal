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
import struct
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
#                     computes at HiFi4 (fp32 accumulation) on a bfloat8_b weight (HiFi2, run 2, LoFi, run 7, and the
#                     LoFi gate/up of layers 14-25, run 6, failed test_e2e_image_text_to_text: the image-token rows
#                     amplify every layer's error ~10x more than text rows; HiFi3 is not bit-exact with HiFi4 on the
#                     bf8 weight and fails too; see ministral3_m_l_p PREFILL_GATE_UP_LOFI_LAYERS).
#                     Fused gate/up computes at HiFi2 (fp32 accumulation), 8 x 8 x 12-tile blocks; the text decoder's
#                     and the vision MLPs' fused gate/up weights are bfloat8_b (the vision gate/up at HiFi2, not run 2's
#                     LoFi, which took the image features' PCC 0.999906 -> 0.999268, an input error the language model
#                     amplified ~60x on the image-token rows; the vision down stays at HiFi2).
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
#                     HiFi4 matmuls (HiFi2, run 2, was part of the image e2e failure above).
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
#                     to bf16 (the gate/up's product fits in L1 too). The fp32 residual stream stays in L1 between
#                     the layers (o_proj / down outputs and both residual adds in L1, consumed tensors freed):
#                     adds 58 -> ~20 us, norms 36 -> 24 us, o_proj 73 -> 61 us, down 125 -> 117 us a layer.
#                     q / k / v, the SDPA output and the head concat sit in L1 too (each freed by its reader):
#                     SDPA 185 -> 167 us, head concat 25 -> 11 us a layer.
#                     Both fp32 residual adds ride in the output stage of the matmul feeding them (o_proj, the MLP
#                     down; tt/minimal_mm.py residual=: the residual block read by the stock readers' FUSE_TERNARY
#                     path, added by the same SFPU add ttnn.add runs, bit-identical): o_proj + add 79.8 -> 67.9 us,
#                     down + add 135.3 -> ~123 us a layer; trace encode 24.62 -> 23.72 ms.
#                     o_proj reads SDPA's head outputs as their concat itself (minimal_mm in0_heads=, kernels/
#                     mm_in0_heads.cpp: the stock in0 reader with its tile index remapped): no nlp_concat_heads
#                     program (11.4 us a layer), bit-identical.
#                     The L1 casts of both norm outputs to bf16 run as tt/local_cast.py (LOCAL_CAST: every core casts
#                     the pages of its own L1 bank with ttnn.typecast's compute, rounding loop unrolled; bit-identical):
#                     10.3 -> 7.2 us a cast, 48 an encode.
#   ministral3_m_l_p many-row fused gate/up (text prefill + vision): tt/minimal_mm.py, minimal_matmul's own program
#                     with silu * up as ONE SFPU pass (kernels/mm_compute.cpp), bit-identical.
#   ministral3_decoder_layer prefill: the residual stays hidden-sharded over the TP mesh: the fp32 adds and the norms
#                     run on the reduce-scatter shard (sum-of-squares all_gather) and the bf16 norm output is all_gathered.
#   ministral3_decoder_layer prefill residual adds: C++ add + next-norm sum of squares in one pass (tt/add_stats.py),
#                     whose stats tile has exact zeros beside the row sum, so the gathered stats need no mask multiply;
#                     rsqrt(stats / N + eps) runs as one unary_chain;
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
#   pixtral_attention CPP_HEADS_ROPE: each vision layer's head split and q / k RoPE run as ONE C++ program
#                     (tt/heads_rope.py: per tile, x * cos + partner * signed sin, replaying ttnn.multiply's SFPU and
#                     ttnn.add's FPU bf16 ops -- bit-identical to the rotate-half chain, no matmul) instead of
#                     nlp_create_qkv_heads + a rotate-half matmul / multiply / multiply / add per tensor. The vision
#                     SDPA runs 128-row query chunks (one 256-row key chunk).
#   ministral3_m_l_p prefill fused gate/up: M blocks that tile each core's M range with the least padding (416-row
#                     prompt: 7 tiles, not 8) and 1 x 4 subblocks -- the SwiGLU epilogue no longer runs padded rows.
#                     The long-prompt QKV / o_proj take N blocks of each core's narrow N range (QKV: 5 tiles, 4 x 1
#                     subblocks) and the MLP down 1 x 4 subblocks.
#   prefill fused gate/up (tt/minimal_mm.py SWIGLU_OVERLAP): each output block's SwiGLU epilogue runs during the next
#                     block's matmul, one gate/up pair between its subblocks (the PACK thread's SFPU beside the FPU), with
#                     a two-block fp32 intermediate and every writer (injectors too, kernels/mm_dm_in*.cpp) deferring
#                     its block write to K step K_blocks / 2 or later: 416-row gate/up 9.57 -> 9.21 ms/layer,
#                     bit-identical (DRAM in / out only: the vision MLP's L1 tensors leave no room for the second block).
#   minimal_mm grid split by per-core work (tt/minimal_mm.py transposed): M along x / N along y or the other way round,
#                     whichever hands each core >= 5% fewer output tiles (stock: M > N). Flips only the vision gate/up
#                     (168 -> 156 tiles a core), o_proj and the patch projection (24 -> 21); the stubs size their blocks to
#                     the chosen ranges and the MLP down keeps the stock split. Trace encode 21.30 -> 21.00 ms.
#   decode vocab top-k (VocabTopK.FUSED): every level sorts packed [bf16 | u16] 32-bit keys (the stock topk's fused-key
#                     mode) instead of value + index tiles: 67.9 -> 54.3 us a step; equal logits rank by position.
#   ministral3_m_l_p prefill MLP down at LoFi on every text layer (PREFILL_DOWN_LOFI; the op is FPU-bound at 608 x 256
#                     tile products a core): trace prefill 1028 -> 937 ms; PCC 0.995097 -> 0.995002 (threshold 0.995).
#                     At LoFi its data movement shows (2.51 ms vs the 2.06 ms roof): M blocks of a third of each core's
#                     M range (13 rows, not 8) re-read every weight block 3 times instead of 5 -- 2.51 -> 2.42 ms a
#                     call, bit-identical (the per-tile K order depends only on the N block); prefill 937 -> 929 ms.
#   ministral3_decoder_layer prefill sharded norm: its bf16 output (34 MB a device) goes to L1 for the hidden
#                     all_gather (PREFILL_AG_IN_L1; the gathered row stays in DRAM): multiply 217 -> 160 us, gather
#                     851 -> 831 us a layer; prefill 929 -> 925 ms, bit-identical.
#   decode SDPA on an 8 x 8 grid (ministral3_attention DECODE_SDPA_ROWS): its 64 (user, kv head) pairs take one core
#                     each, so the 8 x 10 grid launched 16 idle cores; same kernel time, trace decode ~8 us a token less.
#   decode down fold's fabric-link workers on (5, 9), (6, 9) (StreamDownDecode.AR_WORKERS, was (8, 9), (9, 9)): trace
#                     decode ~12 us a token less, bit-identical.
#   decoder_head: the one-tile-row lm_head (decode; prefill's last prompt rows) streams bank-major copies of its
#                     weight on both data-movement RISCs of 64 cores (tt/lm_head_decode.py), bit-identical to ttnn.linear.
#                     The decode gate / up and the lm_head stream through 256 KB weight CBs (the down / o_proj: 112 KB);
#                     the lm_head multiplies at HiFi4 (free under its weight stream); decode down bf4 on layers 16-25.
#   ministral3_decoder_layer decode all_reduces (o_proj, MLP down): ONE all_reduce_async each (tt/decode_all_reduce.py)
#                     instead of reduce-scatter + all_gather: the streamed projections write their bf16 partials into
#                     the residual's 40-core width shards, every device multicasts them into one shared 32 KB/core
#                     scratch, summed in fp32 DEST and rounded once, and the residual add reads the sum natively
#                     (trace decode 15.99 -> 15.75 -> 15.48 ms). The scratch fits L1 because the prefill final norm runs
#                     with no all-ones weight row and minimal_mm keeps one output-block buffer (vision gate / up -86 KB).
#   decode o_proj all_reduce + residual add folded into the MLP's gate / up program (NormGateUpDecode.ar_norm_call):
#                     two fabric-link workers (kernels/ar_send.cpp, all_reduce_async's line-multicast exchange) push the
#                     partial shards into every device's scratch at launch, the norm cores sum the slots and add the
#                     residual exactly as all_reduce_async + ttnn.add do, then norm -- the exchange runs under the weight
#                     streams (trace decode 15.48 -> 15.14 ms, PCC unchanged).
#   ministral3_m_l_p: the block-float4 decode gate / up and down multiply at HiFi2 (the bf4 gate / up is compute-bound;
#                     trace decode 15.09 -> 14.72 ms, PCC 0.995223); decode down bf4 on layers 16-26 (14.67 ms, PCC 0.995097).
#   decode gate / up spread over 104 cores (NormGateUpDecode.SPREAD): the 40 norm cores take 2 columns each once their
#                     norm is done and the stream cores 3 or 2 (was 4), every bank still serving 32 columns: the bf4
#                     layers' post-prologue matmul shrinks by a quarter and the extra readers lift the bf8 stream (bf8
#                     gate / up 214.6 -> 191.5 us; trace decode 14.67 -> 14.16 ms, bit-identical).
#   decode down / o_proj streams spread over 104 cores (StreamDownDecode.SPREAD): per bank 7 cores with two columns
#                     and 6 with one, whose pushes alternate between its two RISCs (bf8 down 110.5 -> 101.4 us, o_proj
#                     15.3 -> 13.6 us; trace decode 14.16 -> 13.92 ms, bit-identical).
#   lm_head stream spread over 104 cores (LMHeadDecode.SPREAD): one column a copy a pass, 5 or 4 passes a core
#                     (419.7 -> 375.7 us, ~474 GB/s; trace decode 13.92 -> 13.88 ms, bit-identical).
#   decode MLP down all_reduce + residual add folded into the down program (StreamDownDecode.ar_call): each core's
#                     columns stream one at a time, the fabric-link workers (kernels/ar_send_waves.cpp) send the
#                     first-column shards while the second columns stream, the residual cores sum the slots and add
#                     the residual (all_reduce_async + ttnn.add bit for bit); the senders multicast half of x on each
#                     NOC (one RISC's 512 KB multicast landed x ~40-60 us in). Trace decode 13.88 -> 13.61 ms.
#   decode gate / up sender multicasts half of the gathered normed row on each NOC (NormGateUpDecode.DUAL_NOC_SENDER):
#                     bf8 gate / up 191.5 -> ~184 us, the bf4 layers' matmul starts earlier; trace 13.61 -> 13.37 ms.
#   decode down fold: a two-column core's second column split into a leading run of pushes for RISCV_0 (18, then 20
#                     of 32) and a trailing one for RISCV_1, whose NOC 1 reads fall behind once only those cores stream
#                     (StreamDownDecode.SPLIT_B_SHARE): down 119.7 -> 116.8 us; trace 13.37 -> 13.22 -> 13.18 ms.
#   decode down fold: each partial tile goes straight into its link worker's send buffer and the worker sends every
#                     shard once its tiles are in (StreamDownDecode.DIRECT_SEND, kernels/ar_send_shards.cpp), not a
#                     wave's shards read back after the whole wave.
#   decode gate / up SwiGLU epilogue as ONE fused SFPU pass a pair (kernels/gate_up_compute.cpp gu_silu_mul, the
#                     prefill's hoisted silu * up): the epilogue is the program's tail after the last weight push
#                     (bf8 math - unpack 3.0 -> 2.5 us); trace decode 13.022 -> 12.990 ms, bit-identical.
#   decode vocab top-k as a 4-ary tree (VocabTopK.TREE3): 64 locals -> 16 -> 4 -> final, 3 + 3 + 3 serial tile merges
#                     instead of 7 + 7 (85 cores): 54.3 -> 51.2 us a call, trace decode 12.990 -> 12.988 ms; the exact
#                     top-32 of [value | position] keys, bit-identical.
#   decode sampler's cross-device top-k as ONE gather (VocabTopK.PACKED_OUT + MergeTopK.packed): the final core emits
#                     its merged packed key tile raw (kernels/topk_final_packed.cpp), a single fp32 all_gather moves
#                     it, and MergeTopK splits the keys in DEST (stock topk_defuse sweep, INT32 index store) + ORs in
#                     each device's vocab offset before its merge -- no position gather, typecast or offset add;
#                     trace decode 12.986 -> 12.977 ms, bit-identical.
#   prefill SDPA reads only each causal q chunk's K / V rows up to its diagonal (tt/sdpa_prefill.py CAUSAL_READ_SKIP,
#                     kernels/sdpa_reader.cpp: the stock reader + SDPA_CAUSAL_READ_SKIP; the compute already skipped
#                     those tiles): 416-row SDPA 529 -> 424 us a call, bit-identical.
#   prefill final norm on the hidden shards (ministral3_model.SHARDED_FINAL_NORM, ministral3_r_m_s_norm
#                     _sharded_prefill): stats all_gather + shard norm * the device's gamma columns + ONE gather of the
#                     bf16 result, instead of gathering the residual and norming + gamma-multiplying the full width on
#                     every device (4x redundant): device_ms 125.45 -> 122.95.
#   prefill norm scale multiply natively HEIGHT-SHARDED (ministral3_decoder_layer.PREFILL_NORM_L1_COPY): the residual
#                     add (add_stats.apply_l1) also writes its fp32 sum into a height-sharded L1 copy (each core its own
#                     whole tile rows, one local NOC write a block), the norm multiplies from it into bf16 shards for the
#                     hidden all_gather and frees it: multiply 178 -> 106 us a call, trace prefill 1132.4 -> 1124.6 ms.
#   prefill SDPA causal rows at EXACT widths (kernels/sdpa_compute.cpp RowCols.need): mask, row max, exp / row sum and
#                     P @ V stop at the diagonal of the row's P @ V subblock instead of rounding up to whole exp groups
#                     (partial last exp group), the exp's SFPU init runs once a pass: SDPA 416 424 -> 418 us, 128
#                     96 -> 86 us, bit-identical.
#   decode sampler MergeTopK on TWO cores (MergeTopK.SPLIT): the leaf merges the second pair of packed key tiles and
#                     copies its winner into the root's buffers while the root merges the first pair, then the root
#                     merges the two winners: 10.0 -> ~7.5 us a token, bit-identical.
#   decode sampler DrawTail on TWO cores, 8 rows a RISC (DrawTail.SPLIT, kernels/draw_tail.cpp SPLIT8): 64-B aligned
#                     row segments in and out, each core's 16 ids written once its two RISCs have them: 8.2 -> ~4.6 us a
#                     token, bit-identical.
#   prefill norm stats and their all_gather in L1 (add_stats.STATS_L1; the norm frees both right after the sum):
#                     416-row stats gather 70-76 -> 57-64 us a call, bit-identical.
#   decode input norm as a generic_op replica of rms_norm's sharded kernel (tt/norm_decode.py, kernels/norm_rep_*.cpp:
#                     same LLK calls, scalers and two-stage reduce) that packs the bf16 TRUNCATION of the normed row --
#                     what the HiFi2 QKV multiplies of an fp32 in0 anyway -- so the QKV reads half the activation bytes:
#                     norm 5.71 -> 6.55 us (factor multicast from the root, MCAST_FACTOR; one arrival semaphore per
#                     gathered slot so the reduces overlap the arrivals, SLOT_SEMS), QKV 24.2 -> ~21.8 us; trace decode
#                     12.98 -> 12.95 ms, bit-identical.
#   decode K / V append as ONE generic_op (ministral3_attention.DECODE_KV_APPEND, tt/kv_append.py,
#                     kernels/kv_append_*.cpp) instead of paged_fused_update_cache: the new rows packed to bfloat8_b by
#                     the packer (a bfp8 face row depends only on its 16 values) and spliced straight into the cache
#                     tiles -- exponent section read, the row's two 16-byte exponent groups and two 16-byte mantissa
#                     face rows written back, one RISC a kv head -- no untilize / patch / re-tilize of the tile row:
#                     6.3 -> ~3.9 us a layer, trace decode 12.95 -> 12.77 ms, bit-identical.
#   prefill norm's per-row factor resharded into the L1 copy's height shards (ministral3_decoder_layer.
#                     PREFILL_NORM_FACTOR_SHARDED_ROWS, <= 64-row shards): the native sharded scale multiply reads no
#                     DRAM: 128-row prompt 54.9 -> 42.7 + 1.9 us a call (at 128-row shards it is slower, so not
#                     there), bit-identical.
#   prefill norm scale as a generic_op replica of ttnn.multiply's SFPU column-broadcast program
#                     (ministral3_decoder_layer.PREFILL_NORM_SCALE_CPP, tt/norm_scale.py, kernels/norm_scale_*.cpp):
#                     the same unpack-to-dest copies, mul_binary_tile and fp32 -> bf16 typecast_tile, 3 x tiles a DEST
#                     acquire instead of 1, factor tiles read straight from r, and the multiply + bf16 rounding in ONE
#                     SFPU pass (FUSED_RNE: the typecast's integer RNE applied to the product in a register): 416-row
#                     prompt 109.8 -> 71.6 us a call, 128-row 44.6 -> 35.2 us, bit-identical.
#   prefill q / k RoPE outputs in L1 for short prompts (ministral3_attention.PREFILL_ROPE_OUT_L1_BYTES, q <= 96 KB a
#                     core), read there by the cache fill and SDPA and freed after SDPA: 128-row prompt q RoPE 39.9 ->
#                     23.1 us, k 13.3 -> 8.7, SDPA 86.0 -> 67.1 us a layer, bit-identical.
#   prefill norm stats gathered COMPACTED (add_stats.COMPACT_STATS, tt/stats_expand.py): each stats tile's column 0
#                     (its 32 row sums; the rest is zeros) written as one row of a compact tile, the all_gather moves
#                     1/32 of the bytes and a DM program rebuilds the exact (..., S, 128) gathered tiles for the sum:
#                     416 rows 96.5 -> 41.4 + 17.3 us, 128 rows 23.7 -> 8.3 + 9.4 us a call, bit-identical.
#   prefill final norm's normed rows and gamma product in L1 for its all_gather (ministral3_r_m_s_norm.
#                     FINAL_GAMMA_OUT_L1): 416 rows x * r 132 -> ~97 us, gamma multiply 205-240 -> ~121 us; 128 rows
#                     ~49 -> 37 and 68.8 -> 39 us, bit-identical.
#   prefill residual add + norm stats in ONE DEST pass (add_stats.ONE_PASS, kernels/add_stats_compute.cpp): h, its
#                     writer copy and h^2 from the same DEST tiles (add_binary_tile, copy_dest_values, square_tile),
#                     no row buffer of h packed and unpacked twice: 416-row adds ~2340 -> 2239 us a 2-layer slice,
#                     128-row unchanged (memory bound), bit-identical.
#   prefill norm row sums and factor r in L1 for the C++ scale (ministral3_decoder_layer.PREFILL_NORM_FACTOR_L1; ~16
#                     KB a core, both freed once read): 416 rows sum 12 -> ~10.5 us, rsqrt chain 13 -> 6 us, scale 71
#                     -> 69 us a norm; 128 rows rsqrt 4 -> 3, scale 35 -> 34 us, bit-identical.
#   decode input norm (its width shards feed the HiFi2 QKV) WITHOUT the all-ones weight
#                     (ministral3_r_m_s_norm.DECODE_SHARDED_NO_WEIGHT): rms_norm's FUSE_GAMMA pass only truncated the
#                     fp32 output to TF32, below what the HiFi2 QKV multiplies of in0: norm 6.28 -> 5.71 us a layer,
#                     QKV output bit-identical.
#   short-prompt MLP down on the full grid (ministral3_m_l_p.SHORT_DOWN_FULL_GRID): blocks cut to each core's exact
#                     13 x 15-tile range (M blocks of 7, 1 x 3 subblocks) over 110 cores instead of whole 16 x 16-tile
#                     blocks over the 80-core grid.
#   short-prompt MLP down with its operands swapped (ministral3_m_l_p.SHORT_DOWN_SWAPPED, tt/minimal_mm.py swapped=):
#                     (w^T @ x^T)^T on a bf16 copy of the bf8 weight, transposed on the host (exact), so the weight sits
#                     in srcB, whose one 7-bit multiplier pass holds a bf8 significand: HiFi2 forms HiFi4's products
#                     (srcA's two 5-bit passes cover bf16). x is read in transposed tile positions and unpacked
#                     transposed, each output tile packed transposed (kernels/mm_dataflow_common.hpp, mm_compute.cpp).
#                     128-row down 2550 -> 1414 us a call; vs the HiFi4 program PCC 1.000000000, max |diff| 1.2e-4,
#                     85% of elements bit-identical, mean |err| vs a fp32 product 1.1749e-6 (HiFi4: 1.1766e-6).
#                     The long (416-row) prompt's down too on the last thirty layers (LONG_DOWN_SWAPPED_LAYERS 10-39):
#                     the image prompt's chaotic rows amplify a perturbation only through the layers after it.
#   prefill gate / up in 16-deep K blocks on layers 20-39 (minimal_mm.SWIGLU_DEEP_K, PREFILL_GU_DEEP_K_LAYERS).


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
_TOPK_MERGE4_KERNEL = str(Path(__file__).resolve().parent / "kernels" / "topk_merge4.cpp")
_TOPK_MERGE4_DM_KERNEL = str(Path(__file__).resolve().parent / "kernels" / "topk_merge4_dm.cpp")
_TOPK_FINAL_PACKED_KERNEL = str(Path(__file__).resolve().parent / "kernels" / "topk_final_packed.cpp")
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
    # grid try (run 8): two cores, one a 16-row half, each RISC 8 rows of it (kernels/draw_tail.cpp SPLIT8)
    SPLIT = True

    def __init__(self, device, batch, pad_token_id, eos_token_id):
        assert batch == ttnn.TILE_SIZE, batch
        self.device, self.batch = device, batch
        self.pad, self.eos = int(pad_token_id), int(eos_token_id)
        self.core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        self.cores2 = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])

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
        if self.SPLIT:
            cbs = [ttnn.CBDescriptor(total_size=size, core_ranges=self.cores2, format_descriptors=[fmt])]
            rt = [t.buffer_address() for t in io]
            kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=_DRAW_TAIL_KERNEL,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(half, 0), ttnn.CoreCoord(half, 0))]),
                    compile_time_args=[half, self.pad, self.eos, self.CB] + acc,
                    runtime_args=[((half, 0), rt)],
                    config=cfg,
                    defines=[("SPLIT8", "1"), ("SPLIT8_SUB", str(sub)), ("SPLIT8_SEM", "0")],
                )
                for half in (0, 1)
                for sub, cfg in ((0, ttnn.ReaderConfigDescriptor()), (1, ttnn.WriterConfigDescriptor()))
            ]
            sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=self.cores2, initial_value=0)]
            ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems))
            return drawn
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
    # The stock op's fused-key mode at every level (topk_local.cpp / topk_final.cpp ct arg 15, kernels/topk_mid.cpp
    # ct arg 13): right after the local transposes each value and its position become one packed [bf16 | u16] 32-bit
    # key that sorts, merges and travels as a single word (raw float32 tiles, 32-bit DEST); the final core splits them
    # back into the value and position outputs. Equal values then rank by position (the stock stable order).
    FUSED = True
    PACKED, PACKED_BYTES = ttnn.float32, 4096
    # TREE3 (fused keys): a 4-ary tree instead of 8 x 8 -- each column's 8 local cores report in two groups of 4 to
    # 16 first-level middle cores (rows 8 and 9, columns 0..7), those in groups of 4 (a row's columns 0..3 / 4..7) to 4
    # second-level middle cores (column 8, rows 0..3), and the final core (9, 0) merges those 4: 3 + 3 + 3 serial tile
    # merges instead of 7 + 7, one more hand-off. Directions alternate across the siblings of every level, as the
    # 8 x 8 tree's do, so each merge still sees bitonic pairs.
    TREE3 = True
    FANIN = 4
    # PACKED_OUT (TREE3, fused keys; packed()): the final core emits its merged packed key tile RAW (column layout,
    # float32 pages; kernels/topk_final_packed.cpp) instead of splitting it into value + position tiles, so ONE
    # tensor crosses the TP gather and MergeTopK.packed splits the keys itself.
    PACKED_OUT = True

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

    def _cb(self, index, tiles, dtype, cores, tile_bytes=TILE_BYTES):
        return ttnn.CBDescriptor(
            total_size=tiles * tile_bytes,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=tile_bytes)],
        )

    def _tree3(self, x, vals, ids, ht, wt, wt_local, packed_out=False):
        """TREE3 (fused keys): locals (gx, ly) -> first-level middle (gx, 8 + h) for ly in 4h..4h+3 -> second-level
        middle (8, j) for the first-level row 8 + j // 2, columns 4 (j % 2)..+3 -> final (9, 0) <- (8, 0..3). The same
        stock local / final kernels and kernels/topk_mid.cpp, wired by the same gather protocol (semaphores 0/1, 2/3,
        4/5 for the three hand-offs; a middle core receives on one pair and sends on the next)."""
        g, n, k, kt, f = self.GROUPS, self.LOCALS, self.k, self.kt, self.FANIN
        IN, IN_IDX, LT_VAL, LT_IDX, G_VAL, G_IDX, M_VAL, M_IDX, OUT_VAL, OUT_IDX = range(10)
        P, PB = self.PACKED, self.PACKED_BYTES
        rng = lambda x0, y0, x1, y1: ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))
        one = lambda c: ttnn.CoreRangeSet([rng(*c, *c)])
        phys = self.phys
        local_set = ttnn.CoreRangeSet([rng(0, 0, g - 1, n - 1)])
        mid1_set = ttnn.CoreRangeSet([rng(0, n, g - 1, n + 1)])
        mid2 = [(g, j) for j in range(4)]
        mid2_set = ttnn.CoreRangeSet([rng(g, 0, g, 3)])
        final = (g + 1, 0)
        final_set = one(final)
        mids = mid1_set.merge(mid2_set)
        mid_final = mids.merge(final_set)
        all_set = local_set.merge(mid_final)
        sending = local_set.merge(mids)
        # the gather buffers first on every core (written remotely at the sender's own address of them)
        cbs = [
            self._cb(G_VAL, f * kt, P, all_set, PB),
            self._cb(G_IDX, f * kt, self.VAL, all_set),
            self._cb(OUT_VAL, 2, P, sending, PB),
            self._cb(OUT_VAL, 2, P, final_set, PB) if packed_out else self._cb(OUT_VAL, 2, self.VAL, final_set),
            self._cb(OUT_IDX, 2, self.IDX, all_set),
            self._cb(IN, 4, self.VAL, local_set),
            self._cb(IN_IDX, 4, self.IDX, local_set),
            self._cb(LT_VAL, wt_local, P, local_set, PB),
            self._cb(M_VAL, f * kt, P, mid_final, PB),
            self._cb(M_IDX, f * kt, self.IDX, mid_final),
        ]
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=all_set, initial_value=0) for i in range(6)]
        compute_cfg = ttnn.ComputeConfigDescriptor()
        compute_cfg.fp32_dest_acc_en = True
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for i in (LT_VAL, G_VAL, M_VAL):
            modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_cfg.unpack_to_dest_mode = modes
        fdefs = [("TOPK_FUSED_STABLE_KEYS", "1")]
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
        logf = int(math.log2(f * kt))
        mid_ct = sort_ct(f * kt, G_VAL, G_IDX, M_VAL, M_IDX) + [logf, 1, 1]
        kernels = [
            kern(
                f"{_TOPK_KERNELS}/dataflow/reader_create_index_local_topk.cpp",
                local_set,
                [IN, IN_IDX, ht, wt_local, wt] + list(ttnn.TensorAccessorArgs(x).get_compile_time_args()),
                [((cx, cy), [x.buffer_address(), 0, i * wt_local, 0]) for i, (cx, cy) in enumerate(locals_)],
                ttnn.ReaderConfigDescriptor(),
                defines=[("GENERATE_INDICES", "1")],
            ),
            kern(
                f"{_TOPK_KERNELS}/compute/topk_local.cpp",
                local_set,
                sort_ct(wt_local, IN, IN_IDX, LT_VAL, LT_IDX) + [int(math.log2(wt_local)), 1, 1, 1, 1],
                [((cx, cy), [i % 2]) for i, (cx, cy) in enumerate(locals_)],
                compute_cfg,
            ),
            # both middle levels: a middle core's direction alternates across its siblings (column within its group of
            # 4 for the first level, j for the second)
            kern(
                _TOPK_MID_KERNEL,
                mids,
                mid_ct,
                [((gx, n + h), [gx % 2]) for h in range(2) for gx in range(g)] + [(c, [c[1] % 2]) for c in mid2],
                compute_cfg,
                compiler_include_paths=[f"{_TOPK_KERNELS}/compute"],
            ),
            # the second level sends to the final core (semaphores 4/5) ...
            kern(
                f"{_TOPK_KERNELS}/dataflow/writer_local_topk.cpp",
                mid2_set,
                [4, 5, phys(*final).x, phys(*final).y, ht, k, kt, OUT_VAL, OUT_IDX, G_VAL, G_IDX],
                [(c, [c[1]]) for c in mid2],
                ttnn.WriterConfigDescriptor(),
                defines=fdefs,
            ),
            # ... which releases them and collects their 4 results
            kern(
                f"{_TOPK_KERNELS}/dataflow/reader_final_topk.cpp",
                final_set,
                [4, 5, phys(*mid2[0]).x, phys(*mid2[0]).y, phys(*mid2[-1]).x, phys(*mid2[-1]).y]
                + [ht, f * kt, f, G_VAL, G_IDX],
                [],
                ttnn.ReaderConfigDescriptor(),
                defines=fdefs,
            ),
        ]
        if packed_out:  # the merged packed keys out raw, one tensor (vals)
            acc = list(ttnn.TensorAccessorArgs(vals).get_compile_time_args())
            kernels += [
                kern(
                    _TOPK_FINAL_PACKED_KERNEL,
                    final_set,
                    sort_ct(f * kt, G_VAL, G_IDX, M_VAL, M_IDX) + [logf, 1, 1, 1, 1],
                    [],
                    compute_cfg,
                    compiler_include_paths=[f"{_TOPK_KERNELS}/compute"],
                ),
                kern(
                    _TOPK_MERGE4_DM_KERNEL,
                    final_set,
                    [OUT_VAL, OUT_IDX, ht * kt] + acc + acc,
                    [(final, [vals.buffer_address(), vals.buffer_address()])],
                    ttnn.WriterConfigDescriptor(),
                    defines=[("MERGE_TOPK_WRITER", "1"), ("WRITE_ONE", "1")],
                ),
            ]
        else:
            kernels += [
                kern(
                    f"{_TOPK_KERNELS}/compute/topk_final.cpp",
                    final_set,
                    sort_ct(f * kt, G_VAL, G_IDX, M_VAL, M_IDX) + [logf, 1, 1, 1, 1],
                    [],
                    compute_cfg,
                ),
                kern(
                    f"{_TOPK_KERNELS}/dataflow/writer_final_topk.cpp",
                    final_set,
                    [OUT_VAL, OUT_IDX, ht, kt]
                    + list(ttnn.TensorAccessorArgs(vals).get_compile_time_args())
                    + list(ttnn.TensorAccessorArgs(ids).get_compile_time_args()),
                    [(final, [vals.buffer_address(), ids.buffer_address()])],
                    ttnn.WriterConfigDescriptor(),
                ),
            ]
        for gx in range(g):
            for h in range(2):
                m1 = (gx, n + h)
                top, bottom = phys(gx, f * h), phys(gx, f * h + f - 1)
                group = ttnn.CoreRangeSet([rng(gx, f * h, gx, f * h + f - 1)])
                # locals (gx, 4h..4h+3) -> first-level middle (gx, 8 + h), semaphores 0/1
                kernels.append(
                    kern(
                        f"{_TOPK_KERNELS}/dataflow/writer_local_topk.cpp",
                        group,
                        [0, 1, phys(*m1).x, phys(*m1).y, ht, k, kt, OUT_VAL, OUT_IDX, G_VAL, G_IDX],
                        [((gx, ly), [ly - f * h]) for ly in range(f * h, f * h + f)],
                        ttnn.WriterConfigDescriptor(),
                        defines=fdefs,
                    )
                )
                kernels.append(
                    kern(
                        f"{_TOPK_KERNELS}/dataflow/reader_final_topk.cpp",
                        one(m1),
                        [0, 1, top.x, top.y, bottom.x, bottom.y, ht, f * kt, f, G_VAL, G_IDX],
                        [],
                        ttnn.ReaderConfigDescriptor(),
                        defines=fdefs,
                    )
                )
        for j, m2 in enumerate(mid2):
            row, c0 = n + j // 2, f * (j % 2)
            # first-level middles (c0..c0+3, row) -> second-level middle (8, j), semaphores 2/3
            kernels.append(
                kern(
                    f"{_TOPK_KERNELS}/dataflow/writer_local_topk.cpp",
                    ttnn.CoreRangeSet([rng(c0, row, c0 + f - 1, row)]),
                    [2, 3, phys(*m2).x, phys(*m2).y, ht, k, kt, OUT_VAL, OUT_IDX, G_VAL, G_IDX],
                    [((cx, row), [cx - c0]) for cx in range(c0, c0 + f)],
                    ttnn.WriterConfigDescriptor(),
                    defines=fdefs,
                )
            )
            top, bottom = phys(c0, row), phys(c0 + f - 1, row)
            kernels.append(
                kern(
                    f"{_TOPK_KERNELS}/dataflow/reader_final_topk.cpp",
                    one(m2),
                    [2, 3, top.x, top.y, bottom.x, bottom.y, ht, f * kt, f, G_VAL, G_IDX],
                    [],
                    ttnn.ReaderConfigDescriptor(),
                    defines=fdefs,
                )
            )
        io = [x, vals] if packed_out else [x, vals, ids]
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems))

    def supports_packed(self, x):
        grid = self.device.compute_with_storage_grid_size()
        width = int(x.shape[-1])
        wt = width // 32
        return (
            self.PACKED_OUT
            and self.TREE3
            and self.FUSED
            and self.GROUPS == self.LOCALS == 8
            and grid.x >= 10
            and grid.y >= 10
            and int(x.shape[-2]) % 32 == 0
            and width <= 65536
            and wt % 64 == 0
            and (wt // 64) >= 2
            and (wt // 64) & (wt // 64 - 1) == 0
        )

    def packed(self, x):
        """(1, 1, B, k) float32 tensor of the merged packed [bf16 value | u16 position'] keys, column layout (see
        PACKED_OUT): the top k of each row, as the TREE3 program's final core holds them before its split."""
        rows, width = int(x.shape[-2]), int(x.shape[-1])
        ht, wt = rows // 32, width // 32
        wt_local = wt // (self.GROUPS * self.LOCALS)
        keys = ttnn.allocate_tensor_on_device(
            ttnn.Shape((1, 1, rows, self.k)), self.PACKED, ttnn.TILE_LAYOUT, self.device, ttnn.DRAM_MEMORY_CONFIG
        )
        self._tree3(x, keys, None, ht, wt, wt_local, packed_out=True)
        return keys

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
        grid = self.device.compute_with_storage_grid_size()
        if self.TREE3 and self.FUSED and g == n == 8 and grid.x >= 10 and grid.y >= 10:
            self._tree3(x, vals, ids, ht, wt, wt_local)
            return vals, ids

        # CB ids as in the stock factory. The gather buffers (c_4/c_5) are written remotely at the sender's own
        # address of them, so they go first on every core (same L1 address everywhere). A middle core reads that
        # address after its own gather wrapped the buffer, so the buffer holds exactly one gather (g == n).
        assert g == n
        IN, IN_IDX, LT_VAL, LT_IDX, G_VAL, G_IDX, M_VAL, M_IDX, OUT_VAL, OUT_IDX = range(10)
        fused = self.FUSED
        P, PB = self.PACKED, self.PACKED_BYTES
        if fused:
            # G_IDX / M_IDX are the final core's staging for the split values (bf16) / positions (u16)
            sending = self.local_set.merge(self.mid_set)
            cbs = [
                self._cb(G_VAL, n * kt, P, self.all_set, PB),
                self._cb(G_IDX, n * kt, self.VAL, self.all_set),
                self._cb(OUT_VAL, 2, P, sending, PB),
                self._cb(OUT_VAL, 2, self.VAL, self.final_set),
                self._cb(OUT_IDX, 2, self.IDX, self.all_set),
                self._cb(IN, 4, self.VAL, self.local_set),
                self._cb(IN_IDX, 4, self.IDX, self.local_set),
                self._cb(LT_VAL, wt_local, P, self.local_set, PB),
                self._cb(M_VAL, g * kt, P, self.mid_final_set, PB),
                self._cb(M_IDX, g * kt, self.IDX, self.mid_final_set),
            ]
        else:
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
        compute_cfg.fp32_dest_acc_en = fused  # packed 32-bit keys need 32-bit DEST
        if fused:
            # packed keys unpack straight to DEST (the 16-bit source registers would truncate them)
            modes = [ttnn.UnpackToDestMode.Default] * 64
            for i in (LT_VAL, G_VAL, M_VAL):
                modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
            compute_cfg.unpack_to_dest_mode = modes
        fdefs = [("TOPK_FUSED_STABLE_KEYS", "1")] if fused else []
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
                sort_ct(wt_local, IN, IN_IDX, LT_VAL, LT_IDX)
                + [int(math.log2(wt_local)), 1, 1, int(fused), int(fused)],
                [((cx, cy), [i % 2]) for i, (cx, cy) in enumerate(locals_)],
                compute_cfg,
            ),
            kern(
                _TOPK_MID_KERNEL,
                self.mid_set,
                sort_ct(n * kt, G_VAL, G_IDX, M_VAL, M_IDX) + [int(math.log2(n * kt)), 1, int(fused)],
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
                defines=fdefs,
            ),
            kern(
                f"{_TOPK_KERNELS}/dataflow/reader_final_topk.cpp",
                self.final_set,
                [2, 3, self.phys(0, n).x, self.phys(0, n).y, self.phys(g - 1, n).x, self.phys(g - 1, n).y]
                + [ht, g * kt, g, G_VAL, G_IDX],
                [],
                ttnn.ReaderConfigDescriptor(),
                defines=fdefs,
            ),
            kern(
                f"{_TOPK_KERNELS}/compute/topk_final.cpp",
                self.final_set,
                sort_ct(g * kt, G_VAL, G_IDX, M_VAL, M_IDX) + [int(math.log2(g * kt)), 1, 1, int(fused), int(fused)],
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
                    defines=fdefs,
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
                    defines=fdefs,
                )
            )
        ttnn.generic_op([x, vals, ids], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems))
        return vals, ids


class MergeTopK:
    """The top-K of the TP gathered VocabTopK lists, (1, 1, B, TP * K) bf16 values + uint32 vocab ids, as ONE
    single-core program (generic_op, kernels/topk_merge4.cpp).

    ttnn.topk sorts the gathered row from scratch (a full 64-wide bitonic sort per tile); each gathered tile is
    already a sorted run, so the kernel runs only the merge half of the bitonic algorithm (rebuild / merge /
    rebuild per pair of runs, then merge / rebuild of the two winners): 28 compare steps instead of 63.
    kernels/topk_merge4_dm.cpp moves the tiles a pair at a time under one barrier (the stock reader waits out one
    DRAM round trip per tile).

    scale (the sampler's 1 / temperature): the values leave as fp32 scores value * scale, the sampler's typecast +
    multiply folded into the program (the same SFPU fp32 multiply, bit-identical); None keeps bf16 values."""

    VAL, IDX = ttnn.bfloat16, ttnn.uint32
    IN, IN_IDX, MID_VAL, MID_IDX, FIN_VAL, FIN_IDX, OUT_VAL, OUT_IDX = range(8)
    RMT_VAL, RMT_IDX = 8, 9
    # grid try (run 8, packed keys): the two first-level pair merges on two cores at once -- the leaf core (0, 1)
    # merges the second pair and copies its winner into the root's (0, 0) rmt buffers (semaphore 0); the root merges
    # the first pair, then the two winners. Same steps on the same data: bit-identical.
    SPLIT = True

    def __init__(self, device, k, scale=None):
        self.device, self.k = device, k
        self.core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        self.leaf = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(0, 1))])
        self.both = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))])
        self.root_phys = device.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
        self.scale_bits = 0 if scale is None else struct.unpack("<I", struct.pack("<f", scale))[0]
        self.out_dtype = self.VAL if scale is None else ttnn.float32

    def supports_packed(self, keys):
        rows, width = int(keys.shape[-2]), int(keys.shape[-1])
        sw = getattr(self, "shard_width", 0)
        return (
            self.k == 32
            and rows == 32
            and width == 128
            and keys.dtype == ttnn.float32
            and keys.layout == ttnn.TILE_LAYOUT
            and sw > 0
            and sw & (sw - 1) == 0
            and sw <= 65536
            and 3 * sw < 2**31
        )

    def packed(self, keys):
        """The same top-K from the gathered VocabTopK.packed keys (1, 1, 32, TP * 32) float32 (column layout): the
        kernel splits each key (value word + position, the device's vocab offset OR-ed in) right before the merge."""
        wt = int(keys.shape[-1]) // 32
        out_shape = ttnn.Shape((1, 1, int(keys.shape[-2]), self.k))
        out_vals = ttnn.allocate_tensor_on_device(
            out_shape, self.out_dtype, ttnn.TILE_LAYOUT, self.device, ttnn.DRAM_MEMORY_CONFIG
        )
        out_ids = ttnn.allocate_tensor_on_device(
            out_shape, self.IDX, ttnn.TILE_LAYOUT, self.device, ttnn.DRAM_MEMORY_CONFIG
        )
        V, I = 2048, 4096
        cbs = [
            self._cb(self.IN, wt, self.IDX, I),  # raw 32-bit keys
            self._cb(self.IN_IDX, 1, self.IDX, I),  # unused here
            self._cb(self.MID_VAL, 2, self.VAL, V),
            self._cb(self.MID_IDX, 2, self.IDX, I),
            self._cb(self.FIN_VAL, 1, self.VAL, V),
            self._cb(self.FIN_IDX, 1, self.IDX, I),
            self._cb(self.OUT_VAL, 1, self.out_dtype, V if self.out_dtype == self.VAL else I),
            self._cb(self.OUT_IDX, 1, self.IDX, I),
        ]
        compute_cfg = ttnn.ComputeConfigDescriptor()
        compute_cfg.fp32_dest_acc_en = True
        kern = lambda src, ct, rt, cfg, **kw: ttnn.KernelDescriptor(
            kernel_source=src,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.core,
            compile_time_args=ct,
            runtime_args=[((0, 0), rt)],
            config=cfg,
            **kw,
        )
        acc = lambda *ts: [a for t in ts for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
        shard_log2 = int(self.shard_width).bit_length() - 1
        if self.SPLIT and wt == 4:
            return self._split_packed(keys, out_vals, out_ids, compute_cfg, acc, shard_log2)
        kernels = [
            kern(
                _TOPK_MERGE4_DM_KERNEL,
                [self.IN, self.IN_IDX, wt] + acc(keys, keys),
                [keys.buffer_address(), keys.buffer_address()],
                ttnn.ReaderConfigDescriptor(),
                defines=[("PACKED_IN", "1")],
            ),
            kern(
                _TOPK_MERGE4_KERNEL,
                list(range(8)) + [wt // 2, self.scale_bits, shard_log2],
                [],
                compute_cfg,
                defines=[("PACKED_IN", "1")],
            ),
            kern(
                _TOPK_MERGE4_DM_KERNEL,
                [self.OUT_VAL, self.OUT_IDX, 1] + acc(out_vals, out_ids),
                [out_vals.buffer_address(), out_ids.buffer_address()],
                ttnn.WriterConfigDescriptor(),
                defines=[("MERGE_TOPK_WRITER", "1")],
            ),
        ]
        ttnn.generic_op([keys, out_vals, out_ids], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
        return out_vals, out_ids

    def _split_packed(self, keys, out_vals, out_ids, compute_cfg, acc, shard_log2):
        V, I = 2048, 4096
        out_bytes = V if self.out_dtype == self.VAL else I
        cbs = [
            ttnn.CBDescriptor(
                total_size=tiles * size,
                core_ranges=self.both,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=size)],
            )
            for index, tiles, dtype, size in (
                (self.IN, 2, self.IDX, I),  # one pair of raw 32-bit keys
                (self.IN_IDX, 1, self.IDX, I),  # unused here
                (self.MID_VAL, 1, self.VAL, V),
                (self.MID_IDX, 1, self.IDX, I),
                (self.FIN_VAL, 1, self.VAL, V),
                (self.FIN_IDX, 1, self.IDX, I),
                (self.OUT_VAL, 1, self.out_dtype, out_bytes),
                (self.OUT_IDX, 1, self.IDX, I),
                (self.RMT_VAL, 1, self.VAL, V),
                (self.RMT_IDX, 1, self.IDX, I),
            )
        ]
        kern = lambda src, cores, xy, ct, rt, cfg, defs: ttnn.KernelDescriptor(
            kernel_source=src,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ct,
            runtime_args=[(xy, rt)],
            config=cfg,
            defines=defs,
        )
        rd = [self.IN, self.IN_IDX, 2] + acc(keys, keys)
        ka = keys.buffer_address()
        cct = list(range(8)) + [1, self.scale_bits, shard_log2]
        pk = ("PACKED_IN", "1")
        kernels = [
            kern(_TOPK_MERGE4_DM_KERNEL, self.core, (0, 0), rd, [ka, ka, 0, 0], ttnn.ReaderConfigDescriptor(),
                 [pk, ("MERGE_SPLIT", "1"), ("MERGE_ROOT", "1")]),
            kern(_TOPK_MERGE4_DM_KERNEL, self.leaf, (0, 1), rd, [ka, ka, 2], ttnn.ReaderConfigDescriptor(),
                 [pk, ("MERGE_SPLIT", "1")]),
            kern(_TOPK_MERGE4_KERNEL, self.core, (0, 0), cct, [], compute_cfg, [pk, ("MERGE_ROOT", "1")]),
            kern(_TOPK_MERGE4_KERNEL, self.leaf, (0, 1), cct, [], compute_cfg, [pk, ("MERGE_LEAF", "1")]),
            kern(_TOPK_MERGE4_DM_KERNEL, self.core, (0, 0), [self.OUT_VAL, self.OUT_IDX, 1] + acc(out_vals, out_ids),
                 [out_vals.buffer_address(), out_ids.buffer_address()], ttnn.WriterConfigDescriptor(),
                 [("MERGE_TOPK_WRITER", "1")]),
            kern(_TOPK_MERGE4_DM_KERNEL, self.leaf, (0, 1), [self.MID_VAL, self.MID_IDX, 1] + acc(out_vals, out_ids),
                 [int(self.root_phys.x), int(self.root_phys.y), 0], ttnn.WriterConfigDescriptor(),
                 [("MERGE_LEAF_SEND", "1")]),
        ]
        sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=self.both, initial_value=0)]
        ttnn.generic_op([keys, out_vals, out_ids], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems))
        return out_vals, out_ids

    def supports(self, vals, ids):
        rows, width = int(vals.shape[-2]), int(vals.shape[-1])
        return (
            self.k == 32
            and rows == 32
            and width == 128
            and vals.dtype == self.VAL
            and ids.dtype == self.IDX
            and tuple(ids.shape) == tuple(vals.shape)
            and vals.layout == ids.layout == ttnn.TILE_LAYOUT
        )

    def _cb(self, index, tiles, dtype, tile_bytes):
        return ttnn.CBDescriptor(
            total_size=tiles * tile_bytes,
            core_ranges=self.core,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=tile_bytes)],
        )

    def __call__(self, vals, ids):
        wt = int(vals.shape[-1]) // 32
        out_shape = ttnn.Shape((1, 1, int(vals.shape[-2]), self.k))
        out_vals = ttnn.allocate_tensor_on_device(
            out_shape, self.out_dtype, ttnn.TILE_LAYOUT, self.device, ttnn.DRAM_MEMORY_CONFIG
        )
        out_ids = ttnn.allocate_tensor_on_device(
            out_shape, self.IDX, ttnn.TILE_LAYOUT, self.device, ttnn.DRAM_MEMORY_CONFIG
        )
        V, I = 2048, 4096  # bf16 / uint32 (and fp32) tile bytes
        cbs = [
            self._cb(self.IN, wt, self.VAL, V),
            self._cb(self.IN_IDX, wt, self.IDX, I),
            self._cb(self.MID_VAL, 2, self.VAL, V),
            self._cb(self.MID_IDX, 2, self.IDX, I),
            self._cb(self.FIN_VAL, 1, self.VAL, V),
            self._cb(self.FIN_IDX, 1, self.IDX, I),
            self._cb(self.OUT_VAL, 1, self.out_dtype, V if self.out_dtype == self.VAL else I),
            self._cb(self.OUT_IDX, 1, self.IDX, I),
        ]
        compute_cfg = ttnn.ComputeConfigDescriptor()
        compute_cfg.fp32_dest_acc_en = True  # uint32 ids ride the swaps in 32-bit DEST (as ttnn.topk)
        kern = lambda src, ct, rt, cfg, **kw: ttnn.KernelDescriptor(
            kernel_source=src,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.core,
            compile_time_args=ct,
            runtime_args=[((0, 0), rt)],
            config=cfg,
            **kw,
        )
        acc = lambda *ts: [a for t in ts for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]
        kernels = [
            # a pair of gathered tiles (values + ids) in flight under one barrier, as the compute consumes them
            kern(
                _TOPK_MERGE4_DM_KERNEL,
                [self.IN, self.IN_IDX, wt] + acc(vals, ids),
                [vals.buffer_address(), ids.buffer_address()],
                ttnn.ReaderConfigDescriptor(),
            ),
            kern(_TOPK_MERGE4_KERNEL, list(range(8)) + [wt // 2, self.scale_bits], [], compute_cfg),
            kern(
                _TOPK_MERGE4_DM_KERNEL,
                [self.OUT_VAL, self.OUT_IDX, 1] + acc(out_vals, out_ids),
                [out_vals.buffer_address(), out_ids.buffer_address()],
                ttnn.WriterConfigDescriptor(),
                defines=[("MERGE_TOPK_WRITER", "1")],
            ),
        ]
        ttnn.generic_op([vals, ids, out_vals, out_ids], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
        return out_vals, out_ids


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
            # the merge also emits the sampler's fp32 scores (value / temperature), see sample()
            self._merge_topk = MergeTopK(device, self.top_k, scale=1.0 / self.temperature)
            self._merge_topk.shard_width = vocab // tp  # VocabTopK.packed positions are within a device's shard
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

    # Tile-layout uploads up to this many elements are tilized on the host: the per-request positions / pos /
    # unfinished columns otherwise each cost a device TilizeWithValPadding launch (one of them on a single core).
    HOST_TILIZE_MAX = 1 << 16

    def _put(self, t, dtype, layout):
        t = t.contiguous()
        if layout == ttnn.TILE_LAYOUT and t.numel() <= self.HOST_TILIZE_MAX:
            return ttnn.to_device(ttnn.from_torch(t, dtype=dtype, layout=layout, mesh_mapper=self._repl), self.device)
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=self.device, mesh_mapper=self._repl)

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
        if vals.dtype == ttnn.float32:  # MergeTopK already emitted the scores (value / temperature)
            scores = vals
        else:
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
        if self._vocab_topk.supports_packed(local) and int(local.shape[-1]) == self._merge_topk.shard_width:
            # ONE gather: each device's packed [value | position] key tile (VocabTopK.packed); MergeTopK.packed splits
            # the keys and adds the devices' vocab offsets itself (no id gather, typecast or offset add)
            keys = ttnn.all_gather(self._vocab_topk.packed(local), dim=3, num_links=1, topology=ttnn.Topology.Linear)
            if self._merge_topk.supports_packed(keys):
                return self._merge_topk.packed(keys)
        vals, ids = self._vocab_topk(local)
        vals = ttnn.all_gather(vals, dim=3, num_links=1, topology=ttnn.Topology.Linear)
        ids = ttnn.all_gather(ids, dim=3, num_links=1, topology=ttnn.Topology.Linear)
        ids = ttnn.add(ttnn.typecast(ids, ttnn.uint32), self._shard_base)
        if self._merge_topk.supports(vals, ids):
            # the gathered lists are sorted runs: merge them (MergeTopK) instead of sorting the row again
            return self._merge_topk(vals, ids)
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

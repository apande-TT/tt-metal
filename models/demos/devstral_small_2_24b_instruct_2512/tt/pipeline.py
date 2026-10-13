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
PLANNED_TP = 4
# Performance levers applied inside _stubs/ (a symlink out of this directory, so they are listed here too):
#   ministral3_m_l_p: gate/up folded into one tile-interleaved weight per device; minimal_matmul(fuse_swiglu)
#                     emits silu(gate) * up in the same op (prefill/encode); decode keeps separate gate/up
#                     with the silu fused into the multiply. Prefill MLP chunk 4096 -> 16384 rows (one call).
#   decoder_head: keeps its pre-gather vocab shard (.local) for the per-device sampler top-k (DevstralPipeline._topk).


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
# Prompts are right-padded to a multiple of this many positions (capped at the KV capacity) so prefill SDPA
# can run 512-wide key chunks: narrower key chunks bias attention's output level (see ministral3_attention).
# The padded rows come after every real row, so causal attention leaves rows 0..L-1 unchanged.
PROMPT_PAD = 512

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
    """Prefill length: real_len rounded up to PROMPT_PAD, never past the resident KV capacity (tile multiple)."""
    padded = int(math.ceil(real_len / PROMPT_PAD) * PROMPT_PAD)
    return padded if padded <= kv_capacity else int(math.ceil(real_len / 32) * 32)


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
        self.vision_tower = _st_vision_model.TtPixtralVisionModel(device, m.vision_tower, n_layers=self.vision_layers)
        self.multi_modal_projector = _st_projector.TtMistral3MultiModalProjector(device, m.multi_modal_projector)
        log(f"building language model ({self.text_layers} layers, TP={self._tp()})")
        self.language_model = _st_ministral3_model.TtMinistral3Model(
            device, m.language_model, n_layers=self.text_layers
        )
        self.lm_head = _st_decoder_head.DecoderHead(device, hf_model.lm_head)
        attn0 = self.language_model.layers[0].self_attn
        self.kv_shape = [self.batch, attn0.local_kv_heads, self.kv_capacity, attn0.head_dim]
        self.kv_caches = [
            (
                ttnn.zeros(self.kv_shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
                ttnn.zeros(self.kv_shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
            )
            for _ in self.language_model.layers
        ]
        self._stage = {}
        self._decode_repeats = None
        # Global vocab id of every column of each device's lm_head shard, one row per batch row: the per-device
        # top-k returns these labels directly (ttnn.topk indices_tensor), so the TP candidate lists merge with
        # one more top-k and no index arithmetic.
        self._vocab_ids = None
        if self._tp() > 1:
            vocab = int(tcfg.vocab_size)
            self._vocab_ids = ttnn.from_torch(
                torch.arange(vocab, dtype=torch.int32).reshape(1, 1, 1, vocab).expand(1, 1, self.batch, vocab).contiguous(),
                dtype=ttnn.uint32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ShardTensorToMesh(device, dim=-1),
            )
        log(f"pipeline resident in {time.time() - t0:.1f}s (KV cache {self.kv_shape} per layer, bf16)")

    def _tp(self):
        try:
            return int(self.device.get_num_devices())
        except Exception:
            return 1

    # ------------------------------------------------------------------ host -> device (outside any forward)

    def _put(self, t, dtype, layout):
        return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, device=self.device, mesh_mapper=self._repl)

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
            "noise": self._put(
                gumbel_table(req["seeds"], steps, NOISE_BUCKETS).reshape(steps, b * NOISE_BUCKETS),
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
            ),
        }
        if "pixel_values" in req:
            sizes = req["image_sizes"]
            dev["image_sizes"] = sizes
            dev["pixel_values"] = self._put(
                req["pixel_values"].to(torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT
            )
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
            dev["pixel_values"], image_sizes=dev["image_sizes"], position_ids=dev["patch_positions"]
        )
        out = self.multi_modal_projector(feats, image_sizes=dev["image_sizes"])
        b, d = self.batch, out.shape[-1]
        return ttnn.reshape(out, (b, 1, dev["tokens_per_image"], d))

    def embed_prompt(self, dev, image_embeds=None):
        """(B, 1, S, hidden) prompt embeddings; image features land on the [IMG] rows through an on-device gather."""
        x = self.language_model.embed_tokens(dev["tokens"])  # (B, S, D)
        b, s, d = x.shape
        if image_embeds is None:
            return ttnn.reshape(x, (b, 1, s, d))
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
        last = ttnn.slice(hidden, (0, dev["real_len"] - 1, 0), (b, dev["real_len"], d))
        logits = self.lm_head(ttnn.reshape(last, (1, 1, b, d)))
        return hidden, logits

    def sample(self, logits, state):
        """On-device seeded sampling (Gumbel-max over the top-K of logits / T), EOS bookkeeping, write-back.

        Writes the next input ids into state["tok"] and the finished mask into state["unfinished"] IN PLACE and
        returns the (1, 1, B, 1) uint32 ids the model itself drew (before pad substitution)."""
        b, k, nb = self.batch, self.top_k, NOISE_BUCKETS
        vals, idx = self._topk(logits)
        scores = ttnn.multiply(ttnn.typecast(vals, ttnn.float32), 1.0 / self.temperature)
        noise = ttnn.embedding(state["step"], state["noise"], layout=ttnn.TILE_LAYOUT)  # (1, 1, B*NB) bf16
        noise = ttnn.reshape(ttnn.typecast(noise, ttnn.float32), (1, 1, b, nb))
        ids = ttnn.typecast(idx, ttnn.float32)  # exact: ids < 2^24
        bucket = ttnn.subtract(ids, ttnn.multiply(ttnn.floor(ttnn.multiply(ids, 1.0 / nb)), float(nb)))
        scores = ttnn.add(scores, ttnn.gather(noise, dim=-1, index=ttnn.typecast(bucket, ttnn.uint32)))
        pick = ttnn.to_layout(ttnn.argmax(scores, dim=-1, keepdim=True), ttnn.TILE_LAYOUT)
        drawn = ttnn.gather(idx, dim=-1, index=pick)  # (1, 1, B, 1) uint32
        # generate(): finished rows emit pad; a row finishes on the step it emits eos.
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
        if self._vocab_ids is None or local is None or tuple(local.shape) != tuple(self._vocab_ids.shape):
            return ttnn.topk(logits, k=k, dim=-1, largest=True, sorted=True)
        vals, ids = ttnn.topk(local, k=k, dim=-1, largest=True, sorted=True, indices_tensor=self._vocab_ids)
        vals = ttnn.all_gather(vals, dim=3, num_links=1, topology=ttnn.Topology.Linear)
        ids = ttnn.all_gather(ids, dim=3, num_links=1, topology=ttnn.Topology.Linear)
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
        logits = self.lm_head(hidden)
        ttnn.plus_one(state["cur_pos"])
        ttnn.add(state["pos"], 1.0, output_tensor=state["pos"])
        state["drawn"] = self.sample(logits, state)
        state["logits"] = logits
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
            "pixel_values": self._put(pv.to(torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
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
        return self.decode_step(self._stage["decode"])["logits"]

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


def host_read(t, device):
    """Device -> host read of a replicated tensor (the first replica)."""
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

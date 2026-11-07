# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash image-text-to-text on a 32-chip Blackhole Galaxy (TP=32 x DP=1) -- the ONE shared
TTNN pipeline that both ``demo/demo_image_text_to_text.py`` and ``tests/e2e`` call.

Chain (every arrow is a graduated ``_stubs`` port running on device; nothing in between is host math):

    pixel_values --vision_model[ patch_embed -> rotary -> 24 x vision_block(r_m_s_norm, vision_attention,
                   vision_m_l_p) -> r_m_s_norm -> downsample -> vision_patch_merger ]--> image embeddings
    input_ids --ttnn.embedding--> token embeddings; image rows replaced by the image embeddings (on device)
    --text_model[ 45 x text_decoder_layer(text_hyper_connection(text_unweighted_r_m_s_norm), text_r_m_s_norm,
                  text_linear_attention(text_forget_gate, text_r_m_s_norm_gated) | text_attention(text_indexer),
                  text_m_l_p | text_mo_e(text_topk_router, text_experts, text_m_l_p)),
                  text_hyper_head, text_r_m_s_norm ]--> hidden --lm_head--> logits --ttnn.argmax--> token
    decode: token --ttnn.embedding--> text_model (resident KDA recurrent/conv state + DSA KV cache) --> ...

At build, the host reads the FP8 checkpoint one block at a time (``tt/checkpoint.py``; the bf16 model, ~640 GB,
is larger than host RAM) and uploads each block ONCE; every weight then stays resident on device for the whole
run, sharded over 32 chips (~20 GB/chip). No weight moves host->device during a forward.
"""

from __future__ import annotations

import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

import ttnn
from models.demos.glm_5_3_flash._stubs.text_decoder_layer import Step, _rs
from models.demos.glm_5_3_flash._stubs.text_model import TtTextModel
from models.demos.glm_5_3_flash._stubs.vision_attention import MAX_TOKENS as VISION_MAX_TOKENS
from models.demos.glm_5_3_flash._stubs.vision_model import TtVisionModel
from models.demos.glm_5_3_flash.tt.checkpoint import HF_MODEL_ID, CheckpointReader

# Stages derived from the HF config: Glm5NextForConditionalGeneration is a decoder-only VLM
# (is_encoder_decoder=False) with a vision sub-config -> encode the image, then prefill + decode.
PIPELINE_STAGES = ["vision_encode", "prefill", "decode"]

# The published example (Glm5NextForConditionalGeneration.forward docstring, transformers glm5_next).
EXAMPLE_IMAGE_URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)
EXAMPLE_PROMPT = "Describe the image."
# Optional processor cap on image tokens. None = the processor's own default, i.e. the published example
# exactly as the model card runs it (893 prompt tokens); any prompt up to index_topk is supported.
DEFAULT_MAX_IMAGE_TOKENS = None
# Input SIZE the pipeline is built and gated for: the side of the equivalent square (the image is scaled to
# that area keeping its aspect ratio; the processor's own resizing rule then decides the exact grid). The
# perf harness's area variable overrides it at run time; 0 = the image exactly as given.
DEFAULT_AREA = 1024
CHUNK = 64  # KDA chunk; prefill length is padded to a multiple of it
DEFAULT_BATCH = 32  # samples per call (one 32-row tile of every projection)
# Token rows (samples x padded prompt) one prefill pass carries. Samples are independent in prefill, so a batch
# whose B x S exceeds this runs as sub-batches whose decode state is concatenated on the batch axis (exact).
# 32 x 128 is the largest prefill the 1x32 mesh ran with the weights resident (the fp32 residual streams alone
# are B*S x hc x D); a larger prompt keeps the batch by splitting it rather than running out of DRAM.
PREFILL_TOKENS = 32 * 128
TRACE_REGION_SIZE = 1_500_000_000  # bytes; the 45-layer decode step (the largest traced stage) overflows 200 MB


def _log(msg):
    print(f"[glm5.3-flash] {msg}", flush=True)


# ----------------------------------------------------------------------------- stub invocation ledger
STUB_ENTRYPOINTS = {
    # graduated component -> (module, class, methods that ARE that component's forward)
    "text_model": ("text_model", "TtTextModel", ["forward", "__call__"]),
    "text_decoder_layer": ("text_decoder_layer", "TtTextDecoderLayer", ["forward", "__call__"]),
    "text_hyper_connection": ("text_hyper_connection", "TtTextHyperConnection", ["streams", "__call__"]),
    "text_unweighted_r_m_s_norm": ("text_unweighted_r_m_s_norm", "TtTextUnweightedRMSNorm", ["__call__"]),
    "text_r_m_s_norm": ("text_r_m_s_norm", "TtTextRMSNorm", ["__call__"]),
    "text_linear_attention": ("text_linear_attention", "TtTextLinearAttention", ["prefill", "decode", "__call__"]),
    "text_forget_gate": ("text_forget_gate", "TtTextForgetGate", ["local", "__call__"]),
    "text_r_m_s_norm_gated": ("text_r_m_s_norm_gated", "TtTextRMSNormGated", ["__call__"]),
    "text_attention": ("text_attention", "TtTextAttention", ["prefill", "decode", "__call__"]),
    "text_indexer": ("text_indexer", "TtTextIndexer", ["__call__"]),
    "text_m_l_p": ("text_m_l_p", "TtTextMLP", ["__call__"]),
    "text_mo_e": ("text_mo_e", "TtTextMoE", ["__call__"]),
    "text_topk_router": ("text_topk_router", "TtTextTopkRouter", ["__call__"]),
    "text_experts": ("text_experts", "TtTextExperts", ["__call__"]),
    "text_hyper_head": ("text_hyper_head", "TtTextHyperHead", ["streams", "__call__"]),
    "vision_model": ("vision_model", "TtVisionModel", ["__call__"]),
    "vision_patch_embed": ("vision_patch_embed", "TtVisionPatchEmbed", ["__call__"]),
    "vision_rotary_embedding": ("vision_rotary_embedding", "TtVisionRotaryEmbedding", ["__call__"]),
    "vision_block": ("vision_block", "TtVisionBlock", ["__call__"]),
    "r_m_s_norm": ("r_m_s_norm", "TtRMSNorm", ["__call__"]),
    "vision_attention": ("vision_attention", "TtVisionAttention", ["__call__"]),
    "vision_m_l_p": ("vision_m_l_p", "TtVisionMLP", ["__call__"]),
    "vision_patch_merger": ("vision_patch_merger", "TtVisionPatchMerger", ["__call__"]),
}
STUB_CALLS: Dict[str, int] = defaultdict(int)
_INSTRUMENTED = False


def instrument_stubs():
    """Count every call of every graduated component's forward (Gate 2). Subclass calls (TtTextRMSNorm
    is a TtRMSNorm) are attributed to the most specific graduated class."""
    global _INSTRUMENTED
    if _INSTRUMENTED:
        return STUB_CALLS
    import importlib

    owner = {}
    for name, (mod, cls, _) in STUB_ENTRYPOINTS.items():
        owner[getattr(importlib.import_module(f"models.demos.glm_5_3_flash._stubs.{mod}"), cls)] = name
    for name, (mod, cls, methods) in STUB_ENTRYPOINTS.items():
        klass = getattr(importlib.import_module(f"models.demos.glm_5_3_flash._stubs.{mod}"), cls)
        for meth in methods:
            if meth not in klass.__dict__:
                continue
            orig = klass.__dict__[meth]

            def wrapped(self, *a, __orig=orig, **k):
                STUB_CALLS[owner.get(type(self), "?")] += 1
                return __orig(self, *a, **k)

            setattr(klass, meth, wrapped)
    _INSTRUMENTED = True
    return STUB_CALLS


def reset_stub_calls():
    STUB_CALLS.clear()


# ----------------------------------------------------------------------------- inputs
def example_messages(prompt: str = EXAMPLE_PROMPT, image_url: str = EXAMPLE_IMAGE_URL):
    return [{"role": "user", "content": [{"type": "image", "image": image_url}, {"type": "text", "text": prompt}]}]


def input_area() -> int:
    """The input size to run at: the harness's area variable when set, else DEFAULT_AREA."""
    from models.experimental.perf_automation.agent.perf_adapter import AREA_ENV

    return int(os.environ.get(AREA_ENV) or DEFAULT_AREA)


def sized_image(image_url, area: int):
    """The image (URL, path or PIL) scaled to area x area pixels in total, aspect ratio kept."""
    from PIL import Image
    from transformers.image_utils import load_image

    im = load_image(image_url)
    s = area / math.sqrt(im.width * im.height)
    return im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.Resampling.BICUBIC)


def load_processor(model_id: str = HF_MODEL_ID):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_id)


def encode_example(
    processor=None,
    prompt: str = EXAMPLE_PROMPT,
    image_url: str = EXAMPLE_IMAGE_URL,
    max_image_tokens: Optional[int] = DEFAULT_MAX_IMAGE_TOKENS,
    chat_kwargs: Optional[dict] = None,
    area: Optional[int] = None,
):
    """HF processor over the published example -> dict(input_ids [1,L], pixel_values [N,1176], image_grid_thw [1,3]).
    The image is first scaled to ``area`` (None: input_area(); 0: as given)."""
    processor = processor or load_processor()
    area = input_area() if area is None else int(area)
    image = sized_image(image_url, area) if area else image_url
    enc = processor.apply_chat_template(
        example_messages(prompt, image),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        **(
            {}
            if max_image_tokens is None
            else {"processor_kwargs": {"images_kwargs": {"max_image_tokens": int(max_image_tokens)}}}
        ),
        **(chat_kwargs or {}),
    )
    return {k: enc[k] for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}


def _cat_tree(trees):
    """Concatenate matching (nested tuple/list) state trees of sub-batches on the batch axis (dim 0)."""
    if isinstance(trees[0], (tuple, list)):
        return type(trees[0])(_cat_tree(list(parts)) for parts in zip(*trees))
    return ttnn.concat(list(trees), dim=0)


@dataclass
class Encoded:
    """Host-side encoding of one batch (B samples of the same prompt/image), uploaded once."""

    B: int
    real_len: int
    S: int
    image_span: tuple
    input_ids: torch.Tensor  # [B, real_len] (host copy, for the reference)
    pixel_values: torch.Tensor  # [N, 1176] one image
    image_grid_thw: torch.Tensor  # [1, 3]
    dev: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------- pipeline
class Glm5FlashPipeline:
    """Resident TT pipeline. Construct through ``build_pipeline``."""

    def __init__(
        self,
        mesh_device,
        model_id: str = HF_MODEL_ID,
        layers: Optional[int] = None,
        vision_encode_layers: Optional[int] = None,
        prefill_layers: Optional[int] = None,
        decode_layers: Optional[int] = None,
        batch: int = 1,
        capacity: Optional[int] = None,
    ):
        self.mesh = mesh_device
        self.model_id = model_id
        self.ckpt = CheckpointReader(model_id)
        self.config = self.ckpt.config
        tcfg, vcfg = self.config.text_config, self.config.vision_config
        self.tcfg, self.vcfg = tcfg, vcfg
        self.batch = int(batch)
        # every repeated stack: vision blocks (vision_encode) and the text decoder (prefill + decode share it)
        pick = lambda v, full: full if v is None else max(1, min(int(v), full))
        n_vis = pick(vision_encode_layers if vision_encode_layers is not None else layers, int(vcfg.depth))
        text_req = {v for v in (prefill_layers, decode_layers) if v is not None}
        if len(text_req) > 1:
            raise ValueError("prefill and decode run the same text stack: prefill_layers must equal decode_layers")
        n_text = pick(text_req.pop() if text_req else layers, int(tcfg.num_hidden_layers))
        self.n_vision_layers, self.n_text_layers = n_vis, n_text
        # Index-topk identity: for a context <= index_topk the DSA selection is the causal set (decode relies on it).
        self.capacity = int(capacity or tcfg.index_topk)
        if self.capacity > int(tcfg.index_topk):
            raise ValueError(f"capacity {self.capacity} > index_topk {tcfg.index_topk}")
        from transformers import GenerationConfig

        self.generation_config = GenerationConfig.from_pretrained(model_id)
        self.eos_ids = [
            int(e) for e in (self.generation_config.eos_token_id or self.config.text_config.eos_token_id or [])
        ]
        self.image_token_id = int(self.config.image_token_id)
        # HF reference structure kept reachable (meta-device skeleton: no weight memory) -- ground truth for the
        # section structure: hf.model.visual.blocks (24) and hf.model.language_model.layers (45).
        from models.common.hf_reference_loading import build_hf_skeleton

        self.hf = build_hf_skeleton(model_id, ("AutoModelForImageTextToText",))
        self._build()

    # ------------------------------------------------------------------ build
    def _rep(self, t, dtype, layout):
        return ttnn.from_torch(
            t.contiguous(),
            dtype=dtype,
            layout=layout,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _build(self):
        t0 = time.time()
        vm = self.ckpt.vision_model(depth=self.n_vision_layers)
        self.vision = TtVisionModel(self.mesh, vm)
        del vm
        _log(f"vision tower built ({self.n_vision_layers} blocks) in {time.time() - t0:.0f}s")

        def layer_iter():
            for i in range(self.n_text_layers):
                yield self.ckpt.text_layer(i)

        self.text = TtTextModel.from_layers(
            self.mesh, layer_iter(), self.ckpt.final_norm(), self.tcfg.hc_mult, progress=_log
        )
        # embeddings / lm_head stay replicated (TP rule: embeddings, norms, lm_head replicated)
        self.embed_w = self._rep(self.ckpt.embed_weight(), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        self.lm_head_w = self._rep(self.ckpt.lm_head_weight().t(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self.D = int(self.tcfg.hidden_size)
        self.V = int(self.tcfg.vocab_size)
        _log(
            f"pipeline built: {self.n_text_layers} text layers, {self.n_vision_layers} vision blocks in {time.time() - t0:.0f}s"
        )

    # ------------------------------------------------------------------ encoding (host, outside the model math)
    def encode(self, enc: dict, batch: Optional[int] = None) -> Encoded:
        """Replicate one processed sample B times and upload everything the forward needs (ids, patches,
        grid, padding mask, shape constants). Pure input preparation."""
        B = int(batch or self.batch)
        ids = enc["input_ids"]
        if ids.shape[0] != 1:
            raise ValueError("encode() takes one processed sample")
        L0 = int(ids.shape[1])
        img = (ids[0] == self.image_token_id).nonzero().flatten()
        a, R1 = int(img[0]), int(img.numel())
        if not bool((img == torch.arange(a, a + R1)).all()):
            raise NotImplementedError("one contiguous image span per prompt")
        S = -(-L0 // CHUNK) * CHUNK
        pad_id = int(self.tcfg.pad_token_id)
        ids_b = ids.repeat(B, 1)
        ids_pad = torch.full((B, S), pad_id, dtype=torch.long)
        ids_pad[:, :L0] = ids_b
        grid = enc["image_grid_thw"].clone()
        pv = enc["pixel_values"]
        e = Encoded(B=B, real_len=L0, S=S, image_span=(a, R1), input_ids=ids_b, pixel_values=pv, image_grid_thw=grid)
        rep = lambda t, dt, lay: self._rep(t, dt, lay)
        # Same-size images packed as frames of one grid (attention block-diagonal per frame == separate images),
        # in groups whose patches fit the vision attention port (images are independent, so grouping is exact).
        T, gh, gw = (int(v) for v in grid[0])
        per_group = max(1, VISION_MAX_TOKENS // (T * gh * gw))
        sizes = [min(per_group, B - g0) for g0 in range(0, B, per_group)]
        uploaded = {}
        for n in sorted(set(sizes)):
            grid_n = torch.tensor([[n * T, gh, gw]], dtype=torch.float32).reshape(1, 1, 1, 3)
            uploaded[n] = dict(
                pixel_values=rep(
                    pv.repeat(n, 1).float(), ttnn.float32, ttnn.TILE_LAYOUT
                ),  # fp32 as the processor emits
                grid=rep(grid_n, ttnn.float32, ttnn.TILE_LAYOUT),
                frames=n * T,  # python int: the vision forward must not index host tensors
            )
        e.dev["vision_groups"] = [uploaded[n] for n in sizes]
        e.dev["input_ids"] = rep(ids_pad.to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        valid = torch.zeros(B, S)
        valid[:, :L0] = 1
        e.dev["valid"] = rep(valid.reshape(1, 1, B * S, 1), ttnn.float32, ttnn.TILE_LAYOUT)
        e.dev["pos0"] = rep(torch.full((1, 1, 1, 1), float(L0)), ttnn.float32, ttnn.TILE_LAYOUT)
        # per-shape constants of the sequence-structured ports, uploaded here (outside the forward)
        for layer in self.text.layers:
            if layer.kind == "linear_attention":
                layer.self_attn.kda.prepare(B, S)
            else:
                layer.self_attn.prepare(S, self.capacity)
                layer.self_attn.prepare(self.capacity, self.capacity)
        return e

    # ------------------------------------------------------------------ stages
    def vision_encode(self, e: Encoded):
        """pixel patches -> image embeddings [B*R, D] fp32 (replicated)."""
        pooled = [
            self.vision(g["pixel_values"], grid_thw=g["grid"], frames=g["frames"])[1] for g in e.dev["vision_groups"]
        ]
        return pooled[0] if len(pooled) == 1 else ttnn.concat(pooled, dim=len(pooled[0].shape) - 2)

    def embed_prompt(self, e: Encoded, image_embeds):
        """Token embeddings with the image rows replaced by the image embeddings, all on device -> [1,1,B*S,D]."""
        B, S, D = e.B, e.S, self.D
        a, R1 = e.image_span
        tok = ttnn.embedding(e.dev["input_ids"], self.embed_w)  # [B,S,D] bf16 row-major
        img = ttnn.reshape(
            ttnn.to_layout(ttnn.typecast(image_embeds, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT), [B, R1, D]
        )
        pieces = []
        if a > 0:
            pieces.append(ttnn.slice(tok, [0, 0, 0], [B, a, D]))
        pieces.append(img)
        if a + R1 < S:
            pieces.append(ttnn.slice(tok, [0, a + R1, 0], [B, S, D]))
        x = ttnn.concat(pieces, dim=1) if len(pieces) > 1 else pieces[0]
        return ttnn.to_layout(ttnn.reshape(x, [1, 1, B * S, D]), ttnn.TILE_LAYOUT)

    def _logits(self, h):
        """h [1,1,B,D] fp32 -> logits [1,1,B,V] fp32 (replicated lm_head)."""
        return ttnn.linear(ttnn.typecast(h, ttnn.bfloat16), self.lm_head_w, dtype=ttnn.float32)

    @staticmethod
    def _argmax(logits):
        """Greedy token per sample on device, over the fp32 logits (rounding them to bf16 would manufacture ties
        the pipeline's fp32 activations do not have)."""
        B = logits.shape[-2]
        tok = ttnn.argmax(logits, dim=-1)  # [1,1,B] uint32, first index among exact ties (as torch)
        return ttnn.reshape(tok, [1, B])

    def _sample(self, logits, noise, sampling: "Sampling"):
        """On-device top-p + temperature + Gumbel-max draw. Top-p keeps token i iff the probability mass of
        the tokens strictly more likely than i is < top_p (HF TopPLogitsWarper); the threshold is found by
        bisection on the probability value, so no vocabulary sort is needed."""
        B = logits.shape[-2]
        x = logits if sampling.temperature == 1.0 else ttnn.multiply(logits, 1.0 / sampling.temperature)
        if sampling.top_p < 1.0:
            p = ttnn.softmax(x, dim=-1, numeric_stable=True)
            hi = ttnn.max(p, dim=-1, keepdim=True)  # [1,1,B,1]
            lo = ttnn.multiply(hi, 0.0)
            for _ in range(TOP_P_BISECT_STEPS):
                mid = ttnn.multiply(ttnn.add(lo, hi), 0.5)
                mass = ttnn.sum(ttnn.multiply(p, ttnn.gt(p, mid)), dim=-1, keepdim=True)
                ok = ttnn.lt(mass, sampling.top_p)
                hi = ttnn.where(ok, mid, hi)
                lo = ttnn.where(ok, lo, mid)
            kept = ttnn.gt(p, lo)
            x = ttnn.add(x, ttnn.multiply(ttnn.subtract(kept, 1.0), 1e9))
        tok = ttnn.argmax(ttnn.add(x, noise), dim=-1)
        return ttnn.reshape(tok, [1, B])

    def choose(self, logits, noise=None, sampling=None):
        return self._argmax(logits) if sampling is None else self._sample(logits, noise, sampling)

    def prefill(self, e: Encoded, x, layer_progress=True):
        """x [1,1,B*S,D] -> (logits [1,1,B,V] at each sample's last real token, decode state). Runs as
        sub-batches of at most PREFILL_TOKENS rows; their states are concatenated on the batch axis."""
        B, S = e.B, e.S
        b = max(1, min(B, PREFILL_TOKENS // S))
        rows = lambda t, b0, b1: (
            t if (b0, b1) == (0, B) else ttnn.slice(t, [0, 0, b0 * S, 0], [1, 1, b1 * S, t.shape[3]])
        )
        hs, steps = [], []
        for b0 in range(0, B, b):
            b1 = min(B, b0 + b)
            st = Step(
                "prefill", b1 - b0, S, valid=rows(e.dev["valid"], b0, b1), real_len=e.real_len, capacity=self.capacity
            )
            hs.append(self.text.forward(rows(x, b0, b1), st, layer_progress=layer_progress))  # [1,1,(b1-b0)*S,D]
            steps.append(st)
        h = hs[0] if len(hs) == 1 else ttnn.concat(hs, dim=2)
        step = steps[0]
        if len(steps) > 1:
            step = Step("prefill", B, S, valid=e.dev["valid"], real_len=e.real_len, capacity=self.capacity)
            step.state = {k: _cat_tree([s_.state[k] for s_ in steps]) for k in steps[0].state}
        self.last_prefill_hidden = h
        last = ttnn.slice(_rs(h, [B, 1, S, self.D]), [0, 0, e.real_len - 1, 0], [B, 1, e.real_len, self.D])
        return self._logits(_rs(last, [1, 1, B, self.D])), step

    def decode_step(self, tokens, step, pos):
        """tokens [1,B] uint32 (device) + resident state -> logits [1,1,B,V]. Reads/updates only device state."""
        B = tokens.shape[-1]
        # [1,B] ids -> [1,B,D] tiles straight out of the embedding; the 4-D view keeps the last two dims
        x = ttnn.reshape(ttnn.embedding(tokens, self.embed_w, layout=ttnn.TILE_LAYOUT), [1, 1, B, self.D])
        dstep = Step("decode", B, 1, pos=pos, state=step.state, inplace=getattr(step, "inplace", False))
        h = self.text.forward(x, dstep)
        return self._logits(h)

    # ================================================================== trace contract (per stage)
    # Inputs for every stage come from _captured/e2e_example/ (the processor encoding of the published example
    # that the e2e test and the demo use). The variable (sequence) dim is pinned to capacity C = the prompt
    # padded to a CHUNK multiple for vision_encode/prefill and to self.capacity (<= index_topk) for decode.
    def _trace_example(self):
        return {"enc": load_captured_example(), "batch": self.batch}

    def _setup_common(self, inputs):
        # every stage's persistent inputs stay alive side by side (a perf adapter sets all stages up
        # before stepping any of them); the encoding is identical across stages, so it is shared
        e = self.encode(inputs["enc"], inputs.get("batch"))
        if not getattr(self, "_tr", None):
            self._tr = {}
        self._tr["e"] = e
        return e

    # ---- vision_encode
    def vision_encode_trace_inputs(self):
        return self._trace_example()

    def vision_encode_trace_setup(self, inputs):
        self._setup_common(inputs)

    def vision_encode_trace_step(self):
        return self.vision_encode(self._tr["e"])

    def vision_encode_trace_items(self):
        e = self._tr["e"] if getattr(self, "_tr", None) else self.encode(load_captured_example())
        return sum(int(g["pixel_values"].shape[0]) for g in e.dev["vision_groups"])  # patches, all samples

    # ---- prefill
    def prefill_trace_inputs(self):
        return self._trace_example()

    def prefill_trace_setup(self, inputs):
        e = self._setup_common(inputs)
        self._tr["img"] = self.vision_encode(e)  # persistent image embeddings (stage input)

    def prefill_trace_step(self):
        e = self._tr["e"]
        logits, _ = self.prefill(e, self.embed_prompt(e, self._tr["img"]), layer_progress=False)
        return logits

    def prefill_trace_items(self):
        e = self._tr["e"] if getattr(self, "_tr", None) else self.encode(load_captured_example())
        return int(e.B * e.S)  # token rows the 45 decoder layers process (padded capacity)

    # ---- decode (AR contract: decode_prefill seeds the resident KDA state + DSA KV cache; decode_step reads them)
    def decode_trace_inputs(self):
        return self._trace_example()

    def decode_prefill(self, e: "Encoded", sampling: Optional["Sampling"] = None):
        """Seed the resident decode state: vision + prefill, first token, persistent token/pos/noise buffers."""
        img = self.vision_encode(e)
        logits, step = self.prefill(e, self.embed_prompt(e, img), layer_progress=False)
        step.inplace = True
        noise = None
        if sampling is not None:
            noise = self._rep(
                gumbel_noise(sampling.seeds, 0, self.V).reshape(1, 1, e.B, self.V), ttnn.float32, ttnn.TILE_LAYOUT
            )
        tok = ttnn.clone(self.choose(logits, noise, sampling))
        pos = ttnn.clone(e.dev["pos0"])
        return {"step": step, "tok": tok, "pos": pos, "noise": noise, "sampling": sampling}

    def decode_trace_setup(self, inputs):
        e = self._setup_common(inputs)
        self._tr["dec"] = self.decode_prefill(e)

    def decode_trace_step(self):
        """One host-op-free decode step over the persistent buffers: token in, logits out, next token, state
        and position advanced in place."""
        d = self._tr["dec"]
        logits = self.decode_step(d["tok"], d["step"], d["pos"])
        ttnn.copy(self.choose(logits, d["noise"], d["sampling"]), d["tok"])
        ttnn.copy(ttnn.add(d["pos"], 1.0), d["pos"])
        return logits

    def decode_trace_items(self):
        return int(self._tr["e"].B if getattr(self, "_tr", None) else self.batch)  # one token per sample

    def decode_trace_repeats(self):
        """Decode steps of one real request: the validated run's generation length when recorded, else the
        capacity left after the prompt (the stop rule is eos, so this is the upper bound)."""
        rec = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_captured", "e2e_example", "decode_steps.txt"
        )
        if os.path.exists(rec):
            return int(open(rec).read().strip())
        e = self._tr["e"] if getattr(self, "_tr", None) else None
        return int(self.capacity - (e.real_len if e else 0))

    # ---- snapshot / restore of the decode state (selftest: compare a replay against an eager step)
    def _decode_snapshot(self):
        d = self._tr["dec"]
        leaves = []

        def walk(x):
            if isinstance(x, (tuple, list)):
                for y in x:
                    walk(y)
            else:
                leaves.append(x)

        for st in d["step"].state.values():
            walk(st)
        leaves += [d["tok"], d["pos"]]
        return leaves, [ttnn.clone(t) for t in leaves]

    @staticmethod
    def _decode_restore(snap):
        for live, saved in zip(*snap):
            ttnn.copy(saved, live)


# ----------------------------------------------------------------------------- sampling (seeded, reproducible)
@dataclass
class Sampling:
    """generate(do_sample=True) with the model's own generation_config (temperature, top_p). One seed per
    batch row; the categorical draw is Gumbel-max with noise from that seed, so the TT pipeline and the HF
    reference draw the SAME sample from their own logits (an exact categorical sampler, unlike
    torch.multinomial whose RNG path cannot be replayed on device)."""

    seeds: List[int]
    temperature: float = 1.0
    top_p: float = 1.0

    @classmethod
    def from_generation_config(cls, gen_cfg, seeds):
        return cls(
            seeds=list(seeds),
            temperature=float(getattr(gen_cfg, "temperature", 1.0) or 1.0),
            top_p=float(getattr(gen_cfg, "top_p", 1.0) or 1.0),
        )


def gumbel_noise(seeds, step: int, V: int) -> torch.Tensor:
    """[B, V] fp32 Gumbel(0,1) noise for decode step ``step`` (step 0 = the token after the prompt)."""
    out = torch.empty(len(seeds), V, dtype=torch.float32)
    for b, s in enumerate(seeds):
        g = torch.Generator().manual_seed(int(s) * 1_000_003 + int(step))
        u = torch.rand(V, generator=g, dtype=torch.float64).clamp_(1e-300, 1.0 - 1e-16)
        out[b] = (-torch.log(-torch.log(u))).to(torch.float32)
    return out


def hf_sample(logits: torch.Tensor, noise: torch.Tensor, sampling: Optional[Sampling]) -> torch.Tensor:
    """REFERENCE-side token choice from HF logits [B, V] with HF's own logits warpers."""
    scores = logits.float()
    if sampling is None:
        return scores.argmax(-1)
    from transformers.generation.logits_process import TemperatureLogitsWarper, TopPLogitsWarper

    if sampling.temperature != 1.0:
        scores = TemperatureLogitsWarper(sampling.temperature)(None, scores)
    if sampling.top_p < 1.0:
        scores = TopPLogitsWarper(sampling.top_p)(None, scores)
    return (scores + noise).argmax(-1)


TOP_P_BISECT_STEPS = 48


# ----------------------------------------------------------------------------- captured example input
CAPTURED_EXAMPLE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_captured", "e2e_example"
)


def capture_example(max_image_tokens: Optional[int] = DEFAULT_MAX_IMAGE_TOKENS):
    """Store the published example's image bytes + prompt under _captured/e2e_example/ (fetched once)."""
    import json
    import urllib.request

    os.makedirs(CAPTURED_EXAMPLE_DIR, exist_ok=True)
    img = os.path.join(CAPTURED_EXAMPLE_DIR, "image.jpeg")
    if not os.path.exists(img):
        with urllib.request.urlopen(EXAMPLE_IMAGE_URL) as r, open(img, "wb") as f:
            f.write(r.read())
    with open(os.path.join(CAPTURED_EXAMPLE_DIR, "manifest.json"), "w") as f:
        json.dump(
            {"image_url": EXAMPLE_IMAGE_URL, "prompt": EXAMPLE_PROMPT, "max_image_tokens": max_image_tokens},
            f,
            indent=2,
        )
    return CAPTURED_EXAMPLE_DIR


def load_captured_example(processor=None):
    """Processor encoding of the captured example (same input the e2e test and the demo use)."""
    import json

    man = os.path.join(CAPTURED_EXAMPLE_DIR, "manifest.json")
    if not os.path.exists(man):
        capture_example()
    m = json.load(open(man))
    return encode_example(
        processor, m["prompt"], os.path.join(CAPTURED_EXAMPLE_DIR, "image.jpeg"), m.get("max_image_tokens")
    )


# ----------------------------------------------------------------------------- the explicit chain
@dataclass
class GenerationResult:
    tokens: torch.Tensor  # [B, n_generated] (including the stop token when it was produced)
    logits: List[torch.Tensor]  # per step [B, V] bf16 (host copies)
    stopped_on_eos: List[bool]
    hit_cap: bool
    prompt_len: int
    image_embeds: Optional[torch.Tensor] = None
    prefill_hidden: Optional[torch.Tensor] = None
    prefill_ms: float = 0.0
    decode_ms: List[float] = field(default_factory=list)


def run_image_text_to_text(
    pipe: Glm5FlashPipeline,
    e: Encoded,
    max_new_tokens: Optional[int] = None,
    keep_logits=True,
    keep_image_embeds=False,
    sampling: Optional[Sampling] = None,
    keep_prefill_hidden=False,
    use_trace: bool = True,
):
    """Explicit TT chain: vision -> embed/scatter -> prefill -> token choice -> decode until EVERY sample emitted
    an eos_token_id (generation_config) or the capacity cap. Decode runs over resident device buffers (token,
    position, KDA state, KV cache, sampling noise); after the first (eager, compiling) step it is captured once
    in a trace and replayed. The host only refreshes the noise input buffer and reads the chosen ids back for
    the stop rule; all model math and the token feedback stay on device."""
    mesh = pipe.mesh
    comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
    host = lambda t, n: ttnn.to_torch(t, mesh_composer=comp)[:n]
    B = e.B
    cap = pipe.capacity - e.real_len
    horizon = cap if max_new_tokens is None else min(cap, int(max_new_tokens))
    if sampling is not None and len(sampling.seeds) != B:
        raise ValueError("one seed per batch row")
    t0 = time.time()
    img = pipe.vision_encode(e)
    x = pipe.embed_prompt(e, img)
    logits, step = pipe.prefill(e, x)
    step.inplace = True
    noise_dev = None
    if sampling is not None:
        noise_dev = pipe._rep(
            gumbel_noise(sampling.seeds, 0, pipe.V).reshape(1, 1, B, pipe.V), ttnn.float32, ttnn.TILE_LAYOUT
        )
    tok = ttnn.clone(pipe.choose(logits, noise_dev, sampling))  # resident token buffer
    pos = ttnn.clone(e.dev["pos0"])  # resident position buffer
    ttnn.synchronize_device(mesh)
    prefill_ms = (time.time() - t0) * 1e3
    _log(f"prefill done ({e.B} x {e.real_len} tokens, padded {e.S}) in {prefill_ms / 1e3:.1f}s")
    image_embeds = host(img, img.shape[0]).float() if keep_image_embeds else None
    prefill_hidden = None
    if keep_prefill_hidden:
        prefill_hidden = ttnn.to_torch(pipe.last_prefill_hidden, mesh_composer=comp)[:1].float().reshape(B, e.S, pipe.D)

    def decode_once():
        lg = pipe.decode_step(tok, step, pos)
        ttnn.copy(pipe.choose(lg, noise_dev, sampling), tok)
        ttnn.copy(ttnn.add(pos, 1.0), pos)
        return lg

    out_tokens, out_logits, dec_ms = [], [], []
    done = torch.zeros(B, dtype=torch.bool)
    eos = torch.tensor(pipe.eos_ids)
    hit_cap = False
    tid = None
    n = 0
    try:
        while True:
            t_host = host(tok, 1).reshape(B).long()
            out_tokens.append(t_host)
            if keep_logits:
                out_logits.append(ttnn.to_torch(logits, mesh_composer=comp)[:1].reshape(B, -1).to(torch.bfloat16))
            done |= torch.isin(t_host, eos)
            n += 1
            if bool(done.all()):
                break
            if n >= horizon:
                hit_cap = True
                break
            t1 = time.time()
            if sampling is not None:  # this step's noise into the resident input buffer (input, not model math)
                host_noise = ttnn.from_torch(
                    gumbel_noise(sampling.seeds, n, pipe.V).reshape(1, 1, B, pipe.V),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )
                ttnn.copy_host_to_device_tensor(host_noise, noise_dev)
            if not use_trace or n == 1:
                logits = decode_once()  # eager (also compiles every program before the capture)
            else:
                if tid is None:
                    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                    logits = decode_once()
                    try:
                        ttnn.end_trace_capture(mesh, tid, cq_id=0)
                    except RuntimeError as exc:  # trace region too small for this depth: keep decoding eagerly
                        _log(
                            f"TRACE FALLBACK: decode capture exceeds trace_region_size ({str(exc)[:160]}); eager decode"
                        )
                        use_trace, tid = False, None
                if tid is not None:
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                else:
                    logits = decode_once()
            ttnn.synchronize_device(mesh)
            dec_ms.append((time.time() - t1) * 1e3)
            _log(
                f"decode step {n} ({dec_ms[-1]:.0f} ms{', trace' if tid is not None else ''}) done={int(done.sum())}/{B}"
            )
    finally:
        if tid is not None:
            ttnn.release_trace(mesh, tid)
    toks = torch.stack(out_tokens, dim=1)
    stopped = [bool(torch.isin(toks[b], eos).any()) for b in range(B)]
    return GenerationResult(
        tokens=toks,
        logits=out_logits,
        stopped_on_eos=stopped,
        hit_cap=hit_cap,
        prompt_len=e.real_len,
        image_embeds=image_embeds,
        prefill_hidden=prefill_hidden,
        prefill_ms=prefill_ms,
        decode_ms=dec_ms,
    )


def trim_at_eos(tokens: torch.Tensor, eos_ids) -> List[List[int]]:
    """Per sample: generated ids up to and including the first eos (generate()'s stopping criterion)."""
    out = []
    for row in tokens.tolist():
        cut = next((i for i, t in enumerate(row) if t in set(eos_ids)), None)
        out.append(row if cut is None else row[: cut + 1])
    return out


# ----------------------------------------------------------------------------- factory
def build_pipeline(
    device,
    model=None,
    layers=None,
    vision_encode_layers=None,
    prefill_layers=None,
    decode_layers=None,
    batch=None,
    capacity=None,
    **kwargs,
):
    """Construct and RETURN the resident pipeline object (does not run it). ``layers`` caps every repeated
    stack (None = full depth); ``vision_encode_layers`` (the vision block stack) and ``prefill_layers`` /
    ``decode_layers`` (the text decoder stack both share) override per stack, None falling back to
    ``layers``. Demo kwargs (prompt, text, ...) are accepted and ignored."""
    if batch is None:
        batch = int(os.environ.get("TT_PERF_BATCH", "1") or 1)
    return Glm5FlashPipeline(
        device,
        model_id=model or HF_MODEL_ID,
        layers=layers,
        vision_encode_layers=vision_encode_layers,
        prefill_layers=prefill_layers,
        decode_layers=decode_layers,
        batch=batch,
        capacity=capacity,
    )


# ----------------------------------------------------------------------------- Gate 1: stubs stay ttnn
_HOST_ONLY_FUNCS = {"__init__", "build", "_put", "_f32", "put", "f32", "prepare", "_upload", "_mask", "_norm_w", "_t"}
_FORBIDDEN_TORCH = {
    "matmul",
    "mm",
    "bmm",
    "einsum",
    "softmax",
    "log_softmax",
    "layer_norm",
    "rms_norm",
    "batch_norm",
    "group_norm",
    "embedding",
    "embedding_bag",
    "conv1d",
    "conv2d",
    "conv3d",
    "scaled_dot_product_attention",
    "relu",
    "gelu",
    "silu",
    "tanh",
    "sigmoid",
    "leaky_relu",
    "argmax",
    "topk",
    "multinomial",
    "dropout",
    "exp",
    "log",
    "sqrt",
    "rsqrt",
    "cumsum",
    "sum",
    "mean",
    "where",
    "clamp",
    "linear",
}
SHARDED_STUBS = [
    "text_attention",
    "text_decoder_layer",
    "text_experts",
    "text_forget_gate",
    "text_hyper_connection",
    "text_hyper_head",
    "text_indexer",
    "text_linear_attention",
    "text_m_l_p",
    "text_mo_e",
    "text_model",
    "text_topk_router",
    "vision_attention",
    "vision_block",
    "vision_m_l_p",
    "vision_model",
    "vision_patch_merger",
]


def stub_purity_report():
    """Gate 1. For every graduated stub file: (a) no torch / torch.nn.functional compute and no HF submodule
    .forward call inside any forward-path function (everything except weight build / constant upload helpers);
    (b) uses ttnn ops; (c) the sharded (TP=32) stubs (or the ports they compose) still shard weights with
    ShardTensorToMesh and reduce/gather with a collective. Returns {stub: [violations]}."""
    import ast
    import importlib
    import inspect

    out = {}
    for name, (mod, _cls, _m) in STUB_ENTRYPOINTS.items():
        m = importlib.import_module(f"models.demos.glm_5_3_flash._stubs.{mod}")
        src = inspect.getsource(m)
        tree = ast.parse(src)
        bad = []
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            if fn.name in _HOST_ONLY_FUNCS or fn.name in (mod,):
                continue
            for node in ast.walk(fn):
                if not isinstance(node, ast.Call):
                    continue
                f = node.func
                chain = []
                while isinstance(f, ast.Attribute):
                    chain.append(f.attr)
                    f = f.value
                if isinstance(f, ast.Name):
                    chain.append(f.id)
                chain = chain[::-1]
                if chain and chain[0] in ("torch", "F") and (chain[-1] in _FORBIDDEN_TORCH or "functional" in chain):
                    bad.append(f"{fn.name}: {'.'.join(chain)}")
                if (
                    chain
                    and chain[-1] == "forward"
                    and set(chain[:-1]) & {"hf", "hf_model", "torch_module", "m", "model"}
                ):
                    bad.append(f"{fn.name}: HF forward call {'.'.join(chain)}")
        if "ttnn." not in src:
            bad.append("no ttnn ops")
        if name in SHARDED_STUBS:
            # the sharded body lives in this file or in the graduated ports it composes
            deps = [src] + [
                inspect.getsource(importlib.import_module(f"models.demos.glm_5_3_flash._stubs.{d}"))
                for d in (
                    "text_decoder_layer",
                    "text_attention",
                    "text_experts",
                    "text_topk_router",
                    "vision_attention",
                    "vision_patch_embed",
                )
            ]
            if not any("ShardTensorToMesh" in d for d in deps) or not any("all_gather" in d for d in deps):
                bad.append("sharded stub lost ShardTensorToMesh / all_gather")
        out[name] = bad
    return out


# ----------------------------------------------------------------------------- trace + host-op selftests
def _to_host(t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(t.device(), dim=0))[:1].float()


def trace_capture_selftest(
    device=None, pipe: Optional[Glm5FlashPipeline] = None, layers=None, pcc_target: float = 0.99
):
    """For EACH stage: set up from <stage>_trace_inputs(), run one eager step (compile + reference), capture
    ONE step in a trace, replay it, compare to the eager output, RELEASE the trace before the next stage.
    Returns (ok, report) on the given device. Called bare (the gate's trace probe, a standalone process) it
    borrows the demo entrypoint's device opener and returns the verdict ``ok`` alone."""
    if device is None and pipe is None:
        from models.demos.glm_5_3_flash.demo.demo_image_text_to_text import close_mesh, open_mesh

        if layers is None and os.environ.get("TT_PERF_LAYERS"):
            layers = int(os.environ["TT_PERF_LAYERS"])
        mesh = open_mesh()
        try:
            ok, _ = _trace_capture_selftest_on(mesh, None, layers, pcc_target)
            return ok
        finally:
            close_mesh(mesh)
    return _trace_capture_selftest_on(device if device is not None else pipe.mesh, pipe, layers, pcc_target)


def _trace_capture_selftest_on(device, pipe, layers, pcc_target):
    from models.common.utility_functions import comp_pcc
    from scripts.tt_hw_planner.host_op_observer import observe_host_ops, verdict

    pipe = pipe or build_pipeline(device, layers=layers)
    report, ok = {}, True
    for stage in PIPELINE_STAGES:
        getattr(pipe, f"{stage}_trace_setup")(getattr(pipe, f"{stage}_trace_inputs")())
        step = getattr(pipe, f"{stage}_trace_step")
        snap = pipe._decode_snapshot() if stage == "decode" else None
        ref = _to_host(step())
        ttnn.synchronize_device(device)
        if snap:
            pipe._decode_restore(snap)
        try:
            with observe_host_ops() as ops:
                tid = ttnn.begin_trace_capture(device, cq_id=0)
                out = step()
                ttnn.end_trace_capture(device, tid, cq_id=0)
            if snap:
                pipe._decode_restore(snap)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            got = _to_host(out)
            ttnn.release_trace(device, tid)
        except Exception as exc:  # noqa: BLE001 -- report which stage failed to capture
            report[stage] = {"captured": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            ok = False
            print(f"[trace-selftest] {stage}: capture FAILED: {report[stage]['error']}", flush=True)
            continue
        v = verdict(ops)
        pcc = comp_pcc(ref, got, pcc_target)[1]
        stage_ok = v["on_device"] and pcc >= pcc_target
        report[stage] = {"captured": True, "host_free": v["on_device"], "host_ops": v["host_ops"][:8], "pcc": pcc}
        print(f"[trace-selftest] {stage}: host_free={v['on_device']} replay PCC={pcc:.6f}", flush=True)
        ok &= stage_ok
    return ok, report


def host_op_selftest(device=None, pipe: Optional[Glm5FlashPipeline] = None, layers=None):
    """AUTHORITATIVE fully-on-device check: input encoding (processor, tokenizer, image patches) and the
    one-time weight build happen OUTSIDE the observed region; the model math -- vision tower, embedding +
    image scatter, prefill over every layer, lm_head, token choice, one decode step -- runs INSIDE
    observe_host_ops(). Returns host_op_observer.verdict(ops)."""
    from scripts.tt_hw_planner.host_op_observer import observe_host_ops, verdict

    own_mesh = None
    if pipe is None and device is None:
        # called bare (the gate's observer probe, a standalone process): the pipeline never opens a device
        # itself -- borrow the demo entrypoint's (the standalone run's sole device opener)
        from models.demos.glm_5_3_flash.demo.demo_image_text_to_text import close_mesh, open_mesh

        own_mesh = device = open_mesh()
        if layers is None and os.environ.get("TT_PERF_LAYERS"):
            layers = int(os.environ["TT_PERF_LAYERS"])
    try:
        pipe = pipe or build_pipeline(device, layers=layers)
        e = pipe.encode(load_captured_example())  # encoding + uploads: outside
        with observe_host_ops() as ops:
            img = pipe.vision_encode(e)
            logits, step = pipe.prefill(e, pipe.embed_prompt(e, img), layer_progress=False)
            tok = pipe.choose(logits)
            logits = pipe.decode_step(tok, step, e.dev["pos0"])
            pipe.choose(logits)
        ttnn.synchronize_device(pipe.mesh)
        return verdict(ops)
    finally:
        if own_mesh is not None:
            del pipe
            close_mesh(own_mesh)

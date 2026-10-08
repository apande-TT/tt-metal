# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Inputs for Kolibri-1 text generation, shared by the demo and the e2e test.

Provenance: the user message is the model card's own example (Aleph-Alpha/Kolibri-1 README, "Querying
the server"), rendered through the tokenizer's chat template with enable_thinking=False -- the card's
documented "disable thinking" mode ("The model will then immediately respond"). Every sample gets the
SAME prompt; sample b differs only by its sampling seed (seed b), the axis generation_config exposes
(do_sample=True, top_k=128, top_p=0.97, temperature=1.0).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch

MODEL_ID = "Aleph-Alpha/Kolibri-1"
CARD_MESSAGE = "Erkläre kurz, was ein Mixture-of-Experts-Modell ist."
BATCH_ENV = "TT_PERF_BATCH"
DEFAULT_BATCH = 32


@dataclass(frozen=True)
class SamplingSettings:
    top_k: int
    top_p: float
    temperature: float
    stop_ids: tuple
    pad_id: int


def batch_size() -> int:
    """Batch the caller asked for ($TT_PERF_BATCH, default 32)."""
    return int(os.environ.get(BATCH_ENV, "") or DEFAULT_BATCH)


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_ID)


def encode_prompt(tokenizer, message: str = CARD_MESSAGE) -> list:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": message}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return tokenizer(text, add_special_tokens=False).input_ids


def sampling_settings() -> SamplingSettings:
    """The model's own sampling rule and stop tokens, from generation_config.json."""
    from transformers import GenerationConfig

    g = GenerationConfig.from_pretrained(MODEL_ID)
    eos = g.eos_token_id if isinstance(g.eos_token_id, (list, tuple)) else [g.eos_token_id]
    return SamplingSettings(
        top_k=int(g.top_k),
        top_p=float(g.top_p),
        temperature=float(g.temperature),
        stop_ids=tuple(int(e) for e in eos),
        pad_id=int(g.pad_token_id),
    )


def uniforms_length() -> int:
    """One uniform per sequence position up to the model's own max length (max_position_embeddings)."""
    import json

    from huggingface_hub import hf_hub_download

    return int(json.loads(open(hf_hub_download(MODEL_ID, "config.json")).read())["max_position_embeddings"])


def sampling_uniforms(seeds, length: int) -> torch.Tensor:
    """[length, B] fp32: column b is torch.rand from torch.Generator(seed b). The token sampled from the
    logits at sequence position p uses row p, on the TT side and in the reference alike."""
    cols = []
    for s in seeds:
        g = torch.Generator().manual_seed(int(s))
        cols.append(torch.rand(length, generator=g, dtype=torch.float32))
    return torch.stack(cols, dim=1)

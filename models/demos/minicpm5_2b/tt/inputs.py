# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Inputs for the openbmb/MiniCPM5-2B text-generation pipeline.

Provenance: the content input is the model card's own example (README.md example 1), used verbatim
for every one of the B samples; samples differ only in the sampling seed (BASE_SEED + i), which the
model's generation_config makes the sampling axis (do_sample=True).
"""
from __future__ import annotations

import os

import torch

MODEL_ID = "openbmb/MiniCPM5-2B"

# ---- The model's published example, verbatim (source: README.md example 1) ----
EXAMPLE_MESSAGES = [
    {"role": "user", "content": "Who are you? Please briefly introduce yourself."}
]  # README.md example 1
EXAMPLE_TOKENIZE = True  # README.md example 1
EXAMPLE_ADD_GENERATION_PROMPT = True  # README.md example 1
EXAMPLE_ENABLE_THINKING = True  # README.md example 1
EXAMPLE_RETURN_DICT = True  # README.md example 1
EXAMPLE_RETURN_TENSORS = "pt"  # README.md example 1
EXAMPLE_MAX_NEW_TOKENS = 128  # README.md example 1
EXAMPLE_SKIP_SPECIAL_TOKENS = True  # README.md example 1
# ---- end of the published example ----

# The example declares no seed. Sample i uses BASE_SEED + i (chosen here, recorded beside the inputs).
BASE_SEED = 1234

# The batch the pipeline drives; the gate sets $TT_PERF_BATCH.
BATCH_ENV = "TT_PERF_BATCH"
DEFAULT_BATCH = 32


def batch_size() -> int:
    return int(os.environ.get(BATCH_ENV, DEFAULT_BATCH))


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_ID)


def example_input_ids(tokenizer=None) -> torch.Tensor:
    """[1, S] prompt ids from the published example's apply_chat_template call."""
    tokenizer = tokenizer or load_tokenizer()
    enc = tokenizer.apply_chat_template(
        EXAMPLE_MESSAGES,
        tokenize=EXAMPLE_TOKENIZE,
        add_generation_prompt=EXAMPLE_ADD_GENERATION_PROMPT,
        enable_thinking=EXAMPLE_ENABLE_THINKING,
        return_dict=EXAMPLE_RETURN_DICT,
        return_tensors=EXAMPLE_RETURN_TENSORS,
    )
    return enc["input_ids"]


def batch_inputs(batch: int, tokenizer=None, base_seed: int = BASE_SEED):
    """(input_ids [B, S], seeds [B]): identical content per row, one seed per row."""
    ids = example_input_ids(tokenizer)
    return ids.repeat(batch, 1), [base_seed + i for i in range(batch)]


def gumbel_noise(seeds, steps: int, vocab: int) -> torch.Tensor:
    """[steps, B, vocab] bf16 Gumbel(0, 1) noise, row b drawn from torch.Generator(seeds[b]).

    Sampling from the model's filtered distribution is realised as argmax(filtered_logits + noise)
    (Gumbel-max), on TT and in the HF reference alike, so one seed gives one sample on both sides.
    Rounded to bf16 once here so both sides add the very same values.
    """
    out = torch.empty(steps, len(seeds), vocab, dtype=torch.bfloat16)
    for b, seed in enumerate(seeds):
        g = torch.Generator().manual_seed(int(seed))
        u = torch.rand(steps, vocab, generator=g, dtype=torch.float64).clamp_(1e-12, 1.0 - 1e-12)
        out[:, b, :] = (-torch.log(-torch.log(u))).to(torch.bfloat16)
    return out


# The request the e2e golden was built from, stored beside the bring-up's captured tensors.
CAPTURED_REQUEST = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_captured", "text_generation", "args.pt"
)


def save_request(input_ids, seeds):
    os.makedirs(os.path.dirname(CAPTURED_REQUEST), exist_ok=True)
    torch.save({"input_ids": input_ids[:1].clone(), "base_seed": int(seeds[0])}, CAPTURED_REQUEST)


def stored_request(batch: int):
    """(input_ids [B, S], seeds [B]) from _captured/text_generation/args.pt (the e2e golden's request);
    falls back to the published example when the golden has not been built yet."""
    if os.path.exists(CAPTURED_REQUEST):
        rec = torch.load(CAPTURED_REQUEST)
        return rec["input_ids"].repeat(batch, 1), [rec["base_seed"] + i for i in range(batch)]
    return batch_inputs(batch)

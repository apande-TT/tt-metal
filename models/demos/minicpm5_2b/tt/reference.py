# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF goldens for the MiniCPM5-2B text-generation pipeline (reference only, never the TT forward).

The golden is model.generate() itself: its own TopK / TopP warpers built from generation_config, the
model's eos ids and the published example's max_new_tokens. The one substitution is the random draw:
generate() samples with torch.multinomial from the global RNG, which nothing else can reproduce, so the
draw is done as Gumbel-max with the same per-sample seeded noise the TT pipeline adds (an exact sample
of the same filtered distribution).
"""
from __future__ import annotations

import hashlib
import os

import torch

from models.demos.minicpm5_2b.tt import inputs as tt_inputs
from models.demos.minicpm5_2b.tt.pipeline import eos_token_ids, sampling_params

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_captured", "text_generation")


class GumbelDraw:
    """LogitsProcessor: adds step t's per-row noise so generate()'s argmax is a seeded sample."""

    def __init__(self, noise, prompt_len):
        self.noise, self.prompt_len = noise, prompt_len

    def __call__(self, input_ids, scores):
        t = input_ids.shape[1] - self.prompt_len
        return scores + self.noise[t].to(scores.dtype)


def processors(hf_model, noise, prompt_len):
    from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

    top_k, top_p, temperature = sampling_params(hf_model)
    chain = []
    if temperature != 1.0:
        chain.append(TemperatureLogitsWarper(temperature))
    chain += [TopKLogitsWarper(top_k=top_k), TopPLogitsWarper(top_p=top_p), GumbelDraw(noise, prompt_len)]
    return LogitsProcessorList(chain)


def _key(input_ids, seeds, max_new_tokens):
    h = hashlib.sha256()
    h.update(input_ids.numpy().tobytes())
    h.update(repr((list(seeds), int(max_new_tokens))).encode())
    return h.hexdigest()[:16]


def _hf_reference_text_generation(hf_model, input_ids, seeds, noise, max_new_tokens, cache=True):
    """generate() golden: tokens [B, T] (pad after eos, as generate() emits) and per-row lengths."""
    path = os.path.join(GOLDEN_DIR, f"golden_{_key(input_ids, seeds, max_new_tokens)}.pt")
    if cache and os.path.exists(path):
        return torch.load(path)
    prompt_len = input_ids.shape[-1]
    eos = eos_token_ids(hf_model)
    with torch.no_grad():
        out = hf_model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            do_sample=False,  # the draw is the Gumbel-max in `processors`
            top_k=None,
            top_p=None,
            temperature=None,
            logits_processor=processors(hf_model, noise.float(), prompt_len),
            max_new_tokens=max_new_tokens,
            eos_token_id=eos,
            pad_token_id=int(hf_model.generation_config.pad_token_id),
        )
    tokens = out[:, prompt_len:]
    lengths = torch.full((tokens.shape[0],), tokens.shape[1], dtype=torch.int64)
    for b in range(tokens.shape[0]):
        hit = torch.isin(tokens[b], torch.tensor(eos)).nonzero()
        if len(hit):
            lengths[b] = int(hit[0]) + 1
    golden = {"tokens": tokens, "lengths": lengths, "steps": tokens.shape[1]}
    if cache:
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        torch.save(golden, path)
        tt_inputs.save_request(input_ids, seeds)
    return golden


def _hf_step_logits(hf_model, input_ids, tokens, lengths, chunk=8):
    """HF logits that produced each generated token of `tokens` (one teacher-forced forward per row
    chunk): list over rows of [len_b, V] float32. Row b step t is the logits after prompt + tokens[:t]."""
    prompt_len = input_ids.shape[-1]
    seq = torch.empty(input_ids.shape[0], prompt_len + tokens.shape[1], dtype=input_ids.dtype)
    seq[:, :prompt_len], seq[:, prompt_len:] = input_ids, tokens.to(input_ids.dtype)
    rows = []
    with torch.no_grad():
        for s in range(0, seq.shape[0], chunk):
            part = seq[s : s + chunk, : prompt_len + tokens.shape[1] - 1]
            logits = hf_model(input_ids=part, use_cache=False).logits.float()
            for i in range(part.shape[0]):
                b = s + i
                rows.append(logits[i, prompt_len - 1 : prompt_len - 1 + int(lengths[b])].clone())
            del logits
    return rows


def _hf_prefill_hidden(hf_model, input_ids):
    """Final-norm hidden state of the prompt [1, S, H] (all rows share the prompt)."""
    with torch.no_grad():
        return hf_model.model(input_ids=input_ids[:1], use_cache=False).last_hidden_state.float()


def hf_choice(step_logits, noise_bt, hf_model):
    """The token generate()'s chain picks from raw logits [V] with noise [V]; also the filtered scores."""
    lp = processors(hf_model, noise_bt.float().reshape(1, 1, -1), 0)
    dummy = torch.zeros(1, 0, dtype=torch.long)
    scores = lp(dummy, step_logits.reshape(1, -1).clone())
    return int(scores.argmax()), scores.reshape(-1)


def tie_gap(step_logits, noise_bt, hf_model, hf_tok, tt_tok):
    """How far TT's token was from being generate()'s choice, judged on the reference's own scores.

    0 when they agree. For a token the reference keeps: its Gumbel-score deficit to the reference's
    choice. For a token the reference's top-k/top-p filter drops: the larger of that deficit (scored
    unfiltered) and its raw-logit distance below the filter's cut. A small gap is a numeric near-tie.
    """
    if hf_tok == tt_tok:
        return 0.0
    _, scores = hf_choice(step_logits, noise_bt, hf_model)
    if torch.isfinite(scores[tt_tok]):
        return float(scores[hf_tok] - scores[tt_tok])
    raw = step_logits.float() + noise_bt.float()
    kept = torch.isfinite(scores)
    cut = float(step_logits[kept].min())
    return max(float(raw[hf_tok] - raw[tt_tok]), cut - float(step_logits[tt_tok]))

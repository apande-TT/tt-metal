# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace golden helpers for Voxtral-Mini-3B-2507.

This is the ONLY module in the bring-up that is allowed to run HF forward code.
Everything here is CPU / float32 / `torch.no_grad()`; no ttnn, no device.

What lives here
---------------
  load_hf_model()            cached fp32 `VoxtralForConditionalGeneration` singleton
  hf_reference()             end-to-end greedy golden via the real `model.generate`
  hf_rope_cos_sin()          (cos, sin) from the real `language_model.rotary_emb`
  hf_audio_features_golden() encoder last hidden + projected audio embeds
  hf_prefill_golden()        inputs_embeds / last_hidden / logits for one prefill
  clear_cache()              drop the in-process model + golden caches

Golden caching
--------------
A full 8-stream, 32-token golden costs several minutes of CPU, which the on-device gate
must not pay on every run.  `hf_reference` therefore memoises to

    _captured/e2e_golden_<head>_<sha1>.pt

where sha1 covers (prompt_text/template, clips, max_new_tokens, dtype).  The saved blob
also carries the full `BatchInputs.fingerprint()` (which hashes the actual token ids),
and that fingerprint is re-verified on load -- a stale cache raises instead of silently
grading the device against the wrong prompt.

Self-check / golden pre-computation (from the repo root, `models` has no __init__.py so
run it as a module):

    ./python_env/bin/python -m models.tt_transformers.demo.voxtral_mini_3b_2507.tt.reference
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .inputs import CAPTURED_DIR, EOS_ID, HF_REPO_ID, BatchInputs, build_inputs, get_tokenizer

__all__ = [
    "HFGolden",
    "load_hf_model",
    "hf_reference",
    "hf_rope_cos_sin",
    "hf_audio_features_golden",
    "hf_prefill_golden",
    "clear_cache",
]


# --------------------------------------------------------------------------------------
# Model singleton
# --------------------------------------------------------------------------------------

_MODEL_CACHE: dict[str, torch.nn.Module] = {}
_GOLDEN_MEM_CACHE: dict[str, "HFGolden"] = {}


def load_hf_model(dtype: torch.dtype = torch.float32):
    """Cached fp32 CPU `VoxtralForConditionalGeneration` in eval mode (~12 GB RAM)."""
    from transformers import VoxtralForConditionalGeneration

    key = str(dtype)
    model = _MODEL_CACHE.get(key)
    if model is None:
        t0 = time.time()
        model = VoxtralForConditionalGeneration.from_pretrained(HF_REPO_ID, dtype=dtype, low_cpu_mem_usage=True)
        model.eval()
        model.requires_grad_(False)
        print(f"[reference] loaded {HF_REPO_ID} ({dtype}) in {time.time() - t0:.1f}s")
        _MODEL_CACHE[key] = model
    return model


def clear_cache() -> None:
    """Drop the in-process model and golden caches (files on disk are kept)."""
    _MODEL_CACHE.clear()
    _GOLDEN_MEM_CACHE.clear()
    gc.collect()


# --------------------------------------------------------------------------------------
# Golden container
# --------------------------------------------------------------------------------------


@dataclass
class HFGolden:
    head: str
    tokens: torch.LongTensor  # [B, N] generated ids only (prompt stripped)
    logits: torch.FloatTensor  # [B, N, 131072] float32
    texts: list[str]  # decoded per stream, skip_special_tokens=True
    lengths: list[int]  # generated length before eos (== N if no eos)
    stopped_on_eos: list[bool]
    max_new_tokens: int
    fingerprint: str = ""
    clips: list[str] = field(default_factory=list)
    prompt_text: str = ""
    prompt_len: int = 0
    elapsed_s: float = 0.0

    @property
    def batch_size(self) -> int:
        return int(self.tokens.shape[0])

    def to_dict(self) -> dict:
        return {
            "head": self.head,
            "tokens": self.tokens,
            "logits": self.logits,
            "texts": self.texts,
            "lengths": self.lengths,
            "stopped_on_eos": self.stopped_on_eos,
            "max_new_tokens": self.max_new_tokens,
            "fingerprint": self.fingerprint,
            "clips": self.clips,
            "prompt_text": self.prompt_text,
            "prompt_len": self.prompt_len,
            "elapsed_s": self.elapsed_s,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HFGolden":
        return cls(**d)


# --------------------------------------------------------------------------------------
# Cache key / path
# --------------------------------------------------------------------------------------


def _golden_key(batch_inputs: BatchInputs, max_new_tokens: int, dtype: torch.dtype) -> str:
    payload = json.dumps(
        {
            "template": batch_inputs.prompt_text,
            "clips": list(batch_inputs.clips),
            "max_new_tokens": int(max_new_tokens),
            "dtype": str(dtype),
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()


def golden_path(head: str, batch_inputs: BatchInputs, max_new_tokens: int, dtype: torch.dtype) -> Path:
    return CAPTURED_DIR / f"e2e_golden_{head}_{_golden_key(batch_inputs, max_new_tokens, dtype)}.pt"


# --------------------------------------------------------------------------------------
# End-to-end golden
# --------------------------------------------------------------------------------------


def hf_reference(
    head: str,
    batch_inputs: BatchInputs,
    max_new_tokens: int = 32,
    cache: bool = True,
    dtype: torch.dtype = torch.float32,
) -> HFGolden:
    """Greedy end-to-end golden through the real `model.generate`.

    Uses the true reference chain (audio tower -> projector -> scatter -> Llama -> lm_head)
    with `do_sample=False` and eos (id 2) stopping, and returns per-step logits stacked
    into [B, N, vocab].
    """
    if batch_inputs.head != head:
        raise ValueError(f"head mismatch: asked {head!r}, BatchInputs says {batch_inputs.head!r}")

    path = golden_path(head, batch_inputs, max_new_tokens, dtype)
    fp = batch_inputs.fingerprint()
    mem_key = f"{path.name}:{fp}"

    if cache:
        hit = _GOLDEN_MEM_CACHE.get(mem_key)
        if hit is not None:
            return hit
        if path.exists():
            blob = torch.load(path, map_location="cpu", weights_only=False)
            if blob.get("fingerprint") != fp:
                raise RuntimeError(
                    f"stale golden cache {path}: fingerprint {blob.get('fingerprint')} != {fp}. "
                    "Delete the file and regenerate."
                )
            golden = HFGolden.from_dict(blob)
            _GOLDEN_MEM_CACHE[mem_key] = golden
            print(f"[reference] golden cache hit: {path.name}")
            return golden

    model = load_hf_model(dtype)
    # NB: the working tekken tokenizer, not processor.tokenizer (which is mis-converted
    # for this checkpoint -- see tt/inputs.py's module docstring).
    tok = get_tokenizer()

    t0 = time.time()
    with torch.no_grad():
        result = model.generate(
            input_ids=batch_inputs.input_ids,
            input_features=batch_inputs.input_features.to(dtype),
            attention_mask=batch_inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=EOS_ID,
            return_dict_in_generate=True,
            output_logits=True,
        )
    elapsed = time.time() - t0

    L = batch_inputs.prompt_len
    tokens = result.sequences[:, L:].contiguous().to(torch.long)
    logits = torch.stack(list(result.logits), dim=1).to(torch.float32).contiguous()
    assert logits.shape[:2] == tokens.shape, (tuple(logits.shape), tuple(tokens.shape))

    N = int(tokens.shape[1])
    lengths: list[int] = []
    stopped: list[bool] = []
    texts: list[str] = []
    for row in tokens:
        eos_pos = (row == EOS_ID).nonzero()
        if eos_pos.numel() > 0:
            n = int(eos_pos[0, 0])
            stopped.append(True)
        else:
            n = N
            stopped.append(False)
        lengths.append(n)
        texts.append(tok.decode(row[:n], skip_special_tokens=True))

    golden = HFGolden(
        head=head,
        tokens=tokens,
        logits=logits,
        texts=texts,
        lengths=lengths,
        stopped_on_eos=stopped,
        max_new_tokens=max_new_tokens,
        fingerprint=fp,
        clips=list(batch_inputs.clips),
        prompt_text=batch_inputs.prompt_text,
        prompt_len=L,
        elapsed_s=elapsed,
    )

    if cache:
        CAPTURED_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(golden.to_dict(), path)
        print(f"[reference] golden saved: {path} ({path.stat().st_size / 1e6:.1f} MB, {elapsed:.1f}s)")
        _GOLDEN_MEM_CACHE[mem_key] = golden

    return golden


# --------------------------------------------------------------------------------------
# Sub-graph goldens
# --------------------------------------------------------------------------------------


def hf_rope_cos_sin(model, positions: torch.LongTensor):
    """(cos, sin) straight out of the REAL `model.model.language_model.rotary_emb`.

    `positions` may be [L] or [B, L]; returns tensors shaped [B, L, head_dim].
    These are the exact values the golden used, so they can seed TT trace constants.
    """
    position_ids = positions if positions.dim() == 2 else positions.unsqueeze(0)
    position_ids = position_ids.to(torch.long)
    lang = model.model.language_model
    hidden_size = lang.config.hidden_size
    dummy = torch.zeros(position_ids.shape[0], position_ids.shape[1], hidden_size, dtype=torch.float32)
    with torch.no_grad():
        cos, sin = lang.rotary_emb(dummy, position_ids)
    return cos.to(torch.float32), sin.to(torch.float32)


def hf_audio_features_golden(batch_inputs: BatchInputs):
    """-> (encoder_last_hidden [B, 1500, 1280], audio_embeds [B*375, 3072])."""
    model = load_hf_model()
    with torch.no_grad():
        out = model.model.get_audio_features(batch_inputs.input_features.to(torch.float32), return_dict=True)
    enc = out.last_hidden_state.to(torch.float32)
    embeds = out.pooler_output.to(torch.float32)
    B = batch_inputs.batch_size
    assert enc.shape[0] == B and enc.shape[2] == 1280, tuple(enc.shape)
    assert embeds.shape == (B * batch_inputs.n_audio_tokens, 3072), tuple(embeds.shape)
    return enc, embeds


def hf_prefill_golden(batch_inputs: BatchInputs):
    """-> (inputs_embeds [B, L, 3072], last_hidden [B, L, 3072], logits [B, L, V]).

    Mirrors `VoxtralModel.forward`: token embeddings, audio embeds masked-scattered into
    the [AUDIO] placeholder slots, then the 30-layer Llama with `use_cache=False`.
    """
    model = load_hf_model()
    vox = model.model
    input_ids = batch_inputs.input_ids
    with torch.no_grad():
        inputs_embeds = vox.get_input_embeddings()(input_ids)
        audio_embeds = vox.get_audio_features(
            batch_inputs.input_features.to(torch.float32), return_dict=True
        ).pooler_output
        mask = vox.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, audio_features=audio_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(mask, audio_embeds.to(inputs_embeds.dtype))

        # Run the inner VoxtralModel so we get `last_hidden_state` (post final RMSNorm,
        # i.e. exactly what lm_head consumes) without materialising all 31 hidden states.
        out = vox(
            inputs_embeds=inputs_embeds,
            attention_mask=batch_inputs.attention_mask,
            use_cache=False,
            return_dict=True,
        )
        last_hidden = out.last_hidden_state
        logits = model.lm_head(last_hidden)

    return inputs_embeds.to(torch.float32), last_hidden.to(torch.float32), logits.to(torch.float32)


# --------------------------------------------------------------------------------------
# Self-check / golden pre-computation
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--heads", nargs="*", default=["audio_chat", "transcription"])
    ap.add_argument("--skip-subgraphs", action="store_true")
    args = ap.parse_args()

    torch.set_grad_enabled(False)

    for head in args.heads:
        bi = build_inputs(head, n=args.n)
        print("=" * 92)
        print(f"{head}: {bi.describe()}")
        print(f"  template: {bi.prompt_text}")
        t0 = time.time()
        g = hf_reference(head, bi, max_new_tokens=args.max_new_tokens)
        print(f"  hf_reference wall: {time.time() - t0:.1f}s (generate {g.elapsed_s:.1f}s)")
        print(f"  tokens {tuple(g.tokens.shape)}  logits {tuple(g.logits.shape)} {g.logits.dtype}")
        for clip, txt, n, eos in zip(g.clips, g.texts, g.lengths, g.stopped_on_eos):
            print(f"  [{clip:26s}] len={n:2d} eos={int(eos)}  {txt!r}")

    if not args.skip_subgraphs:
        bi = build_inputs("transcription", n=2)
        model = load_hf_model()
        cos, sin = hf_rope_cos_sin(model, torch.arange(bi.prompt_len).unsqueeze(0))
        print("=" * 92)
        print("rope cos/sin:", tuple(cos.shape), tuple(sin.shape), float(cos[0, 0, 0]), float(sin[0, 1, 0]))
        enc, emb = hf_audio_features_golden(bi)
        print("audio golden:", tuple(enc.shape), tuple(emb.shape))
        ie, lh, lg = hf_prefill_golden(bi)
        print("prefill golden:", tuple(ie.shape), tuple(lh.shape), tuple(lg.shape))
    print("OK")

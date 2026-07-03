# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Host torch reference for ACE-Step live text conditioning.

Formats caption/lyrics with the ACE-Step SFT templates (same as diffusers
``AceStepPipeline.encode_prompt`` / upstream handler), tokenizes via the
Qwen3-Embedding-0.6B tokenizer, and runs the causal Qwen3 text encoder:

  - caption -> full ``text_encoder`` forward -> ``text_hidden_states``
  - lyrics  -> ``embed_tokens`` lookup only -> ``lyric_hidden_states``

Weights resolve from ``ACE-Step/Ace-Step1.5`` bundle ``Qwen3-Embedding-0.6B/``
(with fallbacks documented in ``resolve_text_encoder_path``).
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

import torch

SFT_GEN_PROMPT = "# Instruction\n{}\n\n# Caption\n{}\n\n# Metas\n{}<|endoftext|>\n"
DEFAULT_DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"

TEXT_ENCODER_SUBDIR = "Qwen3-Embedding-0.6B"
DEFAULT_MAX_TEXT_LEN = 256
DEFAULT_MAX_LYRIC_LEN = 2048
_WEIGHT_FILENAMES = ("model.safetensors", "pytorch_model.bin", "model.bin")

_LOG_PATH = "/tmp/acestep_agent_2a.log"


def _log_progress(message: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"{ts} {message}\n"
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError:
        pass
    print(line, end="", flush=True)


def _has_text_encoder_weights(path: str) -> bool:
    if not os.path.isfile(os.path.join(path, "config.json")):
        return False
    return any(os.path.isfile(os.path.join(path, name)) for name in _WEIGHT_FILENAMES)


def resolve_text_encoder_path(*, require_weights: bool = True) -> str:
    """Return a local directory containing Qwen3-Embedding-0.6B weights + tokenizer."""
    env_path = os.environ.get("ACESTEP_TEXT_ENCODER_PATH")
    if env_path:
        if not os.path.isdir(env_path):
            raise FileNotFoundError(f"ACESTEP_TEXT_ENCODER_PATH is not a directory: {env_path}")
        if require_weights and not _has_text_encoder_weights(env_path):
            raise FileNotFoundError(f"ACESTEP_TEXT_ENCODER_PATH missing model weights: {env_path}")
        return env_path

    candidates: list[str] = []
    pipeline_env = os.environ.get("ACESTEP_PIPELINE_DIR")
    if pipeline_env:
        candidates.append(os.path.join(pipeline_env, TEXT_ENCODER_SUBDIR))

    bundle_pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/models--ACE-Step--Ace-Step1.5/snapshots/*/" + TEXT_ENCODER_SUBDIR
    )
    candidates.extend(sorted(glob.glob(bundle_pattern), reverse=True))

    qwen_pattern = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/*")
    candidates.extend(sorted(glob.glob(qwen_pattern), reverse=True))

    seen: set[str] = set()
    weighted: list[str] = []
    config_only: list[str] = []
    for candidate in candidates:
        if candidate in seen or not os.path.isdir(candidate):
            continue
        seen.add(candidate)
        if _has_text_encoder_weights(candidate):
            weighted.append(candidate)
        elif os.path.isfile(os.path.join(candidate, "config.json")):
            config_only.append(candidate)

    if weighted:
        return weighted[0]
    if not require_weights and config_only:
        return config_only[0]

    raise FileNotFoundError(
        "Qwen3-Embedding-0.6B weights not found. Set ACESTEP_TEXT_ENCODER_PATH or ACESTEP_PIPELINE_DIR, "
        "download ACE-Step/Ace-Step1.5 (full bundle), or cache Qwen/Qwen3-Embedding-0.6B."
    )


def have_text_encoder_weights() -> bool:
    try:
        resolve_text_encoder_path(require_weights=True)
    except FileNotFoundError:
        return False
    return True


def _build_metadata_string(
    *,
    bpm: int | None = None,
    keyscale: str | None = None,
    timesignature: str | None = None,
    audio_duration: float | None = None,
) -> str:
    bpm_str = str(bpm) if bpm is not None and bpm > 0 else "N/A"
    ts_str = timesignature if timesignature and timesignature.strip() else "N/A"
    ks_str = keyscale if keyscale and keyscale.strip() else "N/A"
    if audio_duration is not None and audio_duration > 0:
        dur_str = f"{int(audio_duration)} seconds"
    else:
        dur_str = "30 seconds"
    return f"- bpm: {bpm_str}\n- timesignature: {ts_str}\n- keyscale: {ks_str}\n- duration: {dur_str}\n"


def format_prompt_and_lyrics(
    prompt: str,
    lyrics: str = "",
    *,
    vocal_language: str = "en",
    audio_duration: float = 60.0,
    instruction: str | None = None,
    bpm: int | None = None,
    keyscale: str | None = None,
    timesignature: str | None = None,
) -> tuple[str, str]:
    """Apply ACE-Step processor string templates before tokenization."""
    instr = instruction or DEFAULT_DIT_INSTRUCTION
    if not instr.endswith(":"):
        instr = instr + ":"
    metas = _build_metadata_string(
        bpm=bpm,
        keyscale=keyscale,
        timesignature=timesignature,
        audio_duration=audio_duration,
    )
    formatted_text = SFT_GEN_PROMPT.format(instr, prompt, metas)
    formatted_lyrics = f"# Languages\n{vocal_language}\n\n# Lyric\n{lyrics}<|endoftext|>"
    return formatted_text, formatted_lyrics


@dataclass
class AceStepTextEncoderBundle:
    text_encoder: torch.nn.Module
    tokenizer: object
    device: torch.device


_text_encoder_bundle: AceStepTextEncoderBundle | None = None


def load_text_encoder_bundle(*, device: str | torch.device = "cpu") -> AceStepTextEncoderBundle:
    """Load (and cache) the HF Qwen3-Embedding model + tokenizer."""
    global _text_encoder_bundle
    device = torch.device(device)
    if (
        _text_encoder_bundle is not None
        and _text_encoder_bundle.device == device
        and next(_text_encoder_bundle.text_encoder.parameters()).device == device
    ):
        return _text_encoder_bundle

    from transformers import AutoModel, AutoTokenizer

    path = resolve_text_encoder_path()
    _log_progress(f"text_encode: loading Qwen3-Embedding from {path}")
    tokenizer = AutoTokenizer.from_pretrained(path)
    text_encoder = AutoModel.from_pretrained(path, torch_dtype=torch.float32)
    text_encoder.eval()
    text_encoder.to(device)
    _text_encoder_bundle = AceStepTextEncoderBundle(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        device=device,
    )
    return _text_encoder_bundle


def _normalize_batch(
    values: str | Sequence[str] | None,
    *,
    batch_size: int,
    default: str,
) -> list[str]:
    if values is None:
        return [default] * batch_size
    if isinstance(values, str):
        return [values] * batch_size
    out = list(values)
    if not out:
        return [default] * batch_size
    if len(out) == 1 and batch_size > 1:
        return out * batch_size
    if len(out) != batch_size:
        raise ValueError(f"expected {batch_size} strings, got {len(out)}")
    return out


@torch.no_grad()
def encode_text_conditioning(
    *,
    prompts: str | Sequence[str] | None = None,
    lyrics: str | Sequence[str] | None = None,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    vocal_language: str | Sequence[str] = "en",
    audio_duration: float = 60.0,
    instruction: str | None = None,
    bpm: int | None = None,
    keyscale: str | None = None,
    timesignature: str | None = None,
    max_text_length: int = DEFAULT_MAX_TEXT_LEN,
    max_lyric_length: int = DEFAULT_MAX_LYRIC_LEN,
    text_encoder_bundle: AceStepTextEncoderBundle | None = None,
) -> dict[str, torch.Tensor]:
    """Encode live caption + lyrics into DiT conditioning tensors."""
    bundle = text_encoder_bundle or load_text_encoder_bundle(device=device)
    text_encoder = bundle.text_encoder
    tokenizer = bundle.tokenizer
    dev = bundle.device

    prompt_list = _normalize_batch(
        prompts,
        batch_size=batch_size,
        default="upbeat electronic dance track with energetic drums",
    )
    lyric_list = _normalize_batch(lyrics, batch_size=batch_size, default="[verse]\nInstrumental\n[chorus]\n")

    if isinstance(vocal_language, str):
        lang_list = [vocal_language] * batch_size
    else:
        lang_list = list(vocal_language)
        if len(lang_list) == 1 and batch_size > 1:
            lang_list = lang_list * batch_size
        if len(lang_list) != batch_size:
            raise ValueError(f"expected {batch_size} vocal_language values, got {len(lang_list)}")

    text_strs: list[str] = []
    lyric_strs: list[str] = []
    for i in range(batch_size):
        text_str, lyric_str = format_prompt_and_lyrics(
            prompt_list[i],
            lyric_list[i],
            vocal_language=lang_list[i],
            audio_duration=audio_duration,
            instruction=instruction,
            bpm=bpm,
            keyscale=keyscale,
            timesignature=timesignature,
        )
        text_strs.append(text_str)
        lyric_strs.append(lyric_str)

    text_inputs = tokenizer(
        text_strs,
        padding="longest",
        truncation=True,
        max_length=max_text_length,
        return_tensors="pt",
    )
    lyric_inputs = tokenizer(
        lyric_strs,
        padding="longest",
        truncation=True,
        max_length=max_lyric_length,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids.to(dev)
    text_attention_mask = text_inputs.attention_mask.to(dev)
    lyric_input_ids = lyric_inputs.input_ids.to(dev)
    lyric_attention_mask = lyric_inputs.attention_mask.to(dev)

    text_hidden_states = text_encoder(input_ids=text_input_ids).last_hidden_state
    embed_layer = text_encoder.get_input_embeddings()
    lyric_hidden_states = embed_layer(lyric_input_ids)

    mask_dtype = dtype if dtype.is_floating_point else torch.float32
    return {
        "text_hidden_states": text_hidden_states.to(dtype=dtype),
        "text_attention_mask": text_attention_mask.to(dtype=mask_dtype),
        "lyric_hidden_states": lyric_hidden_states.to(dtype=dtype),
        "lyric_attention_mask": lyric_attention_mask.to(dtype=mask_dtype),
    }


@torch.no_grad()
def encode_text_conditioning_hf_reference(
    *,
    prompts: str | Sequence[str] | None = None,
    lyrics: str | Sequence[str] | None = None,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    vocal_language: str | Sequence[str] = "en",
    audio_duration: float = 60.0,
    instruction: str | None = None,
    bpm: int | None = None,
    keyscale: str | None = None,
    timesignature: str | None = None,
    max_text_length: int = DEFAULT_MAX_TEXT_LEN,
    max_lyric_length: int = DEFAULT_MAX_LYRIC_LEN,
) -> dict[str, torch.Tensor]:
    """Independent HF reference path for golden tests (no module-level cache)."""
    from transformers import AutoModel, AutoTokenizer

    path = resolve_text_encoder_path()
    dev = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(path)
    text_encoder = AutoModel.from_pretrained(path, torch_dtype=torch.float32).eval().to(dev)

    prompt_list = _normalize_batch(
        prompts,
        batch_size=batch_size,
        default="upbeat electronic dance track with energetic drums",
    )
    lyric_list = _normalize_batch(lyrics, batch_size=batch_size, default="[verse]\nInstrumental\n[chorus]\n")

    if isinstance(vocal_language, str):
        lang_list = [vocal_language] * batch_size
    else:
        lang_list = list(vocal_language)
        if len(lang_list) == 1 and batch_size > 1:
            lang_list = lang_list * batch_size

    text_strs: list[str] = []
    lyric_strs: list[str] = []
    for i in range(batch_size):
        text_str, lyric_str = format_prompt_and_lyrics(
            prompt_list[i],
            lyric_list[i],
            vocal_language=lang_list[i],
            audio_duration=audio_duration,
            instruction=instruction,
            bpm=bpm,
            keyscale=keyscale,
            timesignature=timesignature,
        )
        text_strs.append(text_str)
        lyric_strs.append(lyric_str)

    text_inputs = tokenizer(
        text_strs,
        padding="longest",
        truncation=True,
        max_length=max_text_length,
        return_tensors="pt",
    )
    lyric_inputs = tokenizer(
        lyric_strs,
        padding="longest",
        truncation=True,
        max_length=max_lyric_length,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids.to(dev)
    text_attention_mask = text_inputs.attention_mask.to(dev)
    lyric_input_ids = lyric_inputs.input_ids.to(dev)
    lyric_attention_mask = lyric_inputs.attention_mask.to(dev)

    text_hidden_states = text_encoder(input_ids=text_input_ids).last_hidden_state
    lyric_hidden_states = text_encoder.get_input_embeddings()(lyric_input_ids)

    mask_dtype = dtype if dtype.is_floating_point else torch.float32
    return {
        "text_hidden_states": text_hidden_states.to(dtype=dtype),
        "text_attention_mask": text_attention_mask.to(dtype=mask_dtype),
        "lyric_hidden_states": lyric_hidden_states.to(dtype=dtype),
        "lyric_attention_mask": lyric_attention_mask.to(dtype=mask_dtype),
    }

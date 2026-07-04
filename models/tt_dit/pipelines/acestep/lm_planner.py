# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Phase 7A: ACE-Step 5Hz LM planner (host reference path).

Generates discrete ``audio_codes`` from caption/lyrics via ``acestep-5Hz-lm-*``,
maps them through the HF quantizer → TT detokenizer path (skipping Call B tokenizer).

**Variant selection** (see `awesome-ace-step` LM table):

| Variant | Use case |
|---------|----------|
| ``0.6B`` | Fast iteration / CI smoke when weights are cached |
| ``1.7B`` | **Default production** — balance of quality and latency |
| ``4B`` | Highest quality when VRAM/time budget allows |

CLI: ``--lm-model {0.6B,1.7B,4B}`` · env: ``ACESTEP_LM_PLANNER_MODEL=acestep-5Hz-lm-1.7B``

Constrained decoding (ACE-Step FSM, on by default): ``ACESTEP_USE_CONSTRAINED_LM_DECODING=1``
Uses ``MetadataConstrainedLogitsProcessor`` for CoT metadata + audio-code phases on host and TT.

Upstream reference: ``ace-step/ACE-Step-1.5`` ``LLMHandler`` + ``AudioCodesMixin``.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import torch

import ttnn

DEFAULT_LM_INSTRUCTION = "Generate audio semantic tokens based on the given conditions:"
AUDIO_CODE_MAX = 63999
_CODES_PER_SECOND = 5  # 5 Hz planner @ 25 Hz latent frames / pool_window_size=5
_LOG_PATH = "/tmp/acestep_agent_7.log"

LM_VARIANTS = {
    "0.6B": "acestep-5Hz-lm-0.6B",
    "1.7B": "acestep-5Hz-lm-1.7B",
    "4B": "acestep-5Hz-lm-4B",
}
DEFAULT_LM_VARIANT = "1.7B"
_WEIGHT_FILENAMES = ("model.safetensors", "pytorch_model.bin", "model.bin")
_STOP_REASONING_TAG = "</think>"
_AUDIO_CODE_PATTERN = re.compile(r"<\|audio_code_(\d+)\|>")


def _log_progress(message: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"{ts} {message}\n"
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError:
        pass
    print(line, end="", flush=True)


def default_use_lm_planner() -> bool:
    value = os.environ.get("ACESTEP_USE_LM_PLANNER")
    if value is not None:
        return value.strip().lower() in ("1", "true", "yes")
    return False


def default_use_tt_lm_planner() -> bool:
    value = os.environ.get("ACESTEP_USE_TT_LM_PLANNER")
    if value is not None:
        return value.strip().lower() in ("1", "true", "yes")
    return False


def default_lm_variant() -> str:
    env = os.environ.get("ACESTEP_LM_PLANNER_MODEL", DEFAULT_LM_VARIANT)
    env = env.strip()
    if env.startswith("acestep-5Hz-lm-"):
        env = env.removeprefix("acestep-5Hz-lm-")
    if env not in LM_VARIANTS:
        return DEFAULT_LM_VARIANT
    return env


def lm_planner_sets_is_covers() -> bool:
    """When True, LM hints replace silence ``src_latents`` (semantic fill)."""
    value = os.environ.get("ACESTEP_LM_PLANNER_IS_COVERS")
    if value is not None:
        return value.strip().lower() in ("1", "true", "yes")
    return True


def _has_lm_weights(path: str) -> bool:
    if not os.path.isfile(os.path.join(path, "config.json")):
        return False
    return any(os.path.isfile(os.path.join(path, name)) for name in _WEIGHT_FILENAMES)


def resolve_lm_planner_path(*, model: str | None = None, require_weights: bool = True) -> str:
    """Resolve a local directory for ``acestep-5Hz-lm-{0.6B,1.7B,4B}``."""
    variant = model or default_lm_variant()
    if variant.startswith("acestep-5Hz-lm-"):
        variant = variant.removeprefix("acestep-5Hz-lm-")
    if variant not in LM_VARIANTS:
        raise ValueError(f"Unknown LM variant {variant!r}; expected one of {sorted(LM_VARIANTS)}")

    subdir = LM_VARIANTS[variant]
    env_path = os.environ.get("ACESTEP_LM_PLANNER_PATH")
    if env_path:
        if not os.path.isdir(env_path):
            raise FileNotFoundError(f"ACESTEP_LM_PLANNER_PATH is not a directory: {env_path}")
        if require_weights and not _has_lm_weights(env_path):
            raise FileNotFoundError(f"ACESTEP_LM_PLANNER_PATH missing model weights: {env_path}")
        return env_path

    candidates: list[str] = []
    pipeline_env = os.environ.get("ACESTEP_PIPELINE_DIR")
    if pipeline_env:
        candidates.append(os.path.join(pipeline_env, subdir))

    bundle_pattern = os.path.expanduser(f"~/.cache/huggingface/hub/models--ACE-Step--Ace-Step1.5/snapshots/*/{subdir}")
    candidates.extend(sorted(glob.glob(bundle_pattern), reverse=True))

    gtobar_default = f"/local/ttuser/gtobar/acestep_pipeline/{subdir}"
    if os.path.isdir(gtobar_default):
        candidates.append(gtobar_default)

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen or not os.path.isdir(candidate):
            continue
        seen.add(candidate)
        if not require_weights or _has_lm_weights(candidate):
            return candidate

    raise FileNotFoundError(
        f"{subdir} weights not found. Set ACESTEP_LM_PLANNER_PATH or ACESTEP_PIPELINE_DIR, "
        f"download ACE-Step/Ace-Step1.5 bundle, or cache the model under ~/.cache/huggingface/hub."
    )


def have_lm_planner_weights(*, model: str | None = None) -> bool:
    try:
        resolve_lm_planner_path(model=model, require_weights=True)
        return True
    except FileNotFoundError:
        return False


def parse_audio_code_string(code_str: str) -> list[int]:
    """Extract integer audio codes from tokens like ``<|audio_code_123|>``."""
    if not code_str:
        return []
    codes: list[int] = []
    for match in _AUDIO_CODE_PATTERN.finditer(code_str):
        value = int(match.group(1))
        codes.append(max(0, min(value, AUDIO_CODE_MAX)))
    return codes


def parse_lm_output(output_text: str) -> tuple[dict[str, Any], str]:
    """Parse CoT metadata + serialized audio code string from LM output."""
    metadata: dict[str, Any] = {}
    code_matches = _AUDIO_CODE_PATTERN.findall(output_text)
    audio_codes = "".join(f"<|audio_code_{int(x)}|>" for x in code_matches)

    reasoning_patterns = (
        r"<think>(.*?)</think>",
        r"<reasoning>(.*?)</reasoning>",
    )
    reasoning_text = None
    for pattern in reasoning_patterns:
        match = re.search(pattern, output_text, re.DOTALL)
        if match:
            reasoning_text = match.group(1).strip()
            break
    if reasoning_text is None:
        reasoning_text = output_text.split("<|audio_code_")[0].strip()

    current_key: str | None = None
    current_value_lines: list[str] = []

    def _save_field() -> None:
        nonlocal current_key, current_value_lines
        if current_key and current_value_lines:
            value = "\n".join(current_value_lines).strip()
            metadata[current_key] = value
        current_key = None
        current_value_lines = []

    for line in reasoning_text.split("\n"):
        if line.strip().startswith("<"):
            continue
        if line and not line[0].isspace() and ":" in line:
            _save_field()
            key, val = line.split(":", 1)
            current_key = key.strip().lower()
            if val.strip():
                current_value_lines.append(val)
        elif (line.startswith(" ") or line.startswith("\t")) and current_key:
            current_value_lines.append(line)

    _save_field()
    if "bpm" in metadata:
        try:
            metadata["bpm"] = int(str(metadata["bpm"]).strip())
        except ValueError:
            pass
    if "duration" in metadata:
        try:
            metadata["duration"] = int(str(metadata["duration"]).strip())
        except ValueError:
            pass
    return metadata, audio_codes


def build_formatted_prompt_cot(tokenizer, caption: str, lyrics: str = "") -> str:
    user_prompt = f"# Caption\n{caption}\n\n# Lyric\n{lyrics}\n"
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": f"# Instruction\n{DEFAULT_LM_INSTRUCTION}\n\n"},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_formatted_prompt_codes(tokenizer, caption: str, lyrics: str, cot_text: str) -> str:
    user_prompt = f"# Caption\n{caption}\n\n# Lyric\n{lyrics}\n"
    formatted = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": f"# Instruction\n{DEFAULT_LM_INSTRUCTION}\n\n"},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": cot_text},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    if not formatted.endswith("\n"):
        formatted += "\n"
    return formatted


def _target_code_count(*, audio_duration: float | None, metadata: dict[str, Any]) -> int:
    if audio_duration is not None and audio_duration > 0:
        return max(1, int(round(audio_duration * _CODES_PER_SECOND)))
    if metadata.get("duration"):
        try:
            return max(1, int(metadata["duration"]) * _CODES_PER_SECOND)
        except (TypeError, ValueError):
            pass
    return 50


def _fit_code_string(code_str: str, target_codes: int) -> str:
    codes = parse_audio_code_string(code_str)
    if len(codes) >= target_codes:
        codes = codes[:target_codes]
    elif codes:
        last = codes[-1]
        codes.extend([last] * (target_codes - len(codes)))
    else:
        codes = [0] * target_codes
    return "".join(f"<|audio_code_{idx}|>" for idx in codes)


@dataclass
class _HostLMState:
    model: Any
    tokenizer: Any
    weight_path: str
    constrained_processor: Any | None = None


_STATE: _HostLMState | None = None


def _get_constrained_processor(state: _HostLMState) -> Any | None:
    from models.tt_dit.pipelines.acestep.lm_planner_constrained import (
        create_constrained_processor,
        default_use_constrained_decoding,
    )

    if not default_use_constrained_decoding():
        return None
    if state.constrained_processor is None:
        state.constrained_processor = create_constrained_processor(state.tokenizer)
    return state.constrained_processor


def _sample_next_token(logits: torch.Tensor, *, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    scaled = logits / max(temperature, 1e-5)
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _generate_text(
    state: _HostLMState,
    formatted_prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    stop_suffix: str | None = None,
    seed: int | None = None,
    generation_phase: str = "cot",
    target_duration: float | None = None,
    stop_at_reasoning: bool = False,
) -> str:
    from models.tt_dit.pipelines.acestep.lm_planner_constrained import configure_codes_phase, configure_cot_phase

    tokenizer = state.tokenizer
    model = state.model
    processor = _get_constrained_processor(state)
    if processor is not None:
        if generation_phase == "codes":
            configure_codes_phase(processor, enabled=True, target_duration=target_duration)
        else:
            configure_cot_phase(processor, enabled=True, stop_at_reasoning=stop_at_reasoning)

    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    if seed is not None:
        torch.manual_seed(int(seed))

    generated_ids = input_ids.clone()
    past_key_values = None
    eos_id = tokenizer.eos_token_id
    prompt_len = int(input_ids.shape[1])

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            if past_key_values is None:
                model_input = generated_ids
                model_mask = attention_mask
            else:
                model_input = generated_ids[:, -1:]
                model_mask = attention_mask[:, -1:]

            outputs = model(
                input_ids=model_input,
                attention_mask=model_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_logits = outputs.logits[:, -1, :].float()
            if processor is not None:
                next_logits = processor(generated_ids, next_logits)
            next_token = _sample_next_token(next_logits, temperature=temperature)
            if processor is not None:
                processor.update_state(int(next_token.item()))

            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token, dtype=attention_mask.dtype)],
                dim=1,
            )
            past_key_values = outputs.past_key_values

            token_id = int(next_token.item())
            if eos_id is not None and token_id == eos_id:
                break
            if processor is not None and processor.state.name == "COMPLETED":
                break

            new_text = tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=False)
            if stop_suffix and stop_suffix in new_text:
                break

    new_tokens = generated_ids[0, prompt_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    if stop_suffix and stop_suffix in text:
        text = text.split(stop_suffix, 1)[0] + stop_suffix
    return text


def _load_host_lm(*, model: str | None = None) -> _HostLMState:
    global _STATE
    if _STATE is not None:
        return _STATE

    from transformers import AutoModelForCausalLM, AutoTokenizer

    weight_path = resolve_lm_planner_path(model=model)
    _log_progress(f"lm_planner: loading host LM from {weight_path}")
    tokenizer = AutoTokenizer.from_pretrained(weight_path, trust_remote_code=True)
    lm_model = AutoModelForCausalLM.from_pretrained(
        weight_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    lm_model.eval()
    _STATE = _HostLMState(model=lm_model, tokenizer=tokenizer, weight_path=weight_path)
    return _STATE


@torch.no_grad()
def generate_audio_codes_host(
    *,
    caption: str,
    lyrics: str = "",
    audio_duration: float | None = None,
    model: str | None = None,
    temperature: float = 0.85,
    seed: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Two-phase host LM: CoT metadata then audio code tokens."""
    state = _load_host_lm(model=model)
    cot_prompt = build_formatted_prompt_cot(state.tokenizer, caption, lyrics)
    cot_text = _generate_text(
        state,
        cot_prompt,
        max_new_tokens=512,
        temperature=temperature,
        stop_suffix=_STOP_REASONING_TAG,
        seed=seed,
        generation_phase="cot",
        stop_at_reasoning=True,
    )
    if _STOP_REASONING_TAG not in cot_text:
        cot_text = cot_text.rstrip() + f"\n{_STOP_REASONING_TAG}"
    metadata, _ = parse_lm_output(cot_text)

    codes_prompt = build_formatted_prompt_codes(state.tokenizer, caption, lyrics, cot_text)
    target_codes = _target_code_count(audio_duration=audio_duration, metadata=metadata)
    codes_duration = audio_duration if audio_duration is not None and audio_duration > 0 else None
    if codes_duration is None and metadata.get("duration"):
        try:
            codes_duration = float(metadata["duration"])
        except (TypeError, ValueError):
            codes_duration = None
    codes_text = _generate_text(
        state,
        codes_prompt,
        max_new_tokens=min(target_codes + 32, 512),
        temperature=temperature,
        seed=seed,
        generation_phase="codes",
        target_duration=codes_duration,
    )
    _, audio_codes = parse_lm_output(codes_text)
    audio_codes = _fit_code_string(audio_codes, target_codes)
    _log_progress(
        f"lm_planner: generated {len(parse_audio_code_string(audio_codes))} codes "
        f"(target={target_codes}) metadata_keys={list(metadata.keys())}"
    )
    return audio_codes, metadata


@torch.no_grad()
def generate_audio_codes(
    *,
    caption: str,
    lyrics: str = "",
    audio_duration: float | None = None,
    model: str | None = None,
    temperature: float = 0.85,
    seed: int | None = None,
    mesh_device: ttnn.Device | ttnn.MeshDevice | None = None,
    use_tt: bool | None = None,
) -> tuple[str, dict[str, Any]]:
    """Generate audio codes on TT when ``use_tt`` and ``mesh_device`` are set, else host."""
    if use_tt is None:
        use_tt = default_use_tt_lm_planner()
    if use_tt and mesh_device is not None:
        from models.tt_dit.pipelines.acestep.lm_planner_tt import generate_audio_codes_tt

        return generate_audio_codes_tt(
            mesh_device,
            caption=caption,
            lyrics=lyrics,
            audio_duration=audio_duration,
            model=model,
            temperature=temperature,
            seed=seed,
        )
    return generate_audio_codes_host(
        caption=caption,
        lyrics=lyrics,
        audio_duration=audio_duration,
        model=model,
        temperature=temperature,
        seed=seed,
    )


def audio_codes_to_indices_tensor(
    code_str: str,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return quantizer indices ``[1, T_code, 1]``."""
    code_ids = parse_audio_code_string(code_str)
    if not code_ids:
        raise ValueError("audio code string is empty")
    indices = torch.tensor(code_ids, device=device, dtype=torch.long)
    return indices.unsqueeze(0).unsqueeze(-1)


@torch.no_grad()
def audio_codes_to_lm_quantized(
    hf_model,
    code_str: str,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Map serialized codes → quantizer output (Call B bypass)."""
    quantizer = hf_model.tokenizer.quantizer
    indices = audio_codes_to_indices_tensor(code_str, device=device)
    quantized = quantizer.get_output_from_indices(indices)
    return quantized.to(dtype=dtype)


@torch.no_grad()
def audio_codes_to_lm_hints_25hz(
    hf_model,
    code_str: str,
    *,
    target_latent_length: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Host golden: codes → quantizer → detokenizer → ``lm_hints_25Hz``."""
    quantized = audio_codes_to_lm_quantized(hf_model, code_str, dtype=dtype)
    lm_hints = hf_model.detokenize(quantized)
    if target_latent_length is not None:
        lm_hints = lm_hints[:, :target_latent_length, :]
    return lm_hints.to(dtype=dtype)

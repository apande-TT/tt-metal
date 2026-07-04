# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Phase 7B: acestep-5Hz-lm on TT via ``tt_transformers`` Qwen3 decode stack."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoTokenizer

import ttnn
from models.tt_dit.pipelines.acestep.lm_planner import (
    _STOP_REASONING_TAG,
    _fit_code_string,
    _target_code_count,
    build_formatted_prompt_codes,
    build_formatted_prompt_cot,
    parse_audio_code_string,
    parse_lm_output,
    resolve_lm_planner_path,
)
from models.tt_dit.pipelines.acestep.lm_planner_constrained import (
    configure_codes_phase,
    configure_cot_phase,
    create_constrained_processor,
    default_use_constrained_decoding,
)
from models.tt_transformers.tt.common import PagedAttentionConfig, create_tt_model, num_blocks_in_seq, sample_host
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import DecodersPrecision

_LOG_PATH = "/tmp/acestep_agent_7b.log"
_TT_MAX_SEQ_LEN = 2048
_TT_BLOCK_SIZE = 32


def _log_progress(message: str) -> None:
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"{ts} {message}\n"
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError:
        pass
    print(line, end="", flush=True)


def _build_page_table(max_seq_len: int, *, max_num_blocks: int = 1024) -> torch.Tensor:
    num_blocks = min(num_blocks_in_seq(max_seq_len, _TT_BLOCK_SIZE), max_num_blocks)
    permutation = torch.randperm(max_num_blocks)
    reverse = torch.argsort(permutation)
    return reverse[:num_blocks].unsqueeze(0)


@dataclass
class _TTLMState:
    generator: Generator
    model_args: object
    kv_cache: list
    page_table: torch.Tensor
    tokenizer: object
    weight_path: str
    constrained_processor: Any | None = None
    max_num_blocks: int = 1024


@dataclass
class _TTGenerationSession:
    state: _TTLMState
    token_ids: list[int] = field(default_factory=list)
    prompt_len: int = 0


_STATE_BY_DEVICE: dict[int, _TTLMState] = {}


def release_tt_lm_session(mesh_device: ttnn.Device | ttnn.MeshDevice) -> None:
    """Drop cached TT LM weights/KV for this device (tests / isolation)."""
    _STATE_BY_DEVICE.pop(id(mesh_device), None)


def _get_constrained_processor(state: _TTLMState) -> Any | None:
    if not default_use_constrained_decoding():
        return None
    if state.constrained_processor is None:
        state.constrained_processor = create_constrained_processor(state.tokenizer)
    return state.constrained_processor


def _load_tt_lm(
    mesh_device: ttnn.Device | ttnn.MeshDevice,
    *,
    model: str | None = None,
) -> _TTLMState:
    cache_key = id(mesh_device)
    cached = _STATE_BY_DEVICE.get(cache_key)
    if cached is not None:
        return cached

    weight_path = resolve_lm_planner_path(model=model)
    prev_hf_model = os.environ.get("HF_MODEL")
    os.environ["HF_MODEL"] = weight_path
    try:
        _log_progress(f"lm_planner_tt: loading TT Qwen3 LM from {weight_path}")
        max_num_blocks = max(1024, num_blocks_in_seq(_TT_MAX_SEQ_LEN, _TT_BLOCK_SIZE))
        paged_attention_config = PagedAttentionConfig(
            block_size=_TT_BLOCK_SIZE,
            max_num_blocks=max_num_blocks,
        )
        optimizations = lambda model_args: DecodersPrecision.accuracy(model_args.n_layers, model_args.model_name)
        model_args, tt_model, tt_kv_cache, _state_dict = create_tt_model(
            mesh_device,
            instruct=False,
            max_batch_size=1,
            optimizations=optimizations,
            max_seq_len=_TT_MAX_SEQ_LEN,
            paged_attention_config=paged_attention_config,
            dtype=ttnn.bfloat16,
        )
    finally:
        if prev_hf_model is None:
            os.environ.pop("HF_MODEL", None)
        else:
            os.environ["HF_MODEL"] = prev_hf_model

    page_table = _build_page_table(_TT_MAX_SEQ_LEN, max_num_blocks=max_num_blocks)
    tokenizer = AutoTokenizer.from_pretrained(weight_path, trust_remote_code=True)
    generator = Generator([tt_model], [model_args], mesh_device, tokenizer=tokenizer)
    state = _TTLMState(
        generator=generator,
        model_args=model_args,
        kv_cache=tt_kv_cache,
        page_table=page_table,
        tokenizer=tokenizer,
        weight_path=weight_path,
        max_num_blocks=max_num_blocks,
    )
    _STATE_BY_DEVICE[cache_key] = state
    return state


def _begin_session(mesh_device: ttnn.Device | ttnn.MeshDevice, *, model: str | None) -> _TTGenerationSession:
    state = _load_tt_lm(mesh_device, model=model)
    return _TTGenerationSession(state=state)


def _reset_kv(state: _TTLMState) -> None:
    state.page_table = _build_page_table(_TT_MAX_SEQ_LEN, max_num_blocks=state.max_num_blocks)


def _next_token_from_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    processor: Any | None,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    scores = logits.float()
    if processor is not None:
        scores = processor(input_ids, scores)
    if temperature <= 0:
        token = scores.argmax(dim=-1, keepdim=True).to(torch.int64)
    else:
        _, token = sample_host(scores, temperature=temperature, top_p=0.08, on_host=True)
        token = token.to(torch.int64)
    if token.ndim == 1:
        token = token.unsqueeze(-1)
    return token


def _generate_phase(
    session: _TTGenerationSession,
    *,
    formatted_prompt: str,
    max_new_tokens: int,
    temperature: float,
    generation_phase: str,
    target_duration: float | None,
    stop_suffix: str | None = None,
    stop_at_reasoning: bool = False,
    continue_kv: bool = False,
) -> str:
    processor = _get_constrained_processor(session.state)
    if processor is not None:
        if generation_phase == "codes":
            configure_codes_phase(processor, enabled=True, target_duration=target_duration)
        else:
            configure_cot_phase(processor, enabled=True, stop_at_reasoning=stop_at_reasoning)

    state = session.state
    tokenizer = state.tokenizer
    prompt_tokens = tokenizer(formatted_prompt, return_tensors="pt").input_ids
    prompt_list = prompt_tokens[0].tolist()
    eos_id = tokenizer.eos_token_id

    if continue_kv and session.token_ids and prompt_list == session.token_ids:
        _log_progress(
            f"lm_planner_tt: KV reuse — continuing decode at pos={len(session.token_ids)} " f"phase={generation_phase}"
        )
        out_tok = torch.tensor([[session.token_ids[-1]]], dtype=torch.long)
        current_pos = torch.tensor([len(session.token_ids) - 1])
        generated_start = len(session.token_ids)
        decode_steps = max_new_tokens
        reset_batch_next = True
    else:
        _reset_kv(state)
        seq_len = int(prompt_tokens.shape[1])
        session.token_ids = prompt_list.copy()
        session.prompt_len = seq_len
        generated_start = seq_len

        logits = state.generator.prefill_forward_text(
            prompt_tokens,
            page_table=state.page_table,
            kv_cache=[state.kv_cache],
            prompt_lens=torch.tensor([seq_len]),
            enable_trace=False,
            warmup_prefill=True,
        )
        input_ids = torch.tensor([session.token_ids], dtype=torch.long)
        out_tok = _next_token_from_logits(
            logits,
            temperature=temperature,
            processor=processor,
            input_ids=input_ids,
        )
        token_id = int(out_tok.reshape(-1)[0].item())
        if processor is not None:
            processor.update_state(token_id)
        session.token_ids.append(token_id)
        current_pos = torch.tensor([seq_len])
        decode_steps = max(0, max_new_tokens - 1)
        reset_batch_next = True

        new_text = tokenizer.decode(session.token_ids[generated_start:], skip_special_tokens=False)
        if eos_id is not None and token_id == eos_id:
            return _trim_stop_suffix(new_text, stop_suffix)
        if processor is not None and processor.state.name == "COMPLETED":
            return _trim_stop_suffix(new_text, stop_suffix)
        if stop_suffix and stop_suffix in new_text:
            return _trim_stop_suffix(new_text, stop_suffix)

    for decode_idx in range(decode_steps):
        logits, _log_probs = state.generator.decode_forward(
            out_tok.reshape(1, 1),
            current_pos,
            enable_trace=False,
            page_table=state.page_table,
            kv_cache=[state.kv_cache],
            reset_batch=reset_batch_next,
            sampling_params=None,
            prompt_tokens=prompt_tokens,
            output_tokens=out_tok.reshape(1, 1),
        )
        reset_batch_next = False
        input_ids = torch.tensor([session.token_ids], dtype=torch.long)
        out_tok = _next_token_from_logits(
            logits,
            temperature=temperature,
            processor=processor,
            input_ids=input_ids,
        )
        token_id = int(out_tok.reshape(-1)[0].item())
        if processor is not None:
            processor.update_state(token_id)
        session.token_ids.append(token_id)
        current_pos = current_pos + 1

        if eos_id is not None and token_id == eos_id:
            break
        if processor is not None and processor.state.name == "COMPLETED":
            break
        new_text = tokenizer.decode(session.token_ids[generated_start:], skip_special_tokens=False)
        if stop_suffix and stop_suffix in new_text:
            break

    new_text = tokenizer.decode(session.token_ids[generated_start:], skip_special_tokens=False)
    return _trim_stop_suffix(new_text, stop_suffix)


def _trim_stop_suffix(text: str, stop_suffix: str | None) -> str:
    if stop_suffix and stop_suffix in text:
        return text.split(stop_suffix, 1)[0] + stop_suffix
    return text


@torch.no_grad()
def prefill_last_token_logits_tt(
    mesh_device: ttnn.Device | ttnn.MeshDevice,
    formatted_prompt: str,
    *,
    model: str | None = None,
) -> torch.Tensor:
    """Return TT prefill logits for the last prompt token ``[vocab]``."""
    try:
        session = _begin_session(mesh_device, model=model)
        _reset_kv(session.state)
        prompt_tokens = session.state.tokenizer(formatted_prompt, return_tensors="pt").input_ids
        seq_len = int(prompt_tokens.shape[1])
        logits = session.state.generator.prefill_forward_text(
            prompt_tokens,
            page_table=session.state.page_table,
            kv_cache=[session.state.kv_cache],
            prompt_lens=torch.tensor([seq_len]),
            enable_trace=False,
            warmup_prefill=True,
        )
        return logits.reshape(-1).float().cpu()
    finally:
        release_tt_lm_session(mesh_device)


@torch.no_grad()
def generate_audio_codes_tt(
    mesh_device: ttnn.Device | ttnn.MeshDevice,
    *,
    caption: str,
    lyrics: str = "",
    audio_duration: float | None = None,
    model: str | None = None,
    temperature: float = 0.85,
    seed: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Two-phase TT LM planner: CoT metadata then audio code tokens (single model session)."""
    if seed is not None:
        torch.manual_seed(int(seed))

    session = _begin_session(mesh_device, model=model)
    tokenizer = session.state.tokenizer
    cot_prompt = build_formatted_prompt_cot(tokenizer, caption, lyrics)
    cot_text = _generate_phase(
        session,
        formatted_prompt=cot_prompt,
        max_new_tokens=512,
        temperature=temperature,
        generation_phase="cot",
        target_duration=None,
        stop_suffix=_STOP_REASONING_TAG,
        stop_at_reasoning=True,
    )
    if _STOP_REASONING_TAG not in cot_text:
        cot_text = cot_text.rstrip() + f"\n{_STOP_REASONING_TAG}"
    metadata, _ = parse_lm_output(cot_text)

    codes_prompt = build_formatted_prompt_codes(tokenizer, caption, lyrics, cot_text)
    target_codes = _target_code_count(audio_duration=audio_duration, metadata=metadata)
    codes_duration = audio_duration if audio_duration is not None and audio_duration > 0 else None
    if codes_duration is None and metadata.get("duration"):
        try:
            codes_duration = float(metadata["duration"])
        except (TypeError, ValueError):
            codes_duration = None

    codes_prompt_ids = tokenizer(codes_prompt, return_tensors="pt").input_ids[0].tolist()
    can_continue = bool(session.token_ids) and codes_prompt_ids == session.token_ids

    codes_text = _generate_phase(
        session,
        formatted_prompt=codes_prompt,
        max_new_tokens=min(target_codes + 32, 512),
        temperature=temperature,
        generation_phase="codes",
        target_duration=codes_duration,
        continue_kv=can_continue,
    )
    _, audio_codes = parse_lm_output(codes_text)
    audio_codes = _fit_code_string(audio_codes, target_codes)
    _log_progress(
        f"lm_planner_tt: generated {len(parse_audio_code_string(audio_codes))} codes "
        f"(target={target_codes}) kv_reuse={can_continue} metadata_keys={list(metadata.keys())}"
    )
    return audio_codes, metadata

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Phase 7B: acestep-5Hz-lm on TT via ``tt_transformers`` Qwen3 decode stack."""
from __future__ import annotations

import os
from dataclasses import dataclass
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


_STATE_BY_DEVICE: dict[int, _TTLMState] = {}


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
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(weight_path, trust_remote_code=True)
    generator = Generator([tt_model], [model_args], mesh_device, tokenizer=tokenizer)
    state = _TTLMState(
        generator=generator,
        model_args=model_args,
        kv_cache=tt_kv_cache,
        page_table=page_table,
        tokenizer=tokenizer,
        weight_path=weight_path,
    )
    _STATE_BY_DEVICE[cache_key] = state
    return state


def _next_token_from_logits(logits: torch.Tensor, *, temperature: float) -> torch.Tensor:
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    _, token = sample_host(logits, temperature=temperature, top_p=0.08, on_host=True)
    if token.ndim == 1:
        token = token.unsqueeze(-1)
    return token.to(torch.int64)


def _generate_text_tt(
    mesh_device: ttnn.Device | ttnn.MeshDevice,
    *,
    model: str | None,
    formatted_prompt: str,
    max_new_tokens: int,
    temperature: float,
    stop_suffix: str | None = None,
    seed: int | None = None,
) -> str:
    if seed is not None:
        torch.manual_seed(int(seed))

    state = _load_tt_lm(mesh_device, model=model)
    try:
        tokenizer = state.tokenizer
        generator = state.generator
        prompt_tokens = tokenizer(formatted_prompt, return_tensors="pt").input_ids
        seq_len = int(prompt_tokens.shape[1])
        eos_id = tokenizer.eos_token_id

        logits = generator.prefill_forward_text(
            prompt_tokens,
            page_table=state.page_table,
            kv_cache=[state.kv_cache],
            prompt_lens=torch.tensor([seq_len]),
            enable_trace=False,
            warmup_prefill=True,
        )
        out_tok = _next_token_from_logits(logits, temperature=temperature)
        generated_ids = prompt_tokens[0].tolist() + [int(out_tok.reshape(-1)[0].item())]

        current_pos = torch.tensor([seq_len])
        prompt_for_decode = prompt_tokens

        for iteration in range(1, max_new_tokens):
            logits, _log_probs = generator.decode_forward(
                out_tok.reshape(1, 1),
                current_pos,
                enable_trace=False,
                page_table=state.page_table,
                kv_cache=[state.kv_cache],
                reset_batch=(iteration == 1),
                sampling_params=None,
                prompt_tokens=prompt_for_decode,
                output_tokens=out_tok.reshape(1, 1),
            )
            out_tok = _next_token_from_logits(logits, temperature=temperature)
            token_id = int(out_tok.reshape(-1)[0].item())
            generated_ids.append(token_id)
            current_pos = current_pos + 1

            if eos_id is not None and token_id == eos_id:
                break

            new_text = tokenizer.decode(generated_ids[seq_len:], skip_special_tokens=False)
            if stop_suffix and stop_suffix in new_text:
                break

        new_text = tokenizer.decode(generated_ids[seq_len:], skip_special_tokens=False)
        if stop_suffix and stop_suffix in new_text:
            new_text = new_text.split(stop_suffix, 1)[0] + stop_suffix
        return new_text
    finally:
        _STATE_BY_DEVICE.pop(id(mesh_device), None)


@torch.no_grad()
def prefill_last_token_logits_tt(
    mesh_device: ttnn.Device | ttnn.MeshDevice,
    formatted_prompt: str,
    *,
    model: str | None = None,
) -> torch.Tensor:
    """Return TT prefill logits for the last prompt token ``[vocab]``."""
    state = _load_tt_lm(mesh_device, model=model)
    try:
        prompt_tokens = state.tokenizer(formatted_prompt, return_tensors="pt").input_ids
        seq_len = int(prompt_tokens.shape[1])
        logits = state.generator.prefill_forward_text(
            prompt_tokens,
            page_table=state.page_table,
            kv_cache=[state.kv_cache],
            prompt_lens=torch.tensor([seq_len]),
            enable_trace=False,
            warmup_prefill=True,
        )
        return logits.reshape(-1).float().cpu()
    finally:
        _STATE_BY_DEVICE.pop(id(mesh_device), None)


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
    """Two-phase TT LM planner: CoT metadata then audio code tokens."""
    tokenizer = AutoTokenizer.from_pretrained(resolve_lm_planner_path(model=model), trust_remote_code=True)
    cot_prompt = build_formatted_prompt_cot(tokenizer, caption, lyrics)
    cot_text = _generate_text_tt(
        mesh_device,
        model=model,
        formatted_prompt=cot_prompt,
        max_new_tokens=512,
        temperature=temperature,
        stop_suffix=_STOP_REASONING_TAG,
        seed=seed,
    )
    if _STOP_REASONING_TAG not in cot_text:
        cot_text = cot_text.rstrip() + f"\n{_STOP_REASONING_TAG}"
    metadata, _ = parse_lm_output(cot_text)

    codes_prompt = build_formatted_prompt_codes(tokenizer, caption, lyrics, cot_text)
    target_codes = _target_code_count(audio_duration=audio_duration, metadata=metadata)
    codes_text = _generate_text_tt(
        mesh_device,
        model=model,
        formatted_prompt=codes_prompt,
        max_new_tokens=min(target_codes + 32, 512),
        temperature=temperature,
        seed=(seed + 1) if seed is not None else None,
    )
    _, audio_codes = parse_lm_output(codes_text)
    audio_codes = _fit_code_string(audio_codes, target_codes)
    _log_progress(
        f"lm_planner_tt: generated {len(parse_audio_code_string(audio_codes))} codes "
        f"(target={target_codes}) metadata_keys={list(metadata.keys())}"
    )
    return audio_codes, metadata

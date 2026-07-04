# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Phase 2C: Qwen3-Embedding-0.6B text conditioning on TT via ``tt_transformers``.

Host reference (golden): ``text_encode.encode_text_conditioning``.
ACE-Step needs full-sequence ``last_hidden_state`` for captions and ``embed_tokens``
for lyrics — not the last-token pooling path used by vLLM embedding APIs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

import torch

import ttnn
from models.tt_dit.pipelines.acestep.text_encode import (
    DEFAULT_MAX_LYRIC_LEN,
    DEFAULT_MAX_TEXT_LEN,
    _normalize_batch,
    format_prompt_and_lyrics,
    load_text_encoder_bundle,
    resolve_text_encoder_path,
)
from models.tt_transformers.tt.common import (
    Mode,
    PagedAttentionConfig,
    create_tt_model,
    get_padded_prefill_len,
    num_blocks_in_seq,
)
from models.tt_transformers.tt.model_config import DecodersPrecision

_LOG_PATH = "/tmp/acestep_agent_2c.log"
_TT_MAX_SEQ_LEN = max(DEFAULT_MAX_TEXT_LEN, DEFAULT_MAX_LYRIC_LEN)
_TT_BLOCK_SIZE = 32


def _log_progress(message: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"{ts} {message}\n"
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError:
        pass
    print(line, end="", flush=True)


def default_use_tt_text_encode() -> bool:
    """Return True when TT Qwen3 text encode is enabled (Phase 2C gate passed)."""
    value = os.environ.get("ACESTEP_USE_TT_TEXT_ENCODE")
    if value is not None:
        return value.strip().lower() in ("1", "true", "yes")
    return True


@dataclass
class _TTTextEncoderState:
    model: object
    model_args: object
    kv_cache: list
    page_table: torch.Tensor
    tokenizer: object
    weight_path: str


_STATE_BY_DEVICE: dict[int, _TTTextEncoderState] = {}


def _build_page_table(seq_len: int, *, max_num_blocks: int = 1024) -> torch.Tensor:
    num_blocks = num_blocks_in_seq(seq_len, _TT_BLOCK_SIZE)
    num_blocks = min(num_blocks, max_num_blocks)
    return torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)


def _load_tt_text_encoder(mesh_device: ttnn.Device | ttnn.MeshDevice) -> _TTTextEncoderState:
    cache_key = id(mesh_device)
    cached = _STATE_BY_DEVICE.get(cache_key)
    if cached is not None:
        return cached

    weight_path = resolve_text_encoder_path()
    prev_hf_model = os.environ.get("HF_MODEL")
    os.environ["HF_MODEL"] = weight_path
    try:
        _log_progress(f"text_encode_tt: loading TT Qwen3-Embedding from {weight_path}")
        paged_attention_config = PagedAttentionConfig(
            block_size=_TT_BLOCK_SIZE,
            max_num_blocks=max(1024, num_blocks_in_seq(_TT_MAX_SEQ_LEN, _TT_BLOCK_SIZE)),
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

    page_table = _build_page_table(_TT_MAX_SEQ_LEN)
    tokenizer = model_args.tokenizer
    state = _TTTextEncoderState(
        model=tt_model,
        model_args=model_args,
        kv_cache=tt_kv_cache,
        page_table=page_table,
        tokenizer=tokenizer,
        weight_path=weight_path,
    )
    _STATE_BY_DEVICE[cache_key] = state
    return state


def _tt_tensor_to_torch(model, tt_tensor: ttnn.Tensor, seq_len: int) -> torch.Tensor:
    ttnn.synchronize_device(model.mesh_device)
    host_tensor = tt_tensor.cpu(blocking=True) if tt_tensor.storage_type() != ttnn.StorageType.HOST else tt_tensor
    concatenated = model.concat_host_output(host_tensor)
    hidden_dim = model.args.dim
    if concatenated.shape[-1] > hidden_dim:
        hidden = concatenated[0, 0, :seq_len, :hidden_dim]
    else:
        hidden = concatenated[0, 0, :seq_len, :]
    return hidden.float().unsqueeze(0)


def _prefill_full_hidden_states(
    state: _TTTextEncoderState,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Run Qwen3 decoder prefill and return post-norm hidden states ``[1, S, H]``."""
    model = state.model
    seq_len = int(input_ids.shape[1])
    padded_len = get_padded_prefill_len(seq_len)
    tokens = input_ids.to(torch.int64)
    if padded_len > seq_len:
        pad_value = 0
        pad = torch.full((tokens.shape[0], padded_len - seq_len), pad_value, dtype=tokens.dtype)
        tokens = torch.cat([tokens, pad], dim=1)

    page_table = _build_page_table(padded_len)
    (
        prefill_input,
        rot_mats_global_prefill,
        rot_mats_local_prefill,
        page_table_tt,
        *_rest,
    ) = model.prepare_inputs_prefill(
        tokens,
        page_table=page_table,
        batch_size=1,
        user_id=0,
        last_token_idx=padded_len - 1,
    )

    tt_hidden = model.ttnn_prefill_forward(
        prefill_input,
        rot_mats_global=rot_mats_global_prefill,
        rot_mats_local=rot_mats_local_prefill,
        user_id=0,
        page_table=page_table_tt,
        get_last_token=-1,
        kv_cache=state.kv_cache,
        batch_size=1,
    )
    tt_hidden = model.norm(
        tt_hidden,
        mode=Mode.PREFILL,
        norm_config=model.args.get_norm_config("lm_head", Mode.PREFILL, model.prefetcher),
    )
    tt_hidden = ttnn.to_layout(tt_hidden, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return _tt_tensor_to_torch(model, tt_hidden, seq_len)


def _embed_tokens_tt(state: _TTTextEncoderState, input_ids: torch.Tensor) -> torch.Tensor:
    """Token embedding lookup on TT — matches HF ``get_input_embeddings()(ids)``."""
    model = state.model
    batch, seq_len = input_ids.shape
    tokens = input_ids.reshape(1, 1, 1, batch * seq_len).to(torch.int64)
    tt_ids = ttnn.from_torch(
        tokens,
        device=model.mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(model.mesh_device),
    )
    tt_embd = model.embd(tt_ids)
    ttnn.synchronize_device(model.mesh_device)
    if model.args.num_devices > 1:
        torch_embd = ttnn.to_torch(
            ttnn.get_device_tensors(tt_embd)[0],
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                model.mesh_device,
                dims=(1, 3) if model.args.is_galaxy else (0, 1),
                mesh_shape=model.args.cluster_shape,
            ),
        )
    else:
        torch_embd = ttnn.to_torch(tt_embd)
    hidden_dim = model.args.dim
    flat = torch_embd.float().reshape(batch, seq_len, -1)
    if flat.shape[-1] > hidden_dim:
        flat = flat[..., :hidden_dim]
    return flat


@torch.no_grad()
def encode_text_conditioning_tt(
    mesh_device: ttnn.Device | ttnn.MeshDevice,
    *,
    prompts: str | Sequence[str] | None = None,
    lyrics: str | Sequence[str] | None = None,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float32,
    vocal_language: str | Sequence[str] = "en",
    audio_duration: float = 60.0,
    instruction: str | None = None,
    bpm: int | None = None,
    keyscale: str | None = None,
    timesignature: str | None = None,
    max_text_length: int = DEFAULT_MAX_TEXT_LEN,
    max_lyric_length: int = DEFAULT_MAX_LYRIC_LEN,
) -> dict[str, torch.Tensor]:
    """Encode caption + lyrics on TT; returns the same keys as host ``encode_text_conditioning``."""
    state = _load_tt_text_encoder(mesh_device)
    host_bundle = load_text_encoder_bundle(device="cpu")
    tokenizer = host_bundle.tokenizer

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

    text_hidden_states = _prefill_full_hidden_states(state, text_inputs.input_ids)
    lyric_hidden_states = _embed_tokens_tt(state, lyric_inputs.input_ids)

    mask_dtype = dtype if dtype.is_floating_point else torch.float32
    return {
        "text_hidden_states": text_hidden_states.to(dtype=dtype),
        "text_attention_mask": text_inputs.attention_mask.to(dtype=mask_dtype),
        "lyric_hidden_states": lyric_hidden_states.to(dtype=dtype),
        "lyric_attention_mask": lyric_inputs.attention_mask.to(dtype=mask_dtype),
    }

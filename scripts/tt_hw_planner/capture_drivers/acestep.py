"""ACE-Step capture driver for ``capture_inputs.py``.

``AceStepConditionGenerationModel`` is a diffusion/TTS model whose forward
requires a multi-tensor conditioning set (text/lyric/timbre latents, source
latents, chunk masks, etc.). Generic ``pixel_values`` / ``input_ids`` drivers
do not exercise its encoder/decoder subgraphs.

Tensor shapes follow ``test_forward`` in the cached HF modeling file, scaled
down via ``TT_PLANNER_CAPTURE_ACESTEP_SEQ_LEN`` (default 50).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from . import register_capture_driver


def is_acestep_model(model: Any) -> bool:
    """Return True when *model* looks like ACE-Step v1.5."""
    cfg = getattr(model, "config", None)
    model_type = str(getattr(cfg, "model_type", "") or "").lower()
    if model_type == "acestep":
        return True
    cls_name = type(model).__name__
    return "AceStepConditionGeneration" in cls_name or cls_name == "AceStepConditionGenerationModel"


def _cfg_int(cfg: Any, name: str, default: int) -> int:
    val = getattr(cfg, name, None)
    return int(val) if val is not None else default


def build_acestep_forward_kwargs(
    model: Any,
    *,
    batch_size: Optional[int] = None,
    seq_len: Optional[int] = None,
) -> Dict[str, Any]:
    """Build kwargs for ``training_losses`` / ``forward`` from model config."""
    import torch

    cfg = getattr(model, "config", None)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    if batch_size is None:
        try:
            batch_size = int(os.environ.get("TT_PLANNER_CAPTURE_ACESTEP_BATCH", "1"))
        except ValueError:
            batch_size = 1
    if seq_len is None:
        try:
            seq_len = int(os.environ.get("TT_PLANNER_CAPTURE_ACESTEP_SEQ_LEN", "50"))
        except ValueError:
            seq_len = 50

    pool_window = _cfg_int(cfg, "pool_window_size", 5)
    if seq_len % pool_window != 0:
        seq_len += pool_window - (seq_len % pool_window)

    text_dim = _cfg_int(cfg, "text_hidden_dim", 1024)
    audio_dim = _cfg_int(cfg, "audio_acoustic_hidden_dim", 64)
    timbre_dim = _cfg_int(cfg, "timbre_hidden_dim", 64)
    timbre_frames = min(_cfg_int(cfg, "timbre_fix_frame", 750), max(seq_len * 2, 32))

    text_len = max(16, min(77, seq_len))
    lyric_len = max(24, min(123, seq_len + 8))

    text_hidden_states = torch.randn(batch_size, text_len, text_dim, dtype=dtype, device=device)
    text_attention_mask = torch.ones(batch_size, text_len, dtype=dtype, device=device)
    lyric_hidden_states = torch.randn(batch_size, lyric_len, text_dim, dtype=dtype, device=device)
    lyric_attention_mask = torch.ones(batch_size, lyric_len, dtype=dtype, device=device)

    packed_n = batch_size
    refer_audio = torch.randn(packed_n, timbre_frames, timbre_dim, dtype=dtype, device=device)
    refer_audio_order_mask = torch.arange(packed_n, dtype=torch.long, device=device)

    hidden_states = torch.randn(batch_size, seq_len, audio_dim, dtype=dtype, device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=dtype, device=device)
    chunk_masks = torch.ones(batch_size, seq_len, audio_dim, dtype=dtype, device=device)
    silence_latent = torch.randn(batch_size, seq_len, audio_dim, dtype=dtype, device=device)
    src_latents = torch.randn(batch_size, seq_len, audio_dim, dtype=dtype, device=device)
    is_covers = torch.zeros(batch_size, dtype=torch.long, device=device)

    return {
        "hidden_states": hidden_states,
        "attention_mask": attention_mask,
        "text_hidden_states": text_hidden_states,
        "text_attention_mask": text_attention_mask,
        "lyric_hidden_states": lyric_hidden_states,
        "lyric_attention_mask": lyric_attention_mask,
        "refer_audio_acoustic_hidden_states_packed": refer_audio,
        "refer_audio_order_mask": refer_audio_order_mask,
        "src_latents": src_latents,
        "chunk_masks": chunk_masks,
        "is_covers": is_covers,
        "silence_latent": silence_latent,
        "cfg_ratio": 0.0,
    }


def _prefer_cpu_friendly_attention(model: Any) -> None:
    """Use SDPA on CPU — flash_attention_2 is unavailable there."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return
    device = next(model.parameters()).device
    impl = getattr(cfg, "_attn_implementation", None)
    if device.type == "cpu" and impl in ("flash_attention_2", "flash_attention"):
        cfg._attn_implementation = "sdpa"


def drive_acestep(model: Any, pixel_values: Any = None) -> None:
    """Run ACE-Step-specific capture paths (hooks must already be installed)."""
    import torch

    if not is_acestep_model(model):
        return

    _prefer_cpu_friendly_attention(model)
    kwargs = build_acestep_forward_kwargs(model)
    cfg = model.config
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    pool_window = _cfg_int(cfg, "pool_window_size", 5)
    audio_dim = _cfg_int(cfg, "audio_acoustic_hidden_dim", 64)
    hidden_size = _cfg_int(cfg, "hidden_size", 2048)
    fsq_dim = _cfg_int(cfg, "fsq_dim", hidden_size)

    with torch.no_grad():
        if hasattr(model, "training_losses"):
            model.training_losses(**kwargs)
        else:
            model(**kwargs)

        if hasattr(model, "generate_audio"):
            gen_kw = dict(kwargs)
            gen_kw.pop("hidden_states", None)
            gen_kw.pop("cfg_ratio", None)
            model.generate_audio(
                **gen_kw,
                infer_steps=1,
                use_progress_bar=False,
                seed=0,
            )

        encoder = getattr(model, "encoder", None)
        enc_out = None
        if encoder is not None:
            enc_out = encoder(
                text_hidden_states=kwargs["text_hidden_states"],
                text_attention_mask=kwargs["text_attention_mask"],
                lyric_hidden_states=kwargs["lyric_hidden_states"],
                lyric_attention_mask=kwargs["lyric_attention_mask"],
                refer_audio_acoustic_hidden_states_packed=kwargs["refer_audio_acoustic_hidden_states_packed"],
                refer_audio_order_mask=kwargs["refer_audio_order_mask"],
            )
            lyric_enc = getattr(encoder, "lyric_encoder", None)
            if lyric_enc is not None:
                lyric_enc(
                    inputs_embeds=kwargs["lyric_hidden_states"],
                    attention_mask=kwargs["lyric_attention_mask"],
                )
            timbre_enc = getattr(encoder, "timbre_encoder", None)
            if timbre_enc is not None:
                timbre_enc(
                    refer_audio_acoustic_hidden_states_packed=kwargs["refer_audio_acoustic_hidden_states_packed"],
                    refer_audio_order_mask=kwargs["refer_audio_order_mask"],
                )

        decoder = getattr(model, "decoder", None)
        if decoder is not None and enc_out is not None:
            enc_hidden, enc_mask = enc_out[0], enc_out[1]
            bsz = kwargs["hidden_states"].shape[0]
            context_latents = torch.cat([kwargs["src_latents"], kwargs["chunk_masks"].to(dtype)], dim=-1)
            xt = torch.randn_like(kwargs["hidden_states"])
            t = torch.full((bsz,), 0.5, device=device, dtype=dtype)
            decoder(
                hidden_states=xt,
                timestep=t,
                timestep_r=t,
                attention_mask=kwargs["attention_mask"],
                encoder_hidden_states=enc_hidden,
                encoder_attention_mask=enc_mask,
                context_latents=context_latents,
                use_cache=False,
            )

        tokenizer = getattr(model, "tokenizer", None)
        seq_len = kwargs["src_latents"].shape[1]
        n_patch = seq_len // pool_window
        if tokenizer is not None:
            tok_in = torch.randn(
                kwargs["src_latents"].shape[0],
                n_patch,
                pool_window,
                audio_dim,
                dtype=dtype,
                device=device,
            )
            tokenizer(tok_in)
            pooler = getattr(tokenizer, "attention_pooler", None)
            if pooler is not None:
                pooler_in = torch.randn(
                    kwargs["src_latents"].shape[0],
                    n_patch,
                    pool_window,
                    hidden_size,
                    dtype=dtype,
                    device=device,
                )
                pooler(pooler_in)

        detokenizer = getattr(model, "detokenizer", None)
        if detokenizer is not None:
            detok_in = torch.randn(
                kwargs["src_latents"].shape[0],
                n_patch,
                fsq_dim,
                dtype=dtype,
                device=device,
            )
            detokenizer(detok_in)

        if hasattr(model, "tokenize"):
            model.tokenize(kwargs["src_latents"], kwargs["silence_latent"], kwargs["attention_mask"])
        if hasattr(model, "detokenize"):
            fake_q = torch.randn(
                kwargs["src_latents"].shape[0],
                n_patch,
                hidden_size,
                dtype=dtype,
                device=device,
            )
            model.detokenize(fake_q)


def run_acestep_capture_drivers(model: Any) -> Tuple[bool, List[str]]:
    """ACE-Step driver with per-path attempt logging for ``capture_inputs``."""
    attempts: List[str] = []
    if not is_acestep_model(model):
        return False, attempts
    try:
        drive_acestep(model)
        attempts.append("acestep: ok")
        return True, attempts
    except Exception as exc:
        attempts.append(f"acestep: {type(exc).__name__}: {exc}")
        return False, attempts


@register_capture_driver(matcher=is_acestep_model)
def _registered_acestep_driver(model: Any, pixel_values: Any = None) -> None:
    drive_acestep(model, pixel_values)

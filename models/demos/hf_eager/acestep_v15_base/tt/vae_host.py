# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Host-side ACE-Step VAE (AutoencoderOobleck) bridge.

Decode: DiT target_latents [B, T, C] -> stereo PCM [B, 2, samples].
Encode: reference WAV -> cover-mode tensors for ``generate_audio``.
"""
from __future__ import annotations

import glob
import math
import os
from datetime import datetime, timezone
from typing import Any

import torch

DEFAULT_SAMPLE_RATE = 48000
PHASE2B_LOG = "/tmp/acestep_agent_2b.log"
ACOUSTIC_DIM = 64


def _log_phase2b(message: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {message}\n"
    try:
        with open(PHASE2B_LOG, "a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError:
        pass


def resolve_vae_path() -> str:
    env_path = os.environ.get("ACESTEP_VAE_PATH")
    if env_path:
        if not os.path.isdir(env_path):
            raise FileNotFoundError(f"ACESTEP_VAE_PATH is not a directory: {env_path}")
        return env_path

    pattern = os.path.expanduser("~/.cache/huggingface/hub/models--ACE-Step--Ace-Step1.5/snapshots/*/vae")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError("ACE-Step VAE not found. Set ACESTEP_VAE_PATH or download ACE-Step/Ace-Step1.5.")
    return matches[-1]


def load_oobleck_vae(vae_path: str | None = None, *, device: str | torch.device = "cpu"):
    from diffusers import AutoencoderOobleck

    path = vae_path or resolve_vae_path()
    vae = AutoencoderOobleck.from_pretrained(path, torch_dtype=torch.float32)
    vae.eval()
    return vae.to(device)


def latents_per_second(vae) -> float:
    downsample = math.prod(getattr(vae.config, "downsampling_ratios", (1920,)))
    sample_rate = int(getattr(vae.config, "sampling_rate", DEFAULT_SAMPLE_RATE))
    return float(sample_rate) / float(downsample)


def load_wav(wav_path: str, sample_rate: int = DEFAULT_SAMPLE_RATE) -> torch.Tensor:
    """Load WAV as stereo float32 tensor [2, samples] resampled to ``sample_rate``."""
    _log_phase2b(f"load_wav path={wav_path}")

    audio = None
    loaded_sr = sample_rate

    try:
        import torchaudio

        waveform, loaded_sr = torchaudio.load(wav_path)
        audio = waveform
    except Exception:
        pass

    if audio is None:
        try:
            import soundfile as sf

            data, loaded_sr = sf.read(wav_path, always_2d=True)
            audio = torch.from_numpy(data.T).float()
        except Exception:
            pass

    if audio is None:
        from scipy.io import wavfile

        loaded_sr, data = wavfile.read(wav_path)
        if data.ndim == 1:
            data = data[:, None]
        peak = max(float(abs(data.min())), float(abs(data.max())), 1.0)
        audio = torch.from_numpy(data.T).float() / peak

    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)

    if int(loaded_sr) != sample_rate:
        try:
            import torchaudio

            audio = torchaudio.functional.resample(audio, int(loaded_sr), sample_rate)
        except Exception as exc:
            raise RuntimeError(f"WAV sample rate {loaded_sr} != {sample_rate} and torchaudio resample failed") from exc

    return audio.contiguous()


def _vae_encode_audio(
    vae,
    audio: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Encode stereo audio [2, samples] or [B, 2, samples] -> latents [B, T, 64]."""
    if audio.ndim == 2:
        audio = audio.unsqueeze(0)
    if audio.ndim != 3 or audio.shape[1] != 2:
        raise ValueError(f"expected audio [B,2,samples], got {tuple(audio.shape)}")

    audio = audio.to(device=next(vae.parameters()).device, dtype=vae.dtype)
    latents = vae.encode(audio).latent_dist.sample(generator=generator)
    return latents.transpose(1, 2).to(dtype=dtype)


def _crop_reference_segments(reference_audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Mirror diffusers ``prepare_reference_audio_latents`` 30 s front/mid/back crop."""
    target_frames = 30 * sample_rate
    if reference_audio.shape[-1] < target_frames:
        repeat_times = math.ceil(target_frames / reference_audio.shape[-1])
        reference_audio = reference_audio.repeat(1, repeat_times)

    segment_frames = 10 * sample_rate
    total_frames = reference_audio.shape[-1]
    segment_size = total_frames // 3

    front = reference_audio[:, :segment_frames]
    mid_start = segment_size
    middle = reference_audio[:, mid_start : mid_start + segment_frames]
    back_start = max(total_frames - segment_frames, 0)
    back = reference_audio[:, back_start : back_start + segment_frames]
    return torch.cat([front, middle, back], dim=-1)


def _prepare_reference_latents(
    reference_audio: torch.Tensor,
    vae,
    *,
    batch_size: int,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    sample_rate = int(getattr(vae.config, "sampling_rate", DEFAULT_SAMPLE_RATE))
    cropped = _crop_reference_segments(reference_audio, sample_rate)
    ref_latents = _vae_encode_audio(vae, cropped.unsqueeze(0), generator=generator, dtype=dtype)
    refer_audio = ref_latents.expand(batch_size, -1, -1).contiguous()
    refer_order_mask = torch.arange(batch_size, dtype=torch.long)
    return refer_audio, refer_order_mask


def _pad_latents_for_pool(src_latents: torch.Tensor, pool_window_size: int) -> torch.Tensor:
    pad_len = (-src_latents.shape[1]) % pool_window_size
    if pad_len == 0:
        return src_latents
    pad = torch.zeros(
        src_latents.shape[0],
        pad_len,
        src_latents.shape[2],
        device=src_latents.device,
        dtype=src_latents.dtype,
    )
    return torch.cat([src_latents, pad], dim=1)


def _prepare_cover_src_latents(
    src_audio: torch.Tensor,
    vae,
    hf_model,
    *,
    batch_size: int,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, int]:
    src_latents = _vae_encode_audio(vae, src_audio.unsqueeze(0), generator=generator, dtype=dtype)
    latent_length = src_latents.shape[1]
    if src_latents.shape[0] == 1:
        src_latents = src_latents.expand(batch_size, -1, -1)

    pool_window_size = int(getattr(hf_model.config, "pool_window_size", 5))
    padded = _pad_latents_for_pool(src_latents, pool_window_size)
    quantized, _ = hf_model.tokenizer.tokenize(padded)
    src_latents = hf_model.detokenize(quantized).to(dtype=dtype)
    src_latents = src_latents[:, :latent_length, :].contiguous()
    return src_latents, latent_length


def _build_cover_chunk_masks(
    batch_size: int,
    latent_length: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.ones(batch_size, latent_length, ACOUSTIC_DIM, dtype=dtype)


_learned_silence_latent: torch.Tensor | None = None
DEFAULT_OUTPUT_DURATION_SEC = 30.0


def resolve_silence_latent_path() -> str:
    env_path = os.environ.get("ACESTEP_SILENCE_LATENT_PATH")
    if env_path:
        if not os.path.isfile(env_path):
            raise FileNotFoundError(f"ACESTEP_SILENCE_LATENT_PATH is not a file: {env_path}")
        return env_path

    patterns = (
        "~/.cache/huggingface/hub/models--ACE-Step--acestep-v15-xl-turbo-diffusers/snapshots/*/condition_encoder/diffusion_pytorch_model.safetensors",
        "~/.cache/huggingface/hub/models--ACE-Step--acestep-v15-base-diffusers/snapshots/*/condition_encoder/diffusion_pytorch_model.safetensors",
    )
    for pattern in patterns:
        matches = sorted(glob.glob(os.path.expanduser(pattern)))
        if matches:
            return matches[-1]
    raise FileNotFoundError(
        "Learned silence_latent not found. Download ACE-Step/acestep-v15-xl-turbo-diffusers "
        "or set ACESTEP_SILENCE_LATENT_PATH to a condition_encoder safetensors file."
    )


def load_learned_silence_latent(*, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Load the VAE-encoded silence buffer used for cover/text2music src context."""
    global _learned_silence_latent
    if _learned_silence_latent is not None:
        return _learned_silence_latent.to(dtype=dtype)

    from safetensors import safe_open

    path = resolve_silence_latent_path()
    with safe_open(path, framework="pt") as sf:
        if "silence_latent" not in sf.keys():
            raise KeyError(f"silence_latent not found in {path}")
        tensor = sf.get_tensor("silence_latent").to(dtype=dtype)
    _learned_silence_latent = tensor
    _log_phase2b(f"load_learned_silence_latent shape={tuple(tensor.shape)} from {path}")
    return tensor


def _slice_learned_silence_latent(
    learned: torch.Tensor,
    latent_length: int,
    *,
    batch_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Slice or tile learned silence to ``[B, latent_length, 64]`` (diffusers cover src)."""
    sl = learned.to(dtype=dtype)
    if sl.ndim != 3 or sl.shape[-1] != ACOUSTIC_DIM:
        raise ValueError(f"expected learned silence [1, T, {ACOUSTIC_DIM}], got {tuple(sl.shape)}")
    if sl.shape[1] >= latent_length:
        src = sl[:, :latent_length, :]
    else:
        repeats = (latent_length + sl.shape[1] - 1) // sl.shape[1]
        src = sl.repeat(1, repeats, 1)[:, :latent_length, :]
    return src.expand(batch_size, -1, -1).contiguous()


@torch.no_grad()
def encode_reference_audio(
    wav_path: str,
    *,
    hf_model=None,
    vae=None,
    batch_size: int = 1,
    task_type: str = "cover",
    seed: int = 0,
    use_same_for_src: bool = False,
    output_duration_sec: float | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    """Encode a reference WAV into cover-mode ``generate_audio`` tensors.

    Default (``use_same_for_src=False``): official reference-only cover — timbre from
    the WAV, ``src_latents`` from learned ``silence_latent``, output length from
    ``output_duration_sec`` (default 30s). Matches diffusers ``AceStepPipeline``
    when only ``reference_audio`` is passed.

    ``use_same_for_src=True``: also derive cover ``src_latents`` from the same WAV
    (src-audio cover / repaint-style conditioning).
    """
    _log_phase2b(
        f"encode_reference_audio start wav={wav_path} task={task_type} seed={seed} "
        f"use_same_for_src={use_same_for_src}"
    )

    owned_vae = vae is None
    if owned_vae:
        vae = load_oobleck_vae()

    generator = torch.Generator().manual_seed(int(seed))
    audio = load_wav(wav_path)
    lps = latents_per_second(vae)
    ref_duration_sec = audio.shape[-1] / DEFAULT_SAMPLE_RATE

    refer_audio, refer_order_mask = _prepare_reference_latents(
        audio,
        vae,
        batch_size=batch_size,
        dtype=dtype,
        generator=generator,
    )

    learned_silence = load_learned_silence_latent(dtype=dtype)

    if task_type == "cover" and use_same_for_src:
        if hf_model is None:
            raise ValueError("cover mode requires hf_model for src_latents tokenize/detokenize")
        src_latents, latent_length = _prepare_cover_src_latents(
            audio,
            vae,
            hf_model,
            batch_size=batch_size,
            dtype=dtype,
            generator=generator,
        )
        audio_duration_sec = ref_duration_sec
        silence_latent = learned_silence.expand(batch_size, -1, -1).contiguous()
    else:
        audio_duration_sec = output_duration_sec if output_duration_sec is not None else DEFAULT_OUTPUT_DURATION_SEC
        latent_length = math.ceil(audio_duration_sec * lps)
        src_latents = _slice_learned_silence_latent(
            learned_silence,
            latent_length,
            batch_size=batch_size,
            dtype=dtype,
        )
        silence_latent = learned_silence.expand(batch_size, -1, -1).contiguous()

    chunk_masks = _build_cover_chunk_masks(batch_size, latent_length, dtype=dtype)
    attention_mask = torch.ones(batch_size, latent_length, dtype=dtype)
    # Reference-only cover: raw silence src (diffusers path) — do NOT run LM-hint swap.
    # Src-audio cover (use_same_for_src): is_covers=1 replaces src with tokenize→detokenize hints.
    is_covers = (
        torch.ones(batch_size, dtype=torch.int64) if use_same_for_src else torch.zeros(batch_size, dtype=torch.int64)
    )

    result = {
        "refer_audio_acoustic_hidden_states_packed": refer_audio,
        "refer_audio_order_mask": refer_order_mask,
        "src_latents": src_latents,
        "chunk_masks": chunk_masks,
        "attention_mask": attention_mask,
        "silence_latent": silence_latent,
        "is_covers": is_covers,
        "latent_length": latent_length,
        "audio_duration_sec": audio_duration_sec,
        "ref_duration_sec": ref_duration_sec,
    }

    _log_phase2b(
        "encode_reference_audio done "
        f"refer={tuple(refer_audio.shape)} src={tuple(src_latents.shape)} "
        f"chunk_masks={tuple(chunk_masks.shape)} T={latent_length} dur={audio_duration_sec:.1f}s"
    )
    return result


@torch.no_grad()
def latents_to_waveform(latents: torch.Tensor, vae=None) -> torch.Tensor:
    """Decode acoustic latents [B, T, C] to stereo waveform [B, 2, samples]."""
    if latents.ndim != 3:
        raise ValueError(f"expected latents [B,T,C], got shape {tuple(latents.shape)}")

    owned_vae = vae is None
    if owned_vae:
        vae = load_oobleck_vae()

    latents_bct = latents.to(dtype=torch.float32).transpose(1, 2).contiguous()
    waveform = vae.decode(latents_bct).sample
    return waveform


def save_wav(path: str, waveform: torch.Tensor, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 3 or waveform.shape[1] != 2:
        raise ValueError(f"expected waveform [B,2,samples], got shape {tuple(waveform.shape)}")

    audio = waveform.detach().cpu().float()
    if audio.shape[0] != 1:
        raise ValueError(f"save_wav supports batch size 1, got {audio.shape[0]}")

    # Match diffusers AceStepPipeline post-decode loudness (peak clip + -1 dBFS).
    peak = audio.abs().amax(dim=(1, 2), keepdim=True)
    if torch.any(peak > 1.0):
        audio = audio / peak.clamp(min=1.0)
    target_amp = 10.0 ** (-1.0 / 20.0)
    peak = audio.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
    audio = audio * (target_amp / peak)

    stereo = audio[0].T.contiguous().numpy()

    try:
        import torchaudio

        torchaudio.save(path, torch.from_numpy(stereo.T), sample_rate)
        return
    except Exception:
        pass

    try:
        from scipy.io import wavfile

        peak = max(float(abs(stereo.min())), float(abs(stereo.max())), 1e-8)
        pcm = (stereo / peak * 0.99).clip(-1.0, 1.0)
        wavfile.write(path, sample_rate, (pcm * 32767.0).astype("int16"))
        return
    except Exception:
        pass

    try:
        import soundfile as sf

        sf.write(path, stereo, sample_rate)
        return
    except Exception:
        pass

    import wave

    peak = max(float(abs(stereo.min())), float(abs(stereo.max())), 1e-8)
    pcm = (stereo / peak * 0.99).clip(-1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype("int16")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_i16.tobytes())

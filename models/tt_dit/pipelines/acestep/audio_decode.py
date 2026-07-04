# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ACE-Step latent → PCM decode bridge (host VAE or TT Oobleck).

Phase C integration point between ``AceStepPipeline`` (``target_latents``) and
listen-able waveform output. Host decode (Phase A) unblocks e2e perf before the
TT Oobleck port (Phase B) is complete.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import ttnn

_DEFAULT_SAMPLE_RATE = 48_000


def default_use_tt_vae() -> bool:
    """Default to TT Oobleck on device. Set ``ACESTEP_USE_TT_VAE=0`` for host fallback."""
    import os

    value = os.environ.get("ACESTEP_USE_TT_VAE")
    if value is not None:
        return value.strip().lower() in ("1", "true", "yes")
    return True


def decode_latents_to_waveform(
    device: ttnn.Device | ttnn.MeshDevice,
    latents_BTC: torch.Tensor,
    *,
    use_tt_vae: bool,
) -> tuple[torch.Tensor, float]:
    """Decode DiT latents ``[B, T, C]`` to stereo PCM ``[B, 2, samples]``.

    Returns ``(waveform, vae_decode_s)`` where ``vae_decode_s`` is wall time for
    the decode step only (excludes latent generation).

    When ``use_tt_vae=False``, delegates to Phase A host Oobleck
    (``vae_host.latents_to_waveform``). When ``use_tt_vae=True``, uses the TT
    ``OobleckDecoder`` from Phase B (raises ``NotImplementedError`` until wired).
    """
    start = time.perf_counter()
    if use_tt_vae:
        waveform = _decode_with_tt_oobleck(device, latents_BTC)
    else:
        waveform = _decode_with_host_oobleck(latents_BTC)
    return waveform, time.perf_counter() - start


def _decode_with_host_oobleck(latents_BTC: torch.Tensor) -> torch.Tensor:
    from models.demos.hf_eager.acestep_v15_base.tt.vae_host import latents_to_waveform

    return latents_to_waveform(latents_BTC)


# Lazy cache: one TT OobleckDecoder per mesh device (weights loaded from host checkpoint).
_tt_oobleck_decoder_by_device: dict[int, object] = {}


def _get_tt_oobleck_decoder(device: ttnn.Device | ttnn.MeshDevice):
    from models.tt_dit.models.audio_vae.vae_oobleck import OOBLECK_DECODER_PORT_COMPLETE, OobleckDecoder

    if not OOBLECK_DECODER_PORT_COMPLETE:
        raise NotImplementedError(
            "TT Oobleck VAE decode incomplete — finish Phase B (ConvTranspose padding, "
            "dilated ResUnit parity, PCC ≥ 0.99) and set OOBLECK_DECODER_PORT_COMPLETE=True."
        )

    cache_key = id(device)
    cached = _tt_oobleck_decoder_by_device.get(cache_key)
    if cached is not None:
        return cached

    from models.demos.hf_eager.acestep_v15_base.tt.vae_host import load_oobleck_vae, resolve_vae_path

    torch_vae = load_oobleck_vae(resolve_vae_path(), device="cpu")
    tt_decoder = OobleckDecoder.from_torch(torch_vae.decoder, mesh_device=device)
    _tt_oobleck_decoder_by_device[cache_key] = tt_decoder
    return tt_decoder


def _decode_with_tt_oobleck(
    device: ttnn.Device | ttnn.MeshDevice,
    latents_BTC: torch.Tensor,
) -> torch.Tensor:
    try:
        tt_decoder = _get_tt_oobleck_decoder(device)
    except ImportError as exc:
        raise NotImplementedError("TT Oobleck VAE decoder unavailable — complete Phase B (vae_oobleck.py).") from exc

    latents_BCT = latents_BTC.to(dtype=torch.float32).transpose(1, 2).contiguous()
    return tt_decoder(latents_BCT)


def save_waveform_if_requested(
    waveform: torch.Tensor,
    *,
    path: str = "/tmp/acestep_phase_c.wav",
    sample_rate: int = _DEFAULT_SAMPLE_RATE,
) -> None:
    """Persist waveform to ``path`` (used by e2e / perf tests)."""
    from models.demos.hf_eager.acestep_v15_base.tt.vae_host import save_wav

    save_wav(path, waveform, sample_rate=sample_rate)

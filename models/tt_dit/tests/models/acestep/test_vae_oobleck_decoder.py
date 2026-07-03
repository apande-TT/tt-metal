# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Phase 4: TT OobleckDecoder vs torch diffusers decoder (PCC).

Layer padding/crop fixes are in ``oobleck_layers.py``. The test runs once
``OOBLECK_DECODER_PORT_COMPLETE`` is True after the device gate passes (PCC ≥ 0.99).
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.hf_eager.acestep_v15_base.tt.vae_host import load_oobleck_vae, resolve_vae_path
from models.tt_dit.models.audio_vae.vae_oobleck import OOBLECK_DECODER_PORT_COMPLETE, OobleckDecoder
from models.tt_dit.utils.check import assert_quality

LATENT_T = 32
PCC_TARGET = 0.99


def _require_torch_decoder():
    pytest.importorskip("diffusers")
    try:
        vae = load_oobleck_vae(resolve_vae_path(), device="cpu")
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    return vae.decoder.eval()


@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "mesh_device",
    [pytest.param((1, 1), id="1x1")],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32768}],
    indirect=True,
)
def test_oobleck_decoder_pcc_vs_torch(
    mesh_device: ttnn.MeshDevice,
    silicon_arch_blackhole,
) -> None:
    """Compare TT OobleckDecoder output to torch reference on random latents."""
    if ttnn.get_num_devices() == 0:
        pytest.skip("No Tenstorrent device available")

    if not OOBLECK_DECODER_PORT_COMPLETE:
        pytest.skip(
            "Phase B scaffold: OobleckDecoder structure is wired but port is incomplete. "
            "Pending: ConvTranspose1d padding parity vs torch, dilated ResUnit center-crop, "
            "full-chain PCC ≥ 0.99. Set OOBLECK_DECODER_PORT_COMPLETE=True when fixed."
        )

    torch_decoder = _require_torch_decoder()
    tt_decoder = OobleckDecoder.from_torch(torch_decoder, mesh_device=mesh_device)

    torch.manual_seed(0)
    latents = torch.randn(1, 64, LATENT_T, dtype=torch.float32)

    with torch.no_grad():
        torch_out = torch_decoder(latents)
    tt_out = tt_decoder(latents)

    assert torch_out.shape == tt_out.shape, f"{tuple(torch_out.shape)} vs {tuple(tt_out.shape)}"
    logger.info(f"waveform shape: {tuple(tt_out.shape)}")
    assert_quality(torch_out, tt_out, pcc=PCC_TARGET)

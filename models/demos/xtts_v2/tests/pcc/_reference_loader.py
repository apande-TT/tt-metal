"""Reference loader for ``coqui/XTTS-v2``.

Strategy (see the bring-up decision tree): this repo is **config-less for
transformers** — its ``config.json`` is a Coqui trainer config with no
``model_type``/``auto_map`` key, so ``AutoConfig``/``AutoModel`` raise
``Unrecognized model``. The architecture is *not* in transformers. It ships a
native Coqui checkpoint set (``model.pth`` = GPT + HiFiGAN decoder,
``dvae.pth``, ``mel_stats.pth``, ``speakers_xtts.pth``, ``vocab.json``) that is
only understood by Coqui's own package.

So we take the "model's OWN package" path: import the maintained Coqui TTS fork
(PyPI ``coqui-tts``, imported as ``TTS``), build the native
``TTS.tts.models.xtts.Xtts`` module from the repo's ``XttsConfig`` and load the
REAL shipped weights with ``Xtts.load_checkpoint``. The returned object is a
genuine ``torch.nn.Module`` (~467M params) carrying the trained parameters,
with the two sub-modules a ttnn port targets:

    model.gpt              -> TTS.tts.layers.xtts.gpt.GPT  (GPT-2 autoregressive core)
    model.hifigan_decoder  -> TTS.tts.layers.xtts.hifigan_decoder.HifiDecoder

Both sub-module forwards have been verified to run with these weights: GPT
produces a [B, T, 1024] latent, and the HiFiGAN decoder produces a waveform.

Environment notes
-----------------
``coqui-tts==0.27.x`` targets the transformers 4.57 API, while this environment
ships transformers 5.x. Two things bridge that gap, both handled here without
disturbing the environment's numpy/torch/transformers:

  * ``transformers.pytorch_utils.isin_mps_friendly`` was removed in 5.x; Coqui's
    tortoise autoregressive module imports it. We restore an equivalent
    (``torch.isin`` on non-MPS devices, which is all this CPU/TT flow uses).
    This is applied *inside* the loader (idempotent), never at import time.

  * On torch>=2.9, ``TTS/__init__`` and ``transformers.audio_utils`` require the
    ``torchcodec`` *package metadata* to be present (``is_torchcodec_available()``
    -> reads ``importlib.metadata.version("torchcodec")``). We therefore keep
    ``torchcodec`` installed as a dependency. Note it is only ever queried for
    its version — it is never actually imported, and no audio decoding happens
    during checkpoint load or the module forwards, so a torchcodec whose native
    FFmpeg backend can't load (headless box) is still perfectly fine here.

The loader is import-safe (no work at import), deterministic (fixed shipped
weights, ``eval`` mode, grads disabled) and installs nothing unless a required
package is genuinely missing (best-effort, version-pinned so it cannot alter
the environment's numpy/torch/transformers).
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import sys

_COQUI_TTS_VERSION = "0.27.5"


def _apply_transformers_compat_shims() -> None:
    """Bridge the coqui-tts (transformers 4.57-era) API onto transformers 5.x.

    Idempotent; must run *before* ``import TTS``.
    """
    import torch

    import transformers.pytorch_utils as _pu

    if not hasattr(_pu, "isin_mps_friendly"):

        def isin_mps_friendly(elements, test_elements):  # pragma: no cover - trivial
            test_elements = torch.as_tensor(test_elements, device=elements.device)
            if elements.device.type == "mps":
                return (
                    elements.tile(test_elements.shape[0], 1)
                    .eq(test_elements.unsqueeze(1))
                    .sum(dim=0)
                    .bool()
                    .squeeze()
                )
            return torch.isin(elements, test_elements)

        _pu.isin_mps_friendly = isin_mps_friendly


def _torchcodec_metadata_present() -> bool:
    try:
        importlib.metadata.version("torchcodec")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def _pip_install(*pkgs: str, extra_args=()) -> None:
    import subprocess
    import tempfile

    # Pin numpy/torch/transformers to what is already installed so an install
    # can never silently up/downgrade them.
    pins = []
    for mod in ("numpy", "transformers", "torch"):
        try:
            ver = importlib.import_module(mod).__version__.split("+")[0]
            pins.append(f"{mod}=={ver}")
        except Exception:
            pass
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write("\n".join(pins) + "\n")
        constraints = fh.name

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-c", constraints, *pkgs, *extra_args],
        check=True,
    )


def _ensure_dependencies() -> None:
    """Import ``TTS`` after applying shims; install deps only if truly missing."""
    try:
        _apply_transformers_compat_shims()
        import TTS  # noqa: F401

        if _torchcodec_metadata_present():
            return
    except Exception:
        pass

    # Best-effort provisioning for a fresh environment.
    import torch

    torch_base = torch.__version__.split("+")[0]
    cpu_index = ("--index-url", "https://download.pytorch.org/whl/cpu")

    try:
        importlib.import_module("TTS")
    except Exception:
        _pip_install(f"coqui-tts=={_COQUI_TTS_VERSION}")

    try:
        importlib.import_module("torchaudio")
    except Exception:
        _pip_install(f"torchaudio=={torch_base}", extra_args=cpu_index)

    if not _torchcodec_metadata_present():
        # Version metadata is all transformers/TTS need (they never import it).
        _pip_install("torchcodec", extra_args=cpu_index)

    # If transformers cached torchcodec-unavailable before the install above,
    # refresh the flag. Safe now that the metadata exists.
    if _torchcodec_metadata_present():
        for modname in ("transformers.utils.import_utils", "transformers.utils"):
            try:
                mod = sys.modules.get(modname) or importlib.import_module(modname)
                if hasattr(mod, "is_torchcodec_available"):
                    mod.is_torchcodec_available = lambda: True
            except Exception:
                pass

    _apply_transformers_compat_shims()
    import TTS  # noqa: F401


def load_reference_model(model_id: str):
    """Return an ``nn.Module`` (eval mode) equivalent to the HF reference.

    Loads the native Coqui XTTS-v2 checkpoint the repo actually ships
    (``model.pth`` etc.) into ``TTS.tts.models.xtts.Xtts``. The returned module
    exposes ``.gpt`` (GPT-2 autoregressive core) and ``.hifigan_decoder``.
    """
    _ensure_dependencies()
    _apply_transformers_compat_shims()

    from huggingface_hub import snapshot_download

    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    # Fetch the real shipped weights + config from the Hub (cached if present).
    checkpoint_dir = snapshot_download(
        model_id,
        allow_patterns=[
            "config.json",
            "vocab.json",
            "model.pth",
            "dvae.pth",
            "mel_stats.pth",
            "speakers_xtts.pth",
        ],
    )

    config = XttsConfig()
    config.load_json(os.path.join(checkpoint_dir, "config.json"))

    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_dir=checkpoint_dir,
        use_deepspeed=False,
        eval=True,
    )

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model


if __name__ == "__main__":
    import torch

    m = load_reference_model("coqui/XTTS-v2")
    assert isinstance(m, torch.nn.Module), type(m)
    assert not m.training, "model must be in eval mode"
    children = [name for name, _ in m.named_children()]
    n_params = sum(p.numel() for p in m.parameters())
    print(f"loaded {type(m).__name__}: {n_params:,} params; children={children}")

    torch.manual_seed(0)
    with torch.no_grad():
        # HiFiGAN decoder forward: GPT latent [B, T, 1024] + speaker embed [B, 512, 1].
        latents = torch.randn(1, 32, 1024)
        g = torch.randn(1, 512, 1)
        wav = m.hifigan_decoder(latents, g=g)
        print(f"hifigan_decoder forward -> waveform {tuple(wav.shape)} ({wav.dtype})")

        # GPT core forward: text ids + audio codes + conditioning mel.
        gpt = m.gpt
        text_inputs = torch.randint(0, gpt.number_text_tokens, (1, 10))
        text_lengths = torch.tensor([10])
        audio_codes = torch.randint(0, gpt.num_audio_tokens, (1, 20))
        wav_lengths = torch.tensor([20 * gpt.code_stride_len])
        cond_mels = torch.randn(1, 1, 80, 22)
        cond_lens = torch.tensor([22 * gpt.perceiver_cond_length_compression])
        latent = gpt(
            text_inputs,
            text_lengths,
            audio_codes,
            wav_lengths,
            cond_mels=cond_mels,
            cond_lens=cond_lens,
            return_latent=True,
        )
        print(f"gpt forward -> latent {tuple(latent.shape)} ({latent.dtype})")
    print("SELF-CHECK OK")

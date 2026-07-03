"""Unit tests for ACE-Step capture driver (capture_drivers/acestep.py)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _fake_acestep_config(**overrides):
    base = dict(
        model_type="acestep",
        pool_window_size=5,
        patch_size=2,
        text_hidden_dim=1024,
        audio_acoustic_hidden_dim=64,
        timbre_hidden_dim=64,
        timbre_fix_frame=32,
        hidden_size=128,
        fsq_dim=128,
        in_channels=192,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_is_acestep_model_by_model_type():
    from scripts.tt_hw_planner.capture_drivers.acestep import is_acestep_model

    model = SimpleNamespace(config=SimpleNamespace(model_type="acestep"))
    assert is_acestep_model(model)


def test_is_acestep_model_by_class_name():
    from scripts.tt_hw_planner.capture_drivers.acestep import is_acestep_model

    class AceStepConditionGenerationModel:
        config = SimpleNamespace(model_type="other")

    assert is_acestep_model(AceStepConditionGenerationModel())


def test_is_acestep_model_rejects_llama():
    from scripts.tt_hw_planner.capture_drivers.acestep import is_acestep_model

    model = SimpleNamespace(config=SimpleNamespace(model_type="llama"))
    assert not is_acestep_model(model)


def test_build_acestep_forward_kwargs_shapes():
    import torch

    from scripts.tt_hw_planner.capture_drivers.acestep import build_acestep_forward_kwargs

    cfg = _fake_acestep_config()
    model = SimpleNamespace(config=cfg, parameters=lambda: iter([torch.nn.Parameter(torch.zeros(1))]))
    kw = build_acestep_forward_kwargs(model, batch_size=2, seq_len=50)

    assert kw["hidden_states"].shape == (2, 50, 64)
    assert kw["text_hidden_states"].shape == (2, 50, 1024)
    assert kw["lyric_hidden_states"].shape[0] == 2
    assert kw["refer_audio_acoustic_hidden_states_packed"].shape[0] == 2
    assert kw["chunk_masks"].shape == (2, 50, 64)
    assert kw["is_covers"].dtype == torch.long


def test_build_acestep_forward_kwargs_pads_seq_to_pool_window():
    import torch

    from scripts.tt_hw_planner.capture_drivers.acestep import build_acestep_forward_kwargs

    cfg = _fake_acestep_config(pool_window_size=5)
    model = SimpleNamespace(config=cfg, parameters=lambda: iter([torch.nn.Parameter(torch.zeros(1))]))
    kw = build_acestep_forward_kwargs(model, batch_size=1, seq_len=52)
    assert kw["hidden_states"].shape[1] == 55


def test_run_acestep_capture_drivers_on_mock_submodules():
    import torch

    from scripts.tt_hw_planner.capture_drivers.acestep import run_acestep_capture_drivers

    cfg = _fake_acestep_config(hidden_size=64, fsq_dim=64)
    param = torch.nn.Parameter(torch.zeros(1))

    lyric_encoder = MagicMock(return_value=SimpleNamespace(last_hidden_state=torch.zeros(1, 8, 64)))
    timbre_encoder = MagicMock(return_value=(torch.zeros(1, 1, 64), torch.ones(1, 1)))
    encoder = MagicMock(
        return_value=(torch.zeros(1, 16, 64), torch.ones(1, 16)),
        lyric_encoder=lyric_encoder,
        timbre_encoder=timbre_encoder,
    )
    decoder = MagicMock(return_value=(torch.zeros(1, 25, 64), None))
    tokenizer = MagicMock(
        return_value=(torch.zeros(1, 10, 64), None),
        attention_pooler=MagicMock(return_value=torch.zeros(1, 10, 64)),
    )
    detokenizer = MagicMock(return_value=torch.zeros(1, 50, 64))

    model = MagicMock(
        config=cfg,
        parameters=lambda: iter([param]),
        training_losses=MagicMock(return_value={"diffusion_loss": torch.tensor(1.0)}),
        generate_audio=MagicMock(return_value={"target_latents": torch.zeros(1, 50, 64)}),
        encoder=encoder,
        decoder=decoder,
        tokenizer=tokenizer,
        detokenizer=detokenizer,
        tokenize=MagicMock(return_value=(torch.zeros(1, 10, 64), None, torch.ones(1, 10))),
        detokenize=MagicMock(return_value=torch.zeros(1, 50, 64)),
    )

    ok, attempts = run_acestep_capture_drivers(model)
    assert ok
    assert any("acestep: ok" in a for a in attempts)
    model.training_losses.assert_called_once()
    model.generate_audio.assert_called_once()
    encoder.assert_called()
    lyric_encoder.assert_called()
    timbre_encoder.assert_called()
    decoder.assert_called()
    tokenizer.assert_called()
    detokenizer.assert_called()


def test_registered_driver_resolves_for_acestep():
    from scripts.tt_hw_planner import capture_drivers as cd

    cfg = _fake_acestep_config()
    model = SimpleNamespace(config=cfg)

    resolved = cd.resolve_custom_driver(model)
    assert resolved is not None
    assert resolved.__name__ == "_registered_acestep_driver"


@pytest.mark.skipif(
    not __import__("os").path.isdir(
        __import__("os").path.expanduser("~/.cache/huggingface/hub/models--ACE-Step--acestep-v15-base")
    ),
    reason="ACE-Step weights not in local HF cache",
)
def test_live_acestep_training_losses_runs():
    """Integration smoke test when ACE-Step is cached locally."""
    import torch

    from scripts.tt_hw_planner.agentic.probe import load_hf_model_cascade
    from scripts.tt_hw_planner.capture_drivers.acestep import drive_acestep, is_acestep_model

    model, loader = load_hf_model_cascade("ACE-Step/acestep-v15-base", torch_dtype="float32", verbose=False)
    assert model is not None, loader
    assert is_acestep_model(model)
    model.eval()
    drive_acestep(model)
    assert True

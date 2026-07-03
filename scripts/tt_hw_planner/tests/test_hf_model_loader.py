"""Tests for custom AutoModel / meta-tensor loader fallback."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest


def test_is_meta_tensor_load_error() -> None:
    from scripts.tt_hw_planner.hf_model_loader import is_meta_tensor_load_error

    assert is_meta_tensor_load_error(RuntimeError("Tensor.item() cannot be called on meta tensors"))
    assert not is_meta_tensor_load_error(RuntimeError("something else"))
    assert not is_meta_tensor_load_error(ValueError("meta tensors"))


def test_try_load_via_auto_map_no_auto_map_returns_none() -> None:
    from scripts.tt_hw_planner.hf_model_loader import try_load_via_auto_map

    cfg = mock.Mock()
    cfg.auto_map = {}
    with mock.patch("transformers.AutoConfig.from_pretrained", return_value=cfg):
        model, loader = try_load_via_auto_map("org/model")
    assert model is None
    assert loader is None


def test_try_load_via_auto_map_cpu_init_on_meta_error() -> None:
    from scripts.tt_hw_planner.hf_model_loader import try_load_via_auto_map

    cfg = mock.Mock()
    cfg.auto_map = {"AutoModel": "remote.Model"}
    sentinel = object()

    class _FakeCls:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError("Tensor.item() cannot be called on meta tensors")

    with mock.patch("transformers.AutoConfig.from_pretrained", return_value=cfg), mock.patch(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module",
        return_value=_FakeCls,
    ), mock.patch(
        "scripts.tt_hw_planner.hf_model_loader.load_via_auto_map_cpu_init",
        return_value=(sentinel, "AutoModel+cpu_init"),
    ) as cpu_init:
        model, loader = try_load_via_auto_map("ACE-Step/acestep-v15-base", verbose=False)

    assert model is sentinel
    assert loader == "AutoModel+cpu_init"
    cpu_init.assert_called_once()


def test_load_hf_model_cascade_uses_auto_map_first() -> None:
    from scripts.tt_hw_planner.agentic.probe import load_hf_model_cascade

    sentinel = object()
    with mock.patch(
        "scripts.tt_hw_planner.hf_model_loader.try_load_via_auto_map",
        return_value=(sentinel, "AutoModel+cpu_init"),
    ) as auto_load:
        model, loader = load_hf_model_cascade("ACE-Step/acestep-v15-base", verbose=False)

    assert model is sentinel
    assert loader == "AutoModel+cpu_init"
    auto_load.assert_called_once()


def test_acestep_loads_via_cpu_init_when_cached() -> None:
    """Live check when ACE-Step weights are present in the HF cache."""
    from huggingface_hub.constants import HF_HUB_CACHE

    import os

    cache_glob = os.path.join(HF_HUB_CACHE, "models--ACE-Step--acestep-v15-base")
    if not os.path.isdir(cache_glob):
        pytest.skip("ACE-Step/acestep-v15-base not in local HF cache")

    from scripts.tt_hw_planner.agentic.probe import load_hf_model_cascade

    model, loader = load_hf_model_cascade("ACE-Step/acestep-v15-base", torch_dtype="bfloat16", verbose=False)
    assert model is not None
    assert loader in ("AutoModel", "AutoModel+cpu_init")
    assert hasattr(model, "encoder")
    assert hasattr(model, "decoder")


def test_extract_hf_context_uses_load_hf_model_cascade() -> None:
    from scripts.tt_hw_planner.llm_synth import extract_hf_context

    class _FakeSubmodule:
        def forward(self, x):
            return x

        def named_parameters(self):
            return []

        def named_buffers(self):
            return []

    fake_model = mock.Mock()
    fake_sub = _FakeSubmodule()

    with mock.patch(
        "scripts.tt_hw_planner.agentic.probe.load_hf_model_cascade",
        return_value=(fake_model, "AutoModel+cpu_init"),
    ) as cascade, mock.patch(
        "scripts.tt_hw_planner.llm_synth._resolve",
        return_value=fake_sub,
    ):
        ctx = extract_hf_context(
            model_id="ACE-Step/acestep-v15-base",
            component_name="encoder",
            candidate_paths=["encoder"],
        )

    cascade.assert_called_once_with(
        "ACE-Step/acestep-v15-base",
        torch_dtype="bfloat16",
        verbose=False,
    )
    assert ctx.resolved_path == "encoder"
    assert ctx.class_name.endswith("._FakeSubmodule")


def test_autofill_stubs_op_synth_uses_load_hf_model_cascade(tmp_path: Path) -> None:
    from scripts.tt_hw_planner.bringup_loop import autofill_stubs

    demo_dir = tmp_path / "demos" / "ACE-Step__acestep-v15-base"
    demo_dir.mkdir(parents=True)
    (demo_dir / "_stubs").mkdir()
    (demo_dir / "bringup_status.json").write_text(
        json.dumps(
            {
                "components": [
                    {
                        "name": "encoder",
                        "status": "NEW",
                        "hf_reference": "encoder",
                    }
                ]
            }
        )
    )

    sentinel = mock.Mock()
    with mock.patch(
        "scripts.tt_hw_planner.bringup_loop.find_demo_dir",
        return_value=demo_dir,
    ), mock.patch(
        "scripts.tt_hw_planner.agentic.probe.load_hf_model_cascade",
        return_value=(None, "hf-load-failed: RuntimeError: meta tensors"),
    ) as cascade, mock.patch(
        "scripts.tt_hw_planner.bringup_loop._render_autofill_stub",
        return_value="# stub",
    ):
        autofill_stubs(
            model_id="ACE-Step/acestep-v15-base",
            repo_root=tmp_path,
            op_synth=True,
        )

    cascade.assert_called_once_with(
        "ACE-Step/acestep-v15-base",
        torch_dtype="bfloat16",
        verbose=False,
    )

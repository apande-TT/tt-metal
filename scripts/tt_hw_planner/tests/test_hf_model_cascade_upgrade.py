"""Tests for PCC harness upgrade to load_hf_model_cascade."""

from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _capture_inputs():
    return importlib.import_module("scripts.tt_hw_planner.capture_inputs")


def _legacy_pcc_test_body() -> str:
    return """# test scaffold
import transformers

HF_MODEL_ID = "ACE-Step/acestep-v15-base"

def _build_torch_reference():
    model = None
    _last_err = None
    for _cls_name in ("AutoModelForCausalLM", "AutoModel"):
        try:
            _cls = getattr(transformers, _cls_name)
        except AttributeError:
            continue
        try:
            model = _cls.from_pretrained(HF_MODEL_ID, trust_remote_code=True, torch_dtype="bfloat16", low_cpu_mem_usage=True)
            break
        except Exception as _e:
            _last_err = _e
            continue
    if model is None:
        raise RuntimeError(
            f"Could not load {HF_MODEL_ID} via AutoModelForCausalLM or "
            f"AutoModel; last error: {type(_last_err).__name__}: {_last_err}"
        )
    model.eval()
    return model
"""


def test_pcc_template_uses_load_hf_model_cascade() -> None:
    bl = importlib.import_module("scripts.tt_hw_planner.bringup_loop")
    template = bl._PCC_TEST_TEMPLATE
    assert "def _load_hf_model():" in template
    assert "load_hf_model_cascade" in template
    assert "model = _load_hf_model()" in template
    assert "AutoModel.from_pretrained(HF_MODEL_ID" not in template


def test_upgrade_test_to_use_hf_model_cascade_replaces_legacy_loader() -> None:
    ci = _capture_inputs()
    with tempfile.TemporaryDirectory() as td:
        test_path = Path(td) / "test_x.py"
        test_path.write_text(_legacy_pcc_test_body())
        assert ci.upgrade_test_to_use_hf_model_cascade(test_path) is True
        upgraded = test_path.read_text()
        assert ci.HF_MODEL_LOADER_MARKER in upgraded
        assert "load_hf_model_cascade" in upgraded
        assert "model = _load_hf_model()" in upgraded
        assert "AutoModel.from_pretrained" not in upgraded
        assert ci.upgrade_test_to_use_hf_model_cascade(test_path) is False


def test_upgrade_all_tests_in_demo_calls_hf_model_cascade_upgrader() -> None:
    src = (_REPO_ROOT / "scripts/tt_hw_planner/capture_inputs.py").read_text()
    idx = src.find("def upgrade_all_tests_in_demo")
    fn_body = src[idx : idx + 3500]
    assert "upgrade_test_to_use_hf_model_cascade" in fn_body

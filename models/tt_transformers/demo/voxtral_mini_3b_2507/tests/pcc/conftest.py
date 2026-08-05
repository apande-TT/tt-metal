import builtins
import json as _json
import re as _re
from pathlib import Path as _Path

import transformers


def _captured_submodule_path(component_name):
    safe = _re.sub(r"[^A-Za-z0-9_]+", "_", component_name).strip("_").lower() or "component"
    demo_dir = _Path(__file__).resolve().parents[2]
    manifest_p = demo_dir / "_captured" / safe / "manifest.json"
    if not manifest_p.is_file():
        return None
    try:
        data = _json.loads(manifest_p.read_text())
        path = data.get("submodule_path")
        if isinstance(path, str) and path:
            return path
    except Exception:
        pass
    return None


builtins._captured_submodule_path = _captured_submodule_path

_orig_auto_causal_lm = transformers.AutoModelForCausalLM.from_pretrained.__func__


@classmethod
def _voxtral_causal_lm_fallback(cls, pretrained_model_name_or_path, *args, **kwargs):
    try:
        return _orig_auto_causal_lm(cls, pretrained_model_name_or_path, *args, **kwargs)
    except Exception:
        return transformers.VoxtralForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )


transformers.AutoModelForCausalLM.from_pretrained = _voxtral_causal_lm_fallback

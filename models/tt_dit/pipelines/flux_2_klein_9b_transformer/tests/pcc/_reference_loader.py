"""Reference-model loader for the FLUX.2-klein-9B *transformer* component.

Why this file exists
--------------------
The repo at ``/tmp/tt_hw_planner_components/flux_2_klein_9b_transformer`` is a
**diffusers** checkpoint, not a transformers one.  Its ``config.json`` carries no
``model_type`` and no ``auto_map`` -- only::

    {"_class_name": "Flux2Transformer2DModel", "_diffusers_version": "0.37.0.dev0", ...}

so ``transformers.AutoConfig`` can never resolve it ("Unrecognized model ...
Should have a `model_type` key").  That is expected: the architecture lives
*outside* transformers, in the ``diffusers`` package.  The weights the repo ships
are real and complete (``diffusion_pytorch_model-0000{1,2}-of-00002.safetensors``,
233 tensors, 18.16 GB BF16 = 9.0786 B params), so this loader returns the genuine
pretrained module -- no random-weight fallback.

Strategy (case 4 of the loader playbook: config-less custom architecture)
-------------------------------------------------------------------------
1. Read ``_class_name`` from ``config.json``.
2. Find a ``diffusers`` install that actually exports that class.  tt-metal's
   ``python_env`` pins diffusers 0.35.1, which predates ``Flux2Transformer2DModel``;
   a newer build (0.40.0) lives in a side venv.  If the ambient import is too old,
   side-load the newer package *by path* without touching ``sys.path``.
3. ``cls.from_pretrained(model_id, torch_dtype=<sniffed>, low_cpu_mem_usage=True)``.

Two traps this file is written around
-------------------------------------
* ``diffusers/__init__.py`` ends with ``sys.modules[__name__] = _LazyModule(...)``.
  After ``spec.loader.exec_module(m)`` the module object *you* constructed holds only
  the ~46 pre-swap globals, so every lazy export (``Flux2Transformer2DModel``) reads
  as absent and a ``hasattr`` probe silently falls through.  Always take
  ``sys.modules.get("diffusers", module)`` after ``exec_module``.
* Probing a build by *importing* it strands a stale ``diffusers`` in ``sys.modules``.
  So availability is decided by **reading** ``__init__.py`` and substring-matching the
  class name -- lazy-import packages list every export in ``_import_structure``.

Import-safe: nothing here runs at import time, and the module is deterministic
(pretrained weights only, inference mode, grads disabled).
"""

from __future__ import annotations

import glob
import importlib
import importlib.util
import json
import os
import struct
import sys
from typing import List, Optional

import torch
import torch.nn as nn

__all__ = ["load_reference_model"]


# Extra places to look for a diffusers build new enough to hold the Flux2 classes.
# Checked only if the ambient (sys.path) diffusers is too old.  Globs are fine --
# non-existent paths are simply skipped.
_EXTRA_DIFFUSERS_SEARCH_GLOBS = (
    "/home/ttuser/venvs/*/lib/python*/site-packages/diffusers",
    os.path.expanduser("~/venvs/*/lib/python*/site-packages/diffusers"),
)

_SAFETENSORS_FLOAT_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
}


def _read_config(model_id: str) -> dict:
    cfg_path = os.path.join(model_id, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"no config.json under {model_id!r}; cannot identify the architecture")
    with open(cfg_path) as f:
        return json.load(f)


def _pkg_exports_class(pkg_dir: str, class_name: str) -> bool:
    """True if the diffusers package rooted at ``pkg_dir`` exports ``class_name``.

    Reads ``__init__.py`` rather than importing: diffusers is a lazy-import package,
    so every public symbol is named in ``_import_structure`` as plain source text.
    Importing to probe would strand a stale module in ``sys.modules``.
    """
    init_py = os.path.join(pkg_dir, "__init__.py")
    try:
        with open(init_py, encoding="utf-8", errors="replace") as f:
            return class_name in f.read()
    except OSError:
        return False


def _ambient_diffusers_dir() -> Optional[str]:
    """Directory a plain ``import diffusers`` would resolve to, without executing it."""
    if "diffusers" in sys.modules:
        mod_file = getattr(sys.modules["diffusers"], "__file__", None)
        if mod_file:
            return os.path.dirname(mod_file)
    try:
        spec = importlib.util.find_spec("diffusers")  # does not exec the module
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    return os.path.dirname(spec.origin)


def _candidate_diffusers_dirs() -> List[str]:
    dirs: List[str] = []
    ambient = _ambient_diffusers_dir()
    if ambient:
        dirs.append(ambient)
    for pattern in _EXTRA_DIFFUSERS_SEARCH_GLOBS:
        for hit in sorted(glob.glob(pattern)):
            if os.path.isdir(hit) and hit not in dirs:
                dirs.append(hit)
    return dirs


def _sideload_diffusers(pkg_dir: str):
    """Import the diffusers package at ``pkg_dir`` without putting its site-packages on sys.path.

    ``submodule_search_locations`` makes absolute ``diffusers.X`` imports route through
    this ``__path__``, so numpy / PIL / huggingface_hub / transformers keep resolving out
    of the ambient environment.  ``sys.modules['diffusers']`` must be set *before*
    ``exec_module`` so those absolute submodule imports find their parent.
    """
    init_py = os.path.join(pkg_dir, "__init__.py")
    spec = importlib.util.spec_from_file_location("diffusers", init_py, submodule_search_locations=[pkg_dir])
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build an import spec for the diffusers package at {pkg_dir!r}")

    # Drop any stale diffusers (e.g. the too-old ambient build) so its already-imported
    # submodules cannot shadow the ones we are about to load.
    for name in [n for n in sys.modules if n == "diffusers" or n.startswith("diffusers.")]:
        del sys.modules[name]

    module = importlib.util.module_from_spec(spec)
    sys.modules["diffusers"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop("diffusers", None)
        raise
    # diffusers' __init__ replaces itself with a _LazyModule; the object we built holds
    # only the pre-swap globals, so always take whatever now sits in sys.modules.
    return sys.modules.get("diffusers", module)


def _resolve_diffusers(class_name: str):
    """Return a diffusers module that exports ``class_name``."""
    candidates = _candidate_diffusers_dirs()
    if not candidates:
        raise ImportError(
            "diffusers is not installed anywhere this loader can see; "
            f"{class_name} lives in diffusers, not transformers"
        )

    usable = [d for d in candidates if _pkg_exports_class(d, class_name)]
    if not usable:
        raise ImportError(
            f"no diffusers build exporting {class_name} found. Checked: {candidates}. "
            "Install a newer diffusers (>=0.37) or point _EXTRA_DIFFUSERS_SEARCH_GLOBS at one."
        )

    ambient = _ambient_diffusers_dir()
    if ambient in usable:
        # The plain import already works -- take it rather than side-loading.
        return importlib.import_module("diffusers")
    return _sideload_diffusers(usable[0])


def _sniff_dtype(model_id: str) -> torch.dtype:
    """Read the checkpoint's stored floating dtype straight from the safetensors headers.

    Non-float entries (e.g. an I64 ``num_batches_tracked``) are skipped -- taking the
    first header entry blindly would mis-type the whole model.
    """
    shards = sorted(glob.glob(os.path.join(model_id, "*.safetensors")))
    for shard in shards:
        try:
            with open(shard, "rb") as f:
                (header_len,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(header_len))
        except (OSError, ValueError):
            continue
        for key, meta in header.items():
            if key == "__metadata__" or not isinstance(meta, dict):
                continue
            dtype = _SAFETENSORS_FLOAT_DTYPES.get(meta.get("dtype"))
            if dtype is not None:
                return dtype
    return torch.bfloat16


def load_reference_model(model_id: str) -> nn.Module:
    """Return an ``nn.Module`` (eval mode) equivalent to the HF reference for this model.

    Loads the repo's real diffusers checkpoint via the class named in ``config.json``'s
    ``_class_name`` (``Flux2Transformer2DModel``), in the dtype the weights are stored in.
    """
    config = _read_config(model_id)
    class_name = config.get("_class_name")
    if not class_name:
        raise ValueError(
            f"{model_id}/config.json has neither `model_type` nor `_class_name`; "
            "the architecture cannot be identified"
        )

    diffusers = _resolve_diffusers(class_name)
    cls = getattr(diffusers, class_name, None)
    if cls is None:
        raise ImportError(
            f"the resolved diffusers build ({getattr(diffusers, '__version__', '?')}) " f"does not expose {class_name}"
        )

    dtype = _sniff_dtype(model_id)
    model = cls.from_pretrained(model_id, torch_dtype=dtype, low_cpu_mem_usage=True)

    # Do NOT model.to(dtype) here -- from_pretrained already honoured torch_dtype, and a
    # redundant cast makes diffusers warn about modules that should stay float32.
    model.eval()
    model.requires_grad_(False)
    return model

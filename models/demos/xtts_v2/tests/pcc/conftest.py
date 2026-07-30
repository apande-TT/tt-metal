# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared PCC-test harness shims for coqui/XTTS-v2 bring-up.

Every generated ``test_<component>.py`` in this directory calls the bare name
``_captured_submodule_path(COMPONENT_NAME)`` inside ``_build_torch_reference``
(the "resolve the captured submodule path FIRST" branch), but the generator
never emitted a definition for it — so every test raised
``NameError: name '_captured_submodule_path' is not defined`` before it could
build its torch reference.

Rather than hand-patch all 32 identical test files, we define the helper once
here and publish it into ``builtins``. pytest imports this ``conftest.py``
during collection, before any test module is imported/executed, so the bare
name resolves via Python's global -> builtins fallback in every test module
regardless of how it was loaded (including the sharded tests that
``exec_module`` their single-device sibling).
"""

from __future__ import annotations

import builtins
import json
import os


def _captured_submodule_path(component_name: str):
    """Return the ``submodule_path`` recorded in ``_captured/<component>/manifest.json``.

    Capture-inputs records the exact dotted path it resolved the module at, so
    the PCC test can resolve the SAME submodule and the captured golden shapes
    line up. Returns ``None`` when there is no manifest or no recorded path, in
    which case the test falls back to ``_CANDIDATE_SUBMODULE_PATHS``.
    """
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        # tests/pcc/ -> models/demos/xtts_v2/_captured/<component>/manifest.json
        manifest_path = os.path.normpath(
            os.path.join(here, "..", "..", "_captured", component_name, "manifest.json")
        )
        with open(manifest_path) as fh:
            data = json.load(fh)
        path = data.get("submodule_path")
        if isinstance(path, str) and path:
            return path
    except Exception:
        pass
    return None


def _load_captured_inputs(component_name: str):
    """Return ``(args_list, kwargs_dict)`` of the REAL captured torch tensors for
    ``component_name`` (from ``_captured/<comp>/{args,kwargs}.pt``), or
    ``(None, None)`` if unavailable.

    These are the exact inputs the capture step fed the module to produce the
    golden, so their shapes/dtypes always match the real ``forward`` — unlike the
    name-based synthetic builder in each test, which only guesses from arg names
    and silently produces wrong-rank tensors for modules with unusual signatures
    (e.g. an einsum attention taking 4D ``q, k, v``)."""
    import torch

    try:
        here = os.path.dirname(os.path.abspath(__file__))
        base = os.path.normpath(os.path.join(here, "..", "..", "_captured", component_name))
        args = torch.load(os.path.join(base, "args.pt"), map_location="cpu", weights_only=False)
        try:
            kwargs = torch.load(os.path.join(base, "kwargs.pt"), map_location="cpu", weights_only=False)
        except Exception:
            kwargs = {}
        args_list = list(args) if isinstance(args, (list, tuple)) else [args]
        return args_list, dict(kwargs or {})
    except Exception:
        return None, None


def _captured_sample_kwargs(component_name: str, torch_module):
    """Build ``(sample_kwargs, primary)`` from the REAL captured inputs by mapping
    the captured POSITIONAL args onto ``torch_module.forward``'s parameter names.

    Returns ``(None, None)`` when no capture exists, so callers can fall back to
    the synthetic builder. ``primary`` is ``(name, tensor)`` of the first tensor
    arg (the one the PCC harness feeds to the ttnn stub as its positional input);
    remaining args/kwargs ride along as extra kwargs. ``None``-valued captured
    kwargs (e.g. ``mask=None``) are dropped so the module applies its own default."""
    import inspect

    import torch

    args, kwargs = _load_captured_inputs(component_name)
    if args is None:
        return None, None
    try:
        sig = inspect.signature(torch_module.forward)
    except (TypeError, ValueError):
        return None, None
    names = [
        n
        for n, p in sig.parameters.items()
        if n != "self" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)
    ]
    sample = {}
    primary = None
    for i, val in enumerate(args):
        if i >= len(names):
            break
        sample[names[i]] = val
        if primary is None and isinstance(val, torch.Tensor):
            primary = (names[i], val)
    for k, v in (kwargs or {}).items():
        if v is not None:
            sample[k] = v
    if primary is None:
        return None, None
    return sample, primary


# Publish for the bare-name references in every generated test module.
for _name, _fn in (
    ("_captured_submodule_path", _captured_submodule_path),
    ("_load_captured_inputs", _load_captured_inputs),
    ("_captured_sample_kwargs", _captured_sample_kwargs),
):
    if not hasattr(builtins, _name):
        setattr(builtins, _name, _fn)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Harness shim for per-component PCC tests.

Each test file calls the bare name ``_captured_submodule_path(COMPONENT_NAME)``
inside ``_build_torch_reference``. That helper was supposed to be inlined into
every test module by ``scripts/tt_hw_planner/capture_inputs.py`` (see
``CAPTURE_LOADER_SOURCE``), but the emit path for this model skipped that
injection, so every test raises ``NameError: _captured_submodule_path``. We
provide it here and inject it into each test module's globals at collection
time so nothing else has to change.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def _captured_submodule_path(component_name):
    """Read the submodule_path the capture step hooked for this component."""
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", component_name).strip("_").lower() or "component"
    here = Path(__file__).resolve()
    demo_dir = here.parents[2]
    manifest_p = demo_dir / "_captured" / safe / "manifest.json"
    if not manifest_p.is_file():
        return None
    try:
        data = json.loads(manifest_p.read_text())
        path = data.get("submodule_path")
        if isinstance(path, str) and path:
            return path
    except Exception:
        pass
    return None


def pytest_collectstart(collector):
    """Inject ``_captured_submodule_path`` into every test module's globals."""
    mod = getattr(collector, "module", None)
    if mod is not None and not hasattr(mod, "_captured_submodule_path"):
        mod._captured_submodule_path = _captured_submodule_path

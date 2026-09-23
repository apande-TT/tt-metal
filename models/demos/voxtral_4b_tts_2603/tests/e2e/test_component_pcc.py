# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gate 1's companion: every EDITED graduated stub still passes its OWN component PCC test.

Composing the stubs into a chain required editing some of them -- the batch axis had to stop
being a hardcoded 1, and the decode path had to thread a KV cache. An edit that quietly broke a
component would show up here as a per-component failure instead of as a mysterious end-to-end
number, which is a much cheaper place to find it.

Runs Source B's own `tests/pcc/` suite in a SUBPROCESS: those tests open their own device, and
nesting pytest inside a session that already holds one would fight over the command queues.

CLEARING THE GOLDEN CACHES IS THE POINT. Each component test memoises
`(module, kwargs, primary, golden)` in `_captured/<comp>/golden_cache_s0.pt` and reuses it
SILENTLY. On this model, a full-suite re-run with the caches cleared is what turned up
`acoustic_codebook` at PCC -0.87 after the suite had already gone 31/31 green -- a per-component
re-run would never have found it. So this test clears them first.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from models.demos.voxtral_4b_tts_2603.tt import common

pytestmark = [pytest.mark.timeout(3600), pytest.mark.slow]

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))


def _clear_golden_caches():
    cleared = []
    for name in common.graduated_modules():
        path = os.path.join(common.CAPTURED_ROOT, name, "golden_cache_s0.pt")
        if os.path.exists(path):
            os.remove(path)
            cleared.append(name)
    return cleared


def test_all_31_component_pcc_tests_still_pass():
    """The whole suite, caches cleared, in one subprocess."""
    cleared = _clear_golden_caches()
    print(f"\ncleared {len(cleared)} golden caches before re-running")

    suite = os.path.join(common.BRINGUP_ROOT, "tests", "pcc")
    cmd = [sys.executable, "-m", "pytest", suite, "-q", "-p", "no:cacheprovider", "--timeout=3000"]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=3300)

    tail = "\n".join(proc.stdout.strip().splitlines()[-25:])
    print(tail)
    assert proc.returncode == 0, (
        f"Source B's component PCC suite FAILED after this package's stub edits "
        f"(returncode {proc.returncode}). A stub edit that breaks its own component is a Gate 1 "
        f"regression, not an end-to-end one.\n{tail}\n{proc.stderr[-2000:]}"
    )

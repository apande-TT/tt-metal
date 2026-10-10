# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The one place outside the test fixtures that opens a device for MiniCPM5-2B.

Used by the demo entrypoint and by tt/pipeline.py's selftests when they are called without a device
(the gate's probes call them bare). tt/ itself never opens a device: it runs on the one it is given.
"""
from __future__ import annotations

import contextlib
import os

import ttnn

# The QB2's two p300c boards share no ethernet link, so only board 0's chips are made visible.
VISIBLE_DEVICES = "0,1"
DEVICE_PARAMS = {"l1_small_size": 24576, "trace_region_size": 200 * 1024 * 1024}


@contextlib.contextmanager
def opened_device(device_id=0, **overrides):
    """Open one chip with the pipeline's device params (trace region included), close it after."""
    os.environ.setdefault("TT_VISIBLE_DEVICES", VISIBLE_DEVICES)
    dev = ttnn.open_device(device_id=device_id, **{**DEVICE_PARAMS, **overrides})
    try:
        yield dev
    finally:
        ttnn.close_device(dev)

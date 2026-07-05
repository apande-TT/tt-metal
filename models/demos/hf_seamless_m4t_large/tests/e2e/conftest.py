# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared pytest fixture: opens a TT device and yields it."""
from __future__ import annotations

import pytest

import ttnn


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    try:
        yield dev
    finally:
        ttnn.close_device(dev)

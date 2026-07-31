# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Device opener for the module-level self-test hooks in ``tt/pipeline.py``.

It lives OUTSIDE ``tt/`` on purpose: the pipeline package must never open a
device of its own (the test fixture / perf harness owns that), but the
model-agnostic probes call ``pipeline.host_op_selftest()`` /
``pipeline.trace_capture_selftest()`` with NO device argument, so those hooks need
somewhere to stand one up. Importing this module lazily from inside the hooks
keeps ``tt/*.py`` free of any ``open_device`` / ``open_mesh_device`` call.
"""
from __future__ import annotations

import os

import ttnn

# The vocoder trunk is native ttnn.conv1d/conv_transpose2d and the speaker encoder is
# native ttnn.conv2d; both run a sliding-window/halo gather whose sharding + config
# tensors allocate from the dedicated L1_SMALL pool. That pool is 0 B unless reserved
# at device open, and coming up short surfaces as a TT_FATAL "Out of Memory ... bank
# size is 0 B" / "Not enough space to allocate N B L1_SMALL", not an API error.
# It scales with the SYNTHESIZED LENGTH: 32 KB covers the vocoder's 6-latent captured
# case, and the 32-latent (1.5 s) synthesis this pipeline runs needs 128 KB.
L1_SMALL_SIZE = 131072

# Trace budget, sized from the LARGEST stage: the vocode stage captures one HiFi-GAN
# program per decode stream (DECODE_BATCH of them) and the GPT stages capture 30 blocks
# each. Env-overridable so a smaller box can shrink it.
TRACE_REGION_SIZE = int(os.environ.get("XTTS_TRACE_REGION", str(100663296)))

MESH_ROWS = int(os.environ.get("XTTS_MESH_ROWS", "1"))   # DP
MESH_COLS = int(os.environ.get("XTTS_MESH_COLS", "8"))   # TP


def open_selftest_device(trace: bool = False, rows: int = MESH_ROWS, cols: int = MESH_COLS):
    """Open the TP=8 x DP=1 mesh (falling back to a single device), -> (device, is_mesh)."""
    params = {"l1_small_size": L1_SMALL_SIZE}
    if trace:
        params["trace_region_size"] = TRACE_REGION_SIZE
    if rows * cols > 1:
        try:
            # CCLs need the inter-chip fabric enabled BEFORE the mesh is opened, or any
            # all_gather/all_reduce raises TT_FATAL "fabric_context_ != nullptr".
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
            return ttnn.open_mesh_device(ttnn.MeshShape(rows, cols), **params), True
        except Exception as e:  # noqa: BLE001 - single-device box
            print(f"[selftest-device] mesh {rows}x{cols} open failed ({e}); single-device fallback")
            try:
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            except Exception:
                pass
    return ttnn.open_device(device_id=0, **params), False


def close_selftest_device(device, is_mesh: bool):
    if is_mesh:
        ttnn.close_mesh_device(device)
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:
            pass
    else:
        ttnn.close_device(device)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared harness setup for the minicpm5_2b per-component PCC tests."""

import os
import shutil
import subprocess

# On this host the two p300c boards have no working ethernet link between them, so opening a device
# with all 4 chips visible fails at fabric init (TT_FATAL control_plane.cpp:1398: chip not in the
# control-plane mapping). Restrict to one board unless the caller has already chosen devices.
_BOARD0 = "0,1"
_we_chose_devices = "TT_VISIBLE_DEVICES" not in os.environ
os.environ.setdefault("TT_VISIBLE_DEVICES", _BOARD0)

# A whole-box `tt-smi -r` (what the bring-up tooling runs after a hang or an orphan kill) leaves
# chips 0,1 wedged on this host: the first workload then waits forever in
# completion_queue_wait_front. Resetting just that board brings it back, so do that once per
# pytest process before any device is opened. Opt out with TT_PCC_NO_BOARD_RESET=1.
if _we_chose_devices and os.environ.get("TT_PCC_NO_BOARD_RESET", "") in ("", "0"):
    _smi = shutil.which("tt-smi")
    if _smi:
        try:
            subprocess.run([_smi, "-r", _BOARD0], capture_output=True, timeout=120, check=False)
        except (OSError, subprocess.TimeoutExpired):
            pass

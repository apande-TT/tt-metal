# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device selection for the MiniCPM5-2B e2e tests on this host.

The two p300c boards of the QB2 have no ethernet link between them, so opening a device with all four
chips visible fails at fabric init (control_plane.cpp:1398). The run opens ONE chip of board 0.
"""
import os

os.environ.setdefault("TT_VISIBLE_DEVICES", "0,1")

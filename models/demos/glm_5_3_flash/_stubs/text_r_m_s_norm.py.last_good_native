# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn port of `Glm5NextTextRMSNorm` (GLM-5.3-Flash text RMSNorm).

y = weight * (x * rsqrt(mean(x^2, -1) + eps)) -- identical math to the vision RMSNorm, so this reuses
the graduated `r_m_s_norm` port (ttnn.rms_norm, HiFi4, fp32 accumulation; weight replicated).
"""
from __future__ import annotations

from models.demos.glm_5_3_flash._stubs.r_m_s_norm import TtRMSNorm


class TtTextRMSNorm(TtRMSNorm):
    pass


def build(device, torch_module=None):
    return TtTextRMSNorm.build(device, torch_module)


def text_r_m_s_norm(device, torch_module=None):
    return TtTextRMSNorm.build(device, torch_module)

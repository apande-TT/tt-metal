# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Gate-2 instrumentation: prove every graduated module is actually INVOKED.

Wraps the __call__ of each of the 13 graduated stub classes with a counter.
The e2e test enters this context around the pipeline run and asserts every one
of the 13 modules has count > 0 (no graduated module left out)."""
from __future__ import annotations

import contextlib
import importlib

_STUB_PKG = "models.demos.hf_eager.acestep_v15_base._stubs"

# (module, class) for each of the 13 graduated (NEW) components.
GRADUATED_MODULES = [
    ("ace_step_di_t_model", "AceStepDiTModel"),
    ("ace_step_di_t_layer", "AceStepDiTLayer"),
    ("ace_step_encoder_layer", "AceStepEncoderLayer"),
    ("ace_step_condition_encoder", "AceStepConditionEncoder"),
    ("ace_step_lyric_encoder", "AceStepLyricEncoder"),
    ("ace_step_timbre_encoder", "AceStepTimbreEncoder"),
    ("ace_step_audio_tokenizer", "AceStepAudioTokenizer"),
    ("audio_token_detokenizer", "AudioTokenDetokenizer"),
    ("attention_pooler", "AttentionPooler"),
    ("timestep_embedding", "TimestepEmbedding"),
    ("lambda", "Lambda"),
    ("residual_f_s_q", "ResidualFSQ"),
    ("f_s_q", "FSQ"),
]


class InvocationTracker:
    def __init__(self):
        self.counts = {name: 0 for name, _ in GRADUATED_MODULES}
        self._orig = {}

    def all_invoked(self):
        return all(v > 0 for v in self.counts.values())

    def missing(self):
        return [k for k, v in self.counts.items() if v == 0]

    def report(self):
        lines = ["Gate 2 invocation table (graduated module -> __call__ count):"]
        for name, _ in GRADUATED_MODULES:
            flag = "OK " if self.counts[name] > 0 else "MISSING"
            lines.append(f"  [{flag}] {name}: {self.counts[name]}")
        return "\n".join(lines)


@contextlib.contextmanager
def track_invocations():
    tracker = InvocationTracker()
    patched = []
    for mod_name, cls_name in GRADUATED_MODULES:
        try:
            mod = importlib.import_module(f"{_STUB_PKG}.{mod_name}")
            cls = getattr(mod, cls_name)
        except Exception as e:
            raise RuntimeError(f"invocation_tracker: cannot import {mod_name}.{cls_name}: {e}")
        orig_call = cls.__call__

        def make_wrapper(orig, key):
            def wrapper(self, *args, **kwargs):
                tracker.counts[key] += 1
                return orig(self, *args, **kwargs)

            return wrapper

        cls.__call__ = make_wrapper(orig_call, mod_name)
        patched.append((cls, orig_call))
    try:
        yield tracker
    finally:
        for cls, orig_call in patched:
            cls.__call__ = orig_call

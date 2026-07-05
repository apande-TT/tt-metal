# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end pipeline test for `mistralai/Voxtral-Mini-3B-2507`.

Runs the ONE shared chained pipeline (`tt/pipeline.py` — the same code the demo
runs) on real processor input and asserts:

  Gate 1 — every routed graduated stub is still the native TTNN body
           (LIVE _stubs/<name>.py == .last_good_native) and logged no CPU fallback.
  Gate 2 — all 7 graduated stubs are INVOKED on real device tensors.
  Gate 3 — e2e prefill last-token logits PCC vs the HF golden >= 0.95, and the
           TT greedy sequence matches model.generate() token-for-token
           (both capped to the same small horizon N).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from models.common.utility_functions import comp_pcc
from models.demos.hf_eager.voxtral_mini_3b_2507.tt import pipeline as P

PCC_TARGET = 0.95
N = int(os.environ.get("TT_E2E_MAX_NEW_TOKENS", "6"))
_DEMO_DIR = Path(__file__).resolve().parents[2]
_STUBS = _DEMO_DIR / "_stubs"


def _gate1_stubs_native():
    """LIVE stub == graduated .last_good_native (not rewritten to a torch fallback)."""
    for name in P.GRADUATED_STUBS:
        live = (_STUBS / f"{name}.py").read_bytes()
        grad = (_STUBS / f"{name}.py.last_good_native").read_bytes()
        assert live == grad, f"Gate 1: {name}.py drifted from its graduated .last_good_native"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_e2e_voxtral(device_params, device):
    _gate1_stubs_native()

    # route stub CPU-fallback logging to a fresh temp file (Gate 1 runtime check).
    fb_log = _DEMO_DIR / "_e2e_runtime_fallbacks.jsonl"
    if fb_log.exists():
        fb_log.unlink()
    os.environ["TT_HW_PLANNER_RUNTIME_FALLBACK_LOG"] = str(fb_log)

    model = P.load_hf_model()
    proc = P.load_processor()
    input_ids, input_features, n_audio, atid, prompt = P.build_inputs(proc, model)
    print(f"[e2e] input_ids={tuple(input_ids.shape)} n_audio_tokens={n_audio} audio_token_id={atid}", flush=True)

    # ---- HF golden (KV-cache; capped to N) ---- #
    with torch.no_grad():
        gen = model.generate(
            input_ids=input_ids,
            input_features=input_features.to(torch.bfloat16),
            max_new_tokens=N,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
    hf_tokens = gen.sequences[0, input_ids.shape[1] :].tolist()
    hf_scores = [s[0].float() for s in gen.scores]  # scores[0] == prefill last-token logits

    # ---- TT pipeline (the shared chained forward; capped to the same N) ---- #
    pipe = P.VoxtralTTPipeline(device, model)
    out = pipe.run(input_ids, input_features, max_new_tokens=N)
    tt_tokens = out["tt_tokens"]

    # ---- Gate 2: every graduated stub invoked ---- #
    print(f"[e2e] invoked stubs = {sorted(out['invoked'])}", flush=True)
    print(f"[e2e] verified-equivalence PCC = " f"{ {k: round(v,5) for k,v in out['verify_pcc'].items()} }", flush=True)
    missing = set(P.GRADUATED_STUBS) - out["invoked"]
    assert not missing, f"Gate 2: graduated stubs NOT invoked in the pipeline: {sorted(missing)}"

    # ---- Gate 3: e2e PCC + token match ---- #
    _, prefill_pcc = comp_pcc(hf_scores[0], out["tt_step_logits"][0], PCC_TARGET)
    prefill_pcc = float(prefill_pcc)
    per_step = []
    for i in range(N):
        _, p = comp_pcc(hf_scores[i], out["tt_step_logits"][i], 0.0)
        per_step.append(float(p))
    token_match = sum(int(a == b) for a, b in zip(tt_tokens, hf_tokens))
    print(f"[e2e] HF tokens = {hf_tokens}", flush=True)
    print(f"[e2e] TT tokens = {tt_tokens}", flush=True)
    print(f"[e2e] per-step logits PCC = {[round(p,5) for p in per_step]}", flush=True)
    print(f"[e2e] token match = {token_match}/{N}", flush=True)

    # Gate 1 runtime: no stub fell back to torch during the run.
    fb_lines = fb_log.read_text().strip() if fb_log.exists() else ""
    assert not fb_lines, f"Gate 1: CPU fallback(s) logged during e2e run:\n{fb_lines}"

    # Always print the achieved e2e PCC on its own line, pass OR fail.
    print(f"e2e PCC={prefill_pcc}", flush=True)
    assert prefill_pcc >= PCC_TARGET, f"Gate 3: e2e prefill logits PCC {prefill_pcc} < {PCC_TARGET}"
    assert token_match == N, (
        f"Gate 3: TT greedy sequence {tt_tokens} != HF golden {hf_tokens} " f"({token_match}/{N} matched)"
    )

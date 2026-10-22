# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Checks shared by the Devstral e2e tests: streaming logits PCC, near-tie-aware token agreement, Gate 1 / 2."""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import torch

from models.demos.devstral_small_2_24b_instruct_2512.tt import pipeline as P

# A token mismatch between TT and the reference is accepted only where the reference's own decision was a
# near-tie: its top-2 noised scores (in logit units) are closer than NEAR_TIE_SIGMAS x the RMS logit error
# measured between TT and the reference over the candidate tokens AT THAT STEP. Anything wider is a defect.
NEAR_TIE_SIGMAS = 4.0


def alive_mask(tokens, eos):
    """[B, T] True up to and including each row's first EOS (the positions generate() actually produced)."""
    is_eos = tokens == eos
    before = torch.cumsum(is_eos.int(), dim=1) - is_eos.int()
    return before == 0


class StreamingPCC:
    """PCC over the concatenation of every compared logits row, overall and per batch row, in float64 sums."""

    def __init__(self, rows):
        self.tot = torch.zeros(5, dtype=torch.float64)
        self.row = torch.zeros(rows, 5, dtype=torch.float64)
        self.n = 0
        self.rn = torch.zeros(rows, dtype=torch.float64)

    def add(self, b, x, y):
        x, y = x.to(torch.float64), y.to(torch.float64)
        s = torch.stack([x.sum(), y.sum(), (x * y).sum(), (x * x).sum(), (y * y).sum()])
        self.tot += s
        self.row[b] += s
        self.n += x.numel()
        self.rn[b] += x.numel()

    @staticmethod
    def _pcc(s, n):
        sx, sy, sxy, sxx, syy = s.tolist()
        cov = sxy - sx * sy / n
        return cov / max(((sxx - sx * sx / n) * (syy - sy * sy / n)) ** 0.5, 1e-30)

    def overall(self):
        return self._pcc(self.tot, self.n)

    def per_row(self):
        return [self._pcc(self.row[b], float(self.rn[b])) for b in range(self.row.shape[0]) if self.rn[b] > 0]


def topk_rms_error(tt_row, ref_row, k=P.SAMPLING_TOP_K):
    idx = ref_row.float().topk(k).indices
    return float((tt_row.float()[idx] - ref_row.float()[idx]).pow(2).mean().sqrt())


def pcc(a, b):
    a, b = a.flatten().to(torch.float64), b.flatten().to(torch.float64)
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def hidden_state_check(tt_hidden, ref_hidden):
    """Per row: PCC and norm ratio |tt| / |ref| of the first stage's output against the reference."""
    pccs, ratios = [], []
    for b in range(tt_hidden.shape[0]):
        pccs.append(pcc(tt_hidden[b], ref_hidden))
        ratios.append(float(tt_hidden[b].float().norm() / ref_hidden.float().norm()))
    # Rows share one content input: a uniform shortfall is systematic, a scattered one is not.
    worst = sorted(range(len(pccs)), key=lambda r: pccs[r])[:4]
    print(f"  per-row PCC max={max(pccs):.6f} worst rows={[(r, round(pccs[r], 6)) for r in worst]}", flush=True)
    return min(pccs), ratios


def assert_tokens_agree(tt_drawn, ref_choice, ref_margin, step_err, alive, label):
    """Exact agreement of the TT-drawn ids with the reference's choice at every alive step, except near-ties."""
    mismatch = (tt_drawn != ref_choice) & alive
    n_alive, n_mis = int(alive.sum()), int(mismatch.sum())
    unexplained = []
    for b, t in mismatch.nonzero().tolist():
        tol = NEAR_TIE_SIGMAS * float(step_err[b, t])
        if float(ref_margin[b, t]) > tol:
            unexplained.append((b, t, int(tt_drawn[b, t]), int(ref_choice[b, t]), float(ref_margin[b, t]), tol))
    print(
        f"[{label}] token agreement: {n_alive - n_mis}/{n_alive} exact, {n_mis - len(unexplained)} near-tie, "
        f"{len(unexplained)} unexplained",
        flush=True,
    )
    assert not unexplained, f"{label}: TT drew a different token than the reference at non-tie steps: {unexplained[:8]}"


# ------------------------------------------------------------------------------------------- Gate 1 / 2


def gate1_native(components):
    """Gate 1: each routed component is the LIVE graduated stub file and its forward path is ttnn, not torch."""
    from scripts.tt_hw_planner.commands.emit_e2e import _G1_TORCH_DELEGATION, _check_hf_fallback

    stub_dir = (P.PACKAGE_DIR / "_stubs").resolve()
    bad = []
    for name in components:
        cls = P.GRADUATED_STUBS[name]
        path = Path(inspect.getsourcefile(cls)).resolve()
        src = path.read_text()
        if path.parent != stub_dir or path.stem != name:
            bad.append(f"{name}: class lives in {path}, not _stubs/{name}.py")
        if any(re.search(p, src) for p in _G1_TORCH_DELEGATION) or _check_hf_fallback(src):
            bad.append(f"{name}: torch delegation / HF fallback in the stub's forward path")
        # A subclass stub (TtLayer, TtPixtralMLP, ...) inherits its forward from another graduated stub: the
        # forward path is ttnn when the stub file that DEFINES __call__ is.
        owner = next(k for k in cls.__mro__ if "__call__" in k.__dict__ and k is not object)
        fwd_src = Path(inspect.getsourcefile(owner)).read_text()
        if "ttnn." not in src and "ttnn." not in fwd_src:
            bad.append(f"{name}: no ttnn op in the stub's forward")
        if Path(inspect.getsourcefile(owner)).resolve().parent != stub_dir:
            bad.append(f"{name}: forward defined outside _stubs ({owner.__module__})")
    assert not bad, "Gate 1 (native ttnn) failed: " + "; ".join(bad)
    print(f"[gate1] {len(components)} routed stubs are the live graduated _stubs files and native ttnn", flush=True)


def gate2_invoked(components):
    """Gate 2: every graduated component of this task ran inside the real forward chain."""
    missing = [n for n in components if P.INVOKED.get(n, 0) == 0]
    print("[gate2] invoked: " + ", ".join(f"{n}={P.INVOKED.get(n, 0)}" for n in components), flush=True)
    assert not missing, f"Gate 2 failed: graduated stubs never invoked in the forward: {missing}"


# ------------------------------------------------------------------------------------------ test drivers


def _osl_capped():
    import os

    return bool((os.environ.get(P.OSL_ENV) or "").strip())


def check_against_golden(task, run, pipe, hf_model, req):
    """TT teacher-forced onto the INDEPENDENT HF golden (model.generate() under the shared stop + sampling rule).

    Asserts, over the whole generated output of all B rows: the golden ended on EOS (not on the cap); TT drew the
    golden's token at every step (near-ties excepted); the first stage's hidden state holds PCC >= 0.995 on every
    row with |tt|/|ref| within 0.5%; logits PCC >= 0.99 overall and per row; every graduated stub of the task ran."""
    from models.demos.devstral_small_2_24b_instruct_2512.tests.e2e.hf_reference import (
        GumbelTopKSampler,
        _hf_reference_generate,
    )

    b, L, eos = pipe.batch, req["real_len"], pipe.eos_token_id
    golden = _hf_reference_generate(hf_model, req, pipe.top_k, pipe.temperature)
    gt = golden["tokens"]
    T = gt.shape[1]
    ended = bool((gt == eos).any(dim=1).all())
    print(f"[{task}] golden: {T} steps, every row ended on EOS={ended}", flush=True)
    assert ended, f"{task}: the HF golden hit the {req['max_new_tokens']}-token safety cap instead of EOS"
    alive = alive_mask(gt, eos)

    sampler = GumbelTopKSampler(
        P.gumbel_table(req["seeds"], req["max_new_tokens"] + 1), pipe.temperature, pipe.top_k, 0
    )
    margins = torch.zeros(b, T)
    step_err = torch.zeros(b, T)
    stats = StreamingPCC(b)

    def on_step(t, state):
        tt = P.host_read(state["logits"], pipe.device).reshape(b, -1).float()
        gl = golden["logits"][t].float()
        choice, margin = sampler.choose(gl, t)
        assert bool(
            (choice[alive[:, t]] == gt[alive[:, t], t]).all()
        ), "golden sampler replay disagrees with generate()"
        margins[:, t] = margin
        for r in range(b):
            if alive[r, t]:
                stats.add(r, tt[r], gl[r])
                step_err[r, t] = topk_rms_error(tt[r], gl[r])

    P.reset_invocations()
    res = run(pipe, req, teacher=gt, on_step=on_step)
    if _osl_capped():
        # The harness caps the horizon for profiling: compare the steps that ran, termination not asserted.
        print(f"[{task}] {P.OSL_ENV} is set: compared {res['steps']} of {T} golden steps, termination not asserted")
        T = res["steps"]
        gt, alive, margins, step_err = gt[:, :T], alive[:, :T], margins[:, :T], step_err[:, :T]
    else:
        assert (
            res["steps"] == T and not res["hit_cap"]
        ), f"{task}: TT run ended after {res['steps']} of {T} golden steps"

    hidden = P.host_read(res["state"]["hidden"], pipe.device)[:, :L].float()
    h_pcc, h_ratio = hidden_state_check(hidden, golden["prefill_hidden"])
    print(
        f"[{task}] prefill hidden PCC min over rows={h_pcc:.6f} norm ratio range=({min(h_ratio):.5f}, "
        f"{max(h_ratio):.5f})",
        flush=True,
    )
    f_pcc, f_ratio = None, None
    if golden["image_features"] is not None:
        feats = P.host_read(res["state"]["image_embeds"], pipe.device).reshape(
            b, -1, golden["image_features"].shape[-1]
        )
        f_pcc, f_ratio = hidden_state_check(feats.float(), golden["image_features"])
        print(
            f"[{task}] encode (vision+projector) PCC min over rows={f_pcc:.6f} norm ratio range=({min(f_ratio):.5f}, "
            f"{max(f_ratio):.5f})",
            flush=True,
        )
    # Both first-stage states are printed before either is asserted, so a failure shows where the error starts.
    assert h_pcc >= 0.995 and all(abs(r - 1.0) <= 0.005 for r in h_ratio), f"{task}: prefill hidden state off level"
    if f_pcc is not None:
        assert f_pcc >= 0.995 and all(abs(r - 1.0) <= 0.005 for r in f_ratio), f"{task}: image features off level"

    assert_tokens_agree(res["drawn"], gt, margins, step_err, alive, f"{task} vs golden")
    gate1_native(P.TASK_STUBS[task])
    gate2_invoked(P.TASK_STUBS[task])

    rows = stats.per_row()
    e2e = stats.overall()
    print(f"[{task}] logits PCC min over rows={min(rows):.6f}", flush=True)
    print(f"e2e PCC={e2e}", flush=True)
    assert e2e >= 0.99 and min(rows) >= 0.99, f"{task}: e2e logits PCC {e2e} (row min {min(rows)}) < 0.99"
    return golden


def check_free_run(task, run, pipe, processor, hf_model, req):
    """TT free-runs its own on-device samples to EOS; the HF reference is then teacher-forced onto the TT
    trajectory and must choose the same token at every step (near-ties excepted)."""
    from models.demos.devstral_small_2_24b_instruct_2512.tests.e2e.hf_reference import _hf_reference_teacher_forced

    b, eos = pipe.batch, pipe.eos_token_id
    tt_logits = []
    res = run(pipe, req, on_step=lambda t, st: tt_logits.append(P.host_read(st["logits"], pipe.device).reshape(b, -1)))
    if _osl_capped():
        print(
            f"[{task}] {P.OSL_ENV} is set: the horizon is capped by the harness, termination not asserted", flush=True
        )
    else:
        assert not res["hit_cap"], f"{task}: TT run ended on the {req['max_new_tokens']}-token cap, not on EOS"
    tokens = res["tokens"]
    alive = alive_mask(tokens, eos)
    texts = P.decode_text(processor, tokens, eos)
    for r, txt in enumerate(texts):
        print(f"[{task}] row {r} (seed {req['seeds'][r]}): {txt!r}", flush=True)
    distinct = len(set(texts))
    print(f"[{task}] {distinct} distinct outputs over {b} rows", flush=True)
    # The seeds have to reach the draws. "The rows differ" was the proxy for that, and it is a property of the PROMPT,
    # not of the sampler: at temperature 0.15 the image prompt's answer differs between seeds at ONE near-tie step
    # (the HF golden draws ' called' or ' named' on a 0.125-logit margin, 9 of its 32 seeds), which a TT logit row
    # within 0.5% of the reference may or may not reproduce. The sampling rule is checked directly instead: every
    # draw must be the rule's choice on TT's OWN logits with that row's noise (an exact tie excepted: either
    # candidate is the rule's choice there). A noise table that did not reach the draws fails this at the first
    # noised near-tie, whatever the prompt.
    from models.demos.devstral_small_2_24b_instruct_2512.tests.e2e.hf_reference import GumbelTopKSampler

    rule = GumbelTopKSampler(P.gumbel_table(req["seeds"], req["max_new_tokens"] + 1), pipe.temperature, pipe.top_k, 0)
    off = []
    for t in range(res["steps"]):
        pick, margin = rule.choose(tt_logits[t].float(), t)
        for r in range(b):
            if alive[r, t] and int(pick[r]) != int(res["drawn"][r, t]) and float(margin[r]) > 1e-4:
                off.append((r, t, int(res["drawn"][r, t]), int(pick[r]), round(float(margin[r]), 4)))
    print(f"[{task}] sampler rule replayed on TT logits: {len(off)} draws off the rule's choice", flush=True)
    assert not off, f"{task}: on-device draws that the sampling rule does not choose on TT's own logits: {off[:8]}"

    ref = _hf_reference_teacher_forced(hf_model, req, tokens, pipe.top_k, pipe.temperature)
    stats = StreamingPCC(b)
    step_err = torch.zeros(b, res["steps"])
    for t in range(res["steps"]):
        for r in range(b):
            if alive[r, t]:
                stats.add(r, tt_logits[t][r].float(), ref["logits"][t, r].float())
                step_err[r, t] = topk_rms_error(tt_logits[t][r], ref["logits"][t, r])
    assert_tokens_agree(res["drawn"], ref["choice"], ref["margin"], step_err, alive, f"{task} free-run")
    e2e = stats.overall()
    print(f"[{task}] free-run logits PCC vs HF on the TT trajectory={e2e}", flush=True)
    assert e2e >= 0.99 and min(stats.per_row()) >= 0.99
    return res

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""E2E correctness gate for Kolibri-1 text generation: the TT pipeline (tt/pipeline.py, the same code the
demo runs) against the fp32 reference model (tests/pcc/_reference_loader.py).

Inputs: the model card's example user message ("Erkläre kurz, was ein Mixture-of-Experts-Modell ist.",
Aleph-Alpha/Kolibri-1 README, "Querying the server") in the tokenizer's chat template with
enable_thinking=False, identical on every row; row b samples with seed b (generation_config: do_sample,
top_k=128, top_p=0.97, temperature=1.0). B comes from $TT_PERF_BATCH (default 32).

What is asserted, per sample, over the WHOLE generated output (both sides stop on generation_config's
eos_token_id; the cap is the KV capacity and the run must not end on it):
  * Gate 1: every routed module is a graduated stub class whose body equals its graduation snapshot.
  * Gate 2: all 9 graduated modules ran inside the generation forward.
  * prefill (the stage every later one builds on) vs the INDEPENDENT reference: final hidden state PCC
    >= 0.995 (min over rows) and |tt|/|ref| within 0.5% of 1; prompt-position logits PCC >= 0.99.
  * free-running TT output vs the INDEPENDENT free-running reference (same uniforms): identical until
    the first divergence, and that divergence must be a near-tie (see below).
  * teacher-forced: the reference's logits for exactly the prefixes the TT steps saw. Final-output
    PCC (all of a sample's step logits) >= 0.99 for every sample: the printed `e2e PCC`.
  * discrete agreement under teacher forcing: the device sampler picks exactly what the host sampler
    picks on the TT logits, and the TT token EQUALS the reference's token (same uniform) at every step
    whose sampling margin exceeds 2x the total-variation gap between the two distributions -- the
    bound below which a correct pipeline can still land on the other side of a CDF boundary. How many
    steps are that clear, and how many tokens agree overall, is held to a bar read from the reference
    itself: its own bf16-activation run against its fp32 run (golden.precision_baseline).
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest
import torch

from models.demos.kolibri_1.demo.mesh import device_params
from models.demos.kolibri_1.tests.e2e import golden as G
from models.demos.kolibri_1.tt import inputs as kin
from models.demos.kolibri_1.tt import model as tt_model
from models.demos.kolibri_1.tt.pipeline import OSL_ENV, build_pipeline, read_host

E2E_CORRECTNESS_GATE = "test_e2e_text_generation"

PCC_TARGET = 0.99
PREFILL_HIDDEN_FLOOR = 0.995
PREFILL_NORM_TOL = 0.005
# The discrete-agreement bar is read from the reference: the same model run at bf16 activations (its
# config dtype) against its fp32 run. TT must match fp32 at least that well, within this margin.
DISCRETE_MARGIN = 0.02
SAMPLER_TIE_EPS = 1e-5  # device fp32 cumsum vs host fp64: |u - CDF boundary| below this is a numerical tie

DEMO_DIR = Path(__file__).resolve().parents[2]
STUB_CLASSES = {
    "token_embed": ("embed", "TtKolibri1Embedding"),
    "decoder_layer": ("layers", "TtKolibri1DecoderLayer"),
    "attention": ("layers.attn", "TtKolibri1Attention"),
    "f_p8_linear": ("layers.attn.o_proj", "TtKolibri1FP8Linear"),
    "sparse_moe_block": ("layers.moe", "TtKolibri1SparseMoeBlock"),
    "router": ("layers.moe.router", "TtKolibri1Router"),
    "m_l_p": ("layers.moe.shared", "TtKolibri1MLP"),
    "r_m_s_norm": ("norm", "TtKolibri1RMSNorm"),
    "decoder_head": ("head", "TtKolibri1LMHead"),
}


def _pcc(a, b) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _resolve(pipe, path):
    objs = [pipe]
    for part in path.split("."):
        nxt = []
        for o in objs:
            v = getattr(o, part)
            nxt.extend(v if isinstance(v, list) else [v])
        objs = nxt
    return objs


def _check_graduated_stubs(pipe):
    """Gate 1: each routed object is an instance of its graduated stub class, and the stub file on disk is
    the graduation snapshot's code (still the native ttnn body that passed its PCC test; compared as ASTs so
    a formatter pass over the live .py does not count as a change)."""
    import importlib

    for name, (path, cls_name) in STUB_CLASSES.items():
        stub = DEMO_DIR / "_stubs" / f"{name}.py"
        snaps = [stub.with_suffix(".py.last_good_sharded"), stub.with_suffix(".py.last_good_native")]
        snap = next((p for p in snaps if p.is_file()), None)
        assert snap is not None, f"{name}: no graduation snapshot"
        same = ast.dump(ast.parse(stub.read_text())) == ast.dump(ast.parse(snap.read_text()))
        assert same, f"{name}: live stub's code differs from {snap.name}"
        cls = getattr(importlib.import_module(f"models.demos.kolibri_1._stubs.{name}"), cls_name)
        objs = _resolve(pipe, path)
        assert objs and all(isinstance(o, cls) for o in objs), f"{name}: {path} is not a {cls_name}"
    fallbacks = json.loads((DEMO_DIR / "_runtime_fallbacks.json").read_text() or "{}")
    assert not fallbacks, f"runtime CPU fallbacks recorded: {fallbacks}"


@pytest.mark.parametrize("device_params", [device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_e2e_text_generation(mesh_device):
    B = kin.batch_size()
    print(f"PERF_BATCH_STREAMS={B}", flush=True)
    assert B <= G.GOLDEN_SEEDS, f"the reference golden covers {G.GOLDEN_SEEDS} seeds, batch is {B}"
    tok = kin.load_tokenizer()
    prompt = kin.encode_prompt(tok)
    settings = kin.sampling_settings()
    seeds = list(range(B))
    T = len(prompt)
    capped = bool(os.environ.get(OSL_ENV))

    pipe = build_pipeline(mesh_device, batch=B)
    _check_graduated_stubs(pipe)

    # ---- the TT run: the real pipeline, free-running, every step's logits read back for the comparison
    tt_model.INVOCATIONS.clear()
    tt_logits = []
    out = pipe.generate(
        prompt, seeds, on_step=lambda i, lg, ids: tt_logits.append(read_host(lg).reshape(-1, lg.shape[-1])[:B].float())
    )
    hid_tt = read_host(pipe.prefill_hidden).float()[:, 0, :T]
    tt_tokens = out["tokens"]
    for b in range(B):
        print(f"[tt] sample {b} ({len(tt_tokens[b])} tokens): {tok.decode(tt_tokens[b])!r}", flush=True)

    # Gate 2: every graduated module inside the real forward.
    counts = {m: tt_model.INVOCATIONS[m] for m in tt_model.GRADUATED}
    print(f"[gate2] graduated module invocations: {counts}", flush=True)
    missing = [m for m, n in counts.items() if n == 0]
    assert not missing, f"graduated modules never invoked: {missing}"

    # ---- independent reference (free-running, cached; nothing in it depends on this run)
    gold = G.independent_golden(pipe.capacity)
    ref_hidden = gold["prefill_hidden"][:B, :T]
    hid_pcc = [_pcc(hid_tt[b], ref_hidden[b]) for b in range(B)]
    norm_ratio = float(hid_tt.norm() / ref_hidden.norm())
    first_pcc = [_pcc(tt_logits[0][b], gold["prefill_logits"][b]) for b in range(B)]
    print(f"[prefill] final hidden PCC(min over rows)={min(hid_pcc):.6f} norm ratio={norm_ratio:.6f}", flush=True)
    print(f"[prefill] prompt-position logits PCC(min over rows)={min(first_pcc):.6f}", flush=True)

    # ---- teacher-forced reference over the TT trajectories
    tf = G._hf_reference_teacher_forced([prompt + t for t in tt_tokens])
    W = tf["lm_head"]
    uni = kin.sampling_uniforms(seeds, kin.uniforms_length())
    per_sample_tt = [[] for _ in range(B)]
    per_sample_ref = [[] for _ in range(B)]
    sampler_bad, decisive_bad, decisive, total, agree = [], [], 0, 0, 0
    first_div = {}
    for t, step_logits in enumerate(tt_logits):
        rows = [b for b in range(B) if t < len(tt_tokens[b])]
        if not rows:
            break
        r_idx = torch.tensor(rows)
        lt = step_logits[r_idx]
        lr = tf["hidden"][r_idx, T - 1 + t] @ W.t()
        st = G.step_agreement(lt, lr, uni[T - 1 + t, r_idx], settings)
        for j, b in enumerate(rows):
            s_tt = tt_tokens[b][t]
            per_sample_tt[b].append(lt[j])
            per_sample_ref[b].append(lr[j])
            total += 1
            agree += int(st["ref"][j] == s_tt)
            if st["cand"][j] != s_tt and st["cand_margin"][j] > SAMPLER_TIE_EPS:
                sampler_bad.append((b, t))
            if st["decisive"][j]:
                decisive += 1
                if st["ref"][j] != s_tt:
                    decisive_bad.append((b, t, s_tt, int(st["ref"][j]), float(st["margin"][j]), float(st["tv"][j])))
            g = gold["tokens"][b]
            if b not in first_div and (t >= len(g) or g[t] != s_tt):
                first_div[b] = (t, bool(st["decisive"][j]))

    final_pcc = [_pcc(torch.stack(per_sample_tt[b]), torch.stack(per_sample_ref[b])) for b in range(B)]
    for b in range(B):
        div = first_div.get(b)
        where = (
            "identical to the independent reference"
            if div is None
            else f"leaves the independent reference at step {div[0]}"
        )
        print(f"[tf] sample {b}: {len(tt_tokens[b])} steps, logits pcc={final_pcc[b]:.6f}, {where}", flush=True)
    agree_frac, decisive_frac = agree / max(total, 1), decisive / max(total, 1)
    print(
        f"[tf] TT vs fp32 reference: token agreement {agree}/{total} ({agree_frac:.3f}); decisive steps {decisive}/{total} "
        f"({decisive_frac:.3f}); decisive mismatches {len(decisive_bad)}; device-vs-host sampler mismatches {len(sampler_bad)}",
        flush=True,
    )
    bar = G.precision_baseline(gold)
    print(
        f"[tf] bar (the reference at bf16 activations vs its fp32 run, on the independent golden's trajectories): "
        f"agreement {bar['agree_frac']:.3f}, decisive {bar['decisive_frac']:.3f}",
        flush=True,
    )

    # ---- assertions
    assert not sampler_bad, f"device sampler disagrees with the host sampler on its own logits at {sampler_bad[:8]}"
    assert (
        not decisive_bad
    ), f"TT token != reference token at decisive steps (b, t, tt, ref, margin, tv): {decisive_bad[:8]}"
    assert (
        decisive_frac >= bar["decisive_frac"] - DISCRETE_MARGIN
    ), f"decisive share {decisive_frac:.3f} below the bf16-reference bar {bar['decisive_frac']:.3f} - {DISCRETE_MARGIN}"
    assert (
        agree_frac >= bar["agree_frac"] - DISCRETE_MARGIN
    ), f"token agreement {agree_frac:.3f} below the bf16-reference bar {bar['agree_frac']:.3f} - {DISCRETE_MARGIN}"
    bad_div = {b: d[0] for b, d in first_div.items() if d[1]}
    assert not bad_div, f"free-running output left the independent reference at a decisive step: {bad_div}"
    assert min(hid_pcc) >= PREFILL_HIDDEN_FLOOR, f"prefill hidden PCC {min(hid_pcc):.6f} < {PREFILL_HIDDEN_FLOOR}"
    assert (
        abs(norm_ratio - 1.0) <= PREFILL_NORM_TOL
    ), f"prefill hidden norm ratio {norm_ratio:.6f} off by > {PREFILL_NORM_TOL}"
    assert min(first_pcc) >= PCC_TARGET, f"prompt-position logits PCC {min(first_pcc):.6f} < {PCC_TARGET}"
    if B > 1:
        assert len({tuple(t) for t in tt_tokens}) > 1, "all samples produced the same output"
    if capped:
        print(f"[stop] {OSL_ENV} is set: the horizon is capped by the harness, so ending on the cap is expected")
    else:
        not_stopped = [b for b in range(B) if not out["ended_on_stop"][b]]
        assert not not_stopped, f"samples {not_stopped} hit the {out['cap']}-token cap instead of a stop token"
    achieved_pcc = min(final_pcc)
    print(f"e2e PCC={achieved_pcc:.6f}", flush=True)
    assert achieved_pcc >= PCC_TARGET, f"e2e PCC {achieved_pcc:.6f} < {PCC_TARGET} (per sample: {final_pcc})"

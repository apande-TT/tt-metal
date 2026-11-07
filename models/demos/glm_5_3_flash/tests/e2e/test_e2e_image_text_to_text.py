# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end image-text-to-text for zai-org/GLM-5.3-Flash on the 32-chip Galaxy (TP=32 x DP=1).

Inputs (provenance): the published example of Glm5NextForConditionalGeneration.forward (transformers
glm5_next docstring) -- image pipeline-cat-chonk.jpeg + "Describe the image." -- through the HF
Glm5NextProcessor chat template, the image scaled to the pipeline's input size ($TT_PERF_AREA, default
P.DEFAULT_AREA = 1024: a 1382-token prompt), IDENTICAL on every batch row; rows differ
only in their sampling seed (TOOL-AUTHORED seeds 0..B-1) under the model's own generation_config
(temperature, top_p). B is read from $TT_PERF_BATCH (else the pipeline default).

Reference: HF's own module classes with the checkpoint weights, streamed one layer at a time
(reference/hf_reference.py), sampled with the same seeded noise through HF's TopPLogitsWarper, at two precisions:
  * fp32 -- the exact math of the checkpoint (bf16 weights, fp32 activations). THE reference.
  * bf16 -- HF's own production precision. On this model a correct bf16 implementation does not reproduce
    the exact math token for token: measured over 13196 sampled steps, HF bf16 picks a different token than
    HF fp32 on 303 of them, 21 with a score margin > 2 (MoE expert choices and nucleus membership flip under
    bf16 rounding). That is the noise floor the discrete check is held to.
"""

from __future__ import annotations

import math
import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.glm_5_3_flash.reference.hf_reference import teacher_forced_references
from models.demos.glm_5_3_flash.tt import pipeline as P

E2E_CORRECTNESS_GATE = "test_image_text_to_text_matches_hf_golden"

PCC_TARGET = 0.99
FIRST_STAGE_PCC = 0.995
NORM_RATIO_TOL = 0.005
# score margins (Gumbel-max scores of the fp32 reference) at which disagreements are counted
MARGIN_LADDER = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def _ref_scores(logits, noise, sampling):
    """The reference's per-token scores whose argmax is its choice (HF warpers, then the seeded Gumbel noise);
    -inf outside the top-p nucleus."""
    from transformers.generation.logits_process import TemperatureLogitsWarper, TopPLogitsWarper

    s = logits.float()[None]
    if sampling.temperature != 1.0:
        s = TemperatureLogitsWarper(sampling.temperature)(None, s)
    if sampling.top_p < 1.0:
        s = TopPLogitsWarper(sampling.top_p)(None, s)
    return (s + noise[None])[0]


def _batch():
    return int(os.environ.get("TT_PERF_BATCH") or P.DEFAULT_BATCH)


def _depth_kwargs():
    n = os.environ.get("TT_PERF_LAYERS")
    return {"layers": int(n)} if n else {}


def _pcc(a, b):
    return comp_pcc(a.float().reshape(-1), b.float().reshape(-1), PCC_TARGET)[1]


def _norm_ratio(tt, ref):
    # double: an fp32 norm over ~1e8 elements (B=32 full-size prompts) drifts by ~1%
    return (tt.double().norm() / ref.double().norm()).item()


def _rel_err(x, ref):
    return ((x.double() - ref.double()).norm() / ref.double().norm()).item()


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": P.TRACE_REGION_SIZE}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 32)], indirect=True)
def test_image_text_to_text_matches_hf_golden(mesh_device):
    """THE correctness gate. Real input -> chained graduated stubs -> generated text, against the HF reference
    (exact fp32 math; HF's own bf16 forward measured alongside as the noise floor):
    Gate 1 (stubs are pure ttnn, sharded ones still sharded), Gate 2 (all 23 graduated stubs invoked in this
    run), Gate 3 (whole-output logits PCC >= 0.99 on every row; the sampled tokens disagree with the exact
    math's choice no more often than HF's own bf16 forward does, at every score margin), the first stage
    (vision tower output) held at PCC >= 0.995 and norm ratio within 0.5%, the prefill hidden held to HF bf16's
    error and to the 0.5% norm ratio, and the stop-rule termination check."""
    assert mesh_device.get_num_devices() == 32, "needs the 32-chip Galaxy mesh (TP=32)"
    B = _batch()
    print(f"PERF_BATCH_STREAMS={B}", flush=True)

    # ---- Gate 1 (static): graduated stubs are pure ttnn and the TP=32 ones keep their sharding
    purity = P.stub_purity_report()
    assert not any(purity.values()), f"Gate 1 violations: { {k: v for k, v in purity.items() if v} }"

    P.instrument_stubs()
    pipe = P.build_pipeline(mesh_device, batch=B, **_depth_kwargs())
    enc = P.encode_example()
    e = pipe.encode(enc)
    sampling = P.Sampling.from_generation_config(pipe.generation_config, seeds=range(B))
    osl = os.environ.get("TT_PERF_OSL_TOKENS")
    P.reset_stub_calls()
    res = P.run_image_text_to_text(
        pipe,
        e,
        max_new_tokens=int(osl) if osl else None,
        keep_image_embeds=True,
        keep_prefill_hidden=True,
        sampling=sampling,
    )

    # ---- Gate 2: every graduated module ran inside this forward
    missing = [k for k in P.STUB_ENTRYPOINTS if P.STUB_CALLS.get(k, 0) == 0]
    print(f"stub invocations: {dict(P.STUB_CALLS)}", flush=True)
    assert not missing, f"Gate 2: graduated stubs never invoked: {missing}"

    # ---- termination: the model's own stop rule ended the run, not the capacity cap
    if osl:
        print(f"TT_PERF_OSL_TOKENS={osl}: horizon capped by the harness, termination assert skipped", flush=True)
    else:
        assert not res.hit_cap, f"run ended on the capacity cap ({pipe.capacity}) before every row emitted eos"
        assert all(res.stopped_on_eos), f"rows without eos: {[b for b, s in enumerate(res.stopped_on_eos) if not s]}"

    # ---- reference: HF modules over prompt + this run's tokens, fp32 (exact) and bf16 (HF's own precision)
    gen = P.trim_at_eos(res.tokens, pipe.eos_ids)
    refs = teacher_forced_references(
        e.input_ids, gen, e.pixel_values, e.image_grid_thw, pipe.ckpt, pipe.n_text_layers, pipe.n_vision_layers
    )
    f32, bf16 = refs["f32"], refs["bf16"]
    n_tok = sum(len(r) for r in gen)

    # ---- discrete output: the sampled token at every step against the exact math's own choice (same prefix,
    # same seeded noise, HF warpers). margin = the fp32 reference's score of its pick minus its score of the
    # token taken (inf: outside its top-p nucleus). HF's own bf16 forward is scored the same way; at EVERY
    # margin threshold TT may not disagree more often than it.
    tt_m, bf_m, worst = [], [], []
    for b in range(B):
        for t in range(len(gen[b])):
            noise = P.gumbel_noise([sampling.seeds[b]], t, pipe.V)[0]
            s32 = _ref_scores(f32["logits"][b, t], noise, sampling)
            r32 = int(s32.argmax())
            r16 = int(_ref_scores(bf16["logits"][b, t], noise, sampling).argmax())
            g = gen[b][t]
            m_tt = 0.0 if g == r32 else (s32[r32] - s32[g]).item()
            m_bf = 0.0 if r16 == r32 else (s32[r32] - s32[r16]).item()
            if g != r32:
                tt_m.append(m_tt)
                worst.append((m_tt, b, t, g, r32, r16))
            if r16 != r32:
                bf_m.append(m_bf)

    def _count(margins, thr):  # thr 0: every disagreement; inf: outside the reference's nucleus
        if thr == 0.0:
            return len(margins)
        return sum(1 for m in margins if (m == math.inf if thr == math.inf else m > thr))

    ladder = [(thr, _count(tt_m, thr), _count(bf_m, thr)) for thr in MARGIN_LADDER + (math.inf,)]
    print(
        f"sampled tokens vs exact-math choice over {n_tok} steps -- [margin > thr: TT disagreements / HF-bf16 "
        f"disagreements]: {[(f'{t:g}', a, c) for t, a, c in ladder]}; TT worst (margin,row,step,tt,fp32,bf16): "
        f"{sorted(worst, reverse=True)[:6]}",
        flush=True,
    )

    if os.environ.get("GLM53_SAVE_RUN"):  # optional dump for post-hoc analysis (not part of the verdict)
        torch.save(
            dict(
                gen=gen, tt_logits=res.logits, img=res.image_embeds, hid=res.prefill_hidden, seeds=list(sampling.seeds)
            ),
            os.environ["GLM53_SAVE_RUN"],
        )

    # ---- first stage held tighter: the vision tower's output, the hidden state the first stage hands every later
    # stage, against the exact (fp32) HF vision tower.
    R = res.image_embeds.shape[0] // B
    img_ref = f32["img"]
    img_pcc = min(_pcc(res.image_embeds.reshape(B, R, -1)[b], img_ref.reshape(B, R, -1)[b]) for b in range(B))
    img_ratio = _norm_ratio(res.image_embeds, img_ref)
    print(f"image embeds vs HF fp32 vision: PCC(min row)={img_pcc:.6f} norm ratio={img_ratio:.5f}", flush=True)
    # The prefill backbone hidden over the prompt: bf16 rounding flips MoE expert choices at image positions, so
    # even HF's own bf16 forward is far from the exact math there. Held to the norm ratio (systematic level error)
    # and to an error no larger than HF bf16's, on every row.
    L0 = e.real_len
    hid_tt = res.prefill_hidden.reshape(B, -1, pipe.D)[:, :L0]
    hid_err_tt = [_rel_err(hid_tt[b], f32["hid"][b]) for b in range(B)]
    hid_err_bf = [_rel_err(bf16["hid"][b], f32["hid"][b]) for b in range(B)]
    hid_ratio = _norm_ratio(hid_tt, f32["hid"])
    print(
        f"prefill hidden vs HF fp32: rel-err max over rows TT {max(hid_err_tt):.4f} (HF bf16 {max(hid_err_bf):.4f}), "
        f"norm ratio {hid_ratio:.5f}",
        flush=True,
    )

    # ---- logits over the WHOLE generated output of every row (all steps of the row concatenated)
    row_pcc, row_pcc_bf, step_tt, step_bf = [], [], [], []
    for b in range(B):
        n_b = len(gen[b])
        tt_rows = torch.stack([res.logits[t][b].float() for t in range(n_b)])
        ref_rows = f32["logits"][b, :n_b].float()
        row_pcc.append(_pcc(tt_rows, ref_rows))
        row_pcc_bf.append(_pcc(tt_rows, bf16["logits"][b, :n_b]))
        step_tt += [_pcc(tt_rows[t], ref_rows[t]) for t in range(n_b)]
        step_bf += [_pcc(bf16["logits"][b, t], ref_rows[t]) for t in range(n_b)]
    st, sb = torch.tensor(step_tt), torch.tensor(step_bf)
    print(
        f"steps compared: {n_tok} (rows {B}, longest {max(len(r) for r in gen)}); per-step logits vs fp32: "
        f"TT mean {st.mean():.6f} p05 {st.quantile(0.05):.6f} lowest {st.min():.4f} | HF bf16 mean {sb.mean():.6f} "
        f"p05 {sb.quantile(0.05):.6f} lowest {sb.min():.4f}",
        flush=True,
    )
    achieved_pcc = min(row_pcc)
    print(f"whole-output logits vs HF bf16: PCC(min row)={min(row_pcc_bf):.6f}", flush=True)
    print(f"e2e PCC={achieved_pcc}", flush=True)

    for thr, n_tt, n_bf in ladder:
        assert n_tt <= n_bf, (
            f"sampled tokens: {n_tt} disagree with the exact math's choice at margin > {thr} vs {n_bf} for HF's own "
            f"bf16 forward; worst {sorted(worst, reverse=True)[:8]}"
        )
    assert (
        img_pcc >= FIRST_STAGE_PCC and abs(img_ratio - 1) <= NORM_RATIO_TOL
    ), f"image embeds PCC {img_pcc} ratio {img_ratio}"
    assert abs(hid_ratio - 1) <= NORM_RATIO_TOL, f"prefill hidden norm ratio {hid_ratio}"
    bad_rows = [b for b in range(B) if hid_err_tt[b] > hid_err_bf[b]]
    assert not bad_rows, f"prefill hidden error above HF bf16's on rows {bad_rows}"
    assert st.mean() >= sb.mean() and st.quantile(0.05) >= sb.quantile(0.05), "per-step logits below HF bf16's"
    assert min(row_pcc_bf) >= PCC_TARGET, f"whole-output logits vs HF bf16 PCC {min(row_pcc_bf)} < {PCC_TARGET}"
    assert achieved_pcc >= PCC_TARGET, f"e2e logits PCC {achieved_pcc} < {PCC_TARGET}"

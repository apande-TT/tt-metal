# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Call 1, end to end: real text -> a real 24 kHz waveform, against Source A's golden.

THE CHAIN UNDER TEST IS THE ONE THE DEMO RUNS. Both call
`pipeline.VoxtralTTSPipeline.run_text_to_speech`, so there is exactly one copy of the wiring and a
green test cannot coexist with a broken demo.

NOTHING IS EVER SPLICED INTO THE TT SIDE. The TT chain runs completely free: prefill -> (acoustic
frame -> audio-token embedding -> decode step) x T -> vocode, each stage fed the previous TT
stage's real output, exactly as the demo runs it. What the reference is asked changes, and only
the reference: it is put on the TT chain's own trajectory, so at every stage it answers "given
exactly the context this pipeline produced, what does torch compute?". That is a well-posed
question, and it is asked at EVERY joint -- prefill hidden, each frame's backbone hidden, each
frame's semantic logits, each frame's flow-sampler output, and the final waveform. A wiring bug
cannot hide behind it: if a TT stage consumed the wrong thing, its output would no longer match
what torch computes from the same inputs, and one of those five comparisons would drop.

EVERY STAGE IS SCORED ON THE INPUT IT CONSUMED -- all three of them, not just the convenient ones.
The backbone gets the prompt (`input_ids`); the acoustic stage gets the hidden the backbone handed
it (`fed_hidden`); the codec gets the codes the acoustic stage emitted (`fed_codes`). Scoring a
stage against a reference run on a DIFFERENT input measures the stage before it, amplified. That
is not a hypothetical here: the flow sampler runs classifier-free guidance at alpha=3
(`v = 3*v_cond - 2*v_uncond`), which measured a 9.4x amplification -- a per-frame hidden at PCC
0.999163 came out as an `x_final` at 0.9259 while the sampler itself was exact on identical inputs
(velocity PCC 1.000000, semantic logits 1.000000, 99.3% of codes bit-identical). The backbone's
error is asserted where it belongs, on the backbone.

WHY THE COMPARISON IS NOT TT-FREE-RUN vs HF-FREE-RUN. That was the previous shape of this gate and
it is below the hardware's noise floor -- not by a little. The chain's discrete bottleneck is a
round onto 21 levels 0.1 apart, so a free-running comparison measures how fast two trajectories
separate, not whether the port is right. Measured on this machine (pure torch on BOTH arms, no
device involved, `tt/golden.py`'s own chain perturbed by a relative epsilon):

    per-frame hidden eps=1e-4 -> codes identical,      min waveform corr 1.000000
    per-frame hidden eps=1e-3 -> code agreement 0.9896, min waveform corr 0.934352
    per-frame hidden eps=1e-2 -> code agreement 0.9452, min waveform corr 0.778929
    x_final          eps=1e-4 -> code agreement 0.9992, min waveform corr 0.997447

and, measured on this device against a float64 reference, ONE matmul (M=32 K=N=3072, HiFi4 +
`fp32_dest_acc_en`, the fidelity this pipeline already runs at):

    fp32 act x fp32 weight -> 1.169e-3      fp32 act x bf16 weight -> 1.738e-3

So a single matmul's rounding is already ~10x larger than the whole 26-layer chain's budget for
identical codes, and the free-running form of this gate is unreachable by ANY implementation on
this hardware -- including torch's own, at fp32. It is not a TT result. The per-stage comparisons
below are, and they are strictly more of the pipeline than the old single waveform number covered.
The free-running numbers are still computed and printed every run, as diagnostics.
"""
from __future__ import annotations

import json
import os

import pytest
import torch

from models.demos.voxtral_4b_tts_2603.reference import golden
from models.demos.voxtral_4b_tts_2603.tt import common, pipeline

pytestmark = pytest.mark.timeout(3600)

PCC_TARGET = 0.99

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def routed_modules(call_name: str) -> set:
    """The graduated modules this call routes, read from the PLAN rather than typed here."""
    with open(os.path.join(PKG_ROOT, "e2e_plan.json")) as f:
        plan = json.load(f)
    for head in plan["task_heads"]:
        if head["name"] == call_name:
            return set(head["graduated_modules_routed"])
    raise KeyError(call_name)


@pytest.fixture(scope="module")
def evidence(device, hf_model):
    """Run the TT pipeline once, then BOTH reference arms, and hand all three to every assertion."""
    common.use_all_cpu_threads()
    pipe = pipeline.build_pipeline(device, model=hf_model, heads=("text_to_speech",))

    batch = pipe.batch
    input_ids, texts = pipe.default_inputs()
    max_frames, provenance = common.resolve_max_frames(hf_model, gate=True)
    x0 = pipe.noise(max_frames, batch=batch)
    cfg_alpha = torch.full((batch,), pipeline.DEFAULT_CFG_ALPHA)

    print(f"\nbatch driven (read from the pipeline): {batch}")
    print(f"prompt tokens: {input_ids.shape[1]}  (32 real + [BEGIN_AUDIO])")
    print(f"max_frames: {max_frames}  <- {provenance}")
    print(f"stop rule: semantic code == end_audio (id {pipe.stop_token_id}), read off the reference")

    tt = pipe.run_text_to_speech(input_ids=input_ids, x0=x0, cfg_alpha=cfg_alpha, max_frames=max_frames, collect=True)

    # The goldens are memoised on disk: this is a 3.4 B torch model on CPU and the gate is iterated
    # many times against an unchanging reference. The key covers the inputs, the chain version AND
    # the reference loader's contract number, so a loader that starts covering more of the
    # checkpoint -- or a chain that starts returning something else -- invalidates every cached
    # golden rather than being compared against a stale one.
    base = dict(
        task="tts",
        ids=input_ids,
        x0=x0,
        cfg=cfg_alpha,
        max_frames=max_frames,
        B=batch,
        chain=golden.CHAIN_VERSION,
    )
    free = common.cached_golden(
        common.golden_key(arm="free", **base),
        lambda: golden.hf_reference_text_to_speech(hf_model, input_ids, x0, cfg_alpha, max_frames),
    )
    # The acoustic stage is scored on the hidden it actually consumed, the same way the codec is
    # scored on the codes it actually rendered. Without this the acoustic comparison measures the
    # BACKBONE amplified by classifier-free guidance -- see the module docstring.
    tt_hidden = torch.stack([d["llm_hidden"] for d in tt["diagnostics"]], dim=-1)
    aligned = common.cached_golden(
        common.golden_key(arm="aligned", codes=tt["codes"], hidden=tt_hidden, **base),
        lambda: golden.hf_reference_text_to_speech(
            hf_model,
            input_ids,
            x0,
            cfg_alpha,
            max_frames,
            fed_codes=tt["codes"],
            fed_hidden=tt_hidden,
        ),
    )
    return {
        "pipe": pipe,
        "tt": tt,
        "hf": aligned,
        "free": free,
        "batch": batch,
        "texts": texts,
        "input_ids": input_ids,
        "x0": x0,
        "cfg_alpha": cfg_alpha,
        "max_frames": max_frames,
        # Every constant the discretization rule needs, READ OFF THE REFERENCE rather than typed
        # here -- a rule spelled out from literals would agree with itself, not with the model.
        "levels": int(hf_model.acoustic_transformer.acoustic_embeddings_levels),
        "n_special": golden.n_special_tokens(),
        "empty_id": common.audio_empty_token_id(hf_model),
        "stop_token_id": pipe.stop_token_id,
        "semantic_size": int(hf_model.acoustic_transformer.model_args.semantic_codebook_size),
    }


def test_golden_is_the_references_own_arithmetic(hf_model):
    """The golden replaces one `torch.randn` and nothing else -- proven, not asserted by comment.

    `decode_one_frame` draws `x_0` INSIDE the module, so its output is a function of the RNG and
    no PCC against it would mean anything. Seed, call the UNMODIFIED module; re-seed, draw the
    same first tensor it would have drawn, feed that to the helper; the frames must be identical.
    """
    common.use_all_cpu_threads()
    ids, _ = common.build_batch_inputs(batch=4)
    with torch.no_grad():
        h = hf_model.model(input_ids=ids).last_hidden_state[:, -1]
    ok, from_module, from_helper, x0 = golden.reference_frame_matches_module(
        hf_model, h, torch.full((4,), pipeline.DEFAULT_CFG_ALPHA), seed=0
    )
    print(f"golden faithfulness: module codes == helper codes -> {ok}  (x0 {tuple(x0.shape)})")
    assert ok, (
        "the golden's Euler loop is NOT the reference's arithmetic:\n"
        f"  module {from_module[0, :8].tolist()}\n  helper {from_helper[0, :8].tolist()}"
    )


def test_gate2_every_call_1_stub_was_invoked(evidence):
    """Gate 2: each routed stub really ran, inside the real forward path."""
    invoked = evidence["pipe"].invoked()
    expected = routed_modules("text_to_speech")
    missing = sorted(expected - set(invoked))
    print(f"\ninvoked {len(invoked)} stubs; counts: {dict(sorted(invoked.items()))}")
    assert not missing, f"graduated modules routed to Call 1 but never invoked: {missing}"
    assert all(count >= 1 for count in invoked.values())


def test_shapes_and_real_task_output(evidence):
    """The output is real audio, not a smoke-test tensor."""
    tt, free, batch = evidence["tt"], evidence["free"], evidence["batch"]
    frames = tt["frames_decoded"]
    print(f"\nframes decoded: TT={frames} HF={free['frames_decoded']}")
    print(f"TT stop reason: {tt['stop_reason']}")
    print(f"HF stop reason: {free['stop_reason']}")
    assert frames == free["frames_decoded"], "TT and HF decoded different lengths -- the stop rule diverged"

    assert tuple(tt["codes"].shape) == (batch, 37, frames)
    assert tuple(tt["waveform"].shape) == (
        batch,
        1,
        frames * 1920,
    ), f"waveform is {tuple(tt['waveform'].shape)}, expected {frames} frames x 1920 samples"
    assert tt["sampling_rate"] == 24000
    duration = tt["waveform"].shape[-1] / tt["sampling_rate"]
    print(f"audio: {duration:.2f} s at {tt['sampling_rate']} Hz ({frames} frames at 12.5 Hz)")

    wav = tt["waveform"]
    assert torch.isfinite(wav).all(), "the waveform contains non-finite samples"
    assert wav.abs().max() <= 1.5, f"waveform out of audio range: max |x| = {float(wav.abs().max())}"
    per_sample_std = wav.reshape(batch, -1).std(dim=1)
    assert float(per_sample_std.min()) > 1e-4, "at least one waveform is constant -- that is not audio"


def test_batch_is_32_independent_samples(evidence):
    """A pipeline that shape-supports B but emits 32 identical outputs is WRONG."""
    tt, batch = evidence["tt"], evidence["batch"]
    assert batch == common.DEFAULT_BATCH == 32
    rows = {tuple(r.tolist()) for r in evidence["input_ids"]}
    assert len(rows) == batch, "the 32 inputs are not pairwise distinct"
    waves = {tt["waveform"][i].numpy().tobytes() for i in range(batch)}
    codes = {tt["codes"][i].numpy().tobytes() for i in range(batch)}
    print(
        f"\ndistinct inputs {len(rows)}/{batch}; distinct code streams {len(codes)}/{batch}; "
        f"distinct waveforms {len(waves)}/{batch}"
    )
    assert len(waves) == batch, f"only {len(waves)} of {batch} waveforms are distinct"


def test_per_stage_pcc(evidence):
    """EVERY joint of the chain, against the reference driven by this pipeline's own trajectory.

    Five comparisons, each over all 32 samples and all decoded frames. Together they cover the
    whole forward path: the 26-layer prefill, the KV-cached decode step and the audio-token
    embedding that feeds it, the semantic head, the flow sampler, and the codec.
    """
    tt, hf, batch = evidence["tt"], evidence["hf"], evidence["batch"]
    frames = tt["frames_decoded"]
    diag_tt, diag_hf = tt["diagnostics"], hf["diagnostics"]
    assert len(diag_tt) == frames and len(diag_hf) == frames

    prefill = min(common.pcc(tt["prefill_hidden"][i], hf["prefill_hidden"][i]) for i in range(batch))
    print(f"\nstage PCC  prefill hidden           (min over {batch} samples) = {prefill:.6f}")
    assert prefill >= PCC_TARGET, f"the text stack is already below target at {prefill:.6f}"

    hidden = min(
        common.pcc(diag_tt[t]["llm_hidden"][i], hf["llm_hiddens"][t][i]) for t in range(frames) for i in range(batch)
    )
    per_frame = [
        min(common.pcc(diag_tt[t]["llm_hidden"][i], hf["llm_hiddens"][t][i]) for i in range(batch))
        for t in range(frames)
    ]
    print(f"stage PCC  decode hidden, per frame  (min over {batch} x {frames})   = {hidden:.6f}")
    print(f"           per-frame trend: {' '.join(f'{v:.5f}' for v in per_frame)}")
    assert hidden >= PCC_TARGET, (
        f"the decode step drifts: frame-wise hidden PCC {hidden:.6f} < {PCC_TARGET}. The reference "
        f"is fed THIS pipeline's own codes, so a drop here is the KV cache, the position ids or "
        f"the audio-token embedding -- not divergence."
    )

    semantic = min(
        common.pcc(diag_tt[t]["semantic_logits"][i], diag_hf[t]["semantic_logits_raw"][i])
        for t in range(frames)
        for i in range(batch)
    )
    print(f"stage PCC  semantic head logits      (min over {batch} x {frames})   = {semantic:.6f}")
    assert semantic >= PCC_TARGET, f"the semantic head is at {semantic:.6f}"

    x_final = min(
        common.pcc(diag_tt[t]["x_final"][i].clamp(-1, 1), diag_hf[t]["x_final"][i])
        for t in range(frames)
        for i in range(batch)
    )
    print(f"stage PCC  flow sampler x_final      (min over {batch} x {frames})   = {x_final:.6f}")
    assert x_final >= PCC_TARGET, f"the acoustic flow sampler is at {x_final:.6f}"


def test_discretization_is_the_references_own_rule(evidence):
    """The DISCRETE step, checked EXACTLY: no tolerance, no sigma, no threshold.

    Comparing TT's codes against the REFERENCE's codes cannot be made exact -- `x_final` is rounded
    onto 21 levels 0.1 apart, so any deviation at all flips whichever elements sit near a boundary,
    and a "how many may differ" bound is a statistic, not a check (a fixed 3 sigma over 15 000
    elements expects ~40 exceedances by chance, and this deviation is heavy-tailed: RMS 2.1e-2,
    worst 4.0e-1).

    So this asserts the thing that IS exact. The reference's rule is
    `round(((clamp(x, -1, 1) + 1) / 2) * (levels - 1)) + n_special`, with a finished row forced to
    `empty_audio`. Apply that rule, in torch, to the x_final THIS PIPELINE produced, and it must
    reproduce the pipeline's codes bit for bit -- because the on-device clamp / rescale / round /
    offset is supposed to BE that rule. A wrong level count, a wrong offset, a missing clamp or a
    misplaced `empty_audio` substitution all break this by a mile, and none of them can hide behind
    a rounding tie. The NUMERIC quality of `x_final` itself is gated separately, in
    `test_per_stage_pcc`.
    """
    tt, hf, batch = evidence["tt"], evidence["hf"], evidence["batch"]
    frames, levels = tt["frames_decoded"], evidence["levels"]
    n_special, stop_id = evidence["n_special"], evidence["stop_token_id"]
    scale = 0.5 * (levels - 1)

    x_tt = torch.stack([tt["diagnostics"][t]["x_final"] for t in range(frames)], dim=-1)
    s_tt = (x_tt.clamp(-1, 1) + 1) * scale
    rule = s_tt.round().long() + n_special
    finished = (tt["codes"][:, 0, :] == stop_id).unsqueeze(1).expand_as(rule)
    rule = torch.where(finished, torch.full_like(rule, evidence["empty_id"] + n_special), rule)

    codes_tt = tt["codes"][:, 1:, :]
    mismatch = int((codes_tt != rule).sum())
    print(
        f"\ndiscretization: the reference's rule applied to THIS pipeline's x_final reproduces "
        f"{rule.numel() - mismatch}/{rule.numel()} codes"
    )
    assert mismatch == 0, (
        f"{mismatch} of {rule.numel()} acoustic codes are not what the reference's own "
        f"clamp/rescale/round/offset gives on the pipeline's own x_final -- the on-device "
        f"discretization is not that rule"
    )

    # The semantic code, the same way: the reference's argmax over the reference's own -inf mask,
    # applied to THIS pipeline's logits.
    lo_tt = torch.stack([tt["diagnostics"][t]["semantic_logits"] for t in range(frames)], dim=-1)
    masked = lo_tt.clone()
    masked[:, evidence["empty_id"], :] = -float("inf")
    masked[:, n_special + evidence["semantic_size"] :, :] = -float("inf")
    sem_rule = masked.argmax(dim=1)
    sem_mismatch = int((tt["codes"][:, 0, :] != sem_rule).sum())
    print(
        f"semantic argmax: the reference's masked argmax on this pipeline's logits reproduces "
        f"{sem_rule.numel() - sem_mismatch}/{sem_rule.numel()} codes"
    )
    assert sem_mismatch == 0, (
        f"{sem_mismatch} semantic codes are not the reference's masked argmax of the pipeline's "
        f"own logits -- the on-device mask or argmax is not that rule"
    )

    # Reported, not asserted: how the codes land against the reference's OWN codes. Both sides
    # ran the acoustic stage on the same hidden here, so this is the deviation's tie rate.
    dev = s_tt - (torch.stack([hf["diagnostics"][t]["x_final"] for t in range(frames)], dim=-1) + 1) * scale
    print(
        f"acoustic codes vs the reference's own: agreement "
        f"{float((codes_tt == hf['codes'][:, 1:, :]).float().mean()):.6f}  "
        f"(rescaled deviation RMS={float(dev.pow(2).mean().sqrt()):.3e}, "
        f"worst={float(dev.abs().max()):.3e}; the grid spacing is 1.0)"
    )
    print(
        f"semantic codes vs the reference's own: agreement "
        f"{float((tt['codes'][:, 0, :] == hf['codes'][:, 0, :]).float().mean()):.6f}"
    )


def test_free_running_divergence_is_reported(evidence):
    """Report the TT-free-run vs HF-free-run numbers, and assert the part of them that is decidable.

    Frame 0 takes NO feedback: both sides compute it from their own prefill hidden and nothing
    else, so its semantic code is a clean discrete comparison with no accumulated trajectory in
    it. Everything after frame 0 is reported, because the measurements in this module's docstring
    show that a perturbation smaller than one matmul's rounding already scrambles it.
    """
    tt, free, batch = evidence["tt"], evidence["free"], evidence["batch"]
    steps = min(tt["codes"].shape[-1], free["codes"].shape[-1])
    agree = float((tt["codes"][..., :steps] == free["codes"][..., :steps]).float().mean())
    n = min(tt["waveform"].shape[-1], free["waveform"].shape[-1])
    corr = min(common.pcc(tt["waveform"][i, ..., :n], free["waveform"][i, ..., :n]) for i in range(batch))
    print(
        f"\nfree-running reference (DIAGNOSTIC, not a gate metric -- see the module docstring): "
        f"code agreement={agree:.6f}  min waveform corr={corr:.6f}"
    )

    frame0 = tt["codes"][:, 0, 0] == free["codes"][:, 0, 0]
    print(f"frame-0 semantic code (no feedback in it): {int(frame0.sum())}/{batch} exact")
    assert bool(frame0.all()), (
        "the very first semantic code already disagrees, before any feedback exists -- that is a "
        "prefill/semantic-head error, not divergence"
    )


def test_gate3_e2e_pcc(evidence):
    """GATE 3: the FINAL output -- the waveform -- against the HF golden, for all 32 samples.

    The reference here is torch's own codec rendering the codes this pipeline emitted, so this
    compares the real end-to-end task output on identical input. Together with `test_per_stage_pcc`
    it covers every stage from the prompt tokens to the audio samples.
    """
    tt, hf, batch = evidence["tt"], evidence["hf"], evidence["batch"]
    assert torch.equal(tt["codes"], hf["fed_codes"]), "the reference was not put on the TT trajectory"

    per_sample = [common.pcc(tt["waveform"][i], hf["waveform"][i]) for i in range(batch)]
    achieved_pcc = min(per_sample)
    worst = int(torch.tensor(per_sample).argmin())

    print(
        f"\nper-sample waveform PCC: min={achieved_pcc:.6f} "
        f"mean={sum(per_sample) / batch:.6f} max={max(per_sample):.6f} (worst sample {worst})"
    )
    print(f"e2e PCC={achieved_pcc}")
    assert achieved_pcc >= PCC_TARGET, (
        f"Gate 3 FAILED: waveform PCC {achieved_pcc:.6f} < {PCC_TARGET} on sample {worst}. "
        f"Both sides render the SAME codes here, so this is the codec stage's own numerics -- "
        f"fidelity/dtype on the vocoder, never a change to the comparison."
    )

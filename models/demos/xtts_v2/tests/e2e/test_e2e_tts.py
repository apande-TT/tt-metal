# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end TTS pipeline test for coqui/XTTS-v2 on the 8-chip mesh (TP=8 x DP=1).

Real input (DVAE cond-mel + text tokens + 16 kHz speaker ref) -> the SHARED
chained TTNN pipeline (tt/pipeline.py) -> real waveform, compared to the Coqui
reference. Asserts Gate 1 (routed stubs still native/sharded), Gate 2 (every
graduated module invoked in the real forward), Gate 3 (final-waveform PCC>=0.95).

DECODE BATCH = 4: the pipeline runs 4 INDEPENDENT streams (4 different sentences,
4 different shipped speaker references) through one decode program, and EVERY
stream is compared to its OWN reference -- its own ``gpt.generate()`` audio codes
and its own deterministic latent+vocode golden. The reported e2e PCC is the WORST
stream, so one good stream cannot carry a broken one.

Run:  ./python_env/bin/python -m pytest models/demos/xtts_v2/tests/e2e/test_e2e_tts.py -s
"""
from __future__ import annotations

import glob
import os

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.xtts_v2.tt import pipeline as P

HERE = os.path.dirname(os.path.abspath(__file__))
XDIR = os.path.normpath(os.path.join(HERE, "..", ".."))
STUBS = os.path.join(XDIR, "_stubs")
PCC_GATE = 0.95


# The speaker encoder's convolutions are native ttnn.conv2d, whose sliding-window/halo
# path allocates from the L1_SMALL region -- that region is 0 B unless it is reserved at
# device open, so the pipeline requires it exactly as the per-stub PCC tests do.
# The vocoder trunk is native ttnn.conv1d/conv_transpose2d and the speaker encoder is
# native ttnn.conv2d; both run a sliding-window/halo gather whose sharding + config
# tensors allocate from the dedicated L1_SMALL pool. That pool is 0 B unless reserved
# at device open, and 4 KB only covered the 3x3 conv2d halo -- the vocoder's k=11
# dilated taps over 6656 samples need more, and coming up short surfaces as a
# TT_FATAL "Out of Memory ... bank size is 0 B", not an API error.
_DEV_PARAMS = {"l1_small_size": 131072}


def _open_mesh():
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        return ttnn.open_mesh_device(ttnn.MeshShape(1, 8), **_DEV_PARAMS), True
    except Exception as e:
        print(f"[e2e] mesh open failed ({e}); single-device fallback")
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:
            pass
        return ttnn.open_device(device_id=0, **_DEV_PARAMS), False


def _close(dev, is_mesh):
    if is_mesh:
        ttnn.close_mesh_device(dev)
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:
            pass
    else:
        ttnn.close_device(dev)


def _gate1_stubs_native():
    """Every routed stub file is unchanged from its graduated snapshot AND
    still real ttnn; the composed GPT/perceiver bodies carry ShardTensorToMesh +
    a collective (genuine TP=8, not pure replication)."""
    import filecmp

    routed = P.CANONICAL_STUBS + ["learned_position_embeddings"]
    for name in routed:
        live = os.path.join(STUBS, f"{name}.py")
        snaps = glob.glob(os.path.join(STUBS, f"{name}.py.last_good_*"))
        assert snaps, f"no graduated snapshot for {name}"
        assert filecmp.cmp(live, snaps[0], shallow=False), f"{name} LIVE stub != graduated snapshot"
        src = open(live).read()
        assert "ttnn." in src, f"{name} is not ttnn"
    # pipeline-level TP evidence: the GPT + perceiver stages shard + collect.
    shard = any("ShardTensorToMesh" in open(os.path.join(STUBS, f"{n}.py")).read()
                for n in ("g_p_t", "gpt_gpt_inference", "perceiver_resampler"))
    coll = any(("all_gather" in open(os.path.join(STUBS, f"{n}.py")).read()
                or "all_reduce" in open(os.path.join(STUBS, f"{n}.py")).read())
               for n in ("g_p_t", "gpt_gpt_inference", "perceiver_resampler"))
    assert shard and coll, "pipeline lacks ShardTensorToMesh + a collective (not genuine TP=8)"


def _gate2_coverage(invoked):
    import itertools

    for s in P.CANONICAL_STUBS:
        assert s in invoked, f"canonical stub {s} was NOT invoked in the real forward"
    covered = set(itertools.chain(*P.COVERAGE_MAP.values()))
    grad = {os.path.basename(f).split(".py.")[0]
            for f in glob.glob(os.path.join(STUBS, "*.last_good_sharded"))
            + glob.glob(os.path.join(STUBS, "*.last_good_native"))}
    missing = grad - covered
    assert not missing, f"graduated modules not covered by the pipeline: {sorted(missing)}"
    return len(grad), len(covered)


# A greedy argmax is only as decisive as the gap between the top-2 logits. XTTS's first audio
# code is frequently a near-tie (measured: 6.473 / 6.394 / 6.365 for three candidates), and a
# bf16 logit at that magnitude resolves to ~0.025 -- so the reference's own ordering is inside
# the numeric noise and a matching pipeline can legitimately pick the runner-up. Rather than
# ignore a divergence (or pretend it cannot happen), each diverging step is CERTIFIED: the TT
# step distribution must still match the reference's teacher-forced distribution, the
# reference's pick must be in the TT top-3, and BOTH sides must consider the two candidates
# nearly equal. Anything else is a real error and fails the gate.
STEP_LOGITS_PCC = 0.99
TIE_EPS = 0.30


def _certify_steps(pipe, inputs, gen, b, codes_tt, codes_hf, m):
    """Compare one stream's greedy sequence to its reference, up to the FIRST divergence.

    Both sides are greedy and autoregressive, so a flip at step k re-conditions every later
    step on a DIFFERENT prefix -- steps past k are not comparable token-wise and are not
    evidence of anything. So: assert the exact match on [0, k), then certify step k itself
    against the reference's teacher-forced distribution for the SAME prefix.

    Returns (prefix_len, verdict, step_pcc); verdict is None when the whole horizon matched.
    """
    cond_mel, _ref_wav, text_inputs, _c = inputs
    for s in range(m):
        if int(codes_tt[s]) == int(codes_hf[s]):
            continue
        # reference distribution for this step, teacher-forced on the SHARED prefix
        # (codes_hf[:s] == codes_tt[:s] by construction -- everything before s matched)
        lg_hf = pipe.hf_step_logits(cond_mel[b:b + 1], text_inputs[b:b + 1],
                                   codes_hf[:s] if s else None)
        lg_tt = gen["step_logits"][s][b].reshape(-1)
        pcc = comp_pcc(lg_hf.reshape(1, -1), lg_tt.reshape(1, -1))[1]
        id_tt, id_hf = int(codes_tt[s]), int(codes_hf[s])
        top3 = torch.topk(lg_tt, 3).indices.tolist()
        gap_tt = float(lg_tt[id_tt] - lg_tt[id_hf])
        gap_hf = float(lg_hf[id_hf] - lg_hf[id_tt])
        tie = (pcc >= STEP_LOGITS_PCC and id_hf in top3
               and gap_tt <= TIE_EPS and gap_hf <= TIE_EPS)
        print(f"[e2e]   stream {b}: exact for the first {s}/{m} codes; step {s} TT={id_tt} "
              f"HF={id_hf} step-logits PCC={pcc:.5f} TT top3={top3} gap_tt={gap_tt:.3f} "
              f"gap_hf={gap_hf:.3f} -> {'NEAR-TIE (certified)' if tie else 'REAL DIVERGENCE'}"
              f"  [steps > {s} follow a different prefix on each side: not comparable]")
        return (s, tie, pcc)
    return (m, None, 1.0)


def test_e2e_tts():
    dev, is_mesh = _open_mesh()
    try:
        model = P.load_reference_model()
        pipe = P.build_pipeline(dev, model=model, batch=P.DECODE_BATCH)
        B = pipe.batch

        # ---- real input: B DISTINCT streams, encoded by the model's own front end ----
        inputs = pipe.streams()
        cond_mel, ref_wav, text_inputs, _cap_codes = inputs
        print(f"[e2e] {B} streams: speakers={list(P.SPEAKER_SAMPLES[:B])} "
              f"cond_mel={tuple(cond_mel.shape)} ref_wav={tuple(ref_wav.shape)} "
              f"text={tuple(text_inputs.shape)}")

        # ---- Gate 1 ----
        _gate1_stubs_native()
        print("[e2e] Gate 1 OK: routed stubs are native ttnn / sharded snapshots")

        # ---- TT: ONE batched chained forward, decode feeding latent feeding vocode ----
        # Horizon is model-grounded: the vocoder consumes VOCODE_LEN audio codes, so that
        # is how many codes BOTH sides generate; the model's stop_audio_token truncates
        # earlier if it fires, and the config's max audio-token context caps it.
        horizon = pipe.vocode_len
        gen = pipe.run_tts(cond_mel, ref_wav, text_inputs, audio_codes=None,
                           horizon=horizon, generate=True)
        assert int(gen["batch"]) == B

        # ---- reference goldens, per stream (each stream's OWN generate()) ----
        # gold[b]["codes"] is stream b's own generate() sequence (the decode's behavioural
        # reference); the deterministic latent+vocode golden is built over the codes the TT
        # side actually generated, so the waveform comparison is over the SAME tokens even
        # when greedy flips on a numerical tie. Nothing is injected INTO the TT chain.
        gold = pipe.hf_reference_streams(inputs, horizon=horizon, codes=gen["codes_used"])

        # ---- per-stream comparison against each stream's OWN golden ----
        cl_all = pipe._to_torch(gen["cond_latents_tt"])
        stream_pcc, code_match, cl_pccs, g_pccs, ties = [], [], [], [], []
        for b in range(B):
            gd = gold[b]
            codes_tt = gen["codes_gen"][b].reshape(-1)
            codes_hf = gd["codes"].reshape(-1)
            m = min(int(codes_tt.shape[0]), int(codes_hf.shape[0]))
            match = int((codes_tt[:m] == codes_hf[:m]).sum())
            lat_tt = pipe._to_torch(gen["latents_tt"])[b:b + 1].reshape(gd["latents"].shape)
            code_match.append((match, m, comp_pcc(gd["latents"], lat_tt)[1]))
            cl_pccs.append(comp_pcc(gd["cond_latents"],
                                    cl_all[b:b + 1].reshape(gd["cond_latents"].shape))[1])
            g_pccs.append(comp_pcc(gd["g"].reshape(1, -1), gen["g_host"][b].reshape(1, -1))[1])
            wav_tt = gen["waveforms"][b].reshape(-1)
            wav_hf = gd["waveform"].reshape(-1)
            n = min(int(wav_tt.shape[0]), int(wav_hf.shape[0]))
            stream_pcc.append(comp_pcc(wav_hf[:n].reshape(1, -1), wav_tt[:n].reshape(1, -1))[1])
            print(f"[e2e] stream {b}: codes TT={codes_tt.tolist()}")
            print(f"[e2e] stream {b}:       HF={codes_hf.tolist()}")
            print(f"[e2e] stream {b}: code match={match}/{m}  cond_latents={cl_pccs[b]:.4f} "
                  f"g={g_pccs[b]:.4f} latents={code_match[b][2]:.5f} "
                  f"waveform PCC={stream_pcc[b]:.4f}")
            ties.append(_certify_steps(pipe, inputs, gen, b, codes_tt, codes_hf, m))

        # A pipeline that merely SHAPE-supports B would emit B identical waveforms.
        sigs = {round(float(gen["waveforms"][b].reshape(-1).abs().sum()), 4) for b in range(B)}
        print(f"[e2e] distinct stream outputs: {len(sigs)}/{B}")

        # ---- Gate 2 ----
        n_grad, n_cov = _gate2_coverage(gen["invoked"])
        print(f"[e2e] Gate 2 OK: 6 canonical stubs invoked; {n_cov}/{n_grad} graduated modules covered")

        # ---- Gate 3: the WORST stream is the reported e2e PCC ----
        achieved_pcc = min(stream_pcc)
        print(f"[e2e] per-stream waveform PCC={[round(float(p), 4) for p in stream_pcc]}")
        print(f"e2e PCC={achieved_pcc}")
        assert achieved_pcc >= PCC_GATE, (
            f"Gate 3 FAILED: worst-stream e2e waveform PCC {achieved_pcc:.4f} < {PCC_GATE} "
            f"(per stream: {[round(float(p), 4) for p in stream_pcc]})")
        assert min(cl_pccs) >= PCC_GATE and min(g_pccs) >= PCC_GATE, (
            f"a per-stage PCC degraded: cond={[round(float(p), 4) for p in cl_pccs]} "
            f"g={[round(float(p), 4) for p in g_pccs]}")
        assert len(sigs) == B, (
            f"only {len(sigs)}/{B} distinct stream outputs — the batch axis is not carrying "
            f"independent streams")
        exact = sum(1 for _m, _n, _l in code_match if _m == _n)
        prefixes = [p for p, _v, _pcc in ties]
        print(f"[e2e] greedy parity: {exact}/{B} streams reproduce ALL {horizon} reference "
              f"codes exactly; per-stream exact prefix = {prefixes} of {horizon}, "
              f"every divergence a certified near-tie: "
              f"{all(v is not False for _p, v, _pcc in ties)}")
        assert min(code_match[b][2] for b in range(B)) >= PCC_GATE, (
            f"mel-latent PCC degraded: {[round(float(code_match[b][2]), 5) for b in range(B)]}")
        for b, (s, tie, pcc) in enumerate(ties):
            assert tie is not False, (
                f"stream {b} step {s} is a REAL divergence from its own generate() golden "
                f"(step-logits PCC {pcc:.5f}), not a numerical tie")
        print("[e2e] ALL GATES PASSED")
    finally:
        _close(dev, is_mesh)


if __name__ == "__main__":
    test_e2e_tts()

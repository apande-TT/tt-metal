# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end TTS pipeline test for coqui/XTTS-v2 on the 8-chip mesh (TP=8 x DP=1).

Real input (DVAE cond-mel + text tokens + 16 kHz speaker ref) -> the SHARED
chained TTNN pipeline (tt/pipeline.py) -> real waveform, compared to the Coqui
reference. Asserts Gate 1 (routed stubs still native/sharded), Gate 2 (every
graduated module invoked in the real forward), Gate 3 (final-waveform PCC>=0.95).

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


def _cap(name):
    base = os.path.join(XDIR, "_captured", name)
    a = torch.load(os.path.join(base, "args.pt"), map_location="cpu", weights_only=False)
    return list(a) if isinstance(a, (list, tuple)) else [a]


# The speaker encoder's convolutions are native ttnn.conv2d, whose sliding-window/halo
# path allocates from the L1_SMALL region -- that region is 0 B unless it is reserved at
# device open, so the pipeline requires it exactly as the per-stub PCC tests do.
_DEV_PARAMS = {"l1_small_size": 4096}


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


def test_e2e_tts():
    dev, is_mesh = _open_mesh()
    try:
        model = P.load_reference_model()
        pipe = P.build_pipeline(dev, model=model)

        cond_mel = _cap("conditioning_encoder")[0].float()
        text_inputs = _cap("g_p_t")[0]
        audio_codes = _cap("g_p_t")[2]
        ref_wav = P.load_reference_audio_16k()

        # ---- Gate 1 ----
        _gate1_stubs_native()
        print("[e2e] Gate 1 OK: routed stubs are native ttnn / sharded snapshots")

        # ---- reference goldens ----
        gold = pipe.hf_reference(cond_mel, ref_wav, text_inputs, audio_codes)
        horizon = int(audio_codes.shape[-1])  # decode the same #codes as the reference case
        codes_hf = pipe.hf_greedy_codes(cond_mel, text_inputs, horizon).reshape(-1)

        # ---- deterministic synthesis (captured codes): robust primary chain ----
        det = pipe.run_tts(cond_mel, ref_wav, text_inputs, audio_codes=audio_codes, generate=False)
        cl_pcc = comp_pcc(gold["cond_latents"], pipe._to_torch(det["cond_latents_tt"], gold["cond_latents"].shape))[1]
        g_pcc = comp_pcc(gold["g"].reshape(1, -1), det["g_host"].reshape(1, -1))[1]
        lat_pcc = comp_pcc(gold["latents"], pipe._to_torch(det["latents_tt"], gold["latents"].shape))[1]
        det_pcc = comp_pcc(gold["waveform"], det["waveform"].reshape(gold["waveform"].shape))[1]
        print(f"[e2e] per-stage PCC: cond_latents={cl_pcc:.4f} g={g_pcc:.4f} latents={lat_pcc:.4f}")

        # ---- generative synthesis (decode output feeds the final waveform) ----
        gen = pipe.run_tts(cond_mel, ref_wav, text_inputs, audio_codes=None,
                           horizon=horizon, generate=True)
        codes_tt = gen["codes_gen"].reshape(-1)
        m = min(len(codes_tt), len(codes_hf))
        tok_match = int((codes_tt[:m] == codes_hf[:m]).sum())
        print(f"[e2e] greedy audio codes  TT={codes_tt.tolist()}  HF={codes_hf.tolist()}  match={tok_match}/{m}")
        # golden waveform for the generative branch (HF greedy codes)
        gold_gen = pipe.hf_reference(cond_mel, ref_wav, text_inputs, codes_hf[:len(codes_tt)].reshape(1, -1))
        gen_pcc = comp_pcc(gold_gen["waveform"],
                           gen["waveform"].reshape(gold_gen["waveform"].shape))[1] \
            if gen["waveform"].numel() == gold_gen["waveform"].numel() else float("nan")

        # ---- Gate 2 ----
        n_grad, n_cov = _gate2_coverage(gen["invoked"] | det["invoked"])
        print(f"[e2e] Gate 2 OK: 6 canonical stubs invoked; {n_cov}/{n_grad} graduated modules covered")

        # ---- Gate 3 ----
        achieved_pcc = det_pcc
        print(f"[e2e] deterministic waveform PCC={det_pcc:.4f}  generative waveform PCC={gen_pcc}")
        print(f"e2e PCC={achieved_pcc}")
        assert achieved_pcc >= PCC_GATE, (
            f"Gate 3 FAILED: e2e waveform PCC {achieved_pcc:.4f} < {PCC_GATE}")
        assert cl_pcc >= PCC_GATE and g_pcc >= PCC_GATE and lat_pcc >= PCC_GATE, (
            f"a per-stage PCC degraded: cond={cl_pcc:.4f} g={g_pcc:.4f} lat={lat_pcc:.4f}")
        print("[e2e] ALL GATES PASSED")
    finally:
        _close(dev, is_mesh)


if __name__ == "__main__":
    test_e2e_tts()

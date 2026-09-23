# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Section gate for `tt/acoustic_stage.py` -- the flow-matching acoustic sampler, at B=32.

WHAT IS COMPARED. `llm_hidden` is taken from the REAL text backbone (`hf.model(input_ids)` on 32
distinct prompts from `common.build_batch_inputs`), never invented, so a wrong hidden state cannot
be hidden by a synthetic input. Against that:

  * PCC(TT velocity, reference velocity) per sample, at every one of the 7 Euler steps, fed the
    reference's OWN `x_t` for that step so the field is measured without error accumulation;
  * PCC(TT semantic logits, reference) per sample (the UNMASKED logits -- the reference masks two
    slots to -inf, and -inf is not comparable);
  * EXACT integer equality of all 37 codes per sample against the reference's own
    `decode_one_frame` run on the SAME `x_0`.

WHY `x_0` IS AN ARGUMENT. `decode_one_frame` draws `x_0 = torch.randn(...)` INSIDE the module, so
its output is a function of the RNG and no port can reproduce it. The noise is drawn ONCE on the
host and handed to both sides, which is what a flow-matching sampler takes as input anyway.

THE REPLICATION IS PROVEN, NOT ASSERTED. `_reference_frame` is the reference's Euler loop with the
`randn` lifted out. `test_reference_replication_is_faithful` seeds the RNG, calls the UNMODIFIED
`hf.acoustic_transformer(llm_hidden, cfg_alpha)`, re-seeds identically, draws the same first
tensor the module would have drawn, feeds it to `_reference_frame`, and asserts the two frames are
integer-identical. Without that proof the code-equality gate would only be testing a paraphrase.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import common
from models.demos.voxtral_4b_tts_2603.tt.acoustic_stage import build_acoustic_stage

pytestmark = pytest.mark.timeout(1800)


BATCH = 32
VELOCITY_PCC = 0.99
SEMANTIC_PCC = 0.99
SEED = 0

L1_SMALL_SIZE = 24576
TRACE_REGION_SIZE = 200 * 1024 * 1024

_EXPECTED_STUBS = {
    "flow_matching_audio_transformer",
    "acoustic_transformer_block",
    "bidirectional_attention",
    "feed_forward",
    "time_embedding",
}


@pytest.fixture(scope="module")
def device():
    """The SOLE opener of the device for this file. `tt/` never opens one."""
    dev = ttnn.open_device(
        device_id=0,
        l1_small_size=L1_SMALL_SIZE,
        trace_region_size=TRACE_REGION_SIZE,
        num_command_queues=1,
    )
    try:
        yield dev
    finally:
        ttnn.close_device(dev)


@pytest.fixture(scope="module")
def hf_model():
    common.use_all_cpu_threads()
    torch.manual_seed(SEED)
    return common.load_reference_model()


# --------------------------------------------------------------------------------------
# the reference, with x_0 lifted out of the RNG
# --------------------------------------------------------------------------------------


def _reference_frame(hf_model, llm_hidden, x0, cfg_alpha):
    """`FlowMatchingAudioTransformer.forward` + `decode_one_frame`, with `x_0` supplied.

    Line for line the reference, over the reference's OWN submodules. Returns
    ``(frame [B, 37] int64, raw semantic logits [B, 8320], per-step diagnostics)``.
    """
    at = hf_model.acoustic_transformer
    batch = int(llm_hidden.shape[0])
    n_special = common.n_audio_special_tokens(hf_model)

    raw_logits = at.semantic_codebook_output(llm_hidden).float()
    semantic_logit = raw_logits.clone()
    semantic_logit[:, at._empty_audio_token_id] = -float("inf")
    semantic_logit[:, (n_special + at.model_args.semantic_codebook_size) :] = -float("inf")
    semantic_code = semantic_logit.argmax(dim=-1, keepdim=True)
    should_decode = semantic_code.squeeze(1) != at._end_audio_token_id

    timesteps = at._timesteps.to(dtype=llm_hidden.dtype, device=llm_hidden.device)
    t_emb_table = at.time_embedding(timesteps.view(-1, 1)).to(llm_hidden.dtype)
    t_proj_table = at.time_projection(t_emb_table)
    dts = timesteps[1:] - timesteps[:-1]

    llm_batched = torch.cat([llm_hidden, torch.zeros_like(llm_hidden)], dim=0)
    llm_proj_batched = at.llm_projection(llm_batched)
    alpha = cfg_alpha.to(dtype=llm_hidden.dtype, device=llm_hidden.device).unsqueeze(1)

    sampled = at._noise_scale * x0.to(dtype=llm_hidden.dtype, device=llm_hidden.device)
    steps = []
    for i in range(len(timesteps) - 1):
        t_proj = t_proj_table[i].unsqueeze(0).expand(batch, -1)
        v_all = at._predict_velocity(
            x_t=torch.cat([sampled, sampled], dim=0),
            llm_proj=llm_proj_batched,
            t_proj=torch.cat([t_proj, t_proj], dim=0),
        )
        v_t, uncond = v_all[:batch], v_all[batch:]
        steps.append({"t": float(timesteps[i]), "x_t": sampled.clone(), "v_cond": v_t.clone().float()})
        v_t = alpha * v_t + (1 - alpha) * uncond
        sampled = sampled + v_t * dts[i]

    sampled = torch.clamp(sampled, -1, 1)
    scaled = ((sampled + 1) / 2) * (at.acoustic_embeddings_levels - 1)
    out_codes = scaled.round().long()
    out_codes[~should_decode] = at._empty_audio_token_id
    frame = torch.concatenate([semantic_code, out_codes + n_special], dim=1)
    return frame, raw_logits, steps


# --------------------------------------------------------------------------------------
# inputs -- the REAL ones
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stage_inputs(hf_model):
    """`llm_hidden` from the real 26-layer text backbone, plus the host-drawn `x_0` and alphas.

    `x_0` is the exact first tensor `decode_one_frame` would have drawn after `manual_seed(SEED)`,
    which is what makes the faithfulness proof and this gate share one noise sample.
    `cfg_alpha` VARIES per sample: a constant 1.0 would make the uncond half of the CFG batch
    cancel out and hide a guidance bug.
    """
    at = hf_model.acoustic_transformer
    input_ids, texts = common.build_batch_inputs(BATCH)
    key = common.golden_key(what="acoustic_stage_llm_hidden", input_ids=input_ids, batch=BATCH, seed=SEED)

    def compute():
        with torch.no_grad():
            return hf_model.model(input_ids=input_ids).last_hidden_state[:, -1].float().contiguous()

    llm_hidden = common.cached_golden(key, compute)

    torch.manual_seed(SEED)
    x0 = torch.randn(BATCH, int(at.model_args.n_acoustic_codebook), dtype=llm_hidden.dtype)
    cfg_alpha = 1.0 + 0.5 * torch.arange(BATCH, dtype=llm_hidden.dtype) / BATCH
    return {
        "input_ids": input_ids,
        "texts": texts,
        "llm_hidden": llm_hidden,
        "x0": x0,
        "cfg_alpha": cfg_alpha,
    }


@pytest.fixture(scope="module")
def reference(hf_model, stage_inputs):
    with torch.no_grad():
        frame, raw_logits, steps = _reference_frame(
            hf_model, stage_inputs["llm_hidden"], stage_inputs["x0"], stage_inputs["cfg_alpha"]
        )
    return {"frame": frame, "raw_logits": raw_logits, "steps": steps}


# --------------------------------------------------------------------------------------
# the proof that the replication IS the reference
# --------------------------------------------------------------------------------------


def test_reference_replication_is_faithful(hf_model, stage_inputs):
    """codes_A (the unmodified module, drawing its own noise) == codes_B (the replication)."""
    at = hf_model.acoustic_transformer
    llm_hidden = stage_inputs["llm_hidden"]
    cfg_alpha = stage_inputs["cfg_alpha"]

    torch.manual_seed(SEED)
    with torch.no_grad():
        codes_a = at(llm_hidden, cfg_alpha)

    torch.manual_seed(SEED)
    x0 = torch.randn(BATCH, int(at.model_args.n_acoustic_codebook), dtype=llm_hidden.dtype)
    assert torch.equal(x0, stage_inputs["x0"]), "the gate's x_0 is not the module's own first draw"
    with torch.no_grad():
        codes_b, _, _ = _reference_frame(hf_model, llm_hidden, x0, cfg_alpha)

    equal = int((codes_a == codes_b).sum())
    total = int(codes_a.numel())
    print(f"faithfulness proof: codes_A == codes_B on {equal}/{total} codes")
    assert torch.equal(codes_a, codes_b), "the replicated Euler loop is not the reference's"


# --------------------------------------------------------------------------------------
# the section gate
# --------------------------------------------------------------------------------------


def _to_tt(t, device, dtype=ttnn.float32):
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _per_sample_pcc(tt, ref):
    return [common.pcc(tt[i], ref[i]) for i in range(tt.shape[0])]


def test_acoustic_stage(device, hf_model, stage_inputs, reference):
    llm_hidden = stage_inputs["llm_hidden"]
    x0 = stage_inputs["x0"]
    cfg_alpha = stage_inputs["cfg_alpha"]

    counter = common.InvocationCounter()
    stage = build_acoustic_stage(device, hf_model, counter=counter, batch=BATCH)
    split = stage.split_point(BATCH)
    print(f"\n{stage}")
    print(
        f"batch driven = {BATCH} (read from the inputs); rows [0:{split}] -> whole-section body "
        f"`flow_matching_audio_transformer`, rows [{split}:{BATCH}] -> composed part-chain"
    )
    print(f"block kinds: {[(b.layer_id, b.kind, list(b.stubs)) for b in stage.blocks]}")

    llm_tt = _to_tt(llm_hidden, device)

    # ---- per-step velocity, fed the reference's own x_t so the field is measured alone
    step_pccs = []
    semantic_pcc = None
    for i, step in enumerate(reference["steps"]):
        x_tt = _to_tt(step["x_t"], device)
        t_tt = _to_tt(torch.full((BATCH, 1), step["t"], dtype=torch.float32), device)
        v_tt, sem_tt = stage.velocity(llm_tt, x_tt, t_tt)
        v = ttnn.to_torch(v_tt).float()
        assert tuple(v.shape) == (BATCH, stage.n_acoustic), f"velocity shape {tuple(v.shape)}"
        pccs = _per_sample_pcc(v, step["v_cond"])
        step_pccs.append(pccs)
        print(
            f"velocity step {i} (t={step['t']:.4f}): PCC min={min(pccs):.6f} "
            f"mean={sum(pccs) / len(pccs):.6f} worst_sample={pccs.index(min(pccs))}"
        )
        if semantic_pcc is None:
            sem = ttnn.to_torch(sem_tt).float()
            semantic_pcc = _per_sample_pcc(sem, reference["raw_logits"])

    print(
        f"semantic logits: PCC min={min(semantic_pcc):.6f} "
        f"mean={sum(semantic_pcc) / len(semantic_pcc):.6f} "
        f"worst_sample={semantic_pcc.index(min(semantic_pcc))}"
    )

    # ---- the frame: exact integer codes, all on device
    frame_tt = stage.decode_frame(llm_tt, _to_tt(x0, device), _to_tt(cfg_alpha.reshape(BATCH, 1), device))
    frame = ttnn.to_torch(frame_tt).to(torch.int64)
    ref_frame = reference["frame"]
    assert tuple(frame.shape) == tuple(ref_frame.shape), f"{tuple(frame.shape)} vs {tuple(ref_frame.shape)}"

    match = frame == ref_frame
    code_rate = float(match.float().mean())
    semantic_rate = float(match[:, 0].float().mean())
    acoustic_rate = float(match[:, 1:].float().mean())
    rows_exact = int(match.all(dim=1).sum())
    print(f"code match rate      = {code_rate:.6f} ({int(match.sum())}/{match.numel()} codes)")
    print(f"  semantic code rate = {semantic_rate:.6f} ({int(match[:, 0].sum())}/{BATCH})")
    print(f"  acoustic code rate = {acoustic_rate:.6f}")
    print(f"  samples exact on all 37 codes = {rows_exact}/{BATCH}")
    if rows_exact < BATCH:
        bad = (~match.all(dim=1)).nonzero().flatten().tolist()
        for i in bad[:4]:
            diff = (frame[i] != ref_frame[i]).nonzero().flatten().tolist()
            print(f"  sample {i}: differing code slots {diff}")
            print(f"    tt  = {frame[i].tolist()}")
            print(f"    ref = {ref_frame[i].tolist()}")

    # ---- batch honesty
    rows = {tuple(r) for r in frame.tolist()}
    print(f"distinct frames = {len(rows)}/{BATCH}")

    # ---- Gate 2: every stub of this section really ran
    print(f"stub invocations = {counter.counts}")

    velocity_min = min(min(p) for p in step_pccs)
    print(f"acoustic stage PCC velocity_min={velocity_min} semantic_min={min(semantic_pcc)}")

    assert _EXPECTED_STUBS.issubset(
        set(counter.counts)
    ), f"missing stub invocations: {sorted(_EXPECTED_STUBS - set(counter.counts))}"
    assert all(counter.counts[name] >= 1 for name in _EXPECTED_STUBS)
    for i, pccs in enumerate(step_pccs):
        assert min(pccs) >= VELOCITY_PCC, f"velocity step {i} PCC {min(pccs)} < {VELOCITY_PCC}"
    assert min(semantic_pcc) >= SEMANTIC_PCC, f"semantic PCC {min(semantic_pcc)} < {SEMANTIC_PCC}"
    assert len(rows) == BATCH, f"only {len(rows)} of {BATCH} frames are distinct"
    assert code_rate == 1.0, f"code match rate {code_rate} < 1.0"

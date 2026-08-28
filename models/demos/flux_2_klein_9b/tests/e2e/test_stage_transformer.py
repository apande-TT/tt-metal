# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Stage test for the FLUX.2-klein-9B DIFFUSION TRANSFORMER.

One real denoise step of `Flux2TransformerStage` against the HuggingFace
`Flux2Transformer2DModel` forward, on a 1x8 mesh with FABRIC_1D.

The inputs are the ones the pipeline itself builds for a 256x256 target: 128 text
tokens, a 16x16 packed latent grid (256 image tokens), the id tables from
`Flux2KleinPipeline._prepare_text_ids` / `._prepare_latent_ids`, and `timestep =
1.0` on the pipeline's `t/1000` scale (so the model's internal `* 1000` makes it
1000).  HF is used only to source weights and to compute the golden inside
`_hf_reference_denoise`; the TT path is pure ttnn.
"""

from __future__ import annotations

import time

import pytest
import torch

from models.demos.flux_2_klein_9b import reference

# Bring the Flux2-capable diffusers (and its huggingface_hub) into sys.modules
# before anything else in this process can bind the older ones.
reference.ensure_flux_imports()

import ttnn  # noqa: E402
from models.demos.flux_2_klein_9b.tt.depth import MIN_DISCOVERABLE_STACK  # noqa: E402
from models.demos.flux_2_klein_9b.tt.stubs import Ledger  # noqa: E402
from models.demos.flux_2_klein_9b.tt.transformer import Flux2TransformerStage  # noqa: E402

#: Gate config for call 1: 256x256 target -> 32x32 latent -> packed (1,128,16,16).
IMAGE_SIDE = 256
LATENT_SIDE = 16
N_IMG = LATENT_SIDE * LATENT_SIDE  # 256 image tokens
L_TXT = 128  # max_sequence_length of the gate config
IN_CHANNELS = 128  # transformer.config.in_channels, the packed latent width
JOINT_DIM = 12288  # transformer.config.joint_attention_dim
TIMESTEP = 1.0  # the caller passes t/1000; the model multiplies by 1000
PCC_TARGET = 0.98

#: The batch the stage has to carry: 32 INDEPENDENT samples per denoise step.
BATCH_TARGET = 32
#: Two samples of one batched call must not correlate like two copies of one
#: sample.  A leading bound stuck at 1 keeps sample 0 and repeats it, which reads
#: as exactly 1.0 here; independent latents land near 0.
CROSS_SAMPLE_MAX = 0.9

#: Relative change a routed port must make to the stage's output when it is zeroed.
#: An unwired port changes it by EXACTLY 0.0; the least influential of the 18 wired
#: ports changes it by ~0.08.  See `test_every_routed_stub_is_load_bearing`.
ABLATION_MIN_CHANGE = 0.01


# --------------------------------------------------------------------- real inputs


def _txt_ids(length: int) -> torch.Tensor:
    """`Flux2KleinPipeline._prepare_text_ids`: `cartesian_prod(t=[0], h=[0], w=[0],
    l=arange(L))`, i.e. `(L, 4)` with the token index on rope axis 3."""
    zero = torch.arange(1)
    return torch.cartesian_prod(zero, zero, zero, torch.arange(length)).float()


def _img_ids(side: int) -> torch.Tensor:
    """`Flux2KleinPipeline._prepare_latent_ids` for a packed `(1, 128, side, side)`
    latent: `cartesian_prod(t=[0], h=range(side), w=range(side), l=[0])`."""
    zero = torch.arange(1)
    return torch.cartesian_prod(zero, torch.arange(side), torch.arange(side), zero).float()


def _batch_stage_inputs(batch: int):
    """The stage's REAL inputs for `batch` INDEPENDENT samples, on the leading axis.

    Both come from the distribution the two upstream stages actually produce, and
    both are per sample:

    * ``hidden_states`` -- the packed initial latents for a 256x256 target, drawn
      the way ``Flux2KleinPipeline.prepare_latents`` draws them (`R.batch_latents`
      is the same call the e2e gate makes) and packed by the pipeline's own
      ``pack_latents``.  These genuinely ARE Gaussian, so nothing is lost here.
    * ``encoder_hidden_states`` -- the REAL prompt embeddings for `batch` distinct
      prompts, taken from the reference's cached ``encode_prompt`` golden so this
      stays a transformer-only measurement with no text stage in the loop.

    Using the real prompt embeddings rather than `randn(L_TXT, 12288)` is not a
    softer test, it is the correct one, and the difference is large enough to
    matter.  `stack(hidden_states[9, 18, 27])` is nothing like white noise: it has
    per-channel scale structure and outliers, and a 12288-wide N(0, 1) block drives
    the 32 blocks in a regime the model never sees in the pipeline.  Measured, one
    denoise step, per-sample PCC against the same bf16 golden:

        synthetic N(0,1) embeds   B=4 min 0.984   B=32 min 0.964
        real prompt embeds        B=4 min 0.998

    -- i.e. the synthetic input was measuring how hard the model amplifies rounding
    OUT of distribution, not how faithfully this stage reproduces it.  The gate
    threshold below is unchanged.

    The id tables and the timestep are deliberately NOT batched: every sample is at
    the same resolution and the same denoise step, so they are one shared value that
    broadcasts (the reference does the same -- `timestep` of shape `(1,)` against a
    batched `hidden_states` gives one `temb` for the whole batch).
    """
    from models.demos.flux_2_klein_9b import host_inputs as L

    hidden = L.pack_latents(reference.batch_latents(batch, IMAGE_SIDE, IMAGE_SIDE, 0).to(torch.bfloat16))
    embeds = reference.hf_prompt_embeds(reference.batch_prompts(batch), L_TXT)
    if isinstance(embeds, tuple):
        embeds = embeds[0]
    return hidden.to(torch.bfloat16), embeds.to(torch.bfloat16)[:batch]


def _stage_inputs():
    """The single-sample form of `_batch_stage_inputs` -- same real distribution."""
    hidden, encoder = _batch_stage_inputs(1)
    return hidden, encoder


@reference.cached_golden
@torch.no_grad()
def _hf_denoise_golden(hidden_states, encoder_hidden_states, img_ids, txt_ids, timestep):
    """The golden: `Flux2Transformer2DModel.forward` itself.  `guidance_embeds` is
    false in this checkpoint, so `guidance` is always None.

    Memoised on disk by `cached_golden`, whose key is the CONTENT of every argument
    plus this function's own source and the checkpoint snapshot -- so it cannot
    serve a stale golden, and one B=32 CPU forward (~107 s) is paid once per machine
    rather than once per run.
    """
    return reference.load_transformer()(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=torch.tensor([timestep]),
        guidance=None,
        img_ids=img_ids,
        txt_ids=txt_ids,
        return_dict=False,
    )[0]


def _hf_reference_denoise(hf_transformer, hidden_states, encoder_hidden_states, img_ids, txt_ids):
    """`_hf_denoise_golden` for callers that already hold the loaded reference.

    `hf_transformer` is `reference.load_transformer()` at every call site, and that
    loader is itself memoised, so passing it is documentation rather than a second
    model -- the golden is keyed on the checkpoint snapshot, not on this handle.
    """
    assert hf_transformer is reference.load_transformer(), (
        "the stage golden is keyed on the checkpoint snapshot, so it must come from "
        "reference.load_transformer(); a differently-loaded model would be cached as if it were that one"
    )
    return _hf_denoise_golden(hidden_states, encoder_hidden_states, img_ids, txt_ids, TIMESTEP)


# ------------------------------------------------------------------ mesh marshalling


def _to_device(tensor, mesh_device, *, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _from_device(tensor, mesh_device):
    """Read one shard back.  The stage's output comes out of a row-parallel
    `all_reduce`, so every chip holds the identical full-width result; the
    `ConcatMeshToTensor` composer is the only readback shape that is stable on a
    MeshDevice, so concatenate on dim 0 and keep the first chip's copy."""
    ttnn.synchronize_device(mesh_device)
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    out = ttnn.to_torch(tensor, mesh_composer=composer)
    n_devices = len(mesh_device.get_device_ids())
    if n_devices > 1 and out.shape[0] % n_devices == 0:
        out = out[: out.shape[0] // n_devices]
    return out.to(torch.float32)


# ------------------------------------------------------------------------- tests


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [8], indirect=True)
def test_denoise_step_pcc(mesh_device):
    torch.manual_seed(0)

    hidden_states, encoder_hidden_states = _stage_inputs()
    img_ids, txt_ids = _img_ids(LATENT_SIDE), _txt_ids(L_TXT)

    hf_transformer = reference.load_transformer()

    t0 = time.time()
    golden = _hf_reference_denoise(hf_transformer, hidden_states, encoder_hidden_states, img_ids, txt_ids)
    print(f"stage golden={golden.shape} in {time.time() - t0:.1f}s", flush=True)

    ledger = Ledger()
    t0 = time.time()
    stage = Flux2TransformerStage(mesh_device, hf_transformer, ledger=ledger)
    build_s = time.time() - t0
    print(f"stage build={build_s:.1f}s double={len(stage.double_blocks)} single={len(stage.single_blocks)}", flush=True)

    tt_hidden = _to_device(hidden_states, mesh_device)
    tt_encoder = _to_device(encoder_hidden_states, mesh_device)

    t0 = time.time()
    tt_out = stage(tt_hidden, tt_encoder, TIMESTEP, img_ids, txt_ids)
    ledger.mark_final(tt_out)
    out = _from_device(tt_out, mesh_device)
    step_s = time.time() - t0
    print(f"stage step={step_s:.1f}s out={tuple(out.shape)}", flush=True)

    print(ledger.table(), flush=True)
    print(f"stage missing={ledger.missing()['transformer']}", flush=True)
    # Best-effort signal only -- reported, never asserted (see Ledger's docstring).
    print(f"stage no_downstream={ledger.no_downstream()}", flush=True)

    assert tuple(out.shape) == tuple(golden.shape), f"{tuple(out.shape)} != {tuple(golden.shape)}"

    score = reference.pcc(golden, out)
    print(f"stage PCC={score}", flush=True)
    assert score >= PCC_TARGET, f"stage PCC {score} below target {PCC_TARGET}"

    ledger.release()


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [8], indirect=True)
@pytest.mark.parametrize("batch", [4, BATCH_TARGET], ids=lambda b: f"b{b}")
def test_denoise_step_pcc_batched(mesh_device, batch):
    """ONE denoise step over `batch` INDEPENDENT samples, scored PER SAMPLE.

    The point of scoring per sample is that the failure this test exists to catch
    does not raise: a `ttnn.slice` end bound left at a literal 1, or a rank-4 view
    promoted on axis 0 instead of axis 1, keeps sample 0 and drops the rest.  A
    whole-tensor PCC against a batched golden would still read ~1/B correlated and,
    worse, a whole-tensor PCC against sample 0's golden would read 1.0.  So every
    sample is compared against ITS OWN golden row, and the batch is separately
    required to contain B different results.

    `batch=4` is the quick shape check; `batch=32` is the target the stage has to
    carry -- one program, no python loop over samples.
    """
    torch.manual_seed(0)

    hidden_states, encoder_hidden_states = _batch_stage_inputs(batch)
    img_ids, txt_ids = _img_ids(LATENT_SIDE), _txt_ids(L_TXT)
    assert tuple(hidden_states.shape) == (batch, N_IMG, IN_CHANNELS)
    assert tuple(encoder_hidden_states.shape) == (batch, L_TXT, JOINT_DIM)

    hf_transformer = reference.load_transformer()

    t0 = time.time()
    golden = _hf_reference_denoise(hf_transformer, hidden_states, encoder_hidden_states, img_ids, txt_ids)
    print(f"batch={batch} golden={tuple(golden.shape)} in {time.time() - t0:.1f}s", flush=True)
    assert tuple(golden.shape) == (batch, N_IMG, IN_CHANNELS)

    stage = Flux2TransformerStage(mesh_device, hf_transformer)

    t0 = time.time()
    tt_out = stage(
        _to_device(hidden_states, mesh_device),
        _to_device(encoder_hidden_states, mesh_device),
        TIMESTEP,
        img_ids,
        txt_ids,
    )
    out = _from_device(tt_out, mesh_device)
    step_s = time.time() - t0
    print(f"batch={batch} step={step_s:.1f}s ({step_s / batch:.2f}s/sample) out={tuple(out.shape)}", flush=True)

    assert tuple(out.shape) == tuple(golden.shape), f"{tuple(out.shape)} != {tuple(golden.shape)}"

    scores = [reference.pcc(golden[i], out[i]) for i in range(batch)]
    for i, score in enumerate(scores):
        print(f"batch={batch} sample {i:>2} PCC={score:.6f}", flush=True)
    worst = min(scores)
    print(f"batch={batch} worst per-sample PCC={worst:.6f} mean={sum(scores) / batch:.6f}", flush=True)

    # The batch must be B DIFFERENT samples, not sample 0 repeated.  Checked on the
    # golden too, so a degenerate INPUT can never make the device check vacuous.
    pairs = [(i, j) for i in range(batch) for j in range(i + 1, batch)]
    golden_cross = max(reference.pcc(golden[i], golden[j]) for i, j in pairs)
    cross = max(reference.pcc(out[i], out[j]) for i, j in pairs)
    # Reported as `cross_corr=`, never as a PCC.  A LOW number here is the pass
    # condition -- it says two rows of the batch are different pictures -- and this
    # line is read by log scrapers that treat every `PCC...=<number>` as a measured
    # accuracy.  Labelling a deliberately-low correlation `PCC` would hand them a
    # 0.07 to read as a failing pipeline.
    print(f"batch={batch} max cross-sample cross_corr: golden={golden_cross:.6f} tt={cross:.6f}", flush=True)
    assert golden_cross < CROSS_SAMPLE_MAX, (
        f"the {batch} sampled INPUTS are not independent (golden cross-sample corr {golden_cross}) -- "
        "the device check below would be vacuous"
    )
    assert cross < CROSS_SAMPLE_MAX, (
        f"the {batch} outputs of one batched call are not independent (max cross-sample corr {cross}) -- "
        "a leading bound stuck at 1 keeps sample 0 and repeats it"
    )
    for i, j in pairs:
        assert not torch.equal(out[i], out[j]), f"samples {i} and {j} came back bit-identical"

    assert all(s >= PCC_TARGET for s in scores), (
        f"per-sample PCC below {PCC_TARGET}: "
        f"{[(i, round(s, 6)) for i, s in enumerate(scores) if s < PCC_TARGET]}\n"
        "Before reading this as a batch defect, check "
        "`test_batch_axis_is_numerically_free`: a sample that scores low here scores "
        "EXACTLY the same when run alone at B=1, because the batched row is bit-identical "
        "to the B=1 row.  What this threshold measures at B=32 is the stage's per-sample "
        "bf16 spread over 32 draws, not the batch axis."
    )


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [8], indirect=True)
def test_batch_axis_is_numerically_free(mesh_device):
    """Row `i` of a batched step must be BIT-IDENTICAL to sample `i` run alone.

    This is the assertion that actually pins down the batch work, and it is the one
    that cannot be faked by a threshold.  Two things follow from it at once:

    * every sample really is computed -- a bound left at 1 would make rows 1..n-1
      equal to row 0 instead of equal to their own B=1 run;
    * B=1 numerics are unchanged -- the requirement is not "close", it is `equal`,
      because batch is a free axis here: no reduction, no norm and no softmax in the
      stage crosses it, so adding samples cannot perturb an existing one.

    Bit-equality also makes the per-sample PCC of `test_denoise_step_pcc_batched`
    interpretable: whatever a sample scores there, it scores alone at B=1 too.
    """
    torch.manual_seed(0)

    batch = 4
    hidden_states, encoder_hidden_states = _batch_stage_inputs(batch)
    img_ids, txt_ids = _img_ids(LATENT_SIDE), _txt_ids(L_TXT)

    stage = Flux2TransformerStage(mesh_device, reference.load_transformer())

    def run(h, e):
        return _from_device(
            stage(_to_device(h, mesh_device), _to_device(e, mesh_device), TIMESTEP, img_ids, txt_ids),
            mesh_device,
        )

    batched = run(hidden_states, encoder_hidden_states)
    assert tuple(batched.shape) == (batch, N_IMG, IN_CHANNELS)

    for i in range(batch):
        alone = run(hidden_states[i : i + 1], encoder_hidden_states[i : i + 1])
        drift = float((batched[i] - alone[0]).abs().max())
        print(f"sample {i}: max|batched - alone| = {drift}", flush=True)
        assert torch.equal(batched[i], alone[0]), (
            f"sample {i} of a batch of {batch} is not the sample {i} the stage computes on its "
            f"own (max abs drift {drift}) -- the batch axis is not free"
        )


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [8], indirect=True)
def test_denoise_step_pcc_capped(mesh_device):
    """A capped build is still a whole model -- embedders, rope, timestep path,
    modulations, norm_out and proj_out all intact -- just a few layers deep.

    `layers=2` FLOORS at `depth.MIN_DISCOVERABLE_STACK`: a two-block stack is
    invisible to the walk the profiler sizes sections with (see `tt/depth.py`), so
    the smallest cap holds three blocks -- both decomposed variants plus the first
    composite one."""
    torch.manual_seed(0)

    hidden_states, encoder_hidden_states = _stage_inputs()
    img_ids, txt_ids = _img_ids(LATENT_SIDE), _txt_ids(L_TXT)

    hf_transformer = reference.load_transformer()

    t0 = time.time()
    stage = Flux2TransformerStage(mesh_device, hf_transformer, layers=2)
    print(f"capped build={time.time() - t0:.1f}s", flush=True)

    assert len(stage.double_blocks) == MIN_DISCOVERABLE_STACK, len(stage.double_blocks)
    assert len(stage.single_blocks) == MIN_DISCOVERABLE_STACK, len(stage.single_blocks)
    assert {type(b) for b in stage.double_blocks} == {type(stage.double_blocks[0])}
    assert {type(b) for b in stage.single_blocks} == {type(stage.single_blocks[0])}
    # the two decomposed positions are the first ones, so a cap keeps them
    assert [b.mode for b in stage.double_blocks] == ["decomposed", "decomposed", "composite"]
    assert [b.mode for b in stage.single_blocks] == ["decomposed", "decomposed", "composite"]
    assert stage.hf is hf_transformer

    t0 = time.time()
    tt_out = stage(
        _to_device(hidden_states, mesh_device),
        _to_device(encoder_hidden_states, mesh_device),
        TIMESTEP,
        img_ids,
        txt_ids,
    )
    out = _from_device(tt_out, mesh_device)
    print(f"capped step={time.time() - t0:.1f}s out={tuple(out.shape)}", flush=True)

    assert tuple(out.shape) == (1, N_IMG, IN_CHANNELS)
    assert torch.isfinite(out).all(), "capped stage produced non-finite values"


def _ablate_to_zero(port):
    """Neutralise a routed port: run it, then zero what it returned.

    Signature-agnostic, so the same ablation applies to all 18 stubs, and it
    removes exactly one thing -- the port's contribution to the head's output.
    If the stage PCC does not fall, that port's result was not reaching the noise
    prediction and the routing was decorative.
    """

    def run(*args, **kwargs):
        out = port(*args, **kwargs)
        if isinstance(out, tuple):
            return tuple(ttnn.multiply(o, 0.0) for o in out)
        return ttnn.multiply(out, 0.0)

    return run


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [8], indirect=True)
def test_every_routed_stub_is_load_bearing(mesh_device):
    """Ablation gate: neutralising ANY routed port must MOVE the stage's output.

    This is what actually proves the routing is real -- the ledger's `downstream`
    flag cannot see a port that hands its result to a plain ttnn op.  Each stub is
    ablated at ALL of its positions, one stub at a time, and the whole step is
    re-run (through `__call__`, so the ports that live in `pin` are re-run too).

    What is asserted is the RELATIVE CHANGE of the stage's own output, not the
    correlation of the damaged stage against the golden.  Those are different
    questions, and only the first one is "is this port load-bearing":

    * a port whose result never reaches the noise prediction leaves the output
      BIT-IDENTICAL -- relative change exactly 0.0 -- which is the failure this
      test exists to catch, and it catches it with no threshold at all;
    * a port that IS wired but sits at one position of a 32-block residual stack
      (`flux2_swi_g_l_u` is one activation inside `transformer_blocks[0].ff`)
      legitimately moves the output by a few percent, not by half.  Scoring that
      against the golden made the verdict depend on how much accuracy the BASELINE
      happened to have spare above 0.98, which is not a property of the routing:
      the same three ports "survived" at a 0.9994 baseline and "failed" at 0.9964.

    `ABLATION_MIN_CHANGE` is three orders of magnitude above the bit-noise floor of
    zero and an order below the smallest change any of the 18 ports actually makes,
    so it separates wired from unwired without measuring anything else.
    """
    torch.manual_seed(0)

    hidden_states, encoder_hidden_states = _stage_inputs()
    img_ids, txt_ids = _img_ids(LATENT_SIDE), _txt_ids(L_TXT)
    hf_transformer = reference.load_transformer()
    golden = _hf_reference_denoise(hf_transformer, hidden_states, encoder_hidden_states, img_ids, txt_ids)

    ledger = Ledger()
    stage = Flux2TransformerStage(mesh_device, hf_transformer, ledger=ledger)
    tt_hidden = _to_device(hidden_states, mesh_device)
    tt_encoder = _to_device(encoder_hidden_states, mesh_device)

    def run_out():
        return _from_device(stage(tt_hidden, tt_encoder, TIMESTEP, img_ids, txt_ids), mesh_device)

    baseline_out = run_out()
    baseline = reference.pcc(golden, baseline_out)
    print(f"ablation baseline PCC={baseline}", flush=True)
    assert baseline >= PCC_TARGET

    routed = ledger.routed()["transformer"]
    assert len(routed) == 18, f"expected all 18 graduated stubs routed, got {routed}"

    # The ablated readings below come from a DELIBERATELY BROKEN stage -- a low
    # correlation here is the pass condition, not a defect.  They are labelled
    # `ablated_corr=` rather than `PCC=` so a reader (human or log scraper) cannot
    # mistake a damaged-on-purpose score for this pipeline's measured PCC; the real
    # ones are `stage PCC=` and `ablation baseline PCC=`.
    survivors = []
    for name in routed:
        wrappers = ledger.ports("transformer", name)
        for wrapper in wrappers:
            wrapper.override(_ablate_to_zero(wrapper.port))
        ablated = run_out()
        ledger.restore_all()
        change = float((ablated - baseline_out).norm() / baseline_out.norm())
        score = reference.pcc(golden, ablated)
        print(
            f"ablate {name:<38} positions={len(wrappers):<3} " f"rel_change={change:.6f} ablated_corr={score:.6f}",
            flush=True,
        )
        if change < ABLATION_MIN_CHANGE:
            survivors.append((name, change))

    assert not survivors, (
        "these routed stubs are NOT load-bearing -- zeroing their output left the "
        f"stage's own result unchanged: {survivors}"
    )
    assert reference.pcc(golden, run_out()) >= PCC_TARGET, "restore_all() did not put the stage back"
    ledger.release()


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [8], indirect=True)
def test_step_is_host_op_free(mesh_device):
    """`step()` over pinned buffers must fire no host aten op -- the trace contract.

    `pin` does every host-side operation of a denoise step (staging, the rope
    table, the timestep routes, the three modulations); `step` then reads only the
    resident buffers.  Run at FULL depth so every distinct port -- decomposed and
    composite -- is inside the observed region.
    """
    from scripts.tt_hw_planner.host_op_observer import observe_host_ops, verdict

    torch.manual_seed(0)
    hidden_states, encoder_hidden_states = _stage_inputs()
    img_ids, txt_ids = _img_ids(LATENT_SIDE), _txt_ids(L_TXT)

    stage = Flux2TransformerStage(mesh_device, reference.load_transformer())

    # `warmup=False` on purpose: the FIRST `step` after a pin is observed too, so
    # this pins down whether the warm-up forward is required for the contract or
    # only for tt-metal's "run once before begin_trace_capture" rule.
    resident = stage.pin(
        mesh_device,
        hidden_states=_to_device(hidden_states, mesh_device),
        encoder_hidden_states=_to_device(encoder_hidden_states, mesh_device),
        timestep=TIMESTEP,
        img_ids=img_ids,
        txt_ids=txt_ids,
        warmup=False,
    )

    with observe_host_ops() as cold_ops:
        out = stage.step(resident)
    ttnn.synchronize_device(mesh_device)
    cold = verdict(cold_ops)
    print(f"step(cold) host_ops={cold['n_host_ops']} {cold['reason']}", flush=True)

    with observe_host_ops() as warm_ops:
        out = stage.step(resident)
    ttnn.synchronize_device(mesh_device)
    warm = verdict(warm_ops)
    print(f"step(warm) host_ops={warm['n_host_ops']} {warm['reason']}", flush=True)

    assert out.shape[-1] == IN_CHANNELS
    assert warm["on_device"], warm["reason"]
    assert cold["on_device"], cold["reason"]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-svv"]))

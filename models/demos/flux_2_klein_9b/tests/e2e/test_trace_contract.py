# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Command 3: the trace contract and the fully-on-device check.

Four things:

1. `build_pipeline` returns the resident OBJECT carrying `PIPELINE_STAGES` and the
   per-stage `<stage>_trace_setup / _trace_step / _trace_inputs / _trace_items`
   hooks -- the single entry the perf harness calls.  It must not run the model.
2. `trace_capture_selftest` captures ONE step per stage inside
   begin/end_trace_capture, executes it, PCC-checks it against the eager step, and
   RELEASES the trace before the next stage (stage traces must not co-reside).
3. `host_op_selftest` runs each head's model math under `observe_host_ops()` with
   input encoding and weight build OUTSIDE the observed region, and requires zero
   host aten ops.
4. Every section the CHECKPOINT declares is a stack the profiler's structure walk
   can actually find on the object `build_pipeline` just returned -- at the SMALLEST
   cap, where the stacks are thinnest.

The `layers` knob is proved here too: capping it must actually shrink every repeated
stack (this checkpoint declares five sections: the Qwen3 trunk, the 8 double DiT
blocks, the 24 single DiT blocks, the VAE's 4 encoder down blocks and its 4 decoder
up blocks), and the capped build must still be a runnable, WALKABLE model.
"""

from __future__ import annotations

import os

import pytest
import torch

from models.demos.flux_2_klein_9b import reference as R
from models.demos.flux_2_klein_9b.mesh import VAE_L1_SMALL, open_flux_mesh
from models.demos.flux_2_klein_9b.tt import pipeline as P
from models.demos.flux_2_klein_9b.tt.depth import MIN_DISCOVERABLE_STACK

#: sized from the largest stage: pinned joint capacity x layers
TRACE_REGION_SIZE = 90 * 1024 * 1024

#: The smallest cap that keeps EVERY stage able to run: the prompt-embed trunk needs
#: 4 repeats to hold all of its distinct block variants (layer, two decomposed
#: layers, decoder_layer), so capping below 4 would make a section structurally
#: absent rather than merely shallower.
CAP = 4


@pytest.fixture(scope="module")
def device():
    R.ensure_flux_imports()
    with open_flux_mesh(l1_small_size=VAE_L1_SMALL, trace_region_size=TRACE_REGION_SIZE) as dev:
        yield dev


def test_build_pipeline_returns_the_resident_object(device):
    pipe = P.build_pipeline(device, layers=CAP, text="ignored", prompt="ignored", language="ignored")
    assert isinstance(pipe, P.Flux2KleinTtPipeline)
    assert P.PIPELINE_STAGES == ["encode_text", "vae_encode", "denoise", "vae_decode", "prefill", "decode"]
    for stage in P.PIPELINE_STAGES:
        for suffix in ("_trace_setup", "_trace_step", "_trace_inputs", "_trace_items"):
            assert callable(getattr(pipe, stage + suffix)), f"{stage}{suffix} missing"
        items = getattr(pipe, stage + "_trace_items")()
        assert isinstance(items, int) and items >= 1, (stage, items)
        print(f"{stage:12s} capacity={pipe.capacity(stage):5d} items={items}")
    # the AR head also keeps the decode contract
    assert callable(pipe.decode_trace_setup) and callable(pipe.decode_trace_step)

    # BATCH=32: one traced step retires 32 samples' worth of work, so `_trace_items`
    # -- the ONLY input to the arithmetic ceiling -- has to include the batch.  A
    # stage that reported a single sample would be handed a compute roof 32x too
    # small and then read as memory-bound.
    assert pipe.batch == P.BATCH == 32, pipe.batch
    for stage in P.PIPELINE_STAGES:
        items = getattr(pipe, stage + "_trace_items")()
        want = pipe.stage_batch(stage)
        assert (
            items % want == 0 and items >= want
        ), f"{stage}_trace_items()={items} does not include this stage's batch of {want}"


def test_trace_inputs_are_zero_arg_and_feed_setup(device):
    """`<stage>_trace_inputs()` must return EXACTLY what `<stage>_trace_setup` takes."""
    pipe = P.build_pipeline(device, layers=CAP)
    import inspect

    for stage in P.PIPELINE_STAGES:
        seam = getattr(pipe, stage + "_trace_inputs")
        assert not [
            p
            for p in inspect.signature(seam).parameters.values()
            if p.default is inspect.Parameter.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        ], f"{stage}_trace_inputs must be zero-arg"
        value = seam()
        assert isinstance(value, dict) and value, f"{stage}_trace_inputs returned {value!r}"
        print(f"{stage:12s} inputs={ {k: (tuple(v.shape) if hasattr(v, 'shape') else v) for k, v in value.items()} }")

        # every per-sample tensor carries the leading batch axis.  `timestep` is the
        # deliberate exception: the 32 samples share the resolution and the step
        # count, so they share the flow-match schedule -- that conditioning stays
        # batch-1 and broadcasts, which is what makes it shared rather than faked.
        for key, tensor in value.items():
            if not hasattr(tensor, "shape"):
                continue
            want = pipe.stage_batch(stage)
            assert int(tensor.shape[0]) == want, (
                f"{stage}_trace_inputs()[{key!r}] has leading dim {tensor.shape[0]}, "
                f"expected this stage's batch {want}"
            )


def test_layers_knob_caps_every_stack(device):
    """`layers` is the default depth for every repeated block; `<stage>_layers`
    overrides it per stack.  None means all layers -- never 0, and never below
    `depth.MIN_DISCOVERABLE_STACK` (see `test_every_declared_section_is_a_stack`)."""
    full = P.build_pipeline(device, layers=None)
    assert full.layers is None
    assert full.stage_layers["denoise"] is None

    capped = P.build_pipeline(device, layers=CAP, denoise_layers=2, encode_text_layers=5)
    assert capped._depth("denoise") == 2
    assert capped._depth("encode_text") == 5
    assert capped._depth("vae_decode") == CAP  # falls back to `layers`
    assert capped._depth("prefill") == CAP

    text = capped.text_stage()
    assert len(text.blocks) == 5, len(text.blocks)
    assert text.n_layers == 5
    capped.release_stage()

    dit = capped.transformer_stage()
    # asked for 2, floored to 3: a 2-block stack is not a shallow section, it is an
    # invisible one
    assert len(dit.double_blocks) == MIN_DISCOVERABLE_STACK, len(dit.double_blocks)
    assert len(dit.single_blocks) == MIN_DISCOVERABLE_STACK, len(dit.single_blocks)
    capped.release_stage()

    vae = capped.vae_stage()
    # the 4-stage channel ladder is architecture, not repetition: never shortened
    assert len(vae.down_blocks) == 4 and len(vae.up_blocks) == 4
    capped.release_stage()


def test_every_declared_section_is_a_discoverable_stack(device):
    """The gate the profiler actually applies: every section the CHECKPOINT declares
    must be a stack the structure walk can find on a freshly-built pipeline.

    `find_all_stacks` is what sizes, caps and attributes time to a section, and it
    only recognises a list of at least `MIN_DISCOVERABLE_STACK` same-typed callable
    blocks.  A section it cannot see has its depth inferred for the whole run -- so
    this is checked at the SMALLEST cap, which is where the stacks are thinnest, and
    it is checked on `build_pipeline`'s return value with nothing else called, which
    is exactly what the walk gets.
    """
    from models.experimental.perf_automation.agent.layer_depth import declared_section_depths
    from models.experimental.perf_automation.cc_optimize._op_sig_probe import find_all_stacks

    demo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sections = [d for d in declared_section_depths(model_id=R.HF_REPO, model_dir=demo_root) if d > 0]
    assert len(sections) >= 2, sections

    pipe = P.build_pipeline(device, layers=2)
    found = []
    for info in find_all_stacks(pipe) or []:
        blocks = getattr(info, "stack", None) or []
        if not blocks or isinstance(blocks[0], torch.nn.Module) or not callable(blocks[0]):
            continue  # HF reference weights and data-only lists are not TT stacks
        found.append((info.path, len(blocks), type(blocks[0]).__name__))

    for path, n, kind in found:
        print(f"stack {path:36s} depth={n:3d} {kind}")
    assert len(found) >= len(sections), (
        f"{len(sections)} declared sections (depths {sections}) but only {len(found)} "
        f"discoverable stack(s): {found}"
    )


def test_batch_is_a_default_not_a_constraint(device):
    """`batch` is the leading-axis width the trace contract drives, and a head called
    with N prompts runs at N -- so the knob must be settable and must reach the
    trace seams."""
    narrow = P.build_pipeline(device, layers=CAP, batch=2)
    assert narrow.batch == 2
    assert int(narrow.encode_text_trace_inputs()["input_ids"].shape[0]) == 2
    assert narrow.decode_trace_items() == 2
    assert P.build_pipeline(device, layers=CAP).batch == 32


def test_trace_capture_selftest(device):
    pipe = P.build_pipeline(device, layers=CAP)
    report = pipe.trace_capture_selftest(device)
    for stage, entry in report.items():
        if stage == "all_ok":
            continue
        print(f"{stage:12s} {entry}")
    assert report["all_ok"], report


def test_host_op_selftest(device):
    pipe = P.build_pipeline(device, layers=CAP)
    report = pipe.host_op_selftest()
    for head, verdict in report.items():
        if head == "all_on_device":
            continue
        print(f"{head:16s} on_device={verdict['on_device']} n_host_ops={verdict['n_host_ops']}")
        if verdict["host_ops"]:
            print(f"  host ops: {verdict['host_ops'][:20]}")
    assert report["all_on_device"], report

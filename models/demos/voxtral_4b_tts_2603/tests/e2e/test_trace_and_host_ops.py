# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Command 3 for `mistralai/Voxtral-4B-TTS-2603`: the trace contract and the host-op verdict.

Runs AFTER Gates 1-3 (see `test_e2e_text_generation.py` / `test_e2e_hidden_states.py`).
Covers:

  * `PIPELINE_STAGES` is derived from the HF config, not hardcoded per model.
  * Every stage exposes `<stage>_trace_setup/_trace_step/_trace_inputs/_trace_items`, with
    `_trace_inputs` and `_trace_items` genuinely ZERO-ARG.
  * `trace_capture_selftest(device)`: one capture per stage, executed, PCC-matched, released
    before the next (stage traces must not co-reside).
  * `host_op_selftest()`: EACH task head's forward fires zero host aten ops.
  * The `layers` knob is NOT inert -- capping the depth actually moves the work signal.

The batch driven is read from the pipeline object, never typed as a literal.
"""
from __future__ import annotations

import pytest
import torch

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import common, generation
from models.demos.voxtral_4b_tts_2603.tt import pipeline as pipeline_mod

# Each test here builds or drives a 26-layer 4B model on a device, with an HF golden on CPU next to
# it. The repo-wide 300s default in pytest.ini is a generic CI guard sized for unit tests: this
# module measured 99s idle and timed out at 300s on a loaded box, i.e. the default is a flake, not
# a budget. A bound is still enforced -- a genuine hang fails here instead of hanging the gate.
pytestmark = pytest.mark.timeout(1200)

DEVICE_ID = 0
TRACE_REGION = 200 * 1024 * 1024
L1_SMALL = 24576


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=DEVICE_ID, l1_small_size=L1_SMALL, trace_region_size=TRACE_REGION)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="module")
def hf_model():
    return common.load_reference_model()


@pytest.fixture(scope="module")
def pipe(device, hf_model):
    """Both heads, full depth -- built through the ONE factory the perf harness calls."""
    return pipeline_mod.build_pipeline(device, model=hf_model)


def test_pipeline_stages_follow_the_config(pipe):
    cfg = pipe.reference_model.config
    assert cfg.is_encoder_decoder is False
    # decoder-only causal LM -> [prefill, decode]; no 'encode'. No 'vocode' either: the acoustic
    # transformer IS ported (Call 3) but the audio_tokenizer that would turn its codes into a
    # waveform has no reference to gate a port against, so there is no speech-output stage.
    assert pipeline_mod.PIPELINE_STAGES == ["prefill", "decode"]
    assert pipe.PIPELINE_STAGES == pipeline_mod.PIPELINE_STAGES
    print(f"PIPELINE_STAGES={pipe.PIPELINE_STAGES} describe={pipe.describe()}")


def test_every_stage_exposes_the_contract(pipe):
    import inspect

    for stage in pipe.PIPELINE_STAGES:
        for suffix in ("_trace_setup", "_trace_step", "_trace_inputs", "_trace_items"):
            assert hasattr(pipe, stage + suffix), f"{stage}{suffix} is missing"
        for suffix in ("_trace_inputs", "_trace_items"):
            params = inspect.signature(getattr(pipe, stage + suffix)).parameters
            assert not params, f"{stage}{suffix} must be ZERO-ARG, got {list(params)}"
    # The AR decode contract.
    assert hasattr(pipe, "decode_prefill") and hasattr(pipe, "decode_step")


def test_trace_inputs_is_what_trace_setup_takes(pipe):
    for stage in pipe.PIPELINE_STAGES:
        inputs = getattr(pipe, f"{stage}_trace_inputs")()
        assert set(inputs) == {"input_ids", "capacity"}
        assert inputs["input_ids"].shape[0] == pipe.batch
        buf = getattr(pipe, f"{stage}_trace_setup")(inputs)
        assert buf["capacity"] == inputs["capacity"]
        assert buf["batch"] == pipe.batch
        print(f"{stage}_trace_inputs -> ids{tuple(inputs['input_ids'].shape)} C={buf['capacity']}")
        pipe._release_stage(stage)


def test_trace_items_are_the_real_item_counts(pipe):
    pipe.prefill_trace_setup(pipe.prefill_trace_inputs())
    prefill_items = pipe.prefill_trace_items()
    capacity = pipe._stage_buffers["prefill"]["capacity"]
    assert prefill_items == pipe.batch * capacity
    pipe._release_stage("prefill")

    pipe.decode_trace_setup(pipe.decode_trace_inputs())
    decode_items = pipe.decode_trace_items()
    assert decode_items == pipe.batch
    pipe._release_stage("decode")

    print(f"trace_items: prefill={prefill_items} (B*C) decode={decode_items} (B, one token per sample)")


def test_trace_capture_selftest(pipe, device):
    ok = pipe.trace_capture_selftest(device)
    print(f"trace_capture_selftest={ok} (batch={pipe.batch}, C={pipe.trace_capacity})")
    assert ok, "a stage failed to capture host-free or its traced output did not match eager"


def test_host_op_selftest_every_head(pipe):
    verdicts = pipe.host_op_selftest()
    for head, v in verdicts.items():
        if head == "on_device":
            continue
        print(f"host_op_selftest[{head}]: on_device={v['on_device']} n_host_ops={v['n_host_ops']} {v['reason']}")
    assert verdicts["on_device"], f"host aten ops fired inside a head's forward: {verdicts}"


def test_layers_knob_is_not_inert(device, hf_model):
    """Cap the depth and RE-MEASURE the work signal; a knob the builder ignores is a defect."""
    full = pipeline_mod.build_pipeline(device, model=hf_model, heads=("text_generation",))
    full_depth = len(full.generation.layers)
    full_counts = generation.expected_invocation_counts(full_depth)

    capped = pipeline_mod.build_pipeline(device, model=hf_model, heads=("text_generation",), layers=4)
    capped_depth = len(capped.generation.layers)
    capped_counts = generation.expected_invocation_counts(capped_depth)

    print(f"layers=None -> depth {full_depth} counts {full_counts}")
    print(f"layers=4    -> depth {capped_depth} counts {capped_counts}")
    assert full_depth == len(hf_model.model.layers)
    assert capped_depth == 4
    assert capped_counts["decoder_layer"] < full_counts["decoder_layer"], "the layers cap is INERT"

    # A capped build must remain a MODEL, not a fragment: every block kind still present.
    assert {b.kind for b in capped.generation.layers} == {"decoder_layer", "layer", "split_mlp", "split_m_l_p"}

    # And the capped stack must still actually run.
    ids, _ = common.build_batch_inputs(batch=4)
    out = capped.generation.forward_hidden(ids)
    assert isinstance(out, ttnn.Tensor) and tuple(out.shape) == (4, ids.shape[1], hf_model.config.hidden_size)
    ttnn.deallocate(out)


def test_layers_below_the_minimum_clamps_up(device, hf_model, capsys, expect_error):
    """`layers=2` would leave a graduated block structurally absent, so it clamps to 4 and says so."""
    capped = pipeline_mod.build_pipeline(device, model=hf_model, heads=("text_generation",), layers=2)
    printed = capsys.readouterr().out
    assert len(capped.generation.layers) == 4
    assert "clamping up to 4" in printed, printed[-400:]
    with expect_error(ValueError, "would build a zero-layer model"):
        pipeline_mod.build_pipeline(device, model=hf_model, heads=("text_generation",), layers=0)


def test_stacks_are_discoverable(pipe):
    """Each repeated block is a plain list of SAME-TYPED elements the walk can find and size."""
    for name, layers in pipe.stacks.items():
        assert isinstance(layers, list), f"{name} stack is {type(layers)}, not a plain list"
        kinds = {type(x) for x in layers}
        assert len(kinds) == 1, f"{name} stack holds mixed types {kinds}"
        print(f"stack {name}: {len(layers)} x {next(iter(kinds)).__name__}")
    # The HF reference stays reachable and is ground truth for the section structure.
    assert len(pipe.reference_model.model.layers) == 26


def test_build_pipeline_returns_an_object_not_a_result(device, hf_model):
    obj = pipeline_mod.build_pipeline(
        device, model=hf_model, heads=("hidden_states",), layers=2, text="ignored", language="ignored"
    )
    assert hasattr(obj, "PIPELINE_STAGES") and hasattr(obj, "prefill_trace_step")
    assert not isinstance(obj, (torch.Tensor, dict, list, str))
    print(f"build_pipeline -> {type(obj).__name__} (demo kwargs accepted and ignored)")

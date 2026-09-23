# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""COMMAND 3: the per-stage trace contract and the fully-on-device check.

Both selftests are MODULE-LEVEL functions in `tt/pipeline.py`, because the bring-up observers
(`scripts/tt_hw_planner/_host_op_probe.py`, `_trace_capture_probe.py`) import that module in a
FRESH process and call them by name with no arguments -- a method on the pipeline class does not
count. Opening a device is forbidden anywhere under `tt/`, so the opener lives in
`device_session.py` at the demo root; the selftests fall back to it when handed no device.
"""
from __future__ import annotations

import pytest

from models.demos.voxtral_4b_tts_2603.tt import common, pipeline

pytestmark = pytest.mark.timeout(3600)


@pytest.fixture(scope="module")
def pipe(device, hf_model):
    """ONE build for every contract test in this file.

    Building this model is ~2 minutes of the suite's wall clock (26 text layers + the acoustic and
    codec sections), and four tests below were each building an identical copy. The gate runs
    `tests/e2e` under a single 2700s budget and treats overrunning it as a device hang, so the
    rebuilds were spending the budget on work already done. Nothing is shared BETWEEN tests except
    the built weights; the stage buffers each test needs are re-seeded by the setup call it makes.
    """
    return pipeline.build_pipeline(device, model=hf_model)


def test_pipeline_stages_are_derived_from_the_config(hf_model):
    """The stage list follows Source A, not a per-model map."""
    assert pipeline.PIPELINE_STAGES == ["prefill", "decode", "acoustic", "vocode"]
    assert not getattr(
        hf_model.config, "is_encoder_decoder", False
    ), "an encoder-decoder reference would need an `encode` stage"
    assert (
        type(hf_model).__mro__[1].__name__ == "MistralForCausalLM"
    ), "the ForCausalLM derivation of [prefill, decode] no longer holds"
    # `vocode` because the model card's pipeline_tag is text-to-speech, and `acoustic` because
    # params.json carries acoustic_transformer_args as its own sub-config with its own stack.
    params = common.load_params()
    assert "acoustic_transformer_args" in params["multimodal"]["audio_model_args"]
    assert "audio_tokenizer_args" in params["multimodal"]


def test_every_stage_exposes_the_full_contract(pipe):
    """A stage missing `_trace_inputs` cannot be driven by the perf test at all."""
    for stage in pipe.PIPELINE_STAGES:
        for suffix in ("trace_setup", "trace_step", "trace_inputs", "trace_items"):
            name = f"{stage}_{suffix}"
            assert callable(getattr(pipe, name, None)), f"{name} is missing"
    # The AR decode contract.
    assert callable(getattr(pipe, "decode_prefill", None))
    assert callable(getattr(pipe, "decode_trace_step", None))


def test_trace_inputs_are_zero_arg_and_feed_their_own_setup(pipe):
    """`<stage>_trace_inputs()` must return EXACTLY what `<stage>_trace_setup` takes."""
    for stage in pipe.PIPELINE_STAGES:
        stage_inputs = getattr(pipe, f"{stage}_trace_inputs")
        assert stage_inputs.__code__.co_argcount == 1, f"{stage}_trace_inputs is not zero-arg"
        got = stage_inputs()
        getattr(pipe, f"{stage}_trace_setup")(got)
        items = getattr(pipe, f"{stage}_trace_items")()
        assert isinstance(items, int) and items >= 1
        print(f"{stage:9s} trace_items = {items}")


def test_trace_items_price_the_repeated_blocks_not_the_return_value(pipe):
    """The arithmetic ceiling is 2 x params x items, so a stage that understates items is
    handed a compute roof that is too small and is then misreported as memory-bound."""
    for stage in pipe.PIPELINE_STAGES:
        getattr(pipe, f"{stage}_trace_setup")(getattr(pipe, f"{stage}_trace_inputs")())
    batch = pipe.batch
    assert pipe.prefill_trace_items() == batch * pipe.trace_capacity
    assert pipe.decode_trace_items() == batch
    # One acoustic step is a whole frame: 7 Euler steps x CFG-doubled batch x 3 token rows.
    assert pipe.acoustic_trace_items() == pipe.acoustic.n_steps * 2 * batch * 3
    # The codec's four groups see C, 2C, 4C and 8C frames; 8C is what the last group retires.
    assert pipe.vocode_trace_items() == batch * pipe.vocode_capacity * 8


def test_trace_capture_selftest(device, pipe):
    """Capture, replay and PCC-check ONE step of each stage; traces must not co-reside."""
    # The module's own build is handed over: a second one beside it does not fit in DRAM.
    ok = pipeline.trace_capture_selftest(device, pipe=pipe)
    assert ok, "at least one stage failed to capture host-free or its replay missed the reference"


def test_host_op_selftest(device, pipe):
    """The AUTHORITATIVE fully-on-device check, for EVERY task head.

    Runs on the fixture's device and the module's build. The observer calls the SAME function
    zero-arg in a fresh process, where it opens its own device through `device_session` -- the
    branch this test cannot take, because opening a second device beside the live fixture one
    raises `No MetalContext instance` and leaves the device needing a reset.
    """
    result = pipeline.host_op_selftest(device, pipe=pipe)
    print(f"host_op verdict: {result}")
    for head, verdict in result.items():
        assert pipeline._verdict_ok(verdict), f"{head} fired host aten ops inside the model math: {verdict}"

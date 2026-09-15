# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared device/pipeline fixtures for this model's e2e tests.

Extracted from test_e2e_pipeline.py so both the PCC gate (test_e2e_pipeline.py)
and the other structural gates (other_tests/) share the SAME module-scoped
model build instead of each paying for a separate one.
"""
from __future__ import annotations

import os
import time

import pytest

from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16.tests.e2e import make_golden
from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16.tt import _invocation
from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16.tt import pipeline as P

# Depth actually built by this gate. Forced by DRAM (see README / tt/_hf_ref.py):
# the 23 MoE blocks alone need ~29 GB per chip at TP=2 against ~12 GB available.
GATE_LAYERS = int(os.environ.get("TT_E2E_LAYERS", P.DEFAULT_LAYERS))
# Batch is READ FROM THE PIPELINE, never typed into an assertion message.
GATE_BATCH = int(os.environ.get("TT_E2E_BATCH", P.BATCH))


@pytest.fixture(scope="module")
def device():
    dev = P.open_mesh(2, 2)
    yield dev
    P.close_mesh(dev)


@pytest.fixture(scope="module")
def pipe(device):
    os.environ.setdefault("TT_HW_PLANNER_SHARD_RUN", "1")
    t0 = time.time()
    p = P.build_pipeline(device, layers=GATE_LAYERS, batch=GATE_BATCH)
    print(f"\n[e2e] built depth={p.n_layers} batch={p.batch} in {time.time() - t0:.1f}s")
    print(f"[e2e] block_types={p.block_types}")
    print(f"[e2e] variants={p.variants}")
    return p


@pytest.fixture(scope="module")
def run(pipe):
    """ONE on-device run of the real pipeline, shared by every gate below."""
    input_ids = make_golden.build_input_ids(pipe.batch)
    _invocation.reset()
    t0 = time.time()
    tt = P.NemotronHPipeline.run_text_generation(pipe, input_ids)
    dt = time.time() - t0
    print(f"[e2e] TT decode: {tt['steps']} steps x {pipe.batch} samples in {dt:.1f}s")
    invoked = _invocation.snapshot()

    t0 = time.time()
    ref = pipe._hf_reference_text_generation(input_ids, max_new_tokens=tt["steps"])
    print(f"[e2e] HF golden (free-running generate): {ref['steps']} steps in {time.time() - t0:.1f}s")

    tf = pipe._hf_reference_teacher_forced(tt["sequences"], input_ids.shape[1], tt["steps"])
    print(f"[e2e] HF golden (same-prefix): {tuple(tf.shape)}")
    return {"input_ids": input_ids, "tt": tt, "ref": ref, "tf": tf, "invoked": invoked}

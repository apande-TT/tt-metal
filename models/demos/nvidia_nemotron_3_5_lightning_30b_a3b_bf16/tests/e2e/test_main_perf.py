# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf test for the 'main' (text generation) pipeline: on-device TTNN forward
only, no PCC / correctness comparisons. Lifted from
tests/e2e/test_e2e_pipeline.py + tt/pipeline.py (build_pipeline, open_mesh).
"""
from __future__ import annotations

import os
import time

import torch

import ttnn
from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16.tt import pipeline as P

PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
PERF_ISL_TOKENS = int(os.environ.get("TT_PERF_ISL_TOKENS", "128"))
PERF_OSL_TOKENS = int(os.environ.get("TT_PERF_OSL_TOKENS", "128"))
PERF_BATCH = int(os.environ.get("TT_PERF_BATCH", "0"))

_pl = (os.environ.get("TT_PERF_LAYERS") or "").strip()
PERF_LAYERS = int(_pl) if (_pl.isdigit() and int(_pl) > 0) else None
# This checkpoint's 52-block stack cannot fit in DRAM at all (see tt/_hf_ref.py "DEPTH CAP"): the
# demo's own --layers default is P.DEFAULT_LAYERS (7), never None -- "whole model" for THIS source
# is that hardware-forced depth, not an unattainable 52. Mirror the demo exactly.
_BUILD_LAYERS = PERF_LAYERS if PERF_LAYERS is not None else P.DEFAULT_LAYERS

from models.experimental.perf_automation.agent.perf_adapter import resolve_batch, resolve_mesh_shape

# Source's own topology (tt/pipeline.py open_mesh default + docstring): 4 chips, TP=2 x DP=2.
_MESH_SHAPE = resolve_mesh_shape(default_rows=2, default_cols=2)

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_L1_SMALL_SIZE = 24576
_TRACE_REGION_SIZE = int(os.environ.get("TT_PERF_TRACE_REGION", "41943040"))


def prompt_ids_for_isl(tokenizer, n_tokens: int) -> torch.Tensor:
    seed = tokenizer.encode("The quick brown fox jumps over the lazy dog and keeps on running. ")
    seed = [int(t) for t in seed] or [1]
    ids: list[int] = []
    while len(ids) < n_tokens:
        ids.extend(seed)
    return torch.tensor(ids[:n_tokens], dtype=torch.long)


def test_main_perf():
    if _PERF_TRACE:
        device = P.open_mesh(
            *_MESH_SHAPE,
            l1_small_size=_L1_SMALL_SIZE,
            trace_region_size=_TRACE_REGION_SIZE,
        )
    else:
        device = P.open_mesh(*_MESH_SHAPE, l1_small_size=_L1_SMALL_SIZE)

    try:
        os.environ.setdefault("TT_HW_PLANNER_SHARD_RUN", "1")

        def _eager_forward():
            counter = [0]
            _orig = []

            def _draining(fn):
                def inner(*a, **k):
                    r = fn(*a, **k)
                    counter[0] += 1
                    if PERF_FLUSH_EVERY and counter[0] % PERF_FLUSH_EVERY == 0:
                        try:
                            ttnn.ReadDeviceProfiler(device)
                        except Exception:
                            pass
                    return r

                return inner

            pipe = P.build_pipeline(device, layers=_BUILD_LAYERS)
            # --- per-stage marks (injected) ---------------------------------------------------
            # Runs HERE, at the end of the function that built the pipeline, because that object is a LOCAL of
            # this scope: an earlier version copied the test's own PipelineStageAdapter(...) arguments into the
            # profiling branch and raised NameError, since the generator had defined them inside another
            # function. Handed locals() rather than a name, so nothing depends on how the test spells things.
            print("STAGE_MARKS_ENTER", flush=True)
            try:
                from models.experimental.perf_automation.agent import stage_marks as _tt_sm2

                print("STAGE_MARKS_RESULT=%d" % _tt_sm2.mark_stages_in_scope(locals(), device), flush=True)
            except Exception as _tt_e2:  # noqa: BLE001
                print("STAGE_MARKS_SKIPPED=%r" % (_tt_e2,), flush=True)
            n_batch = resolve_batch(pipe, PERF_BATCH)
            tokenizer = P._hf_ref.get_tokenizer()
            prompt_ids = prompt_ids_for_isl(tokenizer, PERF_ISL_TOKENS)
            input_ids = prompt_ids.unsqueeze(0).expand(n_batch, prompt_ids.numel()).contiguous()
            print("PERF_ISL_TOKENS=%d" % PERF_ISL_TOKENS, flush=True)
            print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
            print("PERF_BATCH_STREAMS=%d" % n_batch, flush=True)

            _mods = [ttnn] + [getattr(ttnn, _m, None) for _m in ("transformer", "experimental")]
            for _mod in [_m for _m in _mods if _m is not None]:
                for _n in dir(_mod):
                    _op = getattr(_mod, _n, None)
                    if type(_op).__name__ == "FastOperation":
                        _orig.append((_mod, _n, _op))
                        setattr(_mod, _n, _draining(_op))
            _fw0 = time.monotonic()
            try:
                out = pipe.run_text_generation(input_ids, max_new_tokens=PERF_OSL_TOKENS)
                try:
                    ttnn.ReadDeviceProfiler(device)
                except Exception:
                    pass
            finally:
                for _mod, _n, _f in _orig:
                    setattr(_mod, _n, _f)
            print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
            assert out is not None
            assert out["steps"] > 0

        def _traced_forward():
            from models.experimental.perf_automation.agent.trace_replay import measure_adapter
            from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter

            def _build_for_perf(dev):
                return P.build_pipeline(dev, layers=_BUILD_LAYERS)

            tokenizer = P._hf_ref.get_tokenizer()
            _prompt_ids = prompt_ids_for_isl(tokenizer, PERF_ISL_TOKENS)
            print("PERF_ISL_TOKENS=%d" % _prompt_ids.shape[-1], flush=True)
            print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
            measure_adapter(PipelineStageAdapter(_build_for_perf, _prompt_ids, batch=PERF_BATCH), device)

        def _try_traced():
            try:
                _traced_forward()
                return True
            except Exception as _te:  # noqa: BLE001
                print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)
                return False

        _PROFILING = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
        if _PERF_TRACE and not _PROFILING:
            if not _try_traced():
                print("TRACE_REPLAY_FALLBACK=eager  # trace_replay isn't working — timing eagerly", flush=True)
                _eager_forward()
        else:
            # --- stage marks (injected by perf_test_gen) -------------------------------------
            # The measured region is bracketed by the conventional start/stop pair so the main report
            # slices exactly the ops run_head emitted; the pass below is additive and feeds per-stage
            # fidelity only. Injected rather than written by the generator: the skeleton is advisory and
            # a generated test simply omitted this, which is why five earlier attempts measured nothing.
            try:
                from models.experimental.perf_automation.agent import stage_marks as _tt_sm
                from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter as _TtPSA
            except Exception:  # noqa: BLE001
                _tt_sm = None
            if _tt_sm is not None:
                _tt_sm.signpost("start")
            _eager_forward()
            if _tt_sm is not None:
                _tt_sm.signpost("stop")
            if _PERF_TRACE:
                _try_traced()
    finally:
        P.close_mesh(device)

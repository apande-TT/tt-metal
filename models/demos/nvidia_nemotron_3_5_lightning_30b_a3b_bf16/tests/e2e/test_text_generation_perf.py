# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PERFORMANCE test for the `text_generation` pipeline of
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`.

Same wiring as `demo/demo_text_generation.py`: `tt.pipeline.build_pipeline(device)` ->
`NemotronHPipeline.run_text_generation`. Perf only -- no PCC/comp_pcc, no HF reference compare.

The source SELF-OPENS its device (`tt.pipeline.open_mesh` -> `ttnn.open_mesh_device(MeshShape(rows,
cols))` on a fabric-enabled 4-chip TP=2 x DP=2 mesh, gated by `TT_HW_PLANNER_SHARD_RUN`), so this test
lifts that exact open call rather than using a pytest `device`/`device_params` fixture.
"""
from __future__ import annotations

import os
import time

import torch

import ttnn

from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16.tt import _hf_ref
from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16.tt import pipeline as P

PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# ISL / OSL -- THE MEASUREMENT CONDITIONS. 128 in / 128 out is the industry-standard short-context
# benchmark point; both are env-overridable and echoed below so a reader never has to guess.
PERF_ISL_TOKENS = int(os.environ.get("TT_PERF_ISL_TOKENS", "128"))
PERF_OSL_TOKENS = int(os.environ.get("TT_PERF_OSL_TOKENS", "128"))
# BATCH BELONGS TO THE MODEL: 0 = ask the pipeline (it declares BATCH=32 via `self.batch`).
PERF_BATCH = int(os.environ.get("TT_PERF_BATCH", "0"))
# DEPTH. A POSITIVE TT_PERF_LAYERS caps the profiled window; ABSENT means ALL LAYERS, which is
# build_pipeline's own all-layers sentinel (None) -- never a numeric default here.
_pl = (os.environ.get("TT_PERF_LAYERS") or "").strip()
PERF_LAYERS = int(_pl) if (_pl.isdigit() and int(_pl) > 0) else None

# TOPOLOGY. The source's own shape is 4 chips, TP=2 x DP=2 (rows=DP, cols=TP).
from models.experimental.perf_automation.agent.perf_adapter import resolve_batch, resolve_mesh_shape

_MESH_SHAPE = resolve_mesh_shape(default_rows=2, default_cols=2)

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_TRACE_REGION = int(os.environ.get("TT_PERF_TRACE_REGION", "41943040"))


def _build_kwargs() -> dict:
    # PERF_LAYERS absent -> the demo's OWN default depth (P.DEFAULT_LAYERS), exactly what
    # `python -m demo_text_generation` builds with no CLI override -- NOT build_pipeline's raw
    # None="every layer" sentinel, which this model's demo documents as needing "far more DRAM
    # than 4 chips have" (confirmed: layers=None OOMs mid-build on this mesh). A positive
    # TT_PERF_LAYERS still overrides/caps further, exactly as it would override the demo's --layers.
    kw = {"layers": PERF_LAYERS if PERF_LAYERS is not None else P.DEFAULT_LAYERS}
    if PERF_BATCH > 0:
        kw["batch"] = PERF_BATCH
    return kw


def prompt_ids_for_isl(tokenizer, n_tokens: int) -> torch.Tensor:
    """EXACTLY `n_tokens` real token ids from this model's tokenizer (cycled, then truncated).

    The length is the tool's measurement condition, never an example prompt chosen here.
    """
    seed = tokenizer.encode("The quick brown fox jumps over the lazy dog and keeps on running. ")
    seed = [int(t) for t in seed] or [1]
    ids: list[int] = []
    while len(ids) < n_tokens:
        ids.extend(seed)
    return torch.tensor(ids[:n_tokens], dtype=torch.long)


def test_text_generation_perf():
    device = P.open_mesh(
        rows=_MESH_SHAPE[0],
        cols=_MESH_SHAPE[1],
        l1_small_size=24576,
        trace_region_size=(_TRACE_REGION if _PERF_TRACE else 0),
    )
    try:
        _run(device)
    finally:
        P.close_mesh(device)


def _run(device):
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

        # Build + input encoding OUTSIDE the wrapped/timed region.
        pipe = P.build_pipeline(device, **_build_kwargs())
        n_batch = resolve_batch(pipe, PERF_BATCH)
        tok = _hf_ref.get_tokenizer()
        ids = prompt_ids_for_isl(tok, PERF_ISL_TOKENS)
        input_ids = ids.unsqueeze(0).expand(n_batch, ids.numel()).contiguous()
        print("PERF_ISL_TOKENS=%d" % int(input_ids.shape[-1]), flush=True)
        print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
        print("PERF_BATCH=%d" % n_batch, flush=True)

        _mods = [ttnn] + [getattr(ttnn, _m, None) for _m in ("transformer", "experimental")]
        for _mod in [_m for _m in _mods if _m is not None]:
            for _n in dir(_mod):
                _op = getattr(_mod, _n, None)
                if type(_op).__name__ == "FastOperation":
                    _orig.append((_mod, _n, _op))
                    setattr(_mod, _n, _draining(_op))
        _fw0 = time.monotonic()
        try:
            out = P.NemotronHPipeline.run_text_generation(pipe, input_ids, max_new_tokens=PERF_OSL_TOKENS)
            try:
                ttnn.ReadDeviceProfiler(device)
            except Exception:
                pass
        finally:
            for _mod, _n, _f in _orig:
                setattr(_mod, _n, _f)
        print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
        assert out is not None  # perf only — NO PCC

    def _traced_forward():
        from models.experimental.perf_automation.agent.trace_replay import measure_adapter
        from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter

        def _build_for_perf(dev):
            from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16.tt.pipeline import build_pipeline

            pipe = build_pipeline(dev, **_build_kwargs())
            # THE RESIDENT, STAGE-EXPOSING OBJECT (PIPELINE_STAGES=["prefill","decode"] with
            # <stage>_trace_setup/_step/_inputs already implemented on NemotronHPipeline). Its shipped
            # *_trace_inputs() reads a captured golden batch (demo prompts) via _captured_input_ids();
            # override both with a batch-replicated, EXACTLY-PERF_ISL_TOKENS prompt instead, so the
            # traced ISL is the tool's declared measurement condition rather than whatever the golden
            # happens to contain. Host work only -- it runs in setup(), outside every captured region.
            tok = _hf_ref.get_tokenizer()
            ids = prompt_ids_for_isl(tok, PERF_ISL_TOKENS)
            batch_ids = ids.unsqueeze(0).expand(pipe.batch, ids.numel()).contiguous()
            pipe.prefill_trace_inputs = lambda: {"input_ids": batch_ids}
            pipe.decode_trace_inputs = lambda: {"input_ids": batch_ids}
            return pipe

        _prompt_ids = prompt_ids_for_isl(_hf_ref.get_tokenizer(), PERF_ISL_TOKENS)
        print("PERF_ISL_TOKENS=%d" % _prompt_ids.shape[-1], flush=True)
        print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
        print("PERF_STAGES=%s" % (",".join(P.PIPELINE_STAGES),), flush=True)
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
        _eager_forward()
        if _PERF_TRACE:
            _try_traced()

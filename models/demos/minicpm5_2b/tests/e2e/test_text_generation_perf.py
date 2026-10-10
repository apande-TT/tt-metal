# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf test for openbmb/MiniCPM5-2B text generation: the SAME tt/pipeline.py the demo and e2e use.

Perf only, no PCC (tests/e2e/test_e2e_minicpm5_2b.py is the correctness gate). The traced path
captures each PIPELINE_STAGES step (prefill, decode) via PipelineStageAdapter and prints
TRACE_PER_TOKEN_MS; the eager path (profiling, or trace fallback) prints FORWARD_WALL_MS.
"""
import os
import time

import pytest

import ttnn
from models.demos.minicpm5_2b.tt import inputs as tt_inputs
from models.demos.minicpm5_2b.tt.pipeline import build_pipeline, load_hf_model
from models.experimental.perf_automation.agent.perf_adapter import resolve_batch, resolve_mesh_shape

try:
    from models.experimental.perf_automation.agent.perf_test_gen import prompt_ids_for_isl
except Exception:  # noqa: BLE001

    def prompt_ids_for_isl(tokenizer, n_tokens):
        import torch

        seed = tokenizer.encode("The quick brown fox jumps over the lazy dog and keeps on running. ")
        seed = [int(x) for x in seed] or [1]
        ids = []
        while len(ids) < int(n_tokens):
            ids.extend(seed)
        return torch.tensor(ids[: int(n_tokens)], dtype=torch.long)


PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
PERF_ISL_TOKENS = int(os.environ.get("TT_PERF_ISL_TOKENS", "128"))
PERF_OSL_TOKENS = int(os.environ.get("TT_PERF_OSL_TOKENS", "128"))
_EAGER_OSL_TOKENS = min(PERF_OSL_TOKENS, int(os.environ.get("TT_PERF_EAGER_OSL_TOKENS", "8")))
PERF_BATCH = int(os.environ.get("TT_PERF_BATCH", "0"))  # 0 = the pipeline's own batch
_pl = (os.environ.get("TT_PERF_LAYERS") or "").strip()
PERF_LAYERS = int(_pl) if (_pl.isdigit() and int(_pl) > 0) else None

# The pipeline is single-chip (parallelism_manifest.json: chips=1).
_MESH_SHAPE = resolve_mesh_shape(default_rows=1, default_cols=1)

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_DEV_PARAMS = {"l1_small_size": 24576}
if _PERF_TRACE:
    # 42 layers of prefill at B=32 need ~150 MB of trace region (MiniCPM5Pipeline.trace_region_bytes).
    _DEV_PARAMS["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", str(200 * 1024 * 1024)))
    _DEV_PARAMS["num_command_queues"] = 1


@pytest.mark.parametrize("device_params", [_DEV_PARAMS], indirect=True)
def test_text_generation_perf(device_params, device):
    assert _MESH_SHAPE == (1, 1), f"MiniCPM5-2B pipeline is single-chip; mesh {_MESH_SHAPE} not supported"
    tokenizer = tt_inputs.load_tokenizer()
    hf_model = load_hf_model()
    pipe = build_pipeline(
        device,
        model=hf_model,
        layers=PERF_LAYERS,
        batch=PERF_BATCH or None,
        prompt_len=PERF_ISL_TOKENS,
        max_new_tokens=PERF_OSL_TOKENS,
    )
    batch = resolve_batch(pipe, PERF_BATCH)
    prompt = prompt_ids_for_isl(tokenizer, PERF_ISL_TOKENS)

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
                    except Exception:  # noqa: BLE001
                        pass
                return r

            return inner

        _mods = [ttnn] + [getattr(ttnn, _m, None) for _m in ("transformer", "experimental")]
        for _mod in [_m for _m in _mods if _m is not None]:
            for _n in dir(_mod):
                _op = getattr(_mod, _n, None)
                if type(_op).__name__ == "FastOperation":
                    _orig.append((_mod, _n, _op))
                    setattr(_mod, _n, _draining(_op))
        ids = prompt.reshape(1, -1).expand(batch, -1).contiguous()
        seeds = [tt_inputs.BASE_SEED + i for i in range(batch)]
        noise = tt_inputs.gumbel_noise(seeds, _EAGER_OSL_TOKENS, pipe.vocab)
        _fw0 = time.monotonic()
        try:
            out = pipe.run_text_generation(ids, noise, max_new_tokens=_EAGER_OSL_TOKENS, log=lambda *_: None)
            try:
                ttnn.ReadDeviceProfiler(device)
            except Exception:  # noqa: BLE001
                pass
        finally:
            for _mod, _n, _f in _orig:
                setattr(_mod, _n, _f)
        print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
        assert out is not None  # perf only — NO PCC

    def _traced_forward():
        from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter
        from models.experimental.perf_automation.agent.trace_replay import measure_adapter

        def _build_for_perf(dev):
            return pipe  # reuse the resident build; a second full build would not fit beside it

        print("PERF_ISL_TOKENS=%d" % prompt.shape[-1], flush=True)
        print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
        measure_adapter(PipelineStageAdapter(_build_for_perf, prompt, batch=PERF_BATCH), device)

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

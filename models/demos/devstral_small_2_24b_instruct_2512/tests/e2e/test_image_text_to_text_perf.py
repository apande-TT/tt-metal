# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf for the Devstral image_text_to_text pipeline (vision tower -> projector -> [IMG] merge -> language
model -> lm_head -> on-device sampling), built exactly as demo/demo_image_text_to_text.py does on the
self-opened 1x4 mesh. Trace replay (every PIPELINE_STAGES entry: encode, prefill, decode) is the default path;
the op-wrapped eager forward is the tracy capture. Perf only -- correctness is tests/e2e/test_e2e_*.py.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import ttnn
from models.demos.devstral_small_2_24b_instruct_2512.demo.inputs import IMAGE_PROMPT, load_image
from models.demos.devstral_small_2_24b_instruct_2512.mesh import DEFAULT_MESH, close_mesh, open_mesh
from models.demos.devstral_small_2_24b_instruct_2512.tt import pipeline as P

PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
PERF_ISL_TOKENS = int(os.environ.get("TT_PERF_ISL_TOKENS", "128"))
PERF_OSL_TOKENS = int(os.environ.get("TT_PERF_OSL_TOKENS", "128"))
_EAGER_OSL_TOKENS = min(PERF_OSL_TOKENS, int(os.environ.get("TT_PERF_EAGER_OSL_TOKENS", "8")))
PERF_BATCH = int(os.environ.get("TT_PERF_BATCH", "0"))
_pl = (os.environ.get("TT_PERF_LAYERS") or "").strip()
PERF_LAYERS = int(_pl) if (_pl.isdigit() and int(_pl) > 0) else None
_pl_encode = (os.environ.get("TT_PERF_ENCODE_LAYERS") or "").strip()
PERF_ENCODE_LAYERS = int(_pl_encode) if (_pl_encode.isdigit() and int(_pl_encode) > 0) else None
_pl_prefill = (os.environ.get("TT_PERF_PREFILL_LAYERS") or "").strip()
PERF_PREFILL_LAYERS = int(_pl_prefill) if (_pl_prefill.isdigit() and int(_pl_prefill) > 0) else None
_pl_decode = (os.environ.get("TT_PERF_DECODE_LAYERS") or "").strip()
PERF_DECODE_LAYERS = int(_pl_decode) if (_pl_decode.isdigit() and int(_pl_decode) > 0) else None

from models.experimental.perf_automation.agent.perf_adapter import resolve_batch, resolve_mesh_shape  # noqa: E402,F401

# The demo self-opens a 1x4 mesh (mesh.DEFAULT_MESH) through mesh.open_mesh; that is the source shape.
_MESH_SHAPE = resolve_mesh_shape(default_rows=DEFAULT_MESH[0], default_cols=DEFAULT_MESH[1])

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
# The demo opens with trace_region_size=P.TRACE_REGION_SIZE (open_mesh always uses 1 CQ and enables fabric
# only when the mesh spans more than one chip).
_TRACE_REGION = int(os.environ.get("TT_PERF_TRACE_REGION", str(P.TRACE_REGION_SIZE)))


def _build_args():
    return dict(
        layers=PERF_LAYERS,
        encode_layers=PERF_ENCODE_LAYERS,
        prefill_layers=PERF_PREFILL_LAYERS,
        decode_layers=PERF_DECODE_LAYERS,
        batch=PERF_BATCH or None,  # 0 = ask the pipeline (its own declared batch)
    )


# The pipeline's on-device sampler builds generic_op top-k kernels whose compiler_include_paths are
# repo-relative ("ttnn/cpp/..."), resolved against the CWD -- so the run must sit at the tt-metal root, as the
# demo's `python -m models.demos...` invocation does.
_REPO_ROOT = Path(os.environ.get("TT_METAL_HOME") or Path(__file__).resolve().parents[5])


def test_image_text_to_text_perf():
    _cwd = os.getcwd()
    os.chdir(_REPO_ROOT)
    try:
        # Self-opened mesh, exactly as the demo: open_mesh(trace_region_size=...) on the resolved shape.
        device = open_mesh(trace_region_size=_TRACE_REGION if _PERF_TRACE else 0, mesh_shape=_MESH_SHAPE)
        try:
            _run(device)
        finally:
            close_mesh(device)
    finally:
        os.chdir(_cwd)


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
                    except Exception:  # noqa: BLE001
                        pass
                return r

            return inner

        # Build + encode exactly as demo_image_text_to_text.main() does.
        pipe = P.build_pipeline(device, **_build_args())
        processor = P.load_processor()
        seeds = list(range(pipe.batch))
        req = P.encode_request(
            processor,
            IMAGE_PROMPT,
            image=load_image(None),
            batch=pipe.batch,
            seeds=seeds,
            kv_capacity=pipe.kv_capacity,
        )
        _mods = [ttnn] + [getattr(ttnn, _m, None) for _m in ("transformer", "experimental")]
        for _mod in [_m for _m in _mods if _m is not None]:
            for _n in dir(_mod):
                _op = getattr(_mod, _n, None)
                if type(_op).__name__ == "FastOperation":  # every dispatched ttnn op, by type
                    _orig.append((_mod, _n, _op))
                    setattr(_mod, _n, _draining(_op))
        _fw0 = time.monotonic()
        try:
            # run_image_text_to_text's body, bounded: vision encode -> merged prefill -> token 0, then decode
            state = pipe.decode_prefill(req)
            for _ in range(_EAGER_OSL_TOKENS - 1):
                state = pipe.decode_step(state)
            ttnn.synchronize_device(device)
            out = state
            try:
                ttnn.ReadDeviceProfiler(device)
            except Exception:  # noqa: BLE001
                pass
        finally:
            for _mod, _n, _f in _orig:
                setattr(_mod, _n, _f)
        print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0), flush=True)
        assert out is not None  # perf only -- NO PCC

    def _traced_forward():
        from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter
        from models.experimental.perf_automation.agent.perf_test_gen import prompt_ids_for_isl
        from models.experimental.perf_automation.agent.trace_replay import measure_adapter

        def _build_for_perf(dev):
            return P.build_pipeline(dev, **_build_args())

        _prompt_ids = prompt_ids_for_isl(P.load_processor().tokenizer, PERF_ISL_TOKENS)
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
        _eager_forward()
        if _PERF_TRACE:
            _try_traced()

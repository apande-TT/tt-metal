import os
import time
import pytest
import ttnn

PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))

_pl = (os.environ.get("TT_PERF_LAYERS") or "").strip()
PERF_LAYERS = int(_pl) if (_pl.isdigit() and int(_pl) > 0) else None

from models.experimental.perf_automation.agent.perf_adapter import resolve_mesh_shape

_MESH_SHAPE = resolve_mesh_shape(default_rows=2, default_cols=2)

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_DEV_PARAMS = {"l1_small_size": 24576}

if _MESH_SHAPE[0] * _MESH_SHAPE[1] > 1:
    _DEV_PARAMS["fabric_config"] = True

if _PERF_TRACE:
    _DEV_PARAMS["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "23887872"))
    _DEV_PARAMS["num_command_queues"] = int(os.environ.get("TT_PERF_NUM_CQ", "1"))


def test_main_perf():
    device = ttnn.open_mesh_device(ttnn.MeshShape(*_MESH_SHAPE), **_DEV_PARAMS)
    try:
        def _eager_forward():
            counter = [0]
            _orig = []
            def _draining(fn):
                def inner(*a, **k):
                    r = fn(*a, **k); counter[0] += 1
                    if PERF_FLUSH_EVERY and counter[0] % PERF_FLUSH_EVERY == 0:
                        try: ttnn.ReadDeviceProfiler(device)
                        except Exception: pass
                    return r
                return inner
            _mods = [ttnn] + [getattr(ttnn, _m, None) for _m in ("transformer", "experimental")]
            for _mod in [_m for _m in _mods if _m is not None]:
                for _n in dir(_mod):
                    _op = getattr(_mod, _n, None)
                    if type(_op).__name__ == "FastOperation":
                        _orig.append((_mod, _n, _op)); setattr(_mod, _n, _draining(_op))
            _fw0 = time.monotonic()
            try:
                from models.demos.llama3_1_8b_p150.tt.pipeline import build_pipeline
                pipeline = build_pipeline(device, num_layers=PERF_LAYERS)
                # decode_step needs the state decode_prefill returns (KV cache seeded + first
                # token); passing None raised before a single model op was dispatched, so tracy
                # only ever saw the 21 host-prep tilize/typecast ops and every matmul was
                # invisible to the roofline. Run the real prefill + steady-state decode, and run
                # it UNTRACED so each op is dispatched individually and lands in the profiler.
                state = pipeline.decode_prefill([128000, 791, 6864, 315, 9822, 374], enable_trace=False)
                for _ in range(max(PERF_MAX_NEW_TOKENS, 1)):
                    state = pipeline.decode_step(state, enable_trace=False)
                out = state["out_tok"]
                try: ttnn.ReadDeviceProfiler(device)
                except Exception: pass
            finally:
                for _mod, _n, _f in _orig: setattr(_mod, _n, _f)
            print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
            assert out is not None

        def _traced_forward():
            from models.experimental.perf_automation.agent.trace_replay import measure_adapter
            from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter

            def _build_for_perf(dev):
                from models.demos.llama3_1_8b_p150.tt.pipeline import build_pipeline
                return build_pipeline(dev, num_layers=PERF_LAYERS)

            _prompt_ids = [1, 2, 3, 4, 5]
            measure_adapter(PipelineStageAdapter(_build_for_perf, _prompt_ids, batch=1), device, mode="auto")

        def _try_traced():
            try:
                _traced_forward(); return True
            except Exception as _te:
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
    finally:
        ttnn.close_mesh_device(device)

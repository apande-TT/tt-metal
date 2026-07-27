# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Performance test for Llama-3.1-8B-Instruct main pipeline."""
import os
import time

import ttnn

PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))

_pl = (os.environ.get("TT_PERF_LAYERS") or "").strip()
PERF_LAYERS = int(_pl) if (_pl.isdigit() and int(_pl) > 0) else None

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"


def test_main_perf(reset_seeds):
    """Performance test for Llama main pipeline."""
    from models.demos.llama3_1_8b_p150.tt.pipeline import build_pipeline
    from models.experimental.perf_automation.agent.perf_adapter import resolve_mesh_shape

    # Resolve mesh shape using the tool's topology (handles --devices/--mesh)
    rows, cols = resolve_mesh_shape(default_rows=1, default_cols=1)

    # Open device with minimal config to avoid fabric timeouts on single-chip
    _dev_params = {"l1_small_size": 24576, "num_command_queues": 1}
    if _PERF_TRACE:
        _dev_params["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "23887872"))
        _dev_params["num_command_queues"] = int(os.environ.get("TT_PERF_NUM_CQ", "1"))

    mesh_device = ttnn.open_mesh_device(shape=ttnn.MeshShape(rows, cols), **_dev_params)

    try:
        num_devices = mesh_device.get_num_devices() if isinstance(mesh_device, ttnn.MeshDevice) else 1

        # Wrap all FastOperations for profiler draining
        counter = [0]
        _orig = []

        def _draining(fn):
            def inner(*a, **k):
                r = fn(*a, **k)
                counter[0] += 1
                if PERF_FLUSH_EVERY and counter[0] % PERF_FLUSH_EVERY == 0:
                    try:
                        ttnn.ReadDeviceProfiler(mesh_device)
                    except Exception:
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

        out = None
        _fw_ms = 0.0
        try:
            # Build pipeline with bounded config
            pipeline = build_pipeline(
                mesh_device,
                instruct=True,
                max_seq_len=512,
                batch_size=1,
                num_layers=PERF_LAYERS,
            )

            # Small prompt for profiling
            prompt_ids = [1, 2, 3, 4, 5]

            _fw0 = time.monotonic()

            # Forward pass with bounded decode
            out = pipeline(prompt_ids, num_decode_tokens=PERF_MAX_NEW_TOKENS)

            _fw_ms = (time.monotonic() - _fw0) * 1000.0

            try:
                ttnn.ReadDeviceProfiler(mesh_device)
            except Exception:
                pass
        finally:
            for _mod, _n, _f in _orig:
                setattr(_mod, _n, _f)

        print("FORWARD_WALL_MS=%.4f" % _fw_ms)
        assert out is not None

        # Try traced forward if enabled
        if _PERF_TRACE:
            try:
                from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter
                from models.experimental.perf_automation.agent.trace_replay import measure_adapter

                def _build_for_perf(dev):
                    return build_pipeline(
                        dev,
                        instruct=True,
                        max_seq_len=512,
                        batch_size=1,
                        num_layers=PERF_LAYERS,
                    )

                _adapter = PipelineStageAdapter(_build_for_perf, prompt_ids, batch=1)
                measure_adapter(_adapter, mesh_device, mode="auto")
            except Exception as _te:
                print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)

    finally:
        ttnn.close_mesh_device()

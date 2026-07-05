import os
import time

import ttnn
from models.demos.hf_eager.voxtral_mini_3b_2507.tt import pipeline as P

PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# keep the profiled input small — a representative dispatch-dense pass, NOT the demo's
# production waveform length (8s). Under tracy every device op is instrumented, so a big
# audio/prompt forward stalls host sync for minutes. env-overridable, small default.
PERF_SECONDS = float(os.environ.get("TT_PERF_SECONDS", "2.0"))
PERF_PROMPT = os.environ.get("TT_PERF_PROMPT", "\nWhat is said in the audio?")
PERF_DEVICE_ID = int(os.environ.get("TT_PERF_DEVICE_ID", "0"))
# perf-only depth cap: profile a few blocks so a deep model's marker stream (x mesh chips) does not
# overflow / bloat the profiler; pipelines that read TT_PERF_LAYERS honor it, others ignore it. This
# is set in-process here so ONLY the perf run is capped (the correctness/e2e gate runs the full model).
os.environ.setdefault("TT_PERF_LAYERS", "2")

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
# The demo SELF-OPENS a single device via ttnn.open_device(l1_small_size=24576); it does NOT use a
# pytest device fixture. So we open + close the device the SAME way here (lift the exact open call),
# and thread trace_region_size / num_command_queues onto that self-open when TT_PERF_TRACE is set.
_OPEN_KW = {"l1_small_size": 24576}
if _PERF_TRACE:
    _OPEN_KW["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "23887872"))
    _OPEN_KW["num_command_queues"] = int(os.environ.get("TT_PERF_NUM_CQ", "1"))


def test_transcribe_perf():
    # open the device EXACTLY as demo/demo_transcribe.py does (self-open, single device)
    device = ttnn.open_device(device_id=PERF_DEVICE_ID, **_OPEN_KW)
    try:
        # 1) build the pipeline EXACTLY as demo/demo_transcribe.py does
        model = P.load_hf_model()
        proc = P.load_processor()
        input_ids, input_features, n_audio, atid, prompt = P.build_inputs(
            proc, model, seconds=PERF_SECONDS, prompt=PERF_PROMPT
        )
        pipe = P.VoxtralTTPipeline(device, model)

        # 2) drain the device profiler every PERF_FLUSH_EVERY ops. MODEL-AGNOSTIC: wrap EVERY ttnn
        #    operation (type 'FastOperation') across ttnn + its op submodules, so the flush counter
        #    tracks TOTAL device dispatch for ANY op mix. A curated op list under-counts (sdpa/eltwise/
        #    transpose/reduction slip through) and the 12000-marker buffer overflows on some device,
        #    dropping ops -> non-reproducible device_ms. Wrapping by TYPE never misses an op.
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

        _mods = [ttnn] + [getattr(ttnn, _m, None) for _m in ("transformer", "experimental")]
        for _mod in [_m for _m in _mods if _m is not None]:
            for _n in dir(_mod):
                _op = getattr(_mod, _n, None)
                if type(_op).__name__ == "FastOperation":  # every dispatched ttnn op, by type
                    _orig.append((_mod, _n, _op))
                    setattr(_mod, _n, _draining(_op))

        _fw0 = time.monotonic()
        try:
            out = pipe.run(input_ids, input_features, max_new_tokens=PERF_MAX_NEW_TOKENS)
            try:
                ttnn.ReadDeviceProfiler(device)
            except Exception:
                pass
        finally:
            for _mod, _n, _f in _orig:
                setattr(_mod, _n, _f)
        print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
        assert out is not None  # perf only — NO PCC

        if _PERF_TRACE:
            try:
                from models.experimental.perf_automation.agent.perf_adapter import PipelineDecodeAdapter
                from models.experimental.perf_automation.agent.trace_replay import measure_adapter

                def _build_for_perf(dev):
                    from models.demos.hf_eager.voxtral_mini_3b_2507.tt import pipeline as _P

                    _m = _P.load_hf_model()
                    return _P.VoxtralTTPipeline(dev, _m)

                _prompt_ids = input_ids
                # PREFILL trace+2CQ (audio front-end: encoder->projector) — emitted alongside the
                # decode per-token number so the optimize scorecard covers both phases. Best-effort:
                # a failure here (e.g. OOM on a large model) is skipped, never blocks the decode metric.
                try:
                    from models.experimental.perf_automation.agent.trace_replay import measure_prefill

                    measure_prefill(PipelineDecodeAdapter(_build_for_perf, _prompt_ids, batch=1), device)
                except Exception as _pe:  # noqa: BLE001
                    print("PREFILL_TRACE_SKIPPED=%r" % (_pe,), flush=True)
                _adapter = PipelineDecodeAdapter(_build_for_perf, _prompt_ids, batch=1)
                measure_adapter(_adapter, device, mode="auto")
            except Exception as _te:  # noqa: BLE001
                print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)
    finally:
        ttnn.close_device(device)

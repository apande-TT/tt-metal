import os
import time
import pytest
import torch

import ttnn
from models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.tt import pipeline as pl
from models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.tt._hf_compat import install_hf_compat

install_hf_compat()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# perf-only depth cap: profile a few blocks so a deep model's marker stream (x mesh chips) does not
# overflow / bloat the profiler; pipelines that read TT_PERF_LAYERS honor it, others ignore it. This
# is set in-process here so ONLY the perf run is capped (the correctness/e2e gate runs the full model).
os.environ.setdefault("TT_PERF_LAYERS", "2")

# Trace-replay per-token latency (GPU-comparable T/S/U). MODEL-AGNOSTIC + OFF-BY-DEFAULT-SAFE:
# TT_PERF_TRACE=1 (default) adds trace_region_size + num_command_queues to the device open so the
# per-token block below CAN capture a device trace; TT_PERF_TRACE=0 restores the plain eager open
# (exactly the old behavior -> guaranteed non-breaking escape hatch for tight-memory models).
_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_DEV_PARAMS = {"l1_small_size": 24576}
if _PERF_TRACE:
    _DEV_PARAMS["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "120000000"))
    _DEV_PARAMS["num_command_queues"] = int(os.environ.get("TT_PERF_NUM_CQ", "2"))  # 2 = trace+2CQ overlap path

# small representative prompt/seq for perf (do NOT use production/max shapes under tracy)
PERF_PROMPT = os.environ.get("TT_PERF_PROMPT", "The capital of France is")


def _open_pipeline_device():
    # Self-open the mesh EXACTLY as the demo does (pl.open_pipeline_mesh). When TT_PERF_TRACE is set,
    # try to thread trace_region_size / num_command_queues through the SAME open so the trace block can
    # capture a device trace on the identical sharded topology; fall back to the plain open otherwise.
    if _PERF_TRACE:
        try:
            return pl.open_pipeline_mesh(
                l1_small_size=24576,
                trace_region_size=int(os.environ.get("TT_PERF_TRACE_REGION", "120000000")),
                num_command_queues=int(os.environ.get("TT_PERF_NUM_CQ", "2")),
            )
        except TypeError:
            pass
    return pl.open_pipeline_mesh(l1_small_size=24576)


def test_text_generation_perf():
    # 1) build the pipeline EXACTLY as demo/demo_text_generation.py does (self-open mesh -> match topology)
    tok = AutoTokenizer.from_pretrained(pl.HF_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        pl.HF_MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    eos = int(getattr(model.config, "eos_token_id", 2))

    input_ids = tok(PERF_PROMPT, return_tensors="pt")["input_ids"]

    device, is_mesh = _open_pipeline_device()
    try:
        pipe = pl.build_pipeline(device, model, compose=True)
        print(f"[perf] mesh={is_mesh} shard_active={pipe.shard_active}", flush=True)

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
                        ttnn.ReadDeviceProfiler(device)  # 'device' = mesh_device on multi-chip
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
            out, _ = pipe.generate(input_ids, PERF_MAX_NEW_TOKENS, eos_token_id=eos)
            try:
                ttnn.ReadDeviceProfiler(device)
            except Exception:
                pass
        finally:
            for _mod, _n, _f in _orig:
                setattr(_mod, _n, _f)
        print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
        assert out is not None  # perf only — NO PCC

        # ---- clean, GPU-comparable per-token latency via trace-replay (GENERIC + guarded) ----
        # ONE generic adapter (agent/perf_adapter.PipelineDecodeAdapter) wraps the SAME pipeline build:
        # measure_adapter captures one decode step as a device trace + replays it -> prints
        # TRACE_PER_TOKEN_MS (parsed by the tool into per_token_ms + tokens_per_sec_per_user for a GPU
        # side-by-side). There is NO per-model adapter here. The clean number appears only when the built
        # pipeline exposes a trace-capturable `decode_step(state)` (fixed shape, on-device sample, no host
        # reads) -- produced by the structural decode lever / emit-e2e, not written here. A repeat-prefill
        # pipeline has no decode_step, so setup raises, the guard swallows it, and FORWARD_WALL_MS stands.
        if _PERF_TRACE:
            try:
                from models.experimental.perf_automation.agent.trace_replay import measure_adapter
                from models.experimental.perf_automation.agent.perf_adapter import PipelineDecodeAdapter

                # REUSE the already-resident pipeline (its 52 layer children are built once at
                # init) instead of building a second full copy — a 30B second build would OOM
                # the tight 4-chip residency. And run the FULL model here: Section A profiled a
                # TT_PERF_LAYERS-capped forward for device_ms, but decode_step loops all 52 layers,
                # so the trace prefill must seed all 52 (uncap for this steady-state measurement).
                os.environ["TT_PERF_LAYERS"] = "0"
                _adapter = PipelineDecodeAdapter(lambda _dev: pipe, input_ids, batch=1)
                measure_adapter(_adapter, device, mode="auto")  # prints TRACE_PER_TOKEN_MS / TRACE_REPLAY_PATH
            except Exception as _te:  # noqa: BLE001
                print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)
    finally:
        pl.close_pipeline_mesh(device, is_mesh)
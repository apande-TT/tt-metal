import gc
import inspect
import os
import time

import torch

import ttnn
from models.demos.qwen_image_edit.tt import inputs as I

PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
PERF_ISL_TOKENS = int(os.environ.get("TT_PERF_ISL_TOKENS", "128"))
PERF_OSL_TOKENS = int(os.environ.get("TT_PERF_OSL_TOKENS", "128"))
# EAGER-PATH BOUND. Diffusion has no decode loop; the eager path runs ONE bounded forward.
_EAGER_OSL_TOKENS = min(PERF_OSL_TOKENS, int(os.environ.get("TT_PERF_EAGER_OSL_TOKENS", "8")))
# BATCH BELONGS TO THE MODEL. 0 means "ask the pipeline".
PERF_BATCH = int(os.environ.get("TT_PERF_BATCH", "0"))
_pl = (os.environ.get("TT_PERF_LAYERS") or "").strip()
PERF_LAYERS = int(_pl) if (_pl.isdigit() and int(_pl) > 0) else None

# HEAVY AXIS for diffusion = TIMESTEPS. The demo runs the full 50-step schedule; a perf profile only
# needs a representative dispatch-dense pass, so the raw input (num_inference_steps) is trimmed here.
PERF_DENOISE_STEPS = max(1, int(os.environ.get("TT_PERF_DENOISE_STEPS", "2")))
# Condition-image side; defaults to the demo's own area (the pipeline's VAE/halo path is built for it).
PERF_AREA = int(os.environ.get("TT_PERF_AREA", str(I.DEFAULT_AREA)))

from models.experimental.perf_automation.agent.perf_adapter import resolve_batch, resolve_mesh_shape  # noqa: E402,F401

# The demo opens a 2x4 mesh via demo/mesh.open_mesh().
_SRC_MESH = (2, 4)
_MESH_SHAPE = resolve_mesh_shape(default_rows=_SRC_MESH[0], default_cols=_SRC_MESH[1])

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_TRACE_REGION = int(os.environ.get("TT_PERF_TRACE_REGION", "41943040"))


def _open_device():
    """Open the mesh exactly as the demo does (demo.mesh.open_mesh), honouring the resolved shape and
    passing the trace region through when the open function accepts it."""
    from models.demos.qwen_image_edit.demo.mesh import close_mesh, open_mesh

    rows, cols = _MESH_SHAPE
    try:
        params = inspect.signature(open_mesh).parameters
    except (TypeError, ValueError):
        params = {}
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    kw = {}
    if _PERF_TRACE:
        if "trace_region_size" in params or has_var_kw:
            kw["trace_region_size"] = _TRACE_REGION
        if "num_command_queues" in params or has_var_kw:
            kw["num_command_queues"] = 1
    shape_given = False
    if "mesh_shape" in params:
        kw["mesh_shape"] = ttnn.MeshShape(rows, cols)
        shape_given = True
    elif "rows" in params and "cols" in params:
        kw["rows"], kw["cols"] = rows, cols
        shape_given = True
    elif "shape" in params:
        kw["shape"] = (rows, cols)
        shape_given = True

    if shape_given or (rows, cols) == _SRC_MESH:
        return open_mesh(**kw), close_mesh

    # The planned topology differs from the demo's and open_mesh cannot take a shape: open the
    # resolved MeshShape directly with the same knobs.
    dev_kw = {"l1_small_size": 24576}
    if _PERF_TRACE:
        dev_kw["trace_region_size"] = _TRACE_REGION
        dev_kw["num_command_queues"] = 1
    if rows * cols > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols), **dev_kw)

    def _close(d):
        ttnn.close_mesh_device(d)
        if rows * cols > 1:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    return dev, _close


def _load_tokenizer():
    from transformers import AutoTokenizer

    repo = os.environ.get("TT_PERF_TOKENIZER", "Qwen/Qwen-Image-Edit")
    try:
        return AutoTokenizer.from_pretrained(repo, subfolder="tokenizer")
    except Exception:  # noqa: BLE001
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")


def _isl_prompt():
    """Prompt of EXACTLY PERF_ISL_TOKENS tokens (ids + decoded text for the pipeline's string input)."""
    from models.experimental.perf_automation.agent.perf_test_gen import prompt_ids_for_isl

    tok = _load_tokenizer()
    ids = prompt_ids_for_isl(tok, PERF_ISL_TOKENS)
    flat = ids.reshape(-1).tolist() if hasattr(ids, "reshape") else list(ids)
    return ids, tok.decode(flat, skip_special_tokens=True)


def _pipeline_batch(pipe):
    if PERF_BATCH > 0:
        return PERF_BATCH
    for name in ("max_batch_size", "batch_size", "batch"):
        v = getattr(pipe, name, None)
        if isinstance(v, int) and v > 0:
            return v
    b = I.batch_size_from_env()
    return b if b and b > 0 else 1


def test_image_edit_perf():
    device, _close = _open_device()
    try:
        _prompt_ids, _prompt_text = _isl_prompt()
        if not torch.is_tensor(_prompt_ids):
            _prompt_ids = torch.tensor(_prompt_ids, dtype=torch.long).reshape(1, -1)
        print("PERF_ISL_TOKENS=%d" % _prompt_ids.shape[-1], flush=True)
        print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
        print("PERF_DENOISE_STEPS=%d" % PERF_DENOISE_STEPS, flush=True)

        def _eager_forward():
            from models.demos.qwen_image_edit.tt.pipeline import build_pipeline, run_image_edit

            pipe = build_pipeline(device, layers=PERF_LAYERS)
            # --- per-stage marks (injected) ---------------------------------------------------
            # Runs HERE, at the end of the function that built the pipeline, because that object is a LOCAL of
            # this scope: an earlier version copied the test's own PipelineStageAdapter(...) arguments into the
            # profiling branch and raised NameError, since the generator had defined them inside another
            # function. Handed locals() rather than a name, so nothing depends on how the test spells things.
            print("STAGE_MARKS_ENTER", flush=True)
            try:
                from models.experimental.perf_automation.agent import stage_marks as _tt_sm2

                print("STAGE_MARKS_RESULT=%d" % _tt_sm2.mark_stages_in_scope(locals()), flush=True)
            except Exception as _tt_e2:  # noqa: BLE001
                print("STAGE_MARKS_SKIPPED=%r" % (_tt_e2,), flush=True)
            B = _pipeline_batch(pipe)
            enc = I.encode_inputs(
                B,
                area=PERF_AREA,
                prompt=_prompt_text,
                image=None,
                base_seed=I.EXAMPLE_SEED,
                num_inference_steps=PERF_DENOISE_STEPS,
            )
            print(
                f"[perf] {B} sample(s), {enc.width}x{enc.height}, {len(enc.timesteps)} steps",
                flush=True,
            )

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
                    if type(_op).__name__ == "FastOperation":
                        _orig.append((_mod, _n, _op))
                        setattr(_mod, _n, _draining(_op))
            _fw0 = time.monotonic()
            try:
                out = run_image_edit(pipe, enc, use_trace=False)
                ttnn.synchronize_device(device)
                try:
                    ttnn.ReadDeviceProfiler(device)
                except Exception:
                    pass
            finally:
                for _mod, _n, _f in _orig:
                    setattr(_mod, _n, _f)
            print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
            assert out is not None and out.get("image") is not None  # perf only — NO PCC
            del out, pipe
            gc.collect()

        def _traced_forward():
            from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter
            from models.experimental.perf_automation.agent.trace_replay import measure_adapter

            def _build_for_perf(dev):
                from models.demos.qwen_image_edit.tt.pipeline import build_pipeline

                return build_pipeline(dev, layers=PERF_LAYERS)

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
        _close(device)

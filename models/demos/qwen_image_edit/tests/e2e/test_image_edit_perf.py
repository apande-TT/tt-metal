import inspect
import os
import time

import pytest
import ttnn

from models.demos.qwen_image_edit.tt import inputs as I

# ONE VARIABLE FOR ONE THING. PERF_OSL_TOKENS is the single declared output unit; nothing below
# re-declares it under a second name.
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# ISL / OSL -- THE MEASUREMENT CONDITIONS (tool's choice, env-overridable, echoed into the log).
PERF_ISL_TOKENS = int(os.environ.get("TT_PERF_ISL_TOKENS", "128"))
PERF_OSL_TOKENS = int(os.environ.get("TT_PERF_OSL_TOKENS", "128"))
# EAGER-PATH BOUND (profiler host-memory): the eager pass runs at most this many denoising steps.
_EAGER_OSL_TOKENS = min(PERF_OSL_TOKENS, int(os.environ.get("TT_PERF_EAGER_OSL_TOKENS", "8")))
# BATCH BELONGS TO THE MODEL. 0 means "ask the pipeline"; a positive value overrides.
PERF_BATCH = int(os.environ.get("TT_PERF_BATCH", "0"))

# DEPTH, PER STAGE. Absent == all layers (None); a positive value caps that stage's block stack.
_pl = (os.environ.get("TT_PERF_LAYERS") or "").strip()
PERF_LAYERS = int(_pl) if (_pl.isdigit() and int(_pl) > 0) else None
_pl_vision_encode = (os.environ.get("TT_PERF_VISION_ENCODE_LAYERS") or "").strip()
PERF_VISION_ENCODE_LAYERS = (
    int(_pl_vision_encode) if (_pl_vision_encode.isdigit() and int(_pl_vision_encode) > 0) else None
)
_pl_text_encode = (os.environ.get("TT_PERF_TEXT_ENCODE_LAYERS") or "").strip()
PERF_TEXT_ENCODE_LAYERS = int(_pl_text_encode) if (_pl_text_encode.isdigit() and int(_pl_text_encode) > 0) else None
_pl_vae_encode = (os.environ.get("TT_PERF_VAE_ENCODE_LAYERS") or "").strip()
PERF_VAE_ENCODE_LAYERS = int(_pl_vae_encode) if (_pl_vae_encode.isdigit() and int(_pl_vae_encode) > 0) else None
_pl_denoise = (os.environ.get("TT_PERF_DENOISE_LAYERS") or "").strip()
PERF_DENOISE_LAYERS = int(_pl_denoise) if (_pl_denoise.isdigit() and int(_pl_denoise) > 0) else None
_pl_vae_decode = (os.environ.get("TT_PERF_VAE_DECODE_LAYERS") or "").strip()
PERF_VAE_DECODE_LAYERS = int(_pl_vae_decode) if (_pl_vae_decode.isdigit() and int(_pl_vae_decode) > 0) else None

# HEAVY AXES FOR A DIFFUSION EDIT: timesteps and pixels. The demo runs 50 steps at a 1024 condition
# side -- correctness/production sizes. The profiled pass uses a SHORT schedule and a SMALL area
# (the size the e2e gate runs), both env-overridable. The step count never exceeds the declared OSL.
PERF_STEPS = max(1, min(PERF_OSL_TOKENS, int(os.environ.get("TT_PERF_STEPS", "2"))))
PERF_AREA = int(os.environ.get("TT_PERF_AREA", "256"))
# Samples one program carries (the demo's --max-batch); unset == one program, as the demo defaults.
_mb = (os.environ.get("TT_PERF_MAX_BATCH") or "").strip()
PERF_MAX_BATCH = int(_mb) if (_mb.isdigit() and int(_mb) > 0) else None

# TOPOLOGY. The demo self-opens a 2x4 mesh via demo/mesh.open_mesh.
from models.experimental.perf_automation.agent.perf_adapter import (  # noqa: E402
    resolve_batch,
    resolve_mesh_shape,
    resolve_trace_region,
)

_SRC_ROWS, _SRC_COLS = 2, 4
_MESH_SHAPE = resolve_mesh_shape(default_rows=_SRC_ROWS, default_cols=_SRC_COLS)

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"


def _open_mesh_kwargs(open_mesh):
    """kwargs for the source's own open_mesh: planned mesh shape + (under TT_PERF_TRACE) trace params."""
    try:
        params = inspect.signature(open_mesh).parameters
    except (TypeError, ValueError):
        params = {}
    kw = {}
    rows, cols = _MESH_SHAPE
    if "rows" in params and "cols" in params:
        kw["rows"], kw["cols"] = rows, cols
    else:
        for name in ("mesh_shape", "shape"):
            if name in params:
                default = params[name].default
                if isinstance(default, (tuple, list)):
                    kw[name] = (rows, cols)
                else:
                    kw[name] = ttnn.MeshShape(rows, cols)
                break
        else:
            if (rows, cols) != (_SRC_ROWS, _SRC_COLS):
                print(
                    "MESH_SHAPE_WARNING=open_mesh takes no shape argument; requested %dx%d" % (rows, cols),
                    flush=True,
                )
    if _PERF_TRACE:
        src_region = 0
        if "trace_region_size" in params:
            d = params["trace_region_size"].default
            if isinstance(d, int) and d is not inspect.Parameter.empty:
                src_region = d
            kw["trace_region_size"] = resolve_trace_region(src_region)
        if "num_command_queues" in params:
            kw["num_command_queues"] = 1
    return kw


def _load_tokenizer():
    """The Qwen2.5-VL tokenizer the pipeline's text encoder consumes."""
    for name in ("load_tokenizer", "get_tokenizer", "tokenizer"):
        fn = getattr(I, name, None)
        if callable(fn):
            try:
                return fn()
            except Exception:  # noqa: BLE001
                pass
    from transformers import AutoTokenizer

    model_id = None
    for name in ("MODEL_ID", "HF_MODEL_ID", "MODEL_NAME", "REPO_ID"):
        model_id = getattr(I, name, None)
        if isinstance(model_id, str):
            break
        model_id = None
    model_id = model_id or os.environ.get("HF_MODEL", "Qwen/Qwen-Image-Edit")
    try:
        return AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    except Exception:  # noqa: BLE001
        return AutoTokenizer.from_pretrained(model_id)


def _build_kwargs(image_hw):
    return dict(
        image_hw=image_hw,
        layers=PERF_LAYERS,
        vision_encode_layers=PERF_VISION_ENCODE_LAYERS,
        text_encode_layers=PERF_TEXT_ENCODE_LAYERS,
        vae_encode_layers=PERF_VAE_ENCODE_LAYERS,
        denoise_layers=PERF_DENOISE_LAYERS,
        vae_decode_layers=PERF_VAE_DECODE_LAYERS,
    )


def _model_batch(pipe):
    """Batch from the model: explicit positive TT_PERF_BATCH, else what the pipeline declares."""
    if PERF_BATCH > 0:
        return PERF_BATCH
    try:
        b = resolve_batch(pipe, PERF_BATCH)
        if isinstance(b, int) and b > 0:
            return b
    except Exception:  # noqa: BLE001
        pass
    for name in ("max_batch_size", "batch_size", "batch"):
        b = getattr(pipe, name, None)
        if isinstance(b, int) and b > 0:
            return b
    try:
        b = I.batch_size_from_env()
        if isinstance(b, int) and b > 0:
            return b
    except Exception:  # noqa: BLE001
        pass
    return 1


def test_image_edit_perf():
    from models.demos.qwen_image_edit.demo.mesh import close_mesh, open_mesh
    from models.demos.qwen_image_edit.tt.pipeline import build_pipeline, run_image_edit, to_host
    from models.experimental.perf_automation.agent.perf_test_gen import prompt_ids_for_isl

    # ISL: the instruction prompt is EXACTLY PERF_ISL_TOKENS tokens of the text encoder's tokenizer.
    tokenizer = _load_tokenizer()
    _prompt_ids = prompt_ids_for_isl(tokenizer, PERF_ISL_TOKENS)
    _flat_ids = _prompt_ids.reshape(-1).tolist() if hasattr(_prompt_ids, "reshape") else list(_prompt_ids)
    perf_prompt = tokenizer.decode(_flat_ids, skip_special_tokens=True)
    print("PERF_ISL_TOKENS=%d" % _prompt_ids.shape[-1], flush=True)
    print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
    print("PERF_STEPS=%d PERF_AREA=%d" % (PERF_STEPS, PERF_AREA), flush=True)

    def _encode(B, steps):
        return I.encode_inputs(
            B,
            area=PERF_AREA,
            prompt=perf_prompt,
            image=None,
            base_seed=I.EXAMPLE_SEED,
            num_inference_steps=steps,
        )

    # The pipeline is sized for the condition image's resolution; that comes from the input encoding.
    _enc_probe = _encode(1, PERF_STEPS)
    _image_hw = (_enc_probe.height, _enc_probe.width)

    device = open_mesh(**_open_mesh_kwargs(open_mesh))
    try:

        def _eager_forward():
            import torch

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
            out = None
            try:
                t0 = time.time()
                pipe = build_pipeline(device, **_build_kwargs(_image_hw))
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
                print(f"[perf] pipeline built in {time.time() - t0:.0f} s", flush=True)
                B = _model_batch(pipe)
                steps = max(1, min(PERF_STEPS, _EAGER_OSL_TOKENS))
                enc = _encode(B, steps)
                print(
                    f"[perf] {B} sample(s), {enc.width}x{enc.height}, {len(enc.timesteps)} steps",
                    flush=True,
                )
                _fw0 = time.monotonic()
                groups = I.sample_groups(B, PERF_MAX_BATCH)
                parts = []
                for g in groups:
                    r = run_image_edit(pipe, enc.subset(g) if len(groups) > 1 else enc, use_trace=False)
                    ttnn.synchronize_device(device)
                    parts.append(to_host(r["image"], device).float())
                    del r
                out = torch.cat(parts) if len(parts) > 1 else parts[0]
                try:
                    ttnn.ReadDeviceProfiler(device)
                except Exception:
                    pass
            finally:
                for _mod, _n, _f in _orig:
                    setattr(_mod, _n, _f)
            print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
            assert out is not None  # perf only — NO PCC
            del pipe

        def _traced_forward():
            from models.experimental.perf_automation.agent.trace_replay import measure_adapter
            from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter

            def _build_for_perf(dev):
                from models.demos.qwen_image_edit.tt.pipeline import build_pipeline as _bp

                return _bp(dev, **_build_kwargs(_image_hw))

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
            try:
                _eager_forward()
            except Exception as _tt_e:  # noqa: BLE001
                if _tt_sm is None or not _tt_sm.window_reached(_tt_e):
                    raise
            if _tt_sm is not None:
                _tt_sm.signpost("stop")
            if _PERF_TRACE:
                _try_traced()
    finally:
        close_mesh(device)
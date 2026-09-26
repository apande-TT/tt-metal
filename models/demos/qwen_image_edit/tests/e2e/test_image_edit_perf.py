import inspect
import os
import time

import pytest  # noqa: F401
import torch

import ttnn
from models.demos.qwen_image_edit.mesh import close_mesh, open_mesh
from models.demos.qwen_image_edit.tt import pipeline as P
from models.demos.qwen_image_edit.tt.inputs import EditConfig, sample_images, sample_prompts, sample_seeds
from models.experimental.perf_automation.agent.perf_test_gen import prompt_ids_for_isl

# ONE VARIABLE FOR ONE THING: PERF_OSL_TOKENS is both what is printed and what bounds work.
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# ISL / OSL -- the measurement conditions (tool's choice, env-overridable, echoed below).
PERF_ISL_TOKENS = int(os.environ.get("TT_PERF_ISL_TOKENS", "128"))
PERF_OSL_TOKENS = int(os.environ.get("TT_PERF_OSL_TOKENS", "128"))
# EAGER-PATH BOUND (profiler host-memory); never exceeds the declared OSL.
_EAGER_OSL_TOKENS = min(PERF_OSL_TOKENS, int(os.environ.get("TT_PERF_EAGER_OSL_TOKENS", "8")))
# BATCH BELONGS TO THE MODEL: 0 means "ask the pipeline"; a positive value overrides.
PERF_BATCH = int(os.environ.get("TT_PERF_BATCH", "0"))
# DEPTH. Absent means ALL LAYERS (None). Default depth for any stack not named per-stage below.
_pl = (os.environ.get("TT_PERF_LAYERS") or "").strip()
PERF_LAYERS = int(_pl) if (_pl.isdigit() and int(_pl) > 0) else None
# PER-STAGE DEPTH (same None-means-all-layers contract).
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

# HEAVY AXIS for diffusion = TIMESTEPS (and pixels). Small, env-overridable defaults; the demo's
# 50-step schedule is a quality setting, not a perf-profile size.
PERF_STEPS = int(os.environ.get("TT_PERF_STEPS", "2"))
PERF_AREA = int(os.environ.get("TT_PERF_AREA", "256"))
PERF_CFG_SCALE = float(os.environ.get("TT_PERF_CFG_SCALE", "4.0"))

# TOPOLOGY. The demo self-opens a 2x4 T3K mesh via open_mesh().
from models.experimental.perf_automation.agent.perf_adapter import resolve_batch, resolve_mesh_shape  # noqa: E402,F401

_MESH_SHAPE = resolve_mesh_shape(default_rows=2, default_cols=4)

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_DEV_PARAMS = {}
if _PERF_TRACE:
    # Reserve the trace region at device-open, ONCE; single command queue (trace+1cq).
    _DEV_PARAMS["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "41943040"))
    _DEV_PARAMS["num_command_queues"] = 1


def _open_perf_mesh():
    """Open the mesh EXACTLY as the demo does (open_mesh()), passing the resolved shape and the
    trace params only when open_mesh accepts them."""
    try:
        params = inspect.signature(open_mesh).parameters
    except (TypeError, ValueError):
        params = {}
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    kw = {}
    shape = tuple(_MESH_SHAPE)
    if shape != (2, 4):
        for name in ("mesh_shape", "shape"):
            if name in params:
                kw[name] = ttnn.MeshShape(*shape) if name == "mesh_shape" else shape
                break
        else:
            if "rows" in params and "cols" in params:
                kw["rows"], kw["cols"] = shape
    for k, v in _DEV_PARAMS.items():
        if k in params or accepts_var_kw:
            kw[k] = v
    return open_mesh(**kw)


def _perf_batch():
    if PERF_BATCH > 0:
        b = PERF_BATCH
    else:
        try:
            b = int(getattr(EditConfig(), "batch", 0) or 0)
        except Exception:  # noqa: BLE001
            b = 0
        b = b or 32  # the demo's own default batch
    return b + (b % 2)  # VAE runs batch-parallel over the 2 mesh rows


def _tokenizer(hf):
    for attr in ("tokenizer", "processor"):
        tok = getattr(hf, attr, None)
        if tok is not None:
            return getattr(tok, "tokenizer", tok)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen-Image-Edit", subfolder="tokenizer")


def _make_cfg(batch):
    return EditConfig(
        batch=batch,
        area=PERF_AREA * PERF_AREA,
        num_inference_steps=PERF_STEPS,
        true_cfg_scale=PERF_CFG_SCALE,
        negative_prompt=" ",
    )


def _build_kwargs():
    return dict(
        layers=PERF_LAYERS,
        vision_encode_layers=PERF_VISION_ENCODE_LAYERS,
        text_encode_layers=PERF_TEXT_ENCODE_LAYERS,
        vae_encode_layers=PERF_VAE_ENCODE_LAYERS,
        denoise_layers=PERF_DENOISE_LAYERS,
        vae_decode_layers=PERF_VAE_DECODE_LAYERS,
    )


def test_image_edit_perf():
    batch = _perf_batch()
    cfg = _make_cfg(batch)
    hf = P.load_hf_reference(torch.float32)
    tokenizer = _tokenizer(hf)
    _prompt_ids = prompt_ids_for_isl(tokenizer, PERF_ISL_TOKENS)
    print("PERF_ISL_TOKENS=%d" % _prompt_ids.shape[-1], flush=True)
    print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
    print(
        "PERF_STEPS=%d PERF_AREA=%d PERF_BATCH=%d MESH=%s" % (PERF_STEPS, PERF_AREA, batch, tuple(_MESH_SHAPE)),
        flush=True,
    )
    # the edit instruction IS the ISL-sized prompt
    _ids = _prompt_ids.reshape(-1).tolist()
    _isl_prompt = tokenizer.decode(_ids, skip_special_tokens=True)

    device = _open_perf_mesh()
    try:

        def _eager_forward():
            pipe = P.build_pipeline(device, model=hf, cfg=cfg, **_build_kwargs())
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
            n = getattr(pipe, "batch", None) or batch
            n = min(int(n), batch)
            n += n % 2
            images = sample_images(n)
            prompts = [_isl_prompt] * n
            seeds = sample_seeds(n)
            _ = sample_prompts  # demo's own prompts replaced by the ISL-sized prompt

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
            image = None
            _fw0 = time.monotonic()
            try:
                enc = pipe.encode(cfg, images=images, prompts=prompts, seeds=seeds)
                p = pipe.prepare(enc)
                out = pipe.run_image_edit(p)
                image = P.to_host(out).to(torch.float32)
                try:
                    ttnn.ReadDeviceProfiler(device)
                except Exception:
                    pass
            finally:
                for _mod, _n, _f in _orig:
                    setattr(_mod, _n, _f)
            print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
            assert image is not None and image.numel() > 0  # perf only — NO PCC
            del pipe, p, out

        def _traced_forward():
            from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter
            from models.experimental.perf_automation.agent.trace_replay import measure_adapter

            def _build_for_perf(dev):
                from models.demos.qwen_image_edit.tt.pipeline import build_pipeline

                return build_pipeline(dev, model=hf, cfg=cfg, **_build_kwargs())

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
        close_mesh(device)

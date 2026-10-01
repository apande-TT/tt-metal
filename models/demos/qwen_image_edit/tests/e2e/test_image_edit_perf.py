import gc
import inspect
import math
import os
import time

import pytest
import torch
import ttnn

from models.demos.qwen_image_edit.mesh import close_mesh, open_mesh
from models.demos.qwen_image_edit.tt import pipeline as P
from models.demos.qwen_image_edit.tt.inputs import EditConfig, sample_images, sample_prompts, sample_seeds

PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
PERF_ISL_TOKENS = int(os.environ.get("TT_PERF_ISL_TOKENS", "128"))
PERF_OSL_TOKENS = int(os.environ.get("TT_PERF_OSL_TOKENS", "128"))
_EAGER_OSL_TOKENS = min(PERF_OSL_TOKENS, int(os.environ.get("TT_PERF_EAGER_OSL_TOKENS", "8")))
PERF_BATCH = int(os.environ.get("TT_PERF_BATCH", "0"))
_pl = (os.environ.get("TT_PERF_LAYERS") or "").strip()
PERF_LAYERS = int(_pl) if (_pl.isdigit() and int(_pl) > 0) else None

# diffusion: the heavy axis is the TIMESTEP count (one full transformer pass x2 for CFG per step),
# then pixels. Keep both small by default; env-overridable.
PERF_STEPS = int(os.environ.get("TT_PERF_STEPS", "2"))
PERF_AREA = int(os.environ.get("TT_PERF_AREA", "256"))

# demo defaults (demo/demo_image_edit.py)
_DEMO_BATCH = 32
_DEMO_CFG_SCALE = 4.0
_DEMO_NEGATIVE_PROMPT = " "
_DEMO_ROWS, _DEMO_COLS = 8, 4

from models.experimental.perf_automation.agent.perf_adapter import resolve_batch, resolve_mesh_shape  # noqa: E402,F401

_MESH_SHAPE = resolve_mesh_shape(default_rows=_DEMO_ROWS, default_cols=_DEMO_COLS)

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_DEV_PARAMS = {}
if _PERF_TRACE:
    _DEV_PARAMS["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "41943040"))
    _DEV_PARAMS["num_command_queues"] = 1


def _open_perf_mesh():
    """Open the mesh exactly as the demo does (open_mesh), honouring the planned topology and the
    trace params when open_mesh accepts them."""
    rows, cols = _MESH_SHAPE
    try:
        params = inspect.signature(open_mesh).parameters
    except (TypeError, ValueError):
        params = {}
    var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    kwargs = {}
    for k, v in _DEV_PARAMS.items():
        if k in params or var_kw:
            kwargs[k] = v
    if (rows, cols) != (_DEMO_ROWS, _DEMO_COLS):
        if "mesh_shape" in params:
            kwargs["mesh_shape"] = (rows, cols)
        elif "shape" in params:
            kwargs["shape"] = (rows, cols)
        elif "rows" in params and "cols" in params:
            kwargs["rows"], kwargs["cols"] = rows, cols
        else:
            # open_mesh cannot reshape: open the planned topology directly
            if rows * cols > 1:
                ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
            dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols), **_DEV_PARAMS)

            def _close(d=dev):
                ttnn.close_mesh_device(d)
                if rows * cols > 1:
                    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

            return dev, _close
    mesh = open_mesh(**kwargs)
    return mesh, close_mesh


def _perf_batch():
    b = PERF_BATCH if PERF_BATCH > 0 else _DEMO_BATCH
    # the VAE runs batch-parallel over the mesh rows and the denoise splits the batch over the
    # columns: pad up to a multiple of both, exactly as the demo pads to a multiple of 8
    rows, cols = _MESH_SHAPE
    m = rows * cols // math.gcd(rows, cols)
    return b + ((-b) % m)


def _perf_tokenizer(hf):
    for attr in ("tokenizer", "processor"):
        tok = getattr(hf, attr, None)
        if tok is not None:
            return getattr(tok, "tokenizer", tok)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen-Image-Edit", subfolder="tokenizer")


def _free_device_memory(device):
    gc.collect()
    try:
        ttnn.synchronize_device(device)
    except Exception:
        pass


def test_image_edit_perf():
    from models.experimental.perf_automation.agent.perf_test_gen import prompt_ids_for_isl

    hf = P.load_hf_reference(torch.float32)
    batch = _perf_batch()
    cfg = EditConfig(
        batch=batch,
        area=PERF_AREA * PERF_AREA,
        num_inference_steps=PERF_STEPS,
        true_cfg_scale=_DEMO_CFG_SCALE,
        negative_prompt=_DEMO_NEGATIVE_PROMPT,
    )
    tokenizer = _perf_tokenizer(hf)
    _prompt_ids = prompt_ids_for_isl(tokenizer, PERF_ISL_TOKENS)
    if not torch.is_tensor(_prompt_ids):
        _prompt_ids = torch.tensor(_prompt_ids)
    if _prompt_ids.dim() == 1:
        _prompt_ids = _prompt_ids.unsqueeze(0)
    print("PERF_ISL_TOKENS=%d" % _prompt_ids.shape[-1], flush=True)
    print("PERF_OSL_TOKENS=%d" % PERF_OSL_TOKENS, flush=True)
    print(
        "PERF_STEPS=%d PERF_AREA=%d PERF_BATCH=%d MESH=%dx%d" % (PERF_STEPS, PERF_AREA, batch, *_MESH_SHAPE),
        flush=True,
    )
    perf_prompt = tokenizer.decode(_prompt_ids[0].tolist(), skip_special_tokens=True)

    # inputs exactly as the demo builds them for the bundled samples (prompt text replaced by the
    # ISL-sized prompt), padded to the batch with copies of the last sample
    n = min(batch, _DEMO_BATCH)
    images = sample_images(n)
    _ = sample_prompts(n)
    seeds = sample_seeds(n)
    pad = batch - len(images)
    if pad > 0:
        images, seeds = images + images[-1:] * pad, seeds + seeds[-1:] * pad
    prompts = [perf_prompt] * len(images)

    device, _closer = _open_perf_mesh()
    # ONE resident pipeline per device: the trace pass and the eager pass (fallback or profiling)
    # share it. Building it twice on one mesh exhausts DRAM (bank_manager OOM).
    _pipe_cache = {}

    def _get_pipe(dev):
        key = id(dev)
        if key not in _pipe_cache:
            _free_device_memory(dev)
            _pipe_cache[key] = P.build_pipeline(dev, model=hf, cfg=cfg, layers=PERF_LAYERS)
        return _pipe_cache[key]

    try:

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

            pipe = _get_pipe(device)
            enc = pipe.encode(cfg, images=images, prompts=prompts, seeds=seeds)
            p = pipe.prepare(enc)
            _mods = [ttnn] + [getattr(ttnn, _m, None) for _m in ("transformer", "experimental")]
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
            for _mod in [_m for _m in _mods if _m is not None]:
                for _n in dir(_mod):
                    _op = getattr(_mod, _n, None)
                    if type(_op).__name__ == "FastOperation":
                        _orig.append((_mod, _n, _op))
                        setattr(_mod, _n, _draining(_op))
            _fw0 = time.monotonic()
            try:
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
            del p, enc, out
            _free_device_memory(device)

        def _traced_forward():
            from models.experimental.perf_automation.agent.trace_replay import measure_adapter
            from models.experimental.perf_automation.agent.perf_adapter import PipelineStageAdapter

            def _build_for_perf(dev):
                from models.demos.qwen_image_edit.tt.pipeline import build_pipeline  # noqa: F401

                # same build_pipeline(dev, model=hf, cfg=cfg, layers=PERF_LAYERS) call, resident once
                return _get_pipe(dev)

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
            finally:
                # drop any device tensors the failed/finished trace pass still holds via frames
                _free_device_memory(device)

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
            _eager_forward()
            if _tt_sm is not None:
                _tt_sm.signpost("stop")
            if _PERF_TRACE:
                _try_traced()
    finally:
        _pipe_cache.clear()
        gc.collect()
        _closer(device)
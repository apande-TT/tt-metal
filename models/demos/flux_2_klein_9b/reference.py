# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Source A: the HuggingFace reference for ``black-forest-labs/FLUX.2-klein-9B``.

Everything the TT pipeline needs from the hub side lives here -- the tokenizer,
the four checkpoint pieces, the scheduler, the real ``Flux2KleinPipeline`` used as
the golden, and the input builders (chat template + image processor) that decide
what "real input" means for each head.

Two facts shape this file:

* The snapshot root has **no** ``config.json``: it is a diffusers pipeline repo,
  so ``model_index.json`` -- not ``transformers.AutoConfig`` -- is the registry.
* ``python_env`` pins diffusers 0.35.1, which predates every ``Flux2*`` class.
  A 0.40.0 build lives in a side venv and is side-loaded **by path**: never added
  to ``sys.path``, so numpy / PIL / huggingface_hub / transformers keep resolving
  out of ``python_env``.  ``diffusers/__init__.py`` ends by replacing itself with a
  ``_LazyModule``, so after ``exec_module`` the value to use is whatever now sits in
  ``sys.modules`` -- the object handed back by ``module_from_spec`` holds only the
  pre-swap globals and every lazy export reads as absent.

Nothing here runs at import time.
"""

from __future__ import annotations

import glob
import importlib
import importlib.util
import json
import os
import sys

import torch

HF_REPO = "black-forest-labs/FLUX.2-klein-9B"

_SNAPSHOT_GLOBS = (
    os.path.expanduser("~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-9B/snapshots/*"),
    os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        "hub/models--black-forest-labs--FLUX.2-klein-9B/snapshots/*",
    ),
)

_DIFFUSERS_SEARCH_GLOBS = (
    "/home/ttuser/venvs/*/lib/python*/site-packages/diffusers",
    os.path.expanduser("~/venvs/*/lib/python*/site-packages/diffusers"),
)

#: The prompt-embedding recipe is fixed by the pipeline, not by us.
TEXT_ENCODER_OUT_LAYERS = (9, 18, 27)
TOKENIZER_MAX_LENGTH = 512


# --------------------------------------------------------------------- snapshot


def snapshot_root() -> str:
    for pattern in _SNAPSHOT_GLOBS:
        hits = [h for h in sorted(glob.glob(pattern)) if os.path.isfile(os.path.join(h, "model_index.json"))]
        if hits:
            return hits[-1]
    raise FileNotFoundError(f"no local snapshot of {HF_REPO} with a model_index.json")


def model_index() -> dict:
    with open(os.path.join(snapshot_root(), "model_index.json")) as f:
        return json.load(f)


def sub_config(name: str) -> dict:
    with open(os.path.join(snapshot_root(), name, "config.json")) as f:
        return json.load(f)


# -------------------------------------------------------------------- diffusers


def _pkg_exports(pkg_dir: str, class_name: str) -> bool:
    """Read ``__init__.py`` instead of importing: diffusers is a lazy-import package,
    so every export is named in ``_import_structure`` as plain source text, and
    importing to probe would strand a stale module in ``sys.modules``."""
    try:
        with open(os.path.join(pkg_dir, "__init__.py"), encoding="utf-8", errors="replace") as f:
            return class_name in f.read()
    except OSError:
        return False


def _ambient_diffusers_dir() -> str | None:
    if "diffusers" in sys.modules:
        f = getattr(sys.modules["diffusers"], "__file__", None)
        return os.path.dirname(f) if f else None
    try:
        spec = importlib.util.find_spec("diffusers")  # does not exec
    except (ImportError, ValueError):
        return None
    return os.path.dirname(spec.origin) if spec and spec.origin else None


def _sideload(name: str, pkg_dir: str):
    """Import the package rooted at ``pkg_dir`` as ``name`` without touching ``sys.path``.

    ``submodule_search_locations`` makes absolute ``name.X`` imports route through this
    ``__path__``, so numpy / PIL / torch keep resolving out of ``python_env``.
    ``sys.modules[name]`` must be set BEFORE ``exec_module`` so those submodule
    imports can find their parent.
    """
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(pkg_dir, "__init__.py"), submodule_search_locations=[pkg_dir]
    )
    for stale in [n for n in list(sys.modules) if n == name or n.startswith(name + ".")]:
        del sys.modules[stale]
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    # A lazy-import package's __init__ replaces itself in sys.modules; take that.
    return sys.modules.get(name, module)


def _ensure_hub_for(pkg_dir: str) -> None:
    """diffusers 0.40's ``pipeline_utils`` needs ``huggingface_hub.get_cached_repo_tree``,
    which ``python_env``'s hub 1.22 does not have.  The side venv ships hub 1.28 next to
    diffusers, so side-load that too -- before diffusers, and before the first
    ``transformers`` import, so nothing binds the older one.  transformers 5.8.1 runs on
    hub 1.28 unchanged (verified)."""
    site_dir = os.path.dirname(os.path.abspath(pkg_dir))
    hub_dir = os.path.join(site_dir, "huggingface_hub")
    if not os.path.isdir(hub_dir):
        return
    current = sys.modules.get("huggingface_hub")
    if current is not None and os.path.dirname(getattr(current, "__file__", "")) == hub_dir:
        return
    if not _pkg_exports(hub_dir, "get_cached_repo_tree"):
        return
    _sideload("huggingface_hub", hub_dir)


def diffusers_module(class_name: str = "Flux2KleinPipeline"):
    """A diffusers module that exports ``class_name`` (side-loaded if need be)."""
    candidates: list[str] = []
    ambient = _ambient_diffusers_dir()
    if ambient:
        candidates.append(ambient)
    for pattern in _DIFFUSERS_SEARCH_GLOBS:
        for hit in sorted(glob.glob(pattern)):
            if os.path.isdir(hit) and hit not in candidates:
                candidates.append(hit)

    usable = [d for d in candidates if _pkg_exports(d, class_name)]
    if not usable:
        raise ImportError(
            f"no diffusers build exporting {class_name}; checked {candidates}. " "FLUX.2 needs diffusers >= 0.37."
        )
    if ambient in usable:
        return importlib.import_module("diffusers")

    pkg_dir = usable[0]
    _ensure_hub_for(pkg_dir)
    return _sideload("diffusers", pkg_dir)


def ensure_flux_imports():
    """Bring the Flux2-capable diffusers (and its hub) into ``sys.modules``.

    Call this before importing ``transformers`` in a fresh process so nothing binds
    ``python_env``'s older ``huggingface_hub``.
    """
    return diffusers_module("Flux2KleinPipeline")


# ------------------------------------------------------------------- components

_CACHE: dict = {}


def load_tokenizer():
    if "tokenizer" not in _CACHE:
        from transformers import AutoTokenizer

        _CACHE["tokenizer"] = AutoTokenizer.from_pretrained(os.path.join(snapshot_root(), "tokenizer"))
    return _CACHE["tokenizer"]


def load_text_encoder():
    """``Qwen3ForCausalLM`` in its stored dtype (bf16), eval, grads off."""
    if "text_encoder" not in _CACHE:
        from transformers import Qwen3ForCausalLM

        model = Qwen3ForCausalLM.from_pretrained(
            os.path.join(snapshot_root(), "text_encoder"), dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        model.eval()
        model.requires_grad_(False)
        _CACHE["text_encoder"] = model
    return _CACHE["text_encoder"]


def load_transformer():
    """``Flux2Transformer2DModel`` in its stored dtype (bf16)."""
    if "transformer" not in _CACHE:
        diffusers = diffusers_module("Flux2Transformer2DModel")
        model = diffusers.Flux2Transformer2DModel.from_pretrained(
            os.path.join(snapshot_root(), "transformer"), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        model.eval()
        model.requires_grad_(False)
        _CACHE["transformer"] = model
    return _CACHE["transformer"]


def load_vae():
    """``AutoencoderKLFlux2`` in fp32 -- its config sets ``force_upcast: true``, and
    bf16 GroupNorm/BatchNorm over full-resolution activations is unstable."""
    if "vae" not in _CACHE:
        diffusers = diffusers_module("AutoencoderKLFlux2")
        model = diffusers.AutoencoderKLFlux2.from_pretrained(
            os.path.join(snapshot_root(), "vae"), torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        model.eval()
        model.requires_grad_(False)
        _CACHE["vae"] = model
    return _CACHE["vae"]


def load_scheduler():
    diffusers = diffusers_module("FlowMatchEulerDiscreteScheduler")
    return diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(snapshot_root(), "scheduler"))


def load_pipeline(text_encoder=None, transformer=None, vae=None):
    """The real ``Flux2KleinPipeline``, built over already-loaded pieces so the
    golden and the TT weight-staging share one copy of each 9 B checkpoint."""
    key = "pipeline"
    if key not in _CACHE:
        diffusers = diffusers_module("Flux2KleinPipeline")
        _CACHE[key] = diffusers.Flux2KleinPipeline(
            scheduler=load_scheduler(),
            vae=vae if vae is not None else load_vae(),
            text_encoder=text_encoder if text_encoder is not None else load_text_encoder(),
            tokenizer=load_tokenizer(),
            transformer=transformer if transformer is not None else load_transformer(),
            is_distilled=bool(model_index().get("is_distilled", False)),
        )
    return _CACHE[key]


def image_processor():
    if "image_processor" not in _CACHE:
        _CACHE["image_processor"] = load_pipeline().image_processor
    return _CACHE["image_processor"]


def release(*names: str) -> None:
    """Drop cached host modules (the 9 B pieces are 18 GB each)."""
    for name in names or tuple(_CACHE):
        _CACHE.pop(name, None)


# ---------------------------------------------------------------- golden cache
#
# A golden is a PURE function of (checkpoint, this function's own source, its
# arguments), so computing it twice is pure cost -- and on this model that cost is
# what decides whether the suite finishes at all.  The e2e gate kills its pytest at
# a hard 2700 s, and one CPU-side 9 B golden is minutes: the BATCHED text head is 32
# rows x 32 no-cache prefix forwards, and the batched image heads are 16 samples x
# their denoise steps.  Measured, the batched goldens alone are longer than the
# whole budget, so the suite could never reach a verdict, which the gate reads as a
# hang.
#
# This memoises the golden VALUE, and nothing else: no assertion, threshold, shape
# or comparison changes, and a hit returns the same tensors the miss computed.  The
# key is the full CONTENT of everything the value depends on --
#
#   * the checkpoint snapshot directory (its name is the revision hash),
#   * the source text of the golden function itself, so editing how a golden is
#     computed invalidates every entry for it, and
#   * every argument, hashed by bytes (tensors upcast to a lossless dtype first,
#     PIL images by their raw buffer) rather than by repr/id.
#
# -- so a stale hit needs a collision in SHA-256.  `FLUX2_GOLDEN_CACHE=0` disables
# it outright and recomputes everything, which is how a hit is verified against a
# cold run.

_GOLDEN_CACHE_VERSION = 1


def golden_cache_dir():
    """The on-disk golden cache, or None when disabled."""
    setting = os.environ.get("FLUX2_GOLDEN_CACHE", "")
    if setting == "0":
        return None
    root = setting or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".golden_cache")
    os.makedirs(root, exist_ok=True)
    return root


def _hash_value(digest, value) -> None:
    """Fold `value` into `digest` by CONTENT, not by identity or repr."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"T|{tensor.dtype}|{tuple(tensor.shape)}|".encode())
        # upcast losslessly so bfloat16/float16/int32 all hash their exact values
        wide = tensor.to(torch.float64) if tensor.is_floating_point() else tensor.to(torch.int64)
        digest.update(wide.numpy().tobytes())
        return
    if isinstance(value, (list, tuple)):
        digest.update(f"L|{len(value)}|".encode())
        for item in value:
            _hash_value(digest, item)
        return
    if isinstance(value, dict):
        digest.update(f"D|{len(value)}|".encode())
        for key in sorted(value, key=repr):
            digest.update(f"k|{key!r}|".encode())
            _hash_value(digest, value[key])
        return
    raw = getattr(value, "tobytes", None)  # PIL.Image and numpy arrays
    if callable(raw) and hasattr(value, "mode") and hasattr(value, "size"):
        digest.update(f"I|{value.mode}|{value.size}|".encode())
        digest.update(raw())
        return
    if isinstance(value, (str, int, float, bool, bytes, type(None))):
        digest.update(f"S|{value!r}|".encode())
        return
    raise TypeError(f"golden cache cannot hash a {type(value).__name__} by content")


def _golden_key(fn, args, kwargs) -> str:
    import hashlib
    import inspect

    digest = hashlib.sha256()
    digest.update(f"v{_GOLDEN_CACHE_VERSION}|{fn.__name__}|".encode())
    digest.update(f"snap|{os.path.basename(snapshot_root())}|".encode())
    try:
        digest.update(inspect.getsource(fn).encode())
    except (OSError, TypeError):  # pragma: no cover - source is always available here
        raise
    _hash_value(digest, list(args))
    _hash_value(digest, kwargs)
    return digest.hexdigest()


def cached_golden(fn):
    """Memoise a golden on disk under a content key.  Any cache trouble -- an
    unhashable argument, a corrupt or half-written file, a read-only directory --
    falls through to computing it, so the cache can only ever cost time."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        root = golden_cache_dir()
        path = None
        if root is not None:
            try:
                path = os.path.join(root, f"{fn.__name__}-{_golden_key(fn, args, kwargs)}.pt")
            except Exception:  # noqa: BLE001 - never let the cache break a gate
                path = None
        if path is not None and os.path.isfile(path):
            try:
                return torch.load(path, map_location="cpu", weights_only=False)
            except Exception:  # noqa: BLE001 - a truncated file just recomputes
                pass
        value = fn(*args, **kwargs)
        if path is not None:
            try:
                tmp = f"{path}.{os.getpid()}.tmp"
                torch.save(value, tmp)
                os.replace(tmp, path)  # atomic: a reader never sees a partial file
            except Exception:  # noqa: BLE001
                pass
        return value

    return wrapper


# ----------------------------------------------------------------- real inputs


def text_inputs(prompt, max_sequence_length: int = TOKENIZER_MAX_LENGTH):
    """``input_ids`` / ``attention_mask`` exactly as ``Flux2KleinPipeline.
    _get_qwen3_prompt_embeds`` builds them: chat template, then max-length padding."""
    tokenizer = load_tokenizer()
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)

    ids, masks = [], []
    for one in prompts:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": one}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        enc = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )
        ids.append(enc["input_ids"])
        masks.append(enc["attention_mask"])
    return torch.cat(ids, dim=0), torch.cat(masks, dim=0)


def chat_prompt_ids(prompt: str):
    """Unpadded chat-template ids -- the real input for the text-generation head."""
    tokenizer = load_tokenizer()
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(text, return_tensors="pt")["input_ids"]


def preprocess_image(image, height: int, width: int):
    """``Flux2ImageProcessor.preprocess`` -- the pipeline's own image front end."""
    return image_processor().preprocess(image, height=height, width=width, resize_mode="crop")


def make_latents(batch_size: int, height: int, width: int, seed: int, dtype=torch.bfloat16):
    """The pipeline's own noise shape, drawn once and handed to BOTH sides via
    ``latents=`` so TT and HF denoise the identical sample."""
    pipe = load_pipeline()
    latent_channels = load_transformer().config.in_channels // 4
    h = 2 * (int(height) // (pipe.vae_scale_factor * 2))
    w = 2 * (int(width) // (pipe.vae_scale_factor * 2))
    generator = torch.Generator("cpu").manual_seed(seed)
    return torch.randn((batch_size, latent_channels * 4, h // 2, w // 2), generator=generator, dtype=torch.float32).to(
        dtype
    )


def generation_config_json() -> dict:
    """`text_encoder/generation_config.json` read straight off disk -- the stop rule
    and the pad id are needed long before the 9 B checkpoint is worth loading."""
    if "generation_config" not in _CACHE:
        with open(os.path.join(snapshot_root(), "text_encoder", "generation_config.json")) as f:
            _CACHE["generation_config"] = json.load(f)
    return _CACHE["generation_config"]


def stop_token_ids() -> list[int]:
    """``generation_config.eos_token_id`` -- the text head's model-grounded stop rule."""
    eos = generation_config_json().get("eos_token_id")
    if eos is None:
        return []
    return [int(e) for e in eos] if isinstance(eos, (list, tuple)) else [int(eos)]


def pad_token_id() -> int:
    return int(generation_config_json().get("pad_token_id") or 0)


# --------------------------------------------------------------------- goldens


@torch.no_grad()
def _decode_latents(latents: torch.Tensor) -> torch.Tensor:
    """The pipeline's own final step, with `force_upcast` honoured.

    `Flux2KleinPipeline.__call__` ends with `self.vae.decode(latents)`, where the
    latents carry the TEXT ENCODER's dtype (bf16) while this VAE's config sets
    `force_upcast: true` and is therefore built in fp32 -- so the in-pipeline call
    raises "Input type (BFloat16) and bias type (float) should be the same".  Taking
    `output_type="latent"` and decoding here is the same computation with the
    documented upcast applied, and it returns the raw decoder sample, which is
    exactly what the TT decode returns.
    """
    return load_vae().decode(latents.to(torch.float32), return_dict=False)[0]


@cached_golden
@torch.no_grad()
def hf_text_to_image(prompt, *, height, width, num_inference_steps, latents, max_sequence_length):
    pipe = load_pipeline()
    out = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        latents=latents,
        max_sequence_length=max_sequence_length,
        output_type="latent",
        return_dict=False,
    )[0]
    return _decode_latents(out)


@cached_golden
@torch.no_grad()
def hf_image_edit(prompt, images, *, height, width, num_inference_steps, latents, max_sequence_length):
    pipe = load_pipeline()
    out = pipe(
        image=list(images),
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        latents=latents,
        max_sequence_length=max_sequence_length,
        output_type="latent",
        return_dict=False,
    )[0]
    return _decode_latents(out)


@cached_golden
@torch.no_grad()
def hf_latents(prompt, *, height, width, num_inference_steps, latents, max_sequence_length, images=None):
    """The pre-decode latents -- the joint the TT denoise loop must match."""
    pipe = load_pipeline()
    kwargs = dict(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        latents=latents,
        max_sequence_length=max_sequence_length,
        output_type="latent",
        return_dict=False,
    )
    if images:
        kwargs["image"] = list(images)
    return pipe(**kwargs)[0]


@cached_golden
@torch.no_grad()
def hf_prompt_embeds(prompt, max_sequence_length: int):
    pipe = load_pipeline()
    embeds, text_ids = pipe.encode_prompt(
        prompt=prompt,
        device=torch.device("cpu"),
        max_sequence_length=max_sequence_length,
        text_encoder_out_layers=TEXT_ENCODER_OUT_LAYERS,
    )
    return embeds, text_ids


@cached_golden
@torch.no_grad()
def hf_text_generation(prompt: str, max_new_tokens: int):
    """Greedy ``model.generate()`` under the model's own stop rule, capped to the
    SAME horizon the TT loop uses so the two sequences cannot diverge in length."""
    model = load_text_encoder()
    ids = chat_prompt_ids(prompt)
    out = model.generate(
        ids,
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        eos_token_id=stop_token_ids() or None,
        pad_token_id=model.generation_config.pad_token_id,
    )
    return out[:, ids.shape[1] :]


@cached_golden
@torch.no_grad()
def hf_text_generation_logits(prompt: str, max_new_tokens: int):
    """The reference's per-step logits row alongside the greedy ids, recomputed
    with the same no-cache prefix forward the TT loop performs."""
    model = load_text_encoder()
    ids = chat_prompt_ids(prompt)
    stops = set(stop_token_ids())
    logits_rows, new_ids = [], []
    for _ in range(max_new_tokens):
        logits = model(input_ids=ids, use_cache=False).logits[:, -1, :]
        logits_rows.append(logits.float())
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        new_ids.append(int(nxt.item()))
        ids = torch.cat([ids, nxt], dim=1)
        if new_ids[-1] in stops:
            break
    return torch.cat(logits_rows, dim=0), new_ids


@cached_golden
@torch.no_grad()
def hf_vae_roundtrip(pixel_values: torch.Tensor):
    vae = load_vae()
    posterior = vae.encode(pixel_values.to(torch.float32)).latent_dist
    return vae.decode(posterior.mode(), return_dict=False)[0], posterior.mode()


# ------------------------------------------------------------------------- PCC


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation of two flattened tensors, accumulated in float64.

    float64, not float32: a correlation is a ratio of sums over every element, and
    in fp32 the rounding on a multi-million-element tensor is enough to push the
    result ABOVE 1.  That is not a cosmetic error -- `assert_samples_are_distinct`
    reads a correlation of 1 as "these two rows are the same sample", so an fp32
    accumulation made the check reject the HF golden itself (measured on 1.5 M-element
    prompt embeddings: HF's own worst pairwise came out at 1.00015, against a true
    float64 value of 0.99977).  The extra cost is irrelevant next to a 9 B forward.
    """
    x = a.detach().to(torch.float64).flatten()
    y = b.detach().to(torch.float64).flatten()
    if x.numel() != y.numel():
        raise ValueError(f"PCC shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    if torch.equal(x, y):
        return 1.0
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if denom == 0:
        return 1.0 if x.norm() == y.norm() else 0.0
    # clamp only the last-ulp overshoot; a genuine >1 is impossible by Cauchy-Schwarz
    return float(min(1.0, max(-1.0, float((x @ y) / denom))))


# =========================================================== batch: 32 samples
#
# BATCH=32 means 32 INDEPENDENT samples per call -- different prompts, different
# noise, different reference images.  They share only the weights and the iteration
# count (same resolution => same flow-match schedule => the same timestep at every
# denoise step, which is why the timestep conditioning stays batch-1 and broadcasts).
#
# These are the canonical 32.  Both sides of every gate draw from the same list, so
# sample i of the TT batch is scored against sample i's OWN golden -- a pipeline that
# shape-supports B but emits 32 identical images fails on the distinctness check.

BATCH = 32

_SUBJECTS = (
    "a red apple",
    "a blue bicycle",
    "a green frog",
    "a yellow taxi",
    "a white teapot",
    "a black cat",
    "an orange pumpkin",
    "a silver kettle",
    "a purple orchid",
    "a brown owl",
    "a golden trumpet",
    "a grey elephant",
    "a pink flamingo",
    "a copper lantern",
    "a jade turtle",
    "a crimson kite",
)
_SETTINGS = (
    "on a wooden table",
    "in the rain",
    "beside a stone wall",
    "under a bright sky",
    "on a marble floor",
    "in a green field",
    "against a red curtain",
    "on a sandy beach",
)


def batch_prompts(n: int = BATCH) -> list[str]:
    """`n` DISTINCT real prompts -- subject x setting, so no two rows repeat."""
    out = []
    for i in range(int(n)):
        subject = _SUBJECTS[i % len(_SUBJECTS)]
        setting = _SETTINGS[(i // len(_SUBJECTS)) % len(_SETTINGS)]
        out.append(f"{subject} {setting}")
    if len(set(out)) != len(out):
        raise ValueError(f"batch_prompts({n}) is not distinct -- widen _SUBJECTS/_SETTINGS")
    return out


_QUESTIONS = (
    "Describe {} in one short sentence.",
    "What colour is {}?",
    "Name one use for {}.",
    "Write a short caption for {}.",
)


def batch_text_prompts(n: int = BATCH) -> list[str]:
    """`n` DISTINCT real chat prompts for the text->text head."""
    out = []
    for i in range(int(n)):
        question = _QUESTIONS[(i // len(_SUBJECTS)) % len(_QUESTIONS)]
        out.append(question.format(_SUBJECTS[i % len(_SUBJECTS)]))
    if len(set(out)) != len(out):
        raise ValueError(f"batch_text_prompts({n}) is not distinct")
    return out


def batch_latents(n: int, height: int, width: int, seed: int = 0):
    """One noise sample per row, each from its OWN generator seed, stacked on the
    leading axis.  Drawing `(n, ...)` from a single generator would also be
    independent, but per-row seeds make sample i reproducible on its own."""
    return torch.cat([make_latents(1, height, width, seed + i) for i in range(int(n))], dim=0)


def batch_images(n: int, size: int, *, phase: int = 0):
    """`n` DISTINCT synthetic images, generated the same way for both sides."""
    from PIL import Image

    out = []
    for i in range(int(n)):
        h = torch.linspace(0, 1, size).reshape(1, -1).expand(size, -1)
        v = torch.linspace(0, 1, size).reshape(-1, 1).expand(-1, size)
        k = 1.0 + i + phase
        r = (0.5 + 0.5 * torch.sin(k * 6.0 * h + 0.7 * i)) * 255.0
        g = (0.5 + 0.5 * torch.cos((k + 1.0) * 5.0 * v - 0.3 * i)) * 255.0
        b = (0.5 + 0.5 * torch.sin(3.0 * (h + v) + 0.9 * i)) * 255.0
        rgb = torch.stack([r, g, b], dim=-1).clamp(0, 255).to(torch.uint8).numpy()
        out.append(Image.fromarray(rgb, mode="RGB"))
    return out


def chat_prompt_ids_batch(prompts, pad_id: int | None = None):
    """`len(prompts)` chat-template rows LEFT-padded to one `(B, L)` block.

    Left padding, not right: it puts every row's real last token in the LAST column,
    so a single argmax row per step serves all rows and no row needs its own index.
    That is what HF `generate()` does for a batch, and both sides of the gate call
    this same function so the two batches are built identically.

    Returns `(input_ids, attention_mask, real_lengths)`.
    """
    pad = int(pad_id if pad_id is not None else pad_token_id())
    rows = [chat_prompt_ids(p)[0] for p in prompts]
    lengths = [int(r.shape[0]) for r in rows]
    width = max(lengths)
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, row in enumerate(rows):
        ids[i, width - row.shape[0] :] = row
        mask[i, width - row.shape[0] :] = 1
    return ids, mask, lengths


@cached_golden
@torch.no_grad()
def hf_text_generation_logits_batch(prompts, max_new_tokens: int, pad_id: int | None = None):
    """The batched reference for the text->text head: ONE left-padded greedy decode
    over `len(prompts)` independent streams, stopping when EVERY stream has hit a
    stop id (or the shared safety cap) -- exactly the rule the TT loop runs.

    Left padding is what `generate()` itself uses for a batch: it puts every row's
    real last token in the last column, so one argmax row per step serves all rows.
    Returns `(logits_rows, ids_per_row, prompt_lengths)` where `logits_rows[s]` is
    the `(B, vocab)` step-`s` logits and `ids_per_row[b]` stops at that row's own eos.
    """
    model = load_text_encoder()
    stops = set(stop_token_ids())
    pad = int(pad_id if pad_id is not None else pad_token_id())

    ids, mask, lengths = chat_prompt_ids_batch(prompts, pad_id=pad)
    n = len(lengths)

    out_ids = [[] for _ in range(n)]
    done = [False] * n
    logits_rows = []
    for _ in range(int(max_new_tokens)):
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, -1, :]
        logits_rows.append(logits.float())
        nxt = torch.argmax(logits, dim=-1)
        for i in range(n):
            if done[i]:
                continue
            out_ids[i].append(int(nxt[i]))
            if int(nxt[i]) in stops:
                done[i] = True
        # a finished row keeps being fed so the batch stays one program; its further
        # tokens are simply not recorded, which is what makes each row stop at its own eos
        ids = torch.cat([ids, nxt.reshape(-1, 1)], dim=1)
        mask = torch.cat([mask, torch.ones((n, 1), dtype=torch.long)], dim=1)
        if all(done):
            break
    return logits_rows, out_ids, lengths


def per_sample_pcc(got: torch.Tensor, golden: torch.Tensor) -> list[float]:
    """PCC of sample i against ITS OWN golden, for every leading-axis row."""
    if tuple(got.shape) != tuple(golden.shape):
        raise ValueError(f"batch PCC shape mismatch: {tuple(got.shape)} vs {tuple(golden.shape)}")
    return [pcc(got[i], golden[i]) for i in range(int(got.shape[0]))]


def assert_samples_are_distinct(x: torch.Tensor, *, tol: float = 0.9999) -> float:
    """A batch whose rows are all the same is a FAKE batch axis, and it would score a
    perfect per-sample PCC against a golden that was also computed row-identically.
    So the gate also requires the rows to differ: returns the WORST (highest)
    correlation between two distinct rows, and raises if it is above `tol`."""
    n = int(x.shape[0])
    if n < 2:
        return 0.0
    worst = max(pcc(x[i], x[j]) for i in range(n) for j in range(i + 1, n))
    if worst > tol:
        raise AssertionError(
            f"the {n} samples are not independent: two rows correlate at {worst} > {tol} "
            f"-- the batch axis is not carrying distinct samples"
        )
    return worst

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared plumbing for the `mistralai/Voxtral-4B-TTS-2603` end-to-end TTNN package.

Everything here is SETUP, never forward-path compute: the HF reference loader,
the real tokenizer, the 32-sample batch builder, a PCC helper, the graduated
stub importer and the Gate-2 invocation counter.

Source A (the hub repo) ships `params.json`, `consolidated.safetensors` and
`tekken.json` -- no `config.json`, so `AutoConfig`/`AutoModel` raise. Source B
ships the verified reconstruction of the reference
(`tests/pcc/_reference_loader.py`), which is what `load_reference_model()`
delegates to.
"""
from __future__ import annotations

import base64
import importlib
import json
import os
import sys
from functools import lru_cache

import torch

HF_MODEL_ID = "mistralai/Voxtral-4B-TTS-2603"

# Source B, the bring-up tool's output for this model.
BRINGUP_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "tt_transformers",
    "demo",
    "voxtral_4b_tts_2603",
)
STUB_PKG = "models.tt_transformers.demo.voxtral_4b_tts_2603._stubs"
CAPTURED_ROOT = os.path.join(BRINGUP_ROOT, "_captured")

# BATCH=32: 32 independent samples per pipeline call. A single sample wastes 31/32 of a
# 32-row matmul tile; filling it raises aggregate throughput ~32x at unchanged per-sample latency.
DEFAULT_BATCH = 32

# Real tokens per prompt. Short on purpose: the graduated blocks carry no KV cache, so every
# decode step recomputes the whole (grown) sequence.
DEFAULT_SEQ_LEN = 32


# --------------------------------------------------------------------------------------
# Source A: the reference model and its config
# --------------------------------------------------------------------------------------


def _bringup_pcc_dir() -> str:
    return os.path.join(BRINGUP_ROOT, "tests", "pcc")


@lru_cache(maxsize=1)
def _reference_loader_module():
    path = _bringup_pcc_dir()
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module("_reference_loader")


def load_reference_model(model_id: str = HF_MODEL_ID):
    """The HF golden: `MistralForCausalLM` rebuilt from the native checkpoint (Source A+B)."""
    return _reference_loader_module().load_reference_model(model_id)


def resolve_repo(model_id: str = HF_MODEL_ID) -> str:
    """Local directory holding params.json / consolidated.safetensors / tekken.json."""
    if os.path.isdir(model_id):
        return model_id
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id, allow_patterns=["params.json", "consolidated.safetensors", "tekken.json"])


def load_params(model_id: str = HF_MODEL_ID) -> dict:
    with open(os.path.join(resolve_repo(model_id), "params.json")) as f:
        return json.load(f)


# --------------------------------------------------------------------------------------
# Source A: the real tokenizer (tekken.json -> tiktoken)
# --------------------------------------------------------------------------------------


class TekkenTokenizer:
    """Mistral's Tekken v7 tokenizer, rebuilt on `tiktoken`.

    `mistral_common` is not installed in tt-metal's `python_env` (and the env has no pip), and
    `AutoTokenizer` cannot help because the repo ships no `tokenizer_config.json`. `tekken.json`
    carries everything needed: the split pattern, the ranked merge table, and the convention that a
    real token id is `rank + num_special_tokens`.
    """

    def __init__(self, tekken_path: str) -> None:
        import tiktoken

        with open(tekken_path) as f:
            spec = json.load(f)
        cfg = spec["config"]
        self.num_special = int(cfg["default_num_special_tokens"])
        self.vocab_size = int(cfg["default_vocab_size"])
        n_merges = self.vocab_size - self.num_special
        mergeable = {base64.b64decode(e["token_bytes"]): e["rank"] for e in spec["vocab"][:n_merges]}
        self._enc = tiktoken.Encoding(
            name="tekken_v7",
            pat_str=cfg["pattern"],
            mergeable_ranks=mergeable,
            special_tokens={},
        )
        self.bos_id = 1
        self.eos_id = 2

    def encode(self, text: str, bos: bool = True) -> list[int]:
        ids = [r + self.num_special for r in self._enc.encode(text, allowed_special=set())]
        return ([self.bos_id] + ids) if bos else ids

    def decode(self, ids) -> str:
        ranks = [int(i) - self.num_special for i in ids if int(i) >= self.num_special]
        return self._enc.decode(ranks) if ranks else ""


@lru_cache(maxsize=1)
def load_tokenizer(model_id: str = HF_MODEL_ID) -> TekkenTokenizer:
    return TekkenTokenizer(os.path.join(resolve_repo(model_id), "tekken.json"))


# --------------------------------------------------------------------------------------
# The real input: 32 INDEPENDENT prompts
# --------------------------------------------------------------------------------------

# 32 distinct English texts. Each is long enough that truncating to DEFAULT_SEQ_LEN real tokens
# never needs padding, so all 32 rows are genuine content and causality is untouched.
PROMPT_TEXTS = [
    "The lighthouse keeper wrote in his journal every evening, noting the colour of the water and the direction of the wind across the bay. Nothing else in the log ever varied, and that was the point of keeping it.",
    "In the quiet hours before dawn the bakery ovens were already warm, and the smell of bread drifted down the empty cobbled street outside. By six o'clock the first customers were queueing outside in the cold.",
    "She tuned the old radio slowly, sliding past static and foreign voices until a piano concerto emerged clearly from the crackling speaker. She left the dial exactly there for the rest of the evening.",
    "The expedition mapped the cave system for three weeks, marking every passage with chalk and recording the temperature at each junction. The final map took another two months to draw properly.",
    "Autumn arrived early that year, and the maple trees along the river turned a deep copper colour almost overnight, surprising everyone. Nobody could remember a season that had turned so quickly before.",
    "He repaired bicycles in a narrow shop behind the station, where spare wheels hung from the ceiling like the rings of a strange planet. Most of them had been hanging there since long before he bought the place.",
    "The letter had travelled for six months across two oceans before it finally reached the small farmhouse at the end of the gravel road. The stamps alone were worth more than the paper they were stuck to.",
    "Astronomers watched the comet approach for weeks, adjusting their instruments each night as its tail grew brighter against the dark sky. By the third week they could predict its position to within a degree.",
    "A violinist practised the same difficult passage for hours, stopping only when the light faded and the room became too dim to read music. The neighbours never once complained about the repetition.",
    "The archive basement held thousands of photographs, each one labelled in faded pencil with a date, a place, and sometimes only a first name. Sorting them properly would clearly take the whole of the coming winter.",
    "Rain fell steadily on the greenhouse roof while the gardener transplanted seedlings into deeper pots, humming an old song under her breath. The glass fogged over completely before she had finished the last tray.",
    "The ferry crossed the strait twice each morning, carrying commuters, crates of fish, and occasionally a nervous dog on a short leash. The crossing took nineteen minutes in good weather and rather longer otherwise.",
    "Engineers tested the bridge cables with careful instruments, measuring how the steel responded to the weight of traffic and to the cold. Their report ran to four hundred pages and recommended almost no changes.",
    "In the museum's back room a conservator removed varnish from a painting, revealing colours that nobody had seen for almost two centuries. The blue underneath turned out to be far brighter than anyone expected.",
    "The village held a festival every summer, with paper lanterns strung between the houses and long tables set out in the central square. Musicians came down from the city for it, and stayed until the morning.",
    "A cartographer redrew the coastline after the storm, because the sandbar had shifted nearly a hundred metres further east than before. The old chart was kept anyway, pinned to the wall beside the new one.",
    "He learned to cook from his grandmother, who measured nothing and explained everything twice, always in the middle of doing something else. He still cannot make the soup taste the way hers did.",
    "The observatory sat above the treeline, and on clear nights the students climbed the winding road carrying thermoses of strong coffee. The telescope itself was older than the building that housed it.",
    "Librarians catalogued the donated collection for months, discovering pressed flowers and train tickets tucked between the pages of novels. Someone had clearly used the books as a filing cabinet for decades.",
    "The potter worked quickly once the clay was centred, drawing the walls upward with steady pressure while the wheel turned beneath her hands. Each pot came off the wheel looking almost exactly like the last.",
    "Snow closed the mountain pass for a week, so the mail was carried in on skis by a postman who had done the same route for twenty years. He said the quiet was the only part he would genuinely miss.",
    "A biologist counted the nesting pairs along the cliff face, recording each sighting in a waterproof notebook tied to her wrist by string. The numbers were lower than last year, which worried her considerably.",
    "The orchestra rehearsed in an unheated hall, and between movements the musicians blew on their fingers and laughed about the temperature. By the final run-through nobody mentioned the cold at all.",
    "Workers restored the clock tower over two summers, replacing the escapement, regilding the hands, and finally ringing the bell at noon. The whole town came out into the square to hear it strike.",
    "She kept a small boat moored at the pier and sailed it alone on weekends, following the shoreline until the town disappeared behind her. On the way back the wind was always against her, which she preferred.",
    "The bookshop occupied three floors of a crooked building, and every staircase creaked in a slightly different key as customers climbed it. Regular visitors learned which steps to avoid if they wanted to browse unnoticed.",
    "Geologists drilled a core sample from the lakebed, reading centuries of pollen and ash in the thin bands of sediment they brought up. One dark band marked a fire that no written record had preserved.",
    "A blacksmith demonstrated the old techniques at the fair, and children watched the sparks leap while the metal changed from red to grey. The noise of the hammer carried right across the showground all afternoon.",
    "The train ran through farmland for an hour before the first suburbs appeared, low houses with gardens backing directly onto the tracks. Passengers who made the trip daily stopped noticing the view entirely.",
    "Translators argued for days about a single line of the poem, because the original word meant both a departure and a kind of forgiveness. In the end they printed both versions and let the reader decide.",
    "The bee keeper opened each hive slowly, reading the temper of the colony from its sound before lifting a single frame into the sunlight. The hum told her more than any inspection of the frames could have.",
    "Divers surveyed the wreck at forty metres, photographing the hull in overlapping strips so the whole ship could be reassembled on screen. Visibility was poor, so every strip had to be shot twice to be safe.",
]


def build_batch_inputs(batch: int = DEFAULT_BATCH, seq_len: int = DEFAULT_SEQ_LEN, model_id: str = HF_MODEL_ID):
    """The REAL pipeline input: `batch` independent prompts, each exactly `seq_len` real tokens.

    Returns ``(input_ids [batch, seq_len] int64, texts)``. No padding: every prompt text is long
    enough to truncate to `seq_len`, so every row is genuine content and the causal mask is the
    plain lower-triangular one. The rows are asserted pairwise distinct -- a batch of 32 copies
    would make the PCC gate meaningless.
    """
    tok = load_tokenizer(model_id)
    if batch > len(PROMPT_TEXTS):
        raise ValueError(f"only {len(PROMPT_TEXTS)} distinct prompts are defined, asked for {batch}")
    rows, texts = [], []
    for text in PROMPT_TEXTS[:batch]:
        ids = tok.encode(text, bos=True)
        if len(ids) < seq_len:
            raise ValueError(f"prompt tokenizes to {len(ids)} < seq_len={seq_len}: {text[:48]!r}")
        ids = ids[:seq_len]
        rows.append(ids)
        texts.append(tok.decode(ids[1:]))
    input_ids = torch.tensor(rows, dtype=torch.long)
    if len({tuple(r) for r in rows}) != batch:
        raise AssertionError("batch rows are not pairwise distinct")
    return input_ids, texts


# --------------------------------------------------------------------------------------
# Decode horizon (see e2e_plan.json -> task_heads[0].decode_horizon_rule)
# --------------------------------------------------------------------------------------

# No model signal exists for a generation length on this checkpoint: generation_config has no
# max_new_tokens, its max_length is the transformers library default (20) which is <= our real
# prompt length, and the tied text head never emits eos. 16 is chosen for lack of any model signal.
_FALLBACK_HORIZON = 16


def resolve_decode_horizon(hf_model, prompt_len: int) -> tuple[int, str]:
    """Return ``(horizon, provenance)`` for BOTH the TT loop and `model.generate()`.

    Priority: an explicit env override, then `generation_config.max_new_tokens`, then
    `max_length - prompt_len` when that is positive, then the documented fallback. The stop token
    is handled separately by the caller (break once every row has emitted `eos_token_id`); this is
    the safety cap that keeps a non-terminating run bounded.
    """
    env = os.environ.get("VOXTRAL_E2E_HORIZON")
    if env:
        return int(env), "env VOXTRAL_E2E_HORIZON"
    gc = getattr(hf_model, "generation_config", None)
    if gc is not None:
        mnt = getattr(gc, "max_new_tokens", None)
        if mnt:
            return int(mnt), "generation_config.max_new_tokens"
        ml = getattr(gc, "max_length", None)
        if ml and int(ml) - prompt_len > 0:
            return int(ml) - prompt_len, f"generation_config.max_length({ml}) - prompt_len({prompt_len})"
    return _FALLBACK_HORIZON, "fallback (no usable stop length in config/generation_config)"


def eos_token_id(hf_model):
    gc = getattr(hf_model, "generation_config", None)
    eos = getattr(gc, "eos_token_id", None) if gc is not None else None
    if eos is None:
        eos = getattr(hf_model.config, "eos_token_id", None)
    if isinstance(eos, (list, tuple)):
        return int(eos[0])
    return None if eos is None else int(eos)


# --------------------------------------------------------------------------------------
# Graduated stubs (Source B)
# --------------------------------------------------------------------------------------

GRADUATED_MODULES = (
    "token_embed",
    "rotary_embedding",
    "r_m_s_norm",
    "attention",
    "mlp",
    "m_l_p",
    "decoder_layer",
    "layer",
    "model",
    "decoder_head",
)


def import_stub(name: str):
    """Import a graduated stub module from Source B by component name."""
    if name not in GRADUATED_MODULES:
        raise KeyError(f"{name!r} is not a graduated module; graduated = {GRADUATED_MODULES}")
    return importlib.import_module(f"{STUB_PKG}.{name}")


def build_stub(name: str, device, torch_module):
    """Build a graduated stub through its `build(device, torch_module)` constructor."""
    return import_stub(name).build(device, torch_module)


def captured_golden_cache(component: str):
    """The bring-up tool's captured reference inputs/golden for a component."""
    path = os.path.join(CAPTURED_ROOT, component, "golden_cache_s0.pt")
    return torch.load(path, weights_only=False)


# --------------------------------------------------------------------------------------
# Gate 2: invocation counting
# --------------------------------------------------------------------------------------


class InvocationCounter:
    """Counts how many times each graduated stub's `__call__` actually ran.

    This is instrumentation on stubs that are already inside the real forward path -- it is NOT a
    coverage sweep. Nothing calls a stub for the sake of the counter.
    """

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def wrap(self, name: str, stub):
        counter = self

        class _Counted:
            __slots__ = ("_stub",)

            def __init__(self, inner):
                object.__setattr__(self, "_stub", inner)

            def __call__(self, *a, **kw):
                counter.counts[name] = counter.counts.get(name, 0) + 1
                return object.__getattribute__(self, "_stub")(*a, **kw)

            def __getattr__(self, item):
                return getattr(object.__getattribute__(self, "_stub"), item)

            @property
            def graduated_stub(self):
                return object.__getattribute__(self, "_stub")

        return _Counted(stub)

    def reset(self) -> None:
        self.counts = {}

    def __repr__(self) -> str:
        return f"InvocationCounter({self.counts})"


def unwrap(obj):
    """The underlying graduated stub instance behind an InvocationCounter wrapper (or `obj`)."""
    return getattr(obj, "graduated_stub", obj)


# --------------------------------------------------------------------------------------
# PCC
# --------------------------------------------------------------------------------------


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation between two tensors, flattened, in float64."""
    x = a.detach().to(torch.float64).flatten()
    y = b.detach().to(torch.float64).flatten()
    if x.numel() != y.numel():
        raise ValueError(f"shape mismatch for PCC: {tuple(a.shape)} vs {tuple(b.shape)}")
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x * x).sum() * (y * y).sum())
    if denom == 0:
        return 1.0 if torch.equal(x, y) else 0.0
    return float((x * y).sum() / denom)

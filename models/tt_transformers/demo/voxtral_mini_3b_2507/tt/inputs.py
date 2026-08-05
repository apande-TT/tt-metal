# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Real input assembly for Voxtral-Mini-3B-2507 (mistralai/Voxtral-Mini-3B-2507).

This module is the *single* place where the demo's token streams and mel features
are produced.  It is pure host code: no ttnn, no device, torch/numpy/transformers only.

Why we assemble the token stream by hand
----------------------------------------
`VoxtralProcessor.apply_chat_template` and `VoxtralProcessor.apply_transcription_request`
both route through `MistralCommonBackend` (mistral_common).  That package is deliberately
absent from this environment, so the tokenizer loads as a plain `TokenizersBackend` with
`chat_template is None` and `tokenizer.tokenizer.encode_transcription` does not exist.
We therefore emit the tekken control tokens ourselves.  The exact layouts below were
chosen *empirically* (see the module-level comments on AUDIO_CHAT_TEMPLATE and
TRANSCRIPTION_TEMPLATE) by running the real HF model over a candidate grid and keeping
the layout that produced coherent, on-topic text.

!! processor.tokenizer IS BROKEN FOR THIS CHECKPOINT -- DO NOT USE IT FOR TEXT !!
--------------------------------------------------------------------------------
transformers 5.12.1 converts `tekken.json` into a `tokenizers` BPE on the fly
("Converting tekken.json to tokenizer.json"), and the conversion is wrong in two ways:

  * regular vocab entries are given id == rank instead of id == rank + 1000
    (tekken reserves ids [0, 1000) for control tokens), so every text id is 1000 too low
    and collides with the control-token range; and
  * the byte-level pre-tokenizer loses spaces -- `tokenizer("hello world")` returns the
    pieces 'hell' + 'ow' + 'orld' (i.e. "helloworld").

Round trip proof:
    tokenizer.decode(tokenizer("Hello world, the capital of France is Paris.").input_ids)
    -> '<SPECIAL_72> ellow orld ,the capital of France is Paris <SPECIAL_46>'
    ('H' became id 72, '.' became id 46 -- raw byte values.)

Feeding those ids to the model produces fluent-looking nonsense, and decoding the model's
(correct) output with the same tokenizer makes a perfect transcript look like garbage.
So this module ships its own `TekkenTokenizer`: a tiktoken `Encoding` built directly from
`tekken.json` (pattern + base64 token_bytes + the +1000 control-token offset), which is
exactly what mistral_common's Tekkenizer does.  Only `processor.feature_extractor`
(WhisperFeatureExtractor) is taken from the HF processor -- that part is correct.

Uniform 30 s streams -- a deliberate design constraint
------------------------------------------------------
Every clip is truncated **or zero-padded to exactly 480000 samples (30 s @ 16 kHz)**
before mel extraction.  Consequences, all of them wanted for the TT bring-up:

  * each stream yields exactly ONE mel chunk of shape (128, 3000);
  * the Whisper-style audio tower therefore runs at one fixed shape, 1500 encoder
    frames per stream, which after the 4x reshape into the projector becomes exactly
    N_AUDIO_TOKENS_PER_CHUNK = 375 audio embeddings per stream;
  * the [AUDIO] placeholder run has the same length and the same *start index* in every
    stream, so the whole batch shares one prefill program shape and one shared decode
    position -- no ragged prefill, no per-user position bookkeeping, and the trace can
    be captured once.

Real audio is never trimmed silently: clips longer than 30 s are truncated (only
`obama.mp3` in the default set is longer) and shorter clips are right zero-padded, which
is exactly what `WhisperFeatureExtractor` would do internally anyway.

Token stream layouts
--------------------
  audio_chat    : <s> [INST] [BEGIN_AUDIO] [AUDIO]*375 <instruction> [/INST]
  transcription : <s> [INST] [BEGIN_AUDIO] [AUDIO]*375 lang:<xx> [/INST]

Public API
----------
  AUDIO_CLIPS, DEFAULT_INSTRUCTION, N_AUDIO_TOKENS_PER_CHUNK, AUDIO_TOKEN_ID
  BatchInputs
  load_audio_batch, build_audio_chat_inputs, build_transcription_inputs, build_inputs
  captured, get_processor, get_tokenizer
"""

from __future__ import annotations

import base64
import functools
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

HF_REPO_ID = "mistralai/Voxtral-Mini-3B-2507"
AUDIO_DATASET_REPO = "hf-internal-testing/dummy-audio-samples"

SAMPLING_RATE = 16000
CHUNK_SECONDS = 30.0
N_SAMPLES_PER_CHUNK = 480000  # feature_extractor.n_samples
N_MEL_BINS = 128  # feature_extractor.feature_size
N_MEL_FRAMES = 3000  # feature_extractor.nb_max_frames == max_source_positions
MAX_SOURCE_POSITIONS = 3000

N_AUDIO_TOKENS_PER_CHUNK = 375  # 1500 encoder frames / 4  (== 30 s * 12.5 Hz frame rate)

# tekken v7 control token ids (verified against tekken.json special_tokens)
BOS_ID = 1  # <s>
EOS_ID = 2  # </s>
INST_ID = 3  # [INST]
INST_END_ID = 4  # [/INST]
PAD_ID = 11  # <pad>
AUDIO_TOKEN_ID = 24  # [AUDIO]
BEGIN_AUDIO_ID = 25  # [BEGIN_AUDIO]
TRANSCRIBE_ID = 34  # [TRANSCRIBE]

#: The eight distinct clips used for the 8-stream batch, in canonical order.
AUDIO_CLIPS: list[str] = [
    "bcn_weather.mp3",
    "dude_where_is_my_car.wav",
    "fleur_es_sample.wav",
    "mary_had_lamb.mp3",
    "monte_cristo.flac",
    "obama_first_45_secs.mp3",
    "obama.mp3",
    "winning_call.mp3",
]

DEFAULT_INSTRUCTION = "What is this audio about? Answer briefly."

HEADS = ("audio_chat", "transcription")

# --- frozen templates -----------------------------------------------------------------
#
# Both were picked by running the real fp32 HF model over an 11-candidate grid on
# mary_had_lamb.mp3 (an Edison-phonograph narration ending in "Mary had a little lamb...")
# and bcn_weather.mp3 (a Barcelona temperature report), greedy, 32 new tokens.
#
# AUDIO_CHAT_TEMPLATE
#   <s> [INST] [BEGIN_AUDIO] [AUDIO]*375 <instruction> [/INST]
#   This is the layout mistral_common's InstructTokenizerV7 produces for a user message
#   whose content is [AudioChunk, TextChunk]: the audio chunk comes first, inside the
#   [INST]...[/INST] span, prefixed by [BEGIN_AUDIO].  It answers *about* the audio:
#     mary -> "The audio is about a poem, specifically the first words spoken in the
#              original Cornograph, which is a piece of practical poetry."
#     bcn  -> "The audio discusses the significant temperature change in Barcelona,
#              from 35 degrees Celsius to -20 degrees Celsius, over a 24-hour period."
#   The mirrored variant (<s> [INST] <instruction> [BEGIN_AUDIO] [AUDIO]*375 [/INST])
#   also works, so ordering is not load-bearing for quality; we keep audio-first because
#   it matches mistral_common AND it makes audio_start a constant (3) shared with the
#   transcription head, i.e. one prefill layout for both task heads.
AUDIO_CHAT_TEMPLATE = "<s> [INST] [BEGIN_AUDIO] [AUDIO]*375 {instruction} [/INST]"
#
# TRANSCRIPTION_TEMPLATE
#   <s> [INST] [BEGIN_AUDIO] [AUDIO]*375 lang:<xx> [/INST]
#   The decisive observation is that the bare instruct span WITHOUT a language hint
#   (<s> [INST] [BEGIN_AUDIO] [AUDIO]*375 [/INST]) does NOT transcribe -- it summarises:
#     mary -> "The first words spoken in the original Cornograph were a playful poem,
#              which begins with the line, \"Mary had a little lamb.\""
#   Adding the "lang:en" text run flips the model into verbatim transcription:
#     mary -> "The first words I spoke in the original Cornograph were a little piece of
#              practical poetry: \"Mary had a little lamb, its fleece was white as snow"
#     bcn  -> "Yesterday it was 35 degrees in Barcelona, but today the temperature will
#              go down to minus 20 degrees."
#   The [TRANSCRIBE] control token (id 34) is an alternative trigger -- `[/INST]
#   [TRANSCRIBE]` with no language hint also transcribes -- but every [TRANSCRIBE]
#   placement we tried (before [/INST], after [/INST], with or without lang) produced a
#   transcript no better than, and sometimes worse than, this one ("its fleece was quite
#   as slow" instead of "white as snow" for the [/INST]+[TRANSCRIBE] variants).  Since we
#   cannot check the true placement against mistral_common, we freeze the shortest
#   candidate that is correct on both probe clips and that keeps the two heads
#   structurally identical.  "lang:xx" is a *text* run (tekken ids [9909, 1058, 1262] for
#   "lang:en"), not a control token.
TRANSCRIPTION_TEMPLATE = "<s> [INST] [BEGIN_AUDIO] [AUDIO]*375 lang:{language} [/INST]"

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

_TT_DIR = Path(__file__).resolve().parent
DEMO_DIR = _TT_DIR.parent
CAPTURED_DIR = DEMO_DIR / "_captured"


# --------------------------------------------------------------------------------------
# Processor / audio loading
# --------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=2)
def get_processor(repo_id: str = HF_REPO_ID):
    """Cached `AutoProcessor`.

    Kept for API completeness / feature-extractor access.  Its `.tokenizer` is the broken
    TokenizersBackend -- use `get_tokenizer()` for anything text-related.
    """
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(repo_id)


@functools.lru_cache(maxsize=2)
def get_feature_extractor(repo_id: str = HF_REPO_ID):
    """Cached `WhisperFeatureExtractor` (loaded directly: skips the tekken conversion)."""
    from transformers import WhisperFeatureExtractor

    return WhisperFeatureExtractor.from_pretrained(repo_id)


class TekkenTokenizer:
    """Correct tekken v7 BPE, built straight from `tekken.json` with tiktoken.

    Replaces the broken `processor.tokenizer` (see module docstring).  Mirrors
    mistral_common's `Tekkenizer`:

      * the regular vocab is truncated to `default_vocab_size - default_num_special_tokens`
        (= 131072 - 1000 = 130072) entries;
      * token id = BPE rank + 1000, ids in [0, 1000) are the control tokens.
    """

    def __init__(self, tekken_path: str):
        import tiktoken

        with open(tekken_path) as f:
            spec = json.load(f)
        cfg = spec["config"]
        self.num_special_tokens: int = cfg["default_num_special_tokens"]
        self.vocab_size: int = cfg["default_vocab_size"]
        n_regular = self.vocab_size - self.num_special_tokens
        mergeable = {base64.b64decode(entry["token_bytes"]): entry["rank"] for entry in spec["vocab"][:n_regular]}
        self._enc = tiktoken.Encoding(
            name="tekken-voxtral",
            pat_str=cfg["pattern"],
            mergeable_ranks=mergeable,
            special_tokens={},
        )
        self.special_token_str: dict[int, str] = {e["rank"]: e["token_str"] for e in spec["special_tokens"]}

    def encode(self, text: str) -> list[int]:
        """Text -> ids. Never emits BOS/EOS or any control token."""
        return [r + self.num_special_tokens for r in self._enc.encode(text, disallowed_special=())]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        ids = [int(i) for i in (ids.tolist() if hasattr(ids, "tolist") else ids)]
        out: list[str] = []
        buf: list[int] = []

        def flush():
            if buf:
                out.append(
                    self._enc.decode_bytes([i - self.num_special_tokens for i in buf]).decode("utf-8", "replace")
                )
                buf.clear()

        for i in ids:
            if i < self.num_special_tokens:
                flush()
                if not skip_special_tokens:
                    out.append(self.special_token_str.get(i, f"<SPECIAL_{i}>"))
            else:
                buf.append(i)
        flush()
        return "".join(out)

    def batch_decode(self, rows, skip_special_tokens: bool = True) -> list[str]:
        return [self.decode(r, skip_special_tokens=skip_special_tokens) for r in rows]

    def id_to_piece(self, i: int) -> str:
        i = int(i)
        if i < self.num_special_tokens:
            return self.special_token_str.get(i, f"<SPECIAL_{i}>")
        return self._enc.decode_bytes([i - self.num_special_tokens]).decode("utf-8", "replace")


@functools.lru_cache(maxsize=2)
def get_tokenizer(repo_id: str = HF_REPO_ID) -> TekkenTokenizer:
    """Cached working tekken tokenizer (NOT `processor.tokenizer`)."""
    from huggingface_hub import hf_hub_download

    return TekkenTokenizer(hf_hub_download(repo_id=repo_id, filename="tekken.json"))


@functools.lru_cache(maxsize=32)
def clip_path(filename: str) -> str:
    """Local path of a clip from the `hf-internal-testing/dummy-audio-samples` dataset."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=AUDIO_DATASET_REPO, filename=filename, repo_type="dataset")


def _resolve_clips(n: int, clips: list[str] | None) -> list[str]:
    if clips is None:
        if not 1 <= n <= len(AUDIO_CLIPS):
            raise ValueError(f"n must be in [1, {len(AUDIO_CLIPS)}] when clips is None, got {n}")
        return list(AUDIO_CLIPS[:n])
    # An explicit clip list always wins over `n` (which keeps its default of 8).
    clips = list(clips)
    if not clips:
        raise ValueError("clips must be a non-empty list of dataset filenames")
    return clips


def load_audio_batch(n: int = 8, clips: list[str] | None = None, seconds: float = 30.0) -> list[np.ndarray]:
    """Load `n` distinct clips at 16 kHz mono, each truncated/zero-padded to exactly
    `seconds` * 16000 samples (float32).

    `seconds` must be <= 30.0: a longer window would produce more than one mel chunk and
    break the "one prefill shape / one decode position" invariant documented above.
    """
    from transformers.audio_utils import load_audio

    if seconds <= 0 or seconds > CHUNK_SECONDS:
        raise ValueError(f"seconds must be in (0, {CHUNK_SECONDS}], got {seconds}")
    target = int(round(seconds * SAMPLING_RATE))

    names = _resolve_clips(n, clips)
    out: list[np.ndarray] = []
    for name in names:
        arr = np.asarray(load_audio(clip_path(name), sampling_rate=SAMPLING_RATE), dtype=np.float32)
        if arr.ndim == 2:  # defensive: force mono
            arr = arr.mean(axis=-1)
        padded = np.zeros(target, dtype=np.float32)
        keep = min(arr.shape[0], target)
        padded[:keep] = arr[:keep]
        out.append(padded)
    return out


def build_input_features(arrays: list[np.ndarray]) -> torch.FloatTensor:
    """Mel features, mirroring `VoxtralProcessor._retrieve_input_features` exactly.

    Per-stream: WhisperFeatureExtractor -> (1, 128, 3000) -> reshape(128, -1, 3000)
    -> transpose(0, 1) -> (n_chunks, 128, 3000); with a 30 s input n_chunks == 1.
    """
    fe = get_feature_extractor()
    per_stream = []
    for arr in arrays:
        audio_inputs = fe(
            arr,
            sampling_rate=SAMPLING_RATE,
            padding=True,
            truncation=False,
            pad_to_multiple_of=N_SAMPLES_PER_CHUNK,
            return_tensors="pt",
        )
        feats = audio_inputs["input_features"].reshape(fe.feature_size, -1, MAX_SOURCE_POSITIONS).transpose(0, 1)
        if feats.shape != (1, N_MEL_BINS, N_MEL_FRAMES):
            raise AssertionError(
                f"expected exactly one ({N_MEL_BINS}, {N_MEL_FRAMES}) mel chunk per stream, got {tuple(feats.shape)}"
            )
        per_stream.append(feats)
    return torch.stack(per_stream).squeeze(1).to(torch.float32)


# --------------------------------------------------------------------------------------
# BatchInputs
# --------------------------------------------------------------------------------------


@dataclass
class BatchInputs:
    """One ready-to-run batch: identical prompt length across all streams."""

    head: str
    input_ids: torch.LongTensor  # [B, L]
    attention_mask: torch.LongTensor  # [B, L], all ones
    input_features: torch.FloatTensor  # [B, 128, 3000]
    audio_start: int  # index of the first [AUDIO] token
    n_audio_tokens: int  # 375
    prompt_len: int  # L
    clips: list[str]
    prompt_text: str
    meta: dict = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])

    def fingerprint(self) -> str:
        """Stable sha1 over everything that changes the model's answer."""
        h = hashlib.sha1()
        h.update(
            json.dumps(
                {
                    "head": self.head,
                    "prompt_text": self.prompt_text,
                    "clips": list(self.clips),
                    "prompt_len": int(self.prompt_len),
                    "audio_start": int(self.audio_start),
                    "n_audio_tokens": int(self.n_audio_tokens),
                    "shape": list(self.input_ids.shape),
                },
                sort_keys=True,
            ).encode()
        )
        h.update(self.input_ids.to(torch.int64).numpy().tobytes())
        return h.hexdigest()

    def describe(self) -> str:
        return (
            f"BatchInputs(head={self.head}, B={self.batch_size}, L={self.prompt_len}, "
            f"audio=[{self.audio_start}:{self.audio_start + self.n_audio_tokens}), "
            f"features={tuple(self.input_features.shape)})"
        )


# --------------------------------------------------------------------------------------
# Token stream assembly
# --------------------------------------------------------------------------------------


def encode_text(text: str) -> list[int]:
    """Tekken text ids, no BOS/EOS, no control tokens.

    Uses `TekkenTokenizer` (tiktoken over tekken.json), NOT `processor.tokenizer`, which
    is mis-converted for this checkpoint -- see the module docstring.  All ids returned
    here are >= 1000, so they can never be confused with the control tokens we splice in.
    """
    ids = get_tokenizer().encode(text)
    assert all(i >= 1000 for i in ids), "text encoding leaked into the control-token range"
    return ids


def _assemble(
    head: str,
    prefix_ids: list[int],
    suffix_ids: list[int],
    clips: list[str],
    prompt_text: str,
    arrays: list[np.ndarray] | None = None,
    seconds: float = 30.0,
) -> BatchInputs:
    """prefix_ids + [AUDIO]*375 + suffix_ids, replicated over the batch."""
    audio_start = len(prefix_ids)
    ids = prefix_ids + [AUDIO_TOKEN_ID] * N_AUDIO_TOKENS_PER_CHUNK + suffix_ids

    # --- structural assertions (these are the contract the TT pipeline relies on) ---
    assert ids[0] == BOS_ID, "stream must start with <s>"
    assert ids.count(BOS_ID) == 1, f"exactly one <s> expected, found {ids.count(BOS_ID)}"
    assert (
        ids.count(AUDIO_TOKEN_ID) == N_AUDIO_TOKENS_PER_CHUNK
    ), f"expected {N_AUDIO_TOKENS_PER_CHUNK} [AUDIO] tokens, found {ids.count(AUDIO_TOKEN_ID)}"
    audio_positions = [i for i, t in enumerate(ids) if t == AUDIO_TOKEN_ID]
    assert audio_positions == list(
        range(audio_start, audio_start + N_AUDIO_TOKENS_PER_CHUNK)
    ), "[AUDIO] tokens must be contiguous and start at audio_start"

    if arrays is None:
        arrays = load_audio_batch(n=len(clips), clips=clips, seconds=seconds)
    input_features = build_input_features(arrays)

    B = len(clips)
    assert input_features.shape == (B, N_MEL_BINS, N_MEL_FRAMES), tuple(input_features.shape)

    L = len(ids)
    input_ids = torch.tensor([ids] * B, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    assert input_ids.shape == (B, L)
    assert len({int(row.shape[0]) for row in input_ids}) == 1, "all streams must share one prompt_len"

    return BatchInputs(
        head=head,
        input_ids=input_ids,
        attention_mask=attention_mask,
        input_features=input_features,
        audio_start=audio_start,
        n_audio_tokens=N_AUDIO_TOKENS_PER_CHUNK,
        prompt_len=L,
        clips=list(clips),
        prompt_text=prompt_text,
        meta={"seconds": seconds},
    )


def build_audio_chat_inputs(
    instruction: str = DEFAULT_INSTRUCTION,
    n: int = 8,
    clips: list[str] | None = None,
    seconds: float = 30.0,
) -> BatchInputs:
    """`<s> [INST] [BEGIN_AUDIO] [AUDIO]*375 <instruction> [/INST]`"""
    names = _resolve_clips(n, clips)
    prefix = [BOS_ID, INST_ID, BEGIN_AUDIO_ID]
    suffix = encode_text(instruction) + [INST_END_ID]
    prompt_text = AUDIO_CHAT_TEMPLATE.format(instruction=instruction)
    return _assemble("audio_chat", prefix, suffix, names, prompt_text, seconds=seconds)


def build_transcription_inputs(
    language: str = "en",
    n: int = 8,
    clips: list[str] | None = None,
    seconds: float = 30.0,
) -> BatchInputs:
    """`<s> [INST] [BEGIN_AUDIO] [AUDIO]*375 lang:<xx> [/INST]`  (frozen; see module docstring)"""
    names = _resolve_clips(n, clips)
    prefix = [BOS_ID, INST_ID, BEGIN_AUDIO_ID]
    suffix = encode_text(f"lang:{language}") + [INST_END_ID]
    prompt_text = TRANSCRIPTION_TEMPLATE.format(language=language)
    return _assemble("transcription", prefix, suffix, names, prompt_text, seconds=seconds)


def build_inputs(head: str, n: int = 8, **kw) -> BatchInputs:
    """Dispatcher: head in {"audio_chat", "transcription"}."""
    if head == "audio_chat":
        return build_audio_chat_inputs(n=n, **kw)
    if head == "transcription":
        return build_transcription_inputs(n=n, **kw)
    raise ValueError(f"unknown head {head!r}, expected one of {HEADS}")


# --------------------------------------------------------------------------------------
# Captured-tensor access (written by the bring-up decomposition pass)
# --------------------------------------------------------------------------------------


def captured(component: str, which: str):
    """`torch.load` of `_captured/<component>/<which>.pt`, which in {args, kwargs, output}."""
    if which not in ("args", "kwargs", "output"):
        raise ValueError(f"which must be one of args/kwargs/output, got {which!r}")
    path = CAPTURED_DIR / component / f"{which}.pt"
    if not path.exists():
        available = sorted(p.name for p in CAPTURED_DIR.iterdir()) if CAPTURED_DIR.exists() else []
        raise FileNotFoundError(f"{path} not found; captured components: {available}")
    return torch.load(path, map_location="cpu", weights_only=False)


# --------------------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    tok = get_tokenizer()

    probe = "Hello world, the capital of France is Paris."
    rt = tok.decode(tok.encode(probe))
    print("tekken round-trip:", repr(rt))
    assert rt == probe, "TekkenTokenizer round-trip failed"
    print("tekken lang:en    :", tok.encode("lang:en"))

    for head in HEADS:
        bi = build_inputs(head, n=8)
        print("=" * 88)
        print(head, "->", bi.describe())
        print("  prompt_text :", bi.prompt_text)
        print("  fingerprint :", bi.fingerprint())
        print("  clips       :", bi.clips)
        head_ids = bi.input_ids[0, : bi.audio_start].tolist()
        tail_ids = bi.input_ids[0, bi.audio_start + bi.n_audio_tokens :].tolist()
        print("  prefix ids  :", head_ids, "->", [tok.id_to_piece(i) for i in head_ids])
        print("  suffix ids  :", tail_ids, "->", [tok.id_to_piece(i) for i in tail_ids])
        print(
            "  features    :",
            tuple(bi.input_features.shape),
            bi.input_features.dtype,
            "mean",
            float(bi.input_features.mean()),
            "std",
            float(bi.input_features.std()),
        )
        assert bi.input_ids.shape == (8, bi.prompt_len)
        assert bi.attention_mask.sum() == 8 * bi.prompt_len
        assert bi.input_features.shape == (8, 128, 3000)
        # all streams identical token-wise, distinct audio-wise
        assert (bi.input_ids == bi.input_ids[0]).all()
        assert not torch.allclose(bi.input_features[0], bi.input_features[1])

    arrs = load_audio_batch(8)
    print("=" * 88)
    for name, a in zip(AUDIO_CLIPS, arrs):
        print(f"  {name:26s} samples={a.shape[0]} nonzero={int((a != 0).sum())} peak={float(np.abs(a).max()):.4f}")
    print("OK")

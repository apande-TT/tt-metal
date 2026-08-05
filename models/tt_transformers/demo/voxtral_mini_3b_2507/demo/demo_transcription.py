# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Call 2 demo — speech transcription for `mistralai/Voxtral-Mini-3B-2507`.

Feeds 8 real audio clips through the SAME TTNN pipeline the e2e test uses (`tt/pipeline.py`) with
the transcription prompt, and prints the transcript for each stream.

    ./python_env/bin/python -m models.tt_transformers.demo.voxtral_mini_3b_2507.demo.demo_transcription
    ./python_env/bin/python -m models.tt_transformers.demo.voxtral_mini_3b_2507.demo.demo_transcription \\
        --audio mary_had_lamb.mp3 --batch-size 1 --language en --max-new-tokens 64 --compare-hf

There is no second copy of the wiring here: the demo opens a device, calls `build_pipeline` and
`pipe.run_transcription`, and prints. Exits non-zero on any failure (including a `--compare-hf`
PCC below the gate threshold).
"""

from __future__ import annotations

import argparse
import inspect
import sys
import traceback

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.tt_transformers.demo.voxtral_mini_3b_2507.tt.inputs import AUDIO_CLIPS, build_transcription_inputs
from models.tt_transformers.demo.voxtral_mini_3b_2507.tt.pipeline import build_pipeline
from models.tt_transformers.demo.voxtral_mini_3b_2507.tt.reference import hf_reference

HEAD = "transcription"
L1_SMALL_SIZE = 24576
TRACE_REGION_SIZE = 23887872
PCC_THRESHOLD = 0.95


def _supported_kwargs(fn, **kwargs) -> dict:
    """Keep only the kwargs `fn` actually accepts (it may take **kwargs, in which case: all)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _resolve_clips(requested, batch_size: int) -> list[str]:
    """`batch_size` clip names, cycling the requested list (or the 8 built-ins) as needed."""
    base = list(requested) if requested else list(AUDIO_CLIPS)
    if not base:
        raise ValueError("no audio clips: pass --audio or make tt.inputs.AUDIO_CLIPS non-empty")
    if batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {batch_size}")
    if len(base) != batch_size:
        print(f"[demo] {len(base)} clip(s) given, batch-size {batch_size}: truncating/cycling to fit")
    return [base[i % len(base)] for i in range(batch_size)]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="demo_transcription",
        description="Voxtral-Mini-3B-2507 speech transcription (audio -> transcript) on Tenstorrent.",
    )
    parser.add_argument(
        "--audio",
        action="append",
        default=None,
        metavar="CLIP",
        help="audio clip to transcribe (repeatable); default = the 8 built-in clips from tt.inputs.AUDIO_CLIPS",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="number of streams to decode together")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="greedy decode budget per stream")
    parser.add_argument(
        "--language",
        default=None,
        help="transcription language tag (e.g. en); default = the frozen prompt in tt/inputs.py",
    )
    parser.add_argument(
        "--compare-hf",
        action="store_true",
        help="also compute the HF golden and print the aggregate PCC plus the per-stream text diff",
    )
    return parser.parse_args(argv)


def _compare_hf(batch, res, max_new_tokens: int) -> float:
    """Print the HF-vs-TT transcript diff and per-stream PCC; return the aggregate PCC."""
    golden = hf_reference(HEAD, batch, max_new_tokens=max_new_tokens, cache=True)
    tt = res.logits.detach().to(torch.float32)
    hf = golden.logits.detach().to(torch.float32)
    n = min(tt.shape[1], hf.shape[1])
    _, aggregate = comp_pcc(hf[:, :n], tt[:, :n], PCC_THRESHOLD)
    aggregate = float(aggregate)

    print("\n" + "-" * 88)
    print("HF golden vs TT transcript")
    print("-" * 88)
    for i, clip in enumerate(batch.clips):
        _, stream_pcc = comp_pcc(hf[i, :n], tt[i, :n], PCC_THRESHOLD)
        print(f"[stream {i}] {clip}")
        print(f"    HF : {golden.texts[i]!r}")
        print(f"    TT : {res.texts[i]!r}")
        print(f"    PCC: {float(stream_pcc)}")
    print(f"e2e PCC={aggregate}  (threshold {PCC_THRESHOLD}, {n} decode steps)")
    return aggregate


def main(argv=None) -> int:
    args = parse_args(argv)
    device = None
    try:
        clips = _resolve_clips(args.audio, args.batch_size)
        print(f"[demo] head={HEAD} clips={clips} max_new_tokens={args.max_new_tokens}")

        # tt/inputs.py names the stream count `n`; `batch_size` is a drift-tolerant alias that
        # _supported_kwargs drops when the builder does not declare it. `language` is only
        # forwarded when the user set it, so the frozen default prompt stays in tt/inputs.py.
        kwargs = {"n": args.batch_size, "batch_size": args.batch_size, "clips": clips}
        if args.language is not None:
            kwargs["language"] = args.language
        batch = build_transcription_inputs(**_supported_kwargs(build_transcription_inputs, **kwargs))
        if batch.batch_size != args.batch_size:
            raise RuntimeError(f"input builder produced {batch.batch_size} streams, asked for {args.batch_size}")
        print(f"[demo] prompt: {batch.prompt_text!r}")

        device = ttnn.open_device(device_id=0, l1_small_size=L1_SMALL_SIZE, trace_region_size=TRACE_REGION_SIZE)
        try:
            print("[demo] building the pipeline (HF weights + graduated stub uploads; takes minutes)...")
            pipe = build_pipeline(device)
            res = pipe.run_transcription(batch, max_new_tokens=args.max_new_tokens)

            print("\n" + "=" * 88)
            print(f"Voxtral-Mini-3B-2507 — transcription ({len(batch.clips)} streams)")
            print("=" * 88)
            for i, clip in enumerate(batch.clips):
                print(f"[stream {i}] clip       : {clip}")
                print(f"[stream {i}] transcript : {res.texts[i]}")
                print(
                    f"[stream {i}] tokens     : {res.lengths[i]} generated, " f"stopped_on_eos={res.stopped_on_eos[i]}"
                )
                print("-" * 88)

            if args.compare_hf:
                aggregate = _compare_hf(batch, res, args.max_new_tokens)
                if not aggregate >= PCC_THRESHOLD:
                    print(f"[demo] FAIL: e2e PCC {aggregate} < {PCC_THRESHOLD}", file=sys.stderr)
                    return 1
        finally:
            ttnn.close_device(device)
    except Exception:  # noqa: BLE001 -- a demo must exit non-zero, not raise a bare traceback
        traceback.print_exc()
        return 1

    print("[demo] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

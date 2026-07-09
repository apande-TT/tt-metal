# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Runnable demo for `mistralai/Voxtral-Mini-3B-2507` — audio -> text on Tenstorrent.

Loads real processor input (a 16 kHz waveform + a text prompt), runs the SHARED
chained TTNN pipeline (`tt/pipeline.py` — the exact code the e2e test asserts on),
and prints the generated text.

Run:
    cd /home/ttuser/tt-metal
    TT_METAL_HOME=/home/ttuser/tt-metal \
    PYTHONPATH=/home/ttuser/tt-metal-pr46283:/home/ttuser/tt-metal \
    /home/ttuser/tt-metal/python_env/bin/python -m \
        models.demos.hf_eager.voxtral_mini_3b_2507.demo.demo_transcribe --max-new-tokens 16
"""
from __future__ import annotations

import argparse

import numpy as np

import ttnn
from models.demos.hf_eager.voxtral_mini_3b_2507.tt import pipeline as P


def _load_audio_16k(path):
    """Load an audio file (wav/flac/mp3/...) as a 16 kHz mono float32 waveform."""
    import librosa
    import soundfile as sf

    wav, sr = sf.read(path)
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return wav


def main():
    ap = argparse.ArgumentParser(description="Voxtral-Mini-3B audio->text TTNN demo")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument(
        "--audio",
        type=str,
        default=None,
        help="path to a real audio file (wav/flac/mp3). If omitted, a synthetic tone is used.",
    )
    ap.add_argument("--language", type=str, default="en")
    ap.add_argument("--seconds", type=float, default=2.0, help="synthetic waveform length when --audio is omitted")
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args()

    device = ttnn.open_device(device_id=args.device_id, l1_small_size=24576)
    try:
        print("Loading Voxtral model + processor ...", flush=True)
        model = P.load_hf_model()
        proc = P.load_processor()
        audio = _load_audio_16k(args.audio) if args.audio else None
        input_ids, input_features, n_audio, atid, language = P.build_inputs(
            proc, model, audio=audio, seconds=args.seconds, language=args.language
        )
        print(f"Audio: {args.audio or f'synthetic {args.seconds}s tone'} | language={language}", flush=True)
        print(
            f"Input: {input_ids.shape[1]} tokens ({n_audio} audio placeholders), "
            f"input_features={tuple(input_features.shape)}",
            flush=True,
        )

        pipe = P.VoxtralTTPipeline(device, model)
        # stop_at_eos: emit only the real transcription, not low-confidence filler
        # past end-of-speech (which diverges from the reference at any precision).
        out = pipe.run(input_ids, input_features, max_new_tokens=args.max_new_tokens, stop_at_eos=True)

        text = proc.tokenizer.decode(out["tt_tokens"], skip_special_tokens=True)
        print("\n================ GENERATED (TTNN) ================", flush=True)
        print(f"tokens: {out['tt_tokens']}", flush=True)
        print(f"text  : {text!r}", flush=True)
        print("==================================================", flush=True)
        print(f"graduated stubs invoked: {sorted(out['invoked'])}", flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()

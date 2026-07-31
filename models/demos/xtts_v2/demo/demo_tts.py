# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Runnable TTS demo for coqui/XTTS-v2 on the Tenstorrent 8-chip mesh.

Runs the SHARED chained TTNN pipeline (tt/pipeline.py) — the exact same forward
the e2e test asserts — to synthesize speech for DECODE_BATCH (4) independent
streams at once: 4 sentences, each with its own speaker reference, decoded through
ONE batched GPT program, and writes one .wav per stream.

    python -m models.demos.xtts_v2.demo.demo_tts --out-dir /tmp/xtts_tt
    python -m models.demos.xtts_v2.demo.demo_tts --batch 1 \
        --text "It took me quite a long time to develop a voice." --out /tmp/xtts_tt.wav
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch  # noqa: F401  (torch tensors flow through the pipeline API)

import ttnn
from models.demos.xtts_v2.tt import pipeline as P


def _open_mesh():
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        # l1_small_size: the vocoder's native conv1d/conv_transpose2d and the speaker
        # encoder's conv2d run a sliding-window/halo gather that allocates from the
        # dedicated L1_SMALL pool, which is 0 B unless reserved here.
        dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=131072)
        print("[demo] opened 8-chip mesh (TP=8 x DP=1)")
        return dev, True
    except Exception as e:  # pragma: no cover - single-device fallback
        print(f"[demo] mesh open failed ({type(e).__name__}: {e}); falling back to single device")
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:
            pass
        return ttnn.open_device(device_id=0, l1_small_size=131072), False


def _close(dev, is_mesh):
    if is_mesh:
        ttnn.close_mesh_device(dev)
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:
            pass
    else:
        ttnn.close_device(dev)


def main(argv=None):
    ap = argparse.ArgumentParser(description="XTTS-v2 TTNN text-to-speech demo (batched)")
    ap.add_argument("--text", action="append", default=None,
                    help="text to synthesize; repeat to give one per decode stream "
                         "(default: the 4 built-in stream sentences)")
    ap.add_argument("--speaker", action="append", default=None,
                    help="speaker-reference wav NAME shipped in the XTTS-v2 repo "
                         "(e.g. en_sample.wav); repeat for one per stream")
    ap.add_argument("--batch", type=int, default=P.DECODE_BATCH,
                    help=f"independent decode streams (default {P.DECODE_BATCH})")
    ap.add_argument("--language", default="en")
    ap.add_argument("--horizon", type=int, default=None,
                    help="audio-code decode steps (default: the vocoder's consumed "
                         "length; capped by the config's max audio tokens)")
    ap.add_argument("--out", default=None, help="output .wav path (single-stream runs)")
    ap.add_argument("--out-dir", default="/tmp/xtts_tt", help="output dir for per-stream .wav")
    args = ap.parse_args(argv)

    dev, is_mesh = _open_mesh()
    try:
        model = P.load_reference_model()
        pipe = P.build_pipeline(dev, model=model, batch=args.batch)

        # real input, encoded by the model's own front end: XTTS tokenizer for the text,
        # 16 kHz view of the reference clip for the speaker encoder and its 22.05 kHz
        # DVAE conditioning mel.
        cond_mel, ref_wav, text_ids, _codes = P.build_streams(
            model=model, texts=args.text, speakers=args.speaker,
            batch=args.batch, language=args.language)
        print(f"[demo] {args.batch} stream(s): text={tuple(text_ids.shape)} "
              f"cond_mel={tuple(cond_mel.shape)} ref_wav={tuple(ref_wav.shape)}")

        out = pipe.run_tts(cond_mel, ref_wav, text_ids, audio_codes=None,
                           horizon=args.horizon, generate=True)

        os.makedirs(args.out_dir, exist_ok=True)
        for b in range(int(out["batch"])):
            wav = out["waveforms"][b].reshape(-1).numpy().astype(np.float32)
            codes = out["codes_gen"][b].reshape(-1).tolist()
            path = args.out if (args.out and int(out["batch"]) == 1) \
                else os.path.join(args.out_dir, f"stream{b}.wav")
            print(f"[demo] stream {b}: audio codes={codes} waveform={wav.shape[0]} samples "
                  f"({wav.shape[0] / 24000:.2f}s @ 24 kHz)")
            try:
                from scipy.io import wavfile

                wavfile.write(path, 24000, wav)
                print(f"[demo]   wrote {path}")
            except Exception as e:  # pragma: no cover
                print(f"[demo]   could not write wav ({e})")
        print(f"[demo] invoked stubs: {sorted(out['invoked'])}")
    finally:
        _close(dev, is_mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main())

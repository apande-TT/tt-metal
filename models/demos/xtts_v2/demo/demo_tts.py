# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Runnable TTS demo for coqui/XTTS-v2 on the Tenstorrent 8-chip mesh.

Runs the SHARED chained TTNN pipeline (tt/pipeline.py) — the exact same forward
the e2e test asserts — to synthesize speech from a text string + a speaker
reference audio, and writes the waveform to a .wav file.

    python -m models.demos.xtts_v2.demo.demo_tts \
        --text "It took me quite a long time to develop a voice." \
        --horizon 40 --out /tmp/xtts_tt.wav
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

import ttnn
from models.demos.xtts_v2.tt import pipeline as P


def _captured(name):
    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.normpath(os.path.join(here, "..", "_captured", name))
    a = torch.load(os.path.join(base, "args.pt"), map_location="cpu", weights_only=False)
    try:
        k = torch.load(os.path.join(base, "kwargs.pt"), map_location="cpu", weights_only=False)
    except Exception:
        k = {}
    return (list(a) if isinstance(a, (list, tuple)) else [a]), dict(k or {})


def _open_mesh():
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        # l1_small_size: the vocoder's native conv1d/conv_transpose2d and the speaker
        # encoder's conv2d run a sliding-window/halo gather that allocates from the
        # dedicated L1_SMALL pool, which is 0 B unless reserved here.
        dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=32768)
        print("[demo] opened 8-chip mesh (TP=8 x DP=1)")
        return dev, True
    except Exception as e:  # pragma: no cover - single-device fallback
        print(f"[demo] mesh open failed ({type(e).__name__}: {e}); falling back to single device")
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:
            pass
        return ttnn.open_device(device_id=0), False


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
    ap = argparse.ArgumentParser(description="XTTS-v2 TTNN text-to-speech demo")
    ap.add_argument("--text", default="It took me quite a long time to develop a voice.",
                    help="text to synthesize (tokenized only when --tokenize is set; "
                         "otherwise the captured reference tokens are used)")
    ap.add_argument("--language", default="en")
    ap.add_argument("--horizon", type=int, default=40, help="max audio-code decode steps")
    ap.add_argument("--out", default="/tmp/xtts_tt.wav", help="output .wav path")
    ap.add_argument("--tokenize", action="store_true",
                    help="tokenize --text with the XTTS tokenizer (else use captured tokens)")
    args = ap.parse_args(argv)

    dev, is_mesh = _open_mesh()
    try:
        model = P.load_reference_model()
        pipe = P.build_pipeline(dev, model=model)

        # inputs: DVAE conditioning mel + text tokens (captured golden case) + 16kHz ref wav
        ce_a, _ = _captured("conditioning_encoder")
        g_a, _ = _captured("g_p_t")
        cond_mel = ce_a[0].float()
        text_inputs = g_a[0]
        if args.tokenize:
            try:
                ids = model.tokenizer.encode(args.text.strip().lower(), lang=args.language)
                text_inputs = torch.IntTensor(ids).unsqueeze(0)
                print(f"[demo] tokenized '{args.text}' -> {tuple(text_inputs.shape)} tokens")
            except Exception as e:
                print(f"[demo] tokenize failed ({e}); using captured reference tokens")
        ref_wav = P.load_reference_audio_16k()

        out = pipe.run_tts(cond_mel, ref_wav, text_inputs, audio_codes=None,
                           horizon=args.horizon, generate=True)
        wav = out["waveform"].reshape(-1).numpy().astype(np.float32)
        print(f"[demo] generated audio codes: {out['codes_gen'].reshape(-1).tolist()}")
        print(f"[demo] waveform shape={tuple(out['waveform'].shape)} @ 24 kHz")
        print(f"[demo] invoked stubs: {sorted(out['invoked'])}")

        try:
            from scipy.io import wavfile
            wavfile.write(args.out, 24000, wav)
            print(f"[demo] wrote {args.out} ({wav.shape[0]} samples, {wav.shape[0]/24000:.2f}s)")
        except Exception as e:
            print(f"[demo] could not write wav ({e})")
    finally:
        _close(dev, is_mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main())

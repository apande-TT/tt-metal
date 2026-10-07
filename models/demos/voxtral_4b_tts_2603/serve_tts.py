# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Minimal OpenAI-compatible /v1/audio/speech server for Voxtral-4B-TTS on Tenstorrent.

Wraps the model's OWN on-device TTS pipeline -- the same calls as demo_text_to_speech
(build_voice_prompt -> build_pipeline -> stage_voice -> run_text_to_speech -> trim_to_end) -- behind
an HTTP endpoint, because vLLM's text API cannot carry audio. Stdlib-only (no fastapi/uvicorn dep).

ARBITRARY INPUT AT BATCH 32. The model was tuned for batch 32 at ONE prompt width (the 18-token
SPEECH_TEXTS); any other width overflows the hand-placed L1. So the server keeps that exact shape:
each request is tokenized and split into <=18-token chunks, the chunks are fanned across the 32
rows (short chunks padded to 18 and closed by a per-row additive mask in prefill AND decode, dummy
rows filling the batch to 32), one batch-32 forward runs per 32 chunks, and each request's chunk
waveforms are concatenated back in order. The mask is validated: additive pad-mask == dropping the
pad keys at PCC 1.0, and a padded row's prefill hidden is neighbour-independent at PCC 0.9996.

    TT_METAL_HOME=... python -m models.demos.voxtral_4b_tts_2603.serve_tts --port 20000
    curl -s -X POST localhost:20000/v1/audio/speech -d '{"input":"Hello from Tenstorrent."}' -o out.wav
"""

from __future__ import annotations

import argparse
import io
import json
import threading
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
import ttnn

from models.demos.voxtral_4b_tts_2603.tt import common, pipeline

_LOCK = threading.Lock()
_STATE: dict = {}

BATCH = int(common.DEFAULT_BATCH)  # 32 -- the tuned batch
WIDTH = 18  # the tuned body width (SPEECH_TEXTS); every chunk is padded to this


def _init(device_id: int = 0) -> None:
    common.use_all_cpu_threads()
    hf = common.load_reference_model()
    mf, _prov = common.resolve_max_frames(hf)
    dev = ttnn.open_device(
        device_id=device_id,
        l1_small_size=24576,
        trace_region_size=200 * 1024 * 1024,
        num_command_queues=1,
    )
    tok = common.load_tokenizer(common.HF_MODEL_ID)
    # A filler body for the dummy rows that pad a partial batch up to 32 (its output is discarded).
    filler = tok.encode("Yes.", bos=False)[:WIDTH] or [0]
    _STATE.update(hf=hf, max_frames=mf, device=dev, tok=tok, filler=filler, pipes={}, sr=None)


def _pipe(prompt_len: int):
    """A batch-32 pipeline for a given prompt length, cached (a voice's n_audio sets the length)."""
    p = _STATE["pipes"].get(prompt_len)
    if p is None:
        p = pipeline.build_pipeline(
            _STATE["device"],
            model=_STATE["hf"],
            heads=("text_to_speech",),
            layers=None,
            batch=BATCH,
            kv_capacity=pipeline.tts_kv_capacity(prompt_len, _STATE["max_frames"]),
        )
        _STATE["pipes"][prompt_len] = p
    return p


def _wav_bytes(samples, sr) -> bytes:
    clipped = torch.clamp(samples.reshape(-1), -1.0, 1.0)
    pcm = (clipped * 32767.0).to(torch.int16).numpy().tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as h:
        h.setnchannels(1)
        h.setsampwidth(2)
        h.setframerate(int(sr))
        h.writeframes(pcm)
    return buf.getvalue()


def _chunk_bodies(text: str):
    """Tokenize `text` and split into <=WIDTH-token bodies, on token boundaries.

    Token-boundary chunking is exact and never exceeds the tuned width; each chunk is one batch
    row, and a request's chunks are concatenated back in order after synthesis.
    """
    tok = _STATE["tok"]
    ids = tok.encode(text, bos=False)
    if not ids:
        ids = _STATE["filler"]
    return [ids[i : i + WIDTH] for i in range(0, len(ids), WIDTH)]


def _run_bodies(bodies, voice):
    """Synthesize a list of <=WIDTH-token bodies, returning one waveform per body (in order)."""
    waves = []
    for start in range(0, len(bodies), BATCH):
        chunk = [list(b) for b in bodies[start : start + BATCH]]
        nreal = len(chunk)
        chunk += [list(_STATE["filler"])] * (BATCH - nreal)  # dummy rows -> discarded
        input_ids, audio_mask, emb, pad_mask = common.build_voice_prompt(
            ["x"] * BATCH, voice, bodies=chunk, pad=True, pad_to=WIDTH, return_pad_mask=True
        )
        pipe = _pipe(int(input_ids.shape[-1]))
        v = pipe.stage_voice(audio_mask, emb, input_ids=input_ids)
        result = pipe.run_text_to_speech(
            input_ids=input_ids, max_frames=_STATE["max_frames"], voice=v, pad_mask=pad_mask
        )
        rows = pipeline.trim_to_end(result)
        _STATE["sr"] = result["sampling_rate"]
        waves.extend(rows[:nreal])
    return waves


def _synthesize(text: str, voice: str) -> bytes:
    bodies = _chunk_bodies(text)
    waves = _run_bodies(bodies, voice)
    samples = torch.cat([torch.as_tensor(w).reshape(-1).float() for w in waves]) if waves else torch.zeros(1)
    return _wav_bytes(samples, _STATE["sr"] or 24000)


class _Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, obj) -> None:
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": common.HF_MODEL_ID, "object": "model"}]})
        elif self.path.rstrip("/") in ("", "/health", "/healthz"):
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/v1/audio/speech":
            self._json(404, {"error": "only POST /v1/audio/speech is supported"})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            text = (body.get("input") or body.get("text") or "").strip()
            voice = body.get("voice") or common.DEFAULT_VOICE
            if not text:
                self._json(400, {"error": "missing 'input'"})
                return
            with _LOCK:
                wav = _synthesize(text, voice)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav)))
            self.end_headers()
            self.wfile.write(wav)
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            self._json(500, {"error": str(e)})

    def log_message(self, *a) -> None:  # quiet
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Voxtral-4B-TTS OpenAI /v1/audio/speech server (TTNN)")
    ap.add_argument("--port", type=int, default=20000)
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args(argv)
    print("[serve_tts] loading model + opening device (first start compiles kernels)…", flush=True)
    _init(args.device_id)
    print(f"[serve_tts] ready — POST http://127.0.0.1:{args.port}/v1/audio/speech  body {{'input': '…'}}", flush=True)
    try:
        # Single-threaded ON PURPOSE: ttnn/tt-metal is not thread-safe and the device is bound to the
        # thread that opened it, so device-open and every synthesis must run on this one thread.
        # Requests serialize (TTS is device-serialized anyway); a threaded server corrupts L1 alloc.
        HTTPServer(("0.0.0.0", args.port), _Handler).serve_forever()
    finally:
        try:
            ttnn.close_device(_STATE.get("device"))
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

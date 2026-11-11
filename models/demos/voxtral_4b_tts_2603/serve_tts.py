# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Minimal OpenAI-compatible /v1/audio/speech server for Voxtral-4B-TTS on Tenstorrent.

Wraps the model's OWN on-device TTS pipeline -- the same calls as demo_text_to_speech
(build_voice_prompt -> build_pipeline -> stage_voice -> run_text_to_speech -> trim_to_end) -- behind
an HTTP endpoint, because vLLM's text API cannot carry audio. Stdlib-only (no fastapi/uvicorn dep).

ARBITRARY INPUT AT BATCH 32. The model was tuned for batch 32 at ONE prompt width (the 18-token
SPEECH_TEXTS); any other width overflows the hand-placed L1. So the server keeps that exact shape:
each request is tokenized and split into <=18-token chunks, the chunks are fanned across the 32
rows (short chunks padded to 18, their pad columns closed in prefill AND decode, dummy rows filling
the batch to 32), one batch-32 forward runs per 32 chunks, and each request's chunk waveforms are
concatenated back in order. A forward stops once its real rows have emitted end_audio; the dummy
rows are not waited for. The voice sets the prompt width too, so only the voices whose embedding is
as long as the default's are served.

CONCURRENT REQUESTS SHARE A BATCH. One worker thread owns the device and runs every forward; the
HTTP threads only tokenize, queue their chunks and wait. Each forward takes up to 32 queued chunks
of one voice, from as many requests as are waiting, so N concurrent requests cost about one batch
wall instead of N.

    TT_METAL_HOME=... python -m models.demos.voxtral_4b_tts_2603.serve_tts --port 20000
    curl -s -X POST localhost:20000/v1/audio/speech -d '{"input":"Hello from Tenstorrent."}' -o out.wav
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import ttnn

from models.demos.voxtral_4b_tts_2603.tt import common, pipeline

# How `tt serve` starts this server (tt_hw_planner publish-hf packages it as tt-model's http-server kind):
# the command with {host}/{port} placeholders, the log line printed once it serves, the device mesh it
# opens (device 0 only) and the packages it needs.
SERVING_HTTP_COMMAND = [
    "python",
    "-m",
    "models.demos.voxtral_4b_tts_2603.serve_tts",
    "--host",
    "{host}",
    "--port",
    "{port}",
]
SERVING_HTTP_READY_LINE = "[serve_tts] ready"
SERVING_HTTP_MESH = "(1, 1)"
# What the server imports beyond tt-metal's own install (the reference loader, the tokenizer, the hub).
SERVING_HTTP_PACKAGES = ["transformers", "safetensors", "tiktoken", "huggingface_hub"]

_STATE: dict = {}

BATCH = int(common.DEFAULT_BATCH)  # 32 -- the tuned batch
WIDTH = 18  # the tuned body width (SPEECH_TEXTS); every chunk is padded to this
WARMUP_TEXT = "Hello from Tenstorrent."
GATHER_S = 0.3  # how long an idle device waits for more requests to share a batch


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
    _STATE.update(hf=hf, max_frames=mf, device=dev, tok=tok, filler=filler, pipes={}, sr=None, voices=servable_voices())


def servable_voices() -> list:
    """The presets whose embedding is as long as the default's: any other length changes the prompt
    width the L1 placement was tuned for."""
    voices = common.available_voices()
    return sorted(v for v, n in voices.items() if n == voices[common.DEFAULT_VOICE])


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


def _run_batch(bodies, voice):
    """ONE batch-32 forward over up to BATCH bodies of one voice; one waveform per body, in order."""
    nreal = len(bodies)
    chunk = [list(b) for b in bodies] + [list(_STATE["filler"])] * (BATCH - nreal)  # dummy rows -> discarded
    input_ids, audio_mask, emb, pad_mask = common.build_voice_prompt(
        ["x"] * BATCH, voice, bodies=chunk, pad=True, pad_to=WIDTH, return_pad_mask=True
    )
    pipe = _pipe(int(input_ids.shape[-1]))
    v = pipe.stage_voice(audio_mask, emb, input_ids=input_ids)
    result = pipe.run_text_to_speech(
        input_ids=input_ids, max_frames=_STATE["max_frames"], voice=v, pad_mask=pad_mask, live_rows=nreal
    )
    _STATE["sr"] = result["sampling_rate"]
    return pipeline.trim_to_end(result)[:nreal]


class _Request:
    """One /v1/audio/speech request: its chunks, the waveforms filled in as batches finish."""

    def __init__(self, bodies, voice):
        self.bodies, self.voice = bodies, voice
        self.waves = [None] * len(bodies)
        self.left = len(bodies)
        self.error = None
        self.done = threading.Event()


class _Batcher:
    """The device thread: opens the device, then runs every forward, each over the queued chunks."""

    def __init__(self, device_id: int):
        self._device_id = device_id
        self._cv = threading.Condition()
        self._queue = collections.deque()  # (request, chunk index), in arrival order
        self.ready = threading.Event()
        self.init_error = None
        threading.Thread(target=self._loop, name="voxtral-device", daemon=True).start()

    def submit(self, bodies, voice) -> _Request:
        req = _Request(bodies, voice)
        with self._cv:
            self._queue.extend((req, i) for i in range(len(bodies)))
            self._cv.notify()
        return req

    def _take(self):
        """Up to BATCH queued chunks sharing the oldest chunk's voice (a batch carries one voice)."""
        with self._cv:
            while not self._queue:
                self._cv.wait()
            # Clients that start together arrive over a few hundred ms; let them fill one batch.
            deadline = time.time() + GATHER_S
            while len(self._queue) < BATCH and time.time() < deadline:
                self._cv.wait(deadline - time.time())
            voice = self._queue[0][0].voice
            picked, rest = [], collections.deque()
            while self._queue:
                item = self._queue.popleft()
                (picked if len(picked) < BATCH and item[0].voice == voice else rest).append(item)
            self._queue = rest
        return voice, picked

    def _loop(self) -> None:
        try:
            _init(self._device_id)
            t0 = time.time()
            _run_batch(_chunk_bodies(WARMUP_TEXT), common.DEFAULT_VOICE)  # compile + build before serving
            print(f"[serve_tts] warm-up synthesis done in {time.time() - t0:.1f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            self.init_error = e
            self.ready.set()
            raise
        self.ready.set()
        while True:
            voice, picked = self._take()
            try:
                waves = _run_batch([req.bodies[i] for req, i in picked], voice)
            except Exception as e:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                waves, error = [None] * len(picked), e
            else:
                error = None
            for (req, i), w in zip(picked, waves):
                req.waves[i] = w
                req.error = req.error or error
                req.left -= 1
                if req.left == 0:
                    req.done.set()


def _synthesize(text: str, voice: str) -> bytes:
    req = _STATE["batcher"].submit(_chunk_bodies(text), voice)
    req.done.wait()
    if req.error is not None:
        raise req.error
    samples = torch.cat([torch.as_tensor(w).reshape(-1).float() for w in req.waves]) if req.waves else torch.zeros(1)
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
            if voice not in _STATE["voices"]:
                self._json(400, {"error": f"voice {voice!r} is not served; use one of {_STATE['voices']}"})
                return
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


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    # A full batch is 32 clients at once; socketserver's default backlog of 5 resets the rest.
    request_queue_size = 128


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Voxtral-4B-TTS OpenAI /v1/audio/speech server (TTNN)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=20000)
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args(argv)
    print("[serve_tts] loading model + opening device (first start compiles kernels)…", flush=True)
    # ttnn/tt-metal is not thread-safe and the device is bound to the thread that opened it, so the
    # batcher's thread opens it and runs every forward; the HTTP threads never touch ttnn.
    batcher = _STATE["batcher"] = _Batcher(args.device_id)
    batcher.ready.wait()
    if batcher.init_error is not None:
        raise SystemExit(f"[serve_tts] start-up failed: {batcher.init_error}")
    print(
        f"{SERVING_HTTP_READY_LINE} — POST http://{args.host}:{args.port}/v1/audio/speech  body {{'input': '…'}}",
        flush=True,
    )
    _Server((args.host, args.port), _Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

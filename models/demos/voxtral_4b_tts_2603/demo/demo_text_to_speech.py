# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Call 1 demo: text -> speech on Tenstorrent hardware.

    python -m models.demos.voxtral_4b_tts_2603.demo.demo_text_to_speech \
        --text "Paris is a beautiful city!" --out-dir /tmp/voxtral_wav

Runs the SAME `pipeline.run_text_to_speech` the e2e test asserts on -- there is one copy of the
wiring, so a green test guarantees this demo works.

VOICE. The model card advertises 20 preset voices, and `tekken.json` records each one's audio-
token count, but the voice prompts themselves are AUDIO CODES that this repo does not ship, and
the checkpoint carries no codec ENCODER to derive them from a reference waveform. `--voice` is
accepted for interface parity and the demo prints that the output is unconditioned.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import common, pipeline


def write_wav(path, samples, sampling_rate):
    """A 16-bit PCM WAV, written with the stdlib so the demo needs no audio dependency."""
    import wave

    clipped = torch.clamp(samples.reshape(-1), -1.0, 1.0)
    pcm = (clipped * 32767.0).to(torch.int16).numpy().tobytes()
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sampling_rate))
        handle.writeframes(pcm)
    return len(pcm) // 2


def main(argv=None):
    parser = argparse.ArgumentParser(description="Voxtral-4B-TTS-2603 text-to-speech on TTNN")
    parser.add_argument(
        "--text",
        action="append",
        default=None,
        help="a prompt to speak; repeat for more. Default: the package's 32 prompts.",
    )
    parser.add_argument("--texts-file", default=None, help="one prompt per line")
    parser.add_argument(
        "--voice",
        default="casual_male",
        help="accepted for interface parity; the OSS checkpoint ships no voice prompts",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=common.DEFAULT_BATCH,
        help=f"independent samples per call (default {common.DEFAULT_BATCH})",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="safety cap in 12.5 Hz frames; default is the codec's own ceiling"
    )
    parser.add_argument(
        "--layers", type=int, default=None, help="cap the depth of EVERY repeated stack (None = every layer)"
    )
    parser.add_argument("--out-dir", default="/tmp/voxtral_4b_tts_2603_wav")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--compare", action="store_true", help="also run the HF golden and print the per-sample PCC")
    args = parser.parse_args(argv)

    texts = list(args.text or [])
    if args.texts_file:
        with open(args.texts_file) as handle:
            texts += [line.strip() for line in handle if line.strip()]

    common.use_all_cpu_threads()
    hf_model = common.load_reference_model()

    device = ttnn.open_device(
        device_id=args.device_id,
        l1_small_size=24576,
        trace_region_size=200 * 1024 * 1024,
        num_command_queues=1,
    )
    try:
        # The cache has to hold the whole run: the voice block, the text, and one slot per frame.
        # Asked for BEFORE the pipeline is built, because the cache is allocated at build time.
        _frames = args.max_frames or 256
        _need = None
        try:
            # The SAME texts the run will speak. The package's prompts are truncated to a common
            # token length before use, and sizing from the untruncated ones raises on the ragged
            # widths -- which this swallowed, leaving the default 128 and a prefill that could not
            # fit its own prompt.
            _texts = texts if texts else common.build_batch_inputs(batch=args.batch)[1]
            _ids, _, _ = common.build_voice_prompt(_texts, args.voice)
            _need = int(_ids.shape[-1]) + int(_frames) + 8
            print(f"kv cache sized for prompt {int(_ids.shape[-1])} + {int(_frames)} frames -> {_need}")
        except Exception as exc:  # noqa: BLE001 -- unvoiced falls back to the stack's own default
            print(f"kv cache: could not size from the prompt ({type(exc).__name__}: {exc})")
        pipe = pipeline.build_pipeline(
            device,
            model=hf_model,
            heads=("text_to_speech",),
            layers=args.layers,
            batch=args.batch,
            voice=args.voice,
            kv_capacity=_need,
        )
        if texts:
            tok = common.load_tokenizer()
            begin = common.begin_audio_token_id()
            length = min(len(tok.encode(t, bos=True)) for t in texts)
            rows = [tok.encode(t, bos=True)[:length] + [begin] for t in texts]
            input_ids = torch.tensor(rows, dtype=torch.long)
            spoken = texts
        else:
            input_ids, spoken = pipe.default_inputs(batch=args.batch)

        max_frames = args.max_frames
        if max_frames is None:
            max_frames, provenance = common.resolve_max_frames(hf_model, gate=False)
            print(f"max_frames {max_frames}  <- {provenance}")

        # THE VOICE, which the repo does ship. `voice_embedding/<id>.pt` is one of 20 presets in
        # the checkpoint repo; the prompt is rebuilt with the [AUDIO] placeholder block those rows
        # substitute into. Without it the model has no speaker to imitate and emits babble.
        audio_mask = voice_embedding = None
        try:
            input_ids, audio_mask, voice_embedding = common.build_voice_prompt(spoken, args.voice)
            print(f"voice={args.voice!r} conditioned on {voice_embedding.shape[0]} audio tokens")
        except Exception as exc:  # noqa: BLE001 -- an unavailable voice must not hide the text path
            print(f"voice={args.voice!r} UNAVAILABLE ({type(exc).__name__}: {exc}); running unconditioned")
        print(f"batch={input_ids.shape[0]} prompt_tokens={input_ids.shape[1]} max_frames={max_frames}")

        result = pipe.run_text_to_speech(
            input_ids=input_ids,
            max_frames=max_frames,
            gate=False,
            audio_mask=audio_mask,
            voice_embedding=voice_embedding,
        )

        print(f"frames decoded: {result['frames_decoded']}  ({result['stop_reason']})")
        os.makedirs(args.out_dir, exist_ok=True)
        # EACH ROW AT ITS OWN END. The decode loop runs until every row has emitted end_audio, so
        # the batch is as long as its longest sentence; writing the full length gives every shorter
        # sample its speech followed by whatever came after its stop token.
        _ends = result.get("end_frame") or []
        _per_frame = int(result["waveform"].shape[-1] // max(1, result["frames_decoded"]))
        for i in range(result["waveform"].shape[0]):
            path = os.path.join(args.out_dir, f"sample_{i:02d}.wav")
            wav = result["waveform"][i]
            if i < len(_ends) and _ends[i] >= 0:
                keep = (int(_ends[i]) + 1) * _per_frame
                if 0 < keep < wav.shape[-1]:
                    wav = wav[..., :keep]
            n = write_wav(path, wav, result["sampling_rate"])
            print(f"  {path}  {n} samples  {n / result['sampling_rate']:.2f} s  <- {spoken[i][:60]!r}")

        if args.compare:
            from models.demos.voxtral_4b_tts_2603.reference import golden

            hf = golden.hf_reference_text_to_speech(
                hf_model, input_ids, result["x0"], result["cfg_alpha"], result["max_frames"]
            )
            scores = [common.pcc(result["waveform"][i], hf["waveform"][i]) for i in range(result["waveform"].shape[0])]
            print(f"codes identical: {bool(torch.equal(result['codes'], hf['codes']))}")
            print(f"e2e PCC={min(scores)}")
        return 0
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())

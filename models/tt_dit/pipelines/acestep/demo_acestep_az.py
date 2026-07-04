# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Phase 5 CLI — ACE-Step A→Z demo: prompt + lyrics + reference WAV → music WAV.

Full TT hot path (Phase 5 signoff): **TT Qwen3 text** + **TT DiT** + **TT Oobleck VAE**;
host reference encode only; tokenizer B+D (no LM planner). Production defaults:
``--infer-steps 30``, ``--guidance-scale 7.0``, ``--audio-duration 30``, ``--shift 3.0``.

Run (device gate — serialize with ``flock /tmp/tt_ace_device.lock``):

    bash docs/acestep-az-phase5-run.sh

Or:

    cd /local/ttuser/dvartanians/ace/tt-metal
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) ARCH_NAME=blackhole
    flock /tmp/tt_ace_device.lock ./python_env/bin/python -m \\
        models.tt_dit.pipelines.acestep.demo_acestep_az \\
        --prompt "smooth jazz pop, female lead vocal, warm piano, soft drums, lounge, 90 bpm" \\
        --lyrics "..." \\
        --reference /tmp/ref_kaazoom_25s.wav \\
        --output /tmp/az_phase5_signoff.wav \\
        --infer-steps 30 --guidance-scale 7.0 --audio-duration 30 --shift 3.0 \\
        --use-tt-vae --use-tt-text-encode --no-traced

Agent log: ``/tmp/acestep_agent_5.log``
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.hf_eager.acestep_v15_base.tt.vae_host import save_wav
from models.tt_dit.pipelines.acestep.lm_planner import default_lm_variant, default_use_lm_planner
from models.tt_dit.pipelines.acestep.pipeline_acestep import AceStepPipeline
from models.tt_dit.pipelines.acestep.text_encode_tt import default_use_tt_text_encode

LOG_PATH = "/tmp/acestep_agent_5.log"
DEFAULT_DEVICE_ID = 0
DEVICE_PARAMS = {"l1_small_size": 24576}


def _configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(LOG_PATH, level="DEBUG", enqueue=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ACE-Step A→Z demo: text prompt + lyrics + reference WAV → output WAV",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Caption / style prompt")
    parser.add_argument("--lyrics", type=str, default="", help="Lyrics text (optional)")
    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Path to reference WAV for timbre / cover conditioning",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/az_final.wav",
        help="Output WAV path",
    )
    parser.add_argument(
        "--infer-steps",
        type=int,
        default=30,
        help="Flow-matching ODE steps (default: 30 production; use 4 for fast smoke)",
    )
    parser.add_argument(
        "--traced",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable trace + 2-CQ on DiT hot path (Phase 6; default: False)",
    )
    parser.add_argument(
        "--use-tt-vae",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="TT Oobleck decode on device (default: True; use --no-use-tt-vae for host)",
    )
    parser.add_argument(
        "--audio-duration",
        type=float,
        default=30.0,
        help="Output duration in seconds for cover generation (default: 30; independent of reference length)",
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=1.0,
        help="ODE timestep shift (turbo/diffusers default: 3.0; base default: 1.0)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.0,
        help="Classifier-free guidance scale (HF default: 7.0; use 1.0 to disable CFG)",
    )
    parser.add_argument(
        "--use-tt-text-encode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="TT Qwen3-Embedding on device (Phase 2C; default from ACESTEP_USE_TT_TEXT_ENCODE)",
    )
    parser.add_argument(
        "--use-lm-planner",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="5Hz LM planner replaces Call B tokenizer (Phase 7; default from ACESTEP_USE_LM_PLANNER)",
    )
    parser.add_argument(
        "--lm-model",
        type=str,
        default=None,
        choices=["0.6B", "1.7B", "4B"],
        help="LM planner variant (default: 1.7B or ACESTEP_LM_PLANNER_MODEL)",
    )
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed")
    parser.add_argument("--device-id", type=int, default=DEFAULT_DEVICE_ID, help="TT device id")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    ref_path = Path(args.reference)
    if not ref_path.is_file():
        raise FileNotFoundError(f"--reference must be an existing WAV file: {ref_path}")

    out_path = Path(args.output)
    if out_path.parent and not out_path.parent.exists():
        raise FileNotFoundError(f"--output parent directory does not exist: {out_path.parent}")

    if args.infer_steps < 1:
        raise ValueError(f"--infer-steps must be >= 1, got {args.infer_steps}")

    if args.audio_duration <= 0:
        raise ValueError(f"--audio-duration must be > 0, got {args.audio_duration}")

    if args.shift <= 0:
        raise ValueError(f"--shift must be > 0, got {args.shift}")


def _run_pipeline(args: argparse.Namespace) -> dict:
    """Run AceStepPipeline latent gen + VAE decode → waveform."""
    torch.manual_seed(args.seed)

    device = ttnn.open_device(device_id=args.device_id, **DEVICE_PARAMS)
    try:
        use_tt = args.use_tt_text_encode if args.use_tt_text_encode is not None else default_use_tt_text_encode()
        use_lm = args.use_lm_planner if args.use_lm_planner is not None else default_use_lm_planner()
        use_vae = args.use_tt_vae if args.use_tt_vae is not None else True
        lm_model = args.lm_model if args.lm_model is not None else default_lm_variant()
        logger.info(
            "Opening AceStepPipeline (infer_steps={}, traced={}, use_tt_vae={}, use_tt_text_encode={}, "
            "use_lm_planner={}, lm_model={}, guidance_scale={}, audio_duration={}s, shift={})",
            args.infer_steps,
            args.traced,
            use_vae,
            use_tt,
            use_lm,
            lm_model,
            args.guidance_scale,
            args.audio_duration,
            args.shift,
        )
        pipe = AceStepPipeline.create_pipeline(
            mesh_device=device,
            num_inference_steps=args.infer_steps,
            guidance_scale=args.guidance_scale,
            cfg_enabled=args.guidance_scale > 1.0,
            audio_duration=args.audio_duration,
            shift=args.shift,
            use_tt_text_encode=(
                args.use_tt_text_encode if args.use_tt_text_encode is not None else default_use_tt_text_encode()
            ),
            use_lm_planner=use_lm,
            lm_model=lm_model,
        )

        t0 = time.perf_counter()
        result = pipe(
            prompts=[args.prompt],
            lyrics=args.lyrics,
            reference_audio=args.reference,
            num_inference_steps=args.infer_steps,
            seed=args.seed,
            traced=args.traced,
            return_waveform=True,
            use_tt_vae=args.use_tt_vae,
        )
        e2e_s = time.perf_counter() - t0
    finally:
        ttnn.close_device(device)

    if not isinstance(result, dict) or "waveform" not in result:
        raise RuntimeError(
            "AceStepPipeline(return_waveform=True) must return dict with 'waveform'; "
            "check pipeline_acestep.py integration"
        )

    result["e2e_s"] = e2e_s
    return result


def main() -> int:
    _configure_logging()
    args = _parse_args()

    try:
        _validate_args(args)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("{}", exc)
        return 2

    logger.info("ACE-Step A→Z demo starting")
    logger.info(
        "Live inputs: prompt={!r}, lyrics_len={}, reference={}",
        args.prompt,
        len(args.lyrics),
        args.reference,
    )
    logger.debug("CLI args: {}", vars(args))

    try:
        result = _run_pipeline(args)
    except Exception:
        logger.exception("Pipeline run failed")
        return 1

    waveform = result["waveform"]
    save_wav(args.output, waveform)

    vae_decode_s = result.get("vae_decode_s", float("nan"))
    e2e_s = result.get("e2e_s", float("nan"))
    latent_gen_s = e2e_s - vae_decode_s if vae_decode_s == vae_decode_s else float("nan")

    logger.info("PHASE_5 latent_gen_s={:.4f}", latent_gen_s)
    logger.info("PHASE_5 vae_decode_s={:.4f}", vae_decode_s)
    logger.info("PHASE_5 e2e_music_s={:.4f}", e2e_s)
    logger.info("PHASE_5 output_wav={}", args.output)

    print(f"PHASE_5 latent_gen_s={latent_gen_s:.4f}", flush=True)
    print(f"PHASE_5 vae_decode_s={vae_decode_s:.4f}", flush=True)
    print(f"PHASE_5 e2e_music_s={e2e_s:.4f}", flush=True)
    print(f"PHASE_5 output_wav={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

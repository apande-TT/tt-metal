# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end traced ACE-Step v1.5 music performance (latent gen + VAE decode).

Extends ``test_e2e_perf_traced_acestep.py`` with a VAE decode step after traced
latent generation and reports end-to-end music wall time.

Manual Phase C commands
-----------------------
Shared env (from repo root):

    cd /local/ttuser/dvartanians/ace/tt-metal
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) ARCH_NAME=blackhole
    flock -n /tmp/tt_ace_device.lock echo FREE || echo BUSY

Phase C full stack — TT Oobleck VAE on device (default):

    flock /tmp/tt_ace_device.lock ./python_env/bin/python -m pytest \\
      models/tt_dit/tests/models/acestep/test_e2e_music_perf_traced_acestep.py \\
      -k 1x1 -s -v --timeout=3600

Optional listenable output is written to /tmp/acestep_phase_c.wav on the final run.
Force host VAE fallback: ``ACESTEP_USE_TT_VAE=0``.
"""

from __future__ import annotations

import statistics

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.hf_eager.acestep_v15_base.tt.common import GATE_CONFIG
from models.demos.hf_eager.acestep_v15_base.tt.traced_decoder import _device_supports_2cq
from models.perf.benchmarking_utils import BenchmarkProfiler

from ....pipelines.acestep.audio_decode import (
    decode_latents_to_waveform,
    default_use_tt_vae,
    save_waveform_if_requested,
)
from ....pipelines.acestep.pipeline_acestep import AceStepPipeline
from ....pipelines.events import profiler_event_callback

INFER_STEPS = GATE_CONFIG["infer_steps"]
NUM_PERF_RUNS = 4
SEED = GATE_CONFIG["seed"]
A100_E2E_MUSIC_REFERENCE_S = 2.0
PHASE_C_WAV_PATH = "/tmp/acestep_phase_c.wav"

TRACE_DEVICE_PARAMS = {
    "l1_small_size": 24576,
    "num_command_queues": 2,
    "trace_region_size": 50_000_000,
}


def _print_music_e2e_table(
    *,
    mesh_device: ttnn.MeshDevice,
    infer_steps: int,
    num_perf_runs: int,
    use_2cq: bool,
    use_tt_vae: bool,
    latent_gen_times: list[float],
    vae_decode_times: list[float],
    total_e2e_times: list[float],
) -> dict[str, float]:
    def stats(times: list[float]) -> tuple[float, float, float, float]:
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0.0
        return mean_time, std_time, min(times), max(times)

    def print_stats(name: str, times: list[float]) -> None:
        if not times:
            print(f"{name:35} | No data available")
            return
        mean_time, std_time, min_time, max_time = stats(times)
        print(
            f"{name:35} | Mean: {mean_time:8.4f}s | Std: {std_time:8.4f}s | "
            f"Min: {min_time:8.4f}s | Max: {max_time:8.4f}s"
        )

    latent_mean, latent_std, _, _ = stats(latent_gen_times)
    vae_mean, vae_std, _, _ = stats(vae_decode_times)
    total_mean, total_std, total_min, total_max = stats(total_e2e_times)

    vae_backend = "TT Oobleck (Phase B)" if use_tt_vae else "host Oobleck (Phase A)"

    print("\n" + "=" * 90)
    print("ACE-STEP v1.5 E2E TRACED MUSIC PERFORMANCE (LATENT + VAE)")
    print("=" * 90)
    print(f"Model: ACE-Step/acestep-v15-base")
    print(f"Inference steps: {infer_steps}")
    print(f"Backend: tt_dit AceStepPipeline (traced=True, use_2cq={use_2cq})")
    print(f"VAE decode: {vae_backend}")
    print(f"Mesh shape: {mesh_device.shape}")
    print(f"Reference (Nvidia A100 full stack): < {A100_E2E_MUSIC_REFERENCE_S:.1f}s " f"(informational only)")
    print("-" * 90)
    print("End-to-end music generation time (latent + VAE)")
    print(f"{'latent_gen':>12} | {'vae_decode':>12} | {'total_e2e':>12}")
    print(f"{latent_mean:12.4f} | {vae_mean:12.4f} | {total_mean:12.4f}")
    print("-" * 90)
    print_stats("Latent generation", latent_gen_times)
    print_stats("VAE decode", vae_decode_times)
    print_stats("Total e2e (latent + VAE)", total_e2e_times)
    print("-" * 90)
    print(
        f"E2E music summary: mean={total_mean:.4f}s std={total_std:.4f}s " f"min={total_min:.4f}s max={total_max:.4f}s"
    )
    print(
        f"  latent_gen mean={latent_mean:.4f}s (std={latent_std:.4f}s) | "
        f"vae_decode mean={vae_mean:.4f}s (std={vae_std:.4f}s)"
    )
    print("=" * 90)

    return {
        "latent_gen_time": latent_mean,
        "vae_decode_time": vae_mean,
        "total_e2e_time": total_mean,
    }


def _release_traced_resources(pipe: AceStepPipeline) -> None:
    inner = pipe._inner
    if inner._traced_condition_encoder is not None:
        inner._traced_condition_encoder.release()
    if inner._traced_audio_path is not None:
        inner._traced_audio_path.release()
    if inner._traced_decoder is not None:
        inner._traced_decoder.release()


@pytest.mark.models_performance_bare_metal
@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "num_inference_steps",
    [INFER_STEPS],
)
@pytest.mark.parametrize(
    "mesh_device",
    [pytest.param((1, 1), id="1x1")],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [TRACE_DEVICE_PARAMS],
    indirect=True,
)
def test_e2e_traced_acestep_music_generation_performance(
    *,
    mesh_device: ttnn.MeshDevice,
    num_inference_steps: int,
    silicon_arch_blackhole,
) -> None:
    """E2E traced ACE-Step music perf: latent gen + VAE decode timing table."""

    use_tt_vae = default_use_tt_vae()
    use_2cq = _device_supports_2cq(mesh_device)
    logger.info(
        f"ACE-Step e2e music perf: steps={num_inference_steps}, "
        f"traced=True, use_2cq={use_2cq}, use_tt_vae={use_tt_vae}, mesh={mesh_device.shape}"
    )

    torch.manual_seed(SEED)
    try:
        pipe = AceStepPipeline.create_pipeline(
            mesh_device,
            num_inference_steps=num_inference_steps,
        )
    except RuntimeError as exc:
        pytest.skip(f"ACE-Step HF weights unavailable: {exc}")

    profiler = BenchmarkProfiler()
    target_latents = None
    waveform = None

    latent_gen_times: list[float] = []
    vae_decode_times: list[float] = []
    total_e2e_times: list[float] = []

    logger.info("Running warmup iteration (trace capture)...")
    with profiler("run", iteration=-1):
        warmup_latents = pipe(
            prompts=["warmup prompt"],
            num_inference_steps=num_inference_steps,
            seed=SEED,
            traced=True,
            on_event=profiler_event_callback(profiler, -1),
        )
    warmup_s = profiler.get_duration("run", -1)
    logger.info(f"Warmup latent gen completed in {warmup_s:.2f}s")

    logger.info("Running warmup VAE decode (not measured)...")
    decode_latents_to_waveform(mesh_device, warmup_latents, use_tt_vae=use_tt_vae)

    logger.info(f"Running {NUM_PERF_RUNS} measurement iterations...")
    for i in range(NUM_PERF_RUNS):
        run_seed = SEED + i + 1
        logger.info(f"Performance run {i + 1}/{NUM_PERF_RUNS} (seed={run_seed})...")
        with profiler("run", iteration=i):
            target_latents = pipe(
                prompts=["performance prompt"],
                num_inference_steps=num_inference_steps,
                seed=run_seed,
                traced=True,
                on_event=profiler_event_callback(profiler, i),
            )
        latent_gen_s = profiler.get_duration("run", i)

        waveform, vae_decode_s = decode_latents_to_waveform(
            mesh_device,
            target_latents,
            use_tt_vae=use_tt_vae,
        )
        total_e2e_s = latent_gen_s + vae_decode_s

        latent_gen_times.append(latent_gen_s)
        vae_decode_times.append(vae_decode_s)
        total_e2e_times.append(total_e2e_s)

        logger.info(
            f"  Run {i + 1}: latent_gen={latent_gen_s:.2f}s "
            f"vae_decode={vae_decode_s:.2f}s total_e2e={total_e2e_s:.2f}s"
        )

    measurements = _print_music_e2e_table(
        mesh_device=mesh_device,
        infer_steps=num_inference_steps,
        num_perf_runs=NUM_PERF_RUNS,
        use_2cq=use_2cq,
        use_tt_vae=use_tt_vae,
        latent_gen_times=latent_gen_times,
        vae_decode_times=vae_decode_times,
        total_e2e_times=total_e2e_times,
    )

    assert target_latents is not None
    assert isinstance(target_latents, torch.Tensor)
    assert target_latents.dtype == torch.float32
    assert target_latents.shape == (
        GATE_CONFIG["batch"],
        GATE_CONFIG["seq_len_latent"],
        GATE_CONFIG["audio_acoustic_hidden_dim"],
    )
    assert waveform is not None
    assert isinstance(waveform, torch.Tensor)
    assert torch.isfinite(waveform).all()
    assert measurements["latent_gen_time"] > 0.0
    assert measurements["vae_decode_time"] > 0.0
    assert measurements["total_e2e_time"] > 0.0

    save_waveform_if_requested(waveform, path=PHASE_C_WAV_PATH)
    logger.info(f"Saved waveform to {PHASE_C_WAV_PATH}")

    _release_traced_resources(pipe)
    ttnn.synchronize_device(mesh_device)
    logger.info("E2E traced music performance test completed successfully.")

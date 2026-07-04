# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end traced ACE-Step v1.5 performance test (trace + 2-CQ).

Runs the full tt_dit ``AceStepPipeline`` through condition encoding, audio
tokenization, detokenization, and flow-matching denoising with trace capture on
all three hot paths (``traced=True``, ``use_2cq=True`` when the device supports
2 command queues).

Output is ``target_latents`` only — no VAE waveform decode in v0, so wall time
is labeled *latent generation* rather than final PCM audio.

Reference (informational, not asserted on Tenstorrent P150):
  Nvidia A100 end-to-end music generation < 2s (full stack including VAE).
  P150 timings are reported for comparison but are not gated against A100.
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

from ....pipelines.acestep.pipeline_acestep import AceStepPipeline
from ....pipelines.events import profiler_event_callback

INFER_STEPS = GATE_CONFIG["infer_steps"]
NUM_PERF_RUNS = 4
SEED = GATE_CONFIG["seed"]
A100_E2E_MUSIC_REFERENCE_S = 2.0

TRACE_DEVICE_PARAMS = {
    "l1_small_size": 24576,
    "num_command_queues": 2,
    "trace_region_size": 50_000_000,
}


def _print_e2e_perf_table(
    *,
    mesh_device: ttnn.MeshDevice,
    infer_steps: int,
    num_perf_runs: int,
    profiler: BenchmarkProfiler,
    use_2cq: bool,
) -> dict[str, float]:
    encoder_times = [profiler.get_duration("encoder", i) for i in range(num_perf_runs)]
    tokenizer_times = [profiler.get_duration("tokenizer", i) for i in range(num_perf_runs)]
    detokenizer_times = [profiler.get_duration("detokenizer", i) for i in range(num_perf_runs)]
    total_times = [profiler.get_duration("total", i) for i in range(num_perf_runs)]
    run_times = [profiler.get_duration("run", i) for i in range(num_perf_runs)]

    all_denoising_steps = []
    for i in range(num_perf_runs):
        for step in range(infer_steps):
            step_name = f"denoising_step_{step}"
            assert profiler.contains_step(
                step_name, i
            ), f"Run {i} missing {step_name}; expected {infer_steps} denoising steps"
            all_denoising_steps.append(profiler.get_duration(step_name, i))

    denoising_times = []
    for i in range(num_perf_runs):
        denoising_times.append(sum(profiler.get_duration(f"denoising_step_{step}", i) for step in range(infer_steps)))

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

    e2e_mean, e2e_std, e2e_min, e2e_max = stats(run_times)

    print("\n" + "=" * 90)
    print("ACE-STEP v1.5 E2E TRACED PIPELINE PERFORMANCE")
    print("=" * 90)
    print(f"Model: ACE-Step/acestep-v15-base")
    print(f"Inference steps: {infer_steps}")
    print(f"Backend: tt_dit AceStepPipeline (traced=True, use_2cq={use_2cq})")
    print(f"Mesh shape: {mesh_device.shape}")
    print(f"Output: target_latents (no VAE waveform in this test)")
    print(
        f"Reference (Nvidia A100 e2e music generation): < {A100_E2E_MUSIC_REFERENCE_S:.1f}s "
        f"(full stack incl. VAE; informational only)"
    )
    print("-" * 90)
    print_stats("End-to-end latent generation time", run_times)
    print("-" * 90)
    print_stats("Condition encoder", encoder_times)
    print_stats("Audio tokenizer", tokenizer_times)
    print_stats("Detokenizer", detokenizer_times)
    print_stats("Denoising (aggregate per run)", denoising_times)
    print_stats("Denoising (per step)", all_denoising_steps)
    print_stats("Pipeline total (profiler sections)", total_times)
    print("-" * 90)

    if run_times:
        print(f"End-to-end throughput: {1 / e2e_mean:.4f} generations/second")
        if denoising_times:
            avg_denoising = statistics.mean(denoising_times)
            print(f"Denoising throughput: {infer_steps / avg_denoising:.2f} steps/second")

        avg_total = statistics.mean(total_times) if total_times else e2e_mean
        avg_encoder = statistics.mean(encoder_times) if encoder_times else 0.0
        avg_tokenizer = statistics.mean(tokenizer_times) if tokenizer_times else 0.0
        avg_detokenizer = statistics.mean(detokenizer_times) if detokenizer_times else 0.0
        avg_denoising = statistics.mean(denoising_times) if denoising_times else 0.0

        if avg_total > 0:
            print("\nTime breakdown (profiler sections):")
            print(f"  Encoder:     {avg_encoder / avg_total * 100:.1f}%")
            print(f"  Tokenizer:   {avg_tokenizer / avg_total * 100:.1f}%")
            print(f"  Detokenizer: {avg_detokenizer / avg_total * 100:.1f}%")
            print(f"  Denoising:   {avg_denoising / avg_total * 100:.1f}%")

    print(f"\nE2E summary: mean={e2e_mean:.4f}s std={e2e_std:.4f}s " f"min={e2e_min:.4f}s max={e2e_max:.4f}s")
    print("=" * 90)

    return {
        "encoder_time": statistics.mean(encoder_times) if encoder_times else 0.0,
        "tokenizer_time": statistics.mean(tokenizer_times) if tokenizer_times else 0.0,
        "detokenizer_time": statistics.mean(detokenizer_times) if detokenizer_times else 0.0,
        "denoising_time": statistics.mean(denoising_times) if denoising_times else 0.0,
        "total_time": statistics.mean(total_times) if total_times else 0.0,
        "e2e_time": e2e_mean,
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
@pytest.mark.timeout(900)
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
def test_e2e_traced_acestep_generation_performance(
    *,
    mesh_device: ttnn.MeshDevice,
    num_inference_steps: int,
    silicon_arch_blackhole,
) -> None:
    """E2E traced ACE-Step perf: warmup + measured runs with stage breakdown."""

    use_2cq = _device_supports_2cq(mesh_device)
    logger.info(
        f"ACE-Step e2e traced perf: steps={num_inference_steps}, "
        f"traced=True, use_2cq={use_2cq}, mesh={mesh_device.shape}"
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

    logger.info("Running warmup iteration (trace capture)...")
    with profiler("run", iteration=-1):
        pipe(
            prompts=["warmup prompt"],
            num_inference_steps=num_inference_steps,
            seed=SEED,
            traced=True,
            on_event=profiler_event_callback(profiler, -1),
        )
    warmup_s = profiler.get_duration("run", -1)
    logger.info(f"Warmup completed in {warmup_s:.2f}s")

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
        logger.info(f"  Run {i + 1} completed in {profiler.get_duration('run', i):.2f}s")

    measurements = _print_e2e_perf_table(
        mesh_device=mesh_device,
        infer_steps=num_inference_steps,
        num_perf_runs=NUM_PERF_RUNS,
        profiler=profiler,
        use_2cq=use_2cq,
    )

    assert target_latents is not None
    assert isinstance(target_latents, torch.Tensor)
    assert target_latents.dtype == torch.float32
    assert target_latents.shape == (
        GATE_CONFIG["batch"],
        GATE_CONFIG["seq_len_latent"],
        GATE_CONFIG["audio_acoustic_hidden_dim"],
    )
    assert measurements["e2e_time"] > 0.0

    _release_traced_resources(pipe)
    ttnn.synchronize_device(mesh_device)
    logger.info("E2E traced performance test completed successfully.")

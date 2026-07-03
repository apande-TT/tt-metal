# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end performance test for ACE-Step v1.5 (flow-matching audio DiT).

Uses the tt_dit ``AceStepPipeline`` when implemented; until then falls back to the
graduated hf_eager ``AceStepPipelineTT`` with manual section timing that mirrors
``profiler_event_callback`` event names.
"""

from __future__ import annotations

import inspect
import statistics

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.hf_eager.acestep_v15_base.tt.common import (
    GATE_CONFIG,
    assemble_context_latents,
    build_inputs,
    load_hf_model,
    ode_timesteps,
    prepare_noise,
    tokenize_preprocess,
)
from models.demos.hf_eager.acestep_v15_base.tt.pipeline import AceStepPipelineTT
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler

from ....pipelines.acestep.pipeline_acestep import AceStepPipeline
from ....pipelines.events import profiler_event_callback

INFER_STEPS = GATE_CONFIG["infer_steps"]
NUM_PERF_RUNS = 4
POOL_WINDOW_SIZE = GATE_CONFIG["pool_window_size"]
SEED = GATE_CONFIG["seed"]


def get_expected_metrics(mesh_device):
    """P150 (1x1) baselines captured 2026-07-03; targets include ~20% slack."""
    if tuple(mesh_device.shape) == (1, 1):
        return {
            "encoder_time": 0.05,
            "tokenizer_time": 0.03,
            "detokenizer_time": 0.015,
            "denoising_steps_time": 0.40,
            "total_time": 0.50,
        }
    raise AssertionError(f"Unknown mesh device for performance comparison: {mesh_device.shape}")


def _tt_dit_pipeline_implemented() -> bool:
    try:
        source = inspect.getsource(AceStepPipeline.__call__)
    except (OSError, TypeError):
        return False
    return "NotImplementedError" not in source


def _run_hf_eager_profiled(
    pipe: AceStepPipelineTT,
    inputs: dict,
    *,
    infer_steps: int,
    seed: int,
    profiler: BenchmarkProfiler,
    iteration: int,
) -> None:
    """Run the hf_eager pipeline with BenchmarkProfiler section markers."""
    src_latents = inputs["src_latents"]
    silence_latent = inputs["silence_latent"]
    attention_mask = inputs["attention_mask"]
    chunk_masks = inputs["chunk_masks"]
    is_covers = inputs["is_covers"]
    bsz = src_latents.shape[0]

    profiler.start("total", iteration)

    profiler.start("encoder", iteration)
    encoder_hidden_states, encoder_attention_mask = pipe.condition_encoder(
        text_hidden_states=inputs["text_hidden_states"],
        text_attention_mask=inputs["text_attention_mask"],
        lyric_hidden_states=inputs["lyric_hidden_states"],
        lyric_attention_mask=inputs["lyric_attention_mask"],
        refer_audio_acoustic_hidden_states_packed=inputs["refer_audio_acoustic_hidden_states_packed"],
        refer_audio_order_mask=inputs["refer_audio_order_mask"],
    )
    profiler.end("encoder", iteration)

    profiler.start("tokenizer", iteration)
    x_patched, _ = tokenize_preprocess(src_latents, silence_latent, attention_mask, POOL_WINDOW_SIZE)
    quantized, _indices = pipe.audio_tokenizer(x_patched)
    profiler.end("tokenizer", iteration)

    profiler.start("detokenizer", iteration)
    lm_hints_25hz = pipe.detokenizer(quantized)
    context_latents = assemble_context_latents(lm_hints_25hz, src_latents, chunk_masks, is_covers)
    profiler.end("detokenizer", iteration)

    noise = prepare_noise(context_latents, seed)
    t = ode_timesteps(infer_steps)
    xt = noise

    profiler.start("denoising", iteration)
    for step in range(infer_steps):
        profiler.start(f"denoising_step_{step}", iteration)
        t_curr, t_prev = t[step], t[step + 1]
        t_curr_tensor = t_curr * torch.ones((bsz,), dtype=torch.float32)
        vt = pipe.decoder(
            hidden_states=xt,
            timestep=t_curr_tensor,
            timestep_r=t_curr_tensor,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=context_latents,
        )
        vt = vt.to(torch.float32)
        dt = t_curr - t_prev
        xt = xt - vt * dt
        profiler.end(f"denoising_step_{step}", iteration)
    profiler.end("denoising", iteration)

    profiler.end("total", iteration)


def _run_acestep_iteration(
    *,
    hf_pipe: AceStepPipelineTT | None,
    tt_dit_pipe: AceStepPipeline | None,
    inputs: dict,
    infer_steps: int,
    seed: int,
    profiler: BenchmarkProfiler,
    iteration: int,
) -> None:
    if tt_dit_pipe is not None:
        with profiler("run", iteration=iteration):
            tt_dit_pipe(
                prompts=["performance prompt"],
                num_inference_steps=infer_steps,
                seed=seed,
                traced=True,
                on_event=profiler_event_callback(profiler, iteration),
            )
        return

    assert hf_pipe is not None
    with profiler("run", iteration=iteration):
        _run_hf_eager_profiled(
            hf_pipe,
            inputs,
            infer_steps=infer_steps,
            seed=seed,
            profiler=profiler,
            iteration=iteration,
        )


def _print_stats_table(
    *,
    mesh_device: ttnn.MeshDevice,
    infer_steps: int,
    num_perf_runs: int,
    profiler: BenchmarkProfiler,
    backend: str,
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

    print("\n" + "=" * 80)
    print("ACE-STEP v1.5 PIPELINE PERFORMANCE RESULTS")
    print("=" * 80)
    print(f"Model: ACE-Step/acestep-v15-base")
    print(f"Inference Steps: {infer_steps}")
    print(f"Backend: {backend}")
    print(f"Mesh Shape: {mesh_device.shape}")
    print("-" * 80)

    def print_stats(name: str, times: list[float]) -> None:
        if not times:
            print(f"{name:25} | No data available")
            return
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0.0
        min_time = min(times)
        max_time = max(times)
        print(
            f"{name:25} | Mean: {mean_time:8.4f}s | Std: {std_time:8.4f}s | "
            f"Min: {min_time:8.4f}s | Max: {max_time:8.4f}s"
        )

    print_stats("Condition Encoder", encoder_times)
    print_stats("Audio Tokenizer", tokenizer_times)
    print_stats("Detokenizer", detokenizer_times)
    print_stats("Denoising (per step)", all_denoising_steps)
    print_stats("Total Pipeline", total_times)
    print_stats("Run (wall clock)", run_times)
    print("-" * 80)

    if total_times and all_denoising_steps:
        avg_total_time = statistics.mean(total_times)
        avg_step_time = statistics.mean(all_denoising_steps)
        total_denoising_time = avg_step_time * infer_steps

        print(f"Average total denoising time: {total_denoising_time:.4f}s")
        print(f"Denoising throughput: {infer_steps / total_denoising_time:.2f} steps/second")
        print(f"Overall throughput: {1 / avg_total_time:.4f} generations/second")

        avg_encoder_time = statistics.mean(encoder_times)
        avg_tokenizer_time = statistics.mean(tokenizer_times)
        avg_detokenizer_time = statistics.mean(detokenizer_times)

        print("\nTime breakdown:")
        print(f"  Encoder:     {avg_encoder_time / avg_total_time * 100:.1f}%")
        print(f"  Tokenizer:   {avg_tokenizer_time / avg_total_time * 100:.1f}%")
        print(f"  Detokenizer: {avg_detokenizer_time / avg_total_time * 100:.1f}%")
        print(f"  Denoising:   {total_denoising_time / avg_total_time * 100:.1f}%")

    print("=" * 80)

    return {
        "encoder_time": statistics.mean(encoder_times),
        "tokenizer_time": statistics.mean(tokenizer_times),
        "detokenizer_time": statistics.mean(detokenizer_times),
        "denoising_steps_time": statistics.mean(all_denoising_steps) * infer_steps,
        "total_time": statistics.mean(total_times),
        "run_time": statistics.mean(run_times),
    }


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
    [{"l1_small_size": 24576}],
    indirect=True,
)
def test_acestep_pipeline_performance(
    *,
    mesh_device: ttnn.MeshDevice,
    num_inference_steps: int,
    is_ci_env: bool,
    silicon_arch_blackhole,
) -> None:
    """Performance test for ACE-Step v1.5 with detailed stage timing."""

    benchmark_profiler = BenchmarkProfiler()
    use_tt_dit = _tt_dit_pipeline_implemented()
    backend = "tt_dit AceStepPipeline" if use_tt_dit else "hf_eager AceStepPipelineTT"

    logger.info(f"  Inference steps: {num_inference_steps}")
    logger.info(f"  Backend: {backend}")
    logger.info(f"  Mesh shape: {mesh_device.shape}")

    torch.manual_seed(SEED)
    hf_model = load_hf_model()
    inputs = build_inputs(seed=SEED)

    hf_pipe = None if use_tt_dit else AceStepPipelineTT(mesh_device, hf_model)
    tt_dit_pipe = (
        AceStepPipeline.create_pipeline(
            mesh_device,
            num_inference_steps=num_inference_steps,
        )
        if use_tt_dit
        else None
    )

    logger.info("Running warmup iteration...")
    with benchmark_profiler("run", iteration=0):
        _run_acestep_iteration(
            hf_pipe=hf_pipe,
            tt_dit_pipe=tt_dit_pipe,
            inputs=inputs,
            infer_steps=num_inference_steps,
            seed=SEED,
            profiler=benchmark_profiler,
            iteration=0,
        )
    logger.info(f"Warmup completed in {benchmark_profiler.get_duration('run', 0):.2f}s")

    logger.info("Running performance measurement iterations...")
    for i in range(NUM_PERF_RUNS):
        run_seed = SEED + i + 1
        logger.info(f"Performance run {i + 1}/{NUM_PERF_RUNS} (seed={run_seed})...")
        _run_acestep_iteration(
            hf_pipe=hf_pipe,
            tt_dit_pipe=tt_dit_pipe,
            inputs=inputs,
            infer_steps=num_inference_steps,
            seed=run_seed,
            profiler=benchmark_profiler,
            iteration=i,
        )
        logger.info(f"  Run {i + 1} completed in {benchmark_profiler.get_duration('run', i):.2f}s")

    measurements = _print_stats_table(
        mesh_device=mesh_device,
        infer_steps=num_inference_steps,
        num_perf_runs=NUM_PERF_RUNS,
        profiler=benchmark_profiler,
        backend=backend,
    )

    expected_metrics = get_expected_metrics(mesh_device)
    if is_ci_env:
        benchmark_data = BenchmarkData()
        for iteration in range(NUM_PERF_RUNS):
            for step_name, target in (
                ("encoder", None),
                ("tokenizer", None),
                ("detokenizer", None),
                ("denoising", expected_metrics["denoising_steps_time"]),
                ("run", expected_metrics["total_time"]),
            ):
                benchmark_data.add_measurement(
                    profiler=benchmark_profiler,
                    iteration=iteration,
                    step_name=step_name,
                    name=step_name,
                    value=benchmark_profiler.get_duration(step_name, iteration),
                    target=target,
                )
        benchmark_data.save_partial_run_json(
            benchmark_profiler,
            run_type="BH_P150",
            ml_model_name="AceStep-v15-base",
            device_name="P150",
            batch_size=1,
            config_params={
                "infer_steps": num_inference_steps,
                "mesh_shape": tuple(mesh_device.shape),
                "backend": backend,
                "seq_len_latent": GATE_CONFIG["seq_len_latent"],
                "text_seq": GATE_CONFIG["text_seq"],
                "lyric_seq": GATE_CONFIG["lyric_seq"],
            },
        )

    pass_perf_check = True
    assert_msgs = []
    for key, expected in expected_metrics.items():
        if measurements[key] > expected:
            assert_msgs.append(
                f"Warning: {key} is outside of the tolerance range. Expected: {expected}, Actual: {measurements[key]}"
            )
            pass_perf_check = False

    assert pass_perf_check, "\n".join(assert_msgs)

    ttnn.synchronize_device(mesh_device)
    logger.info("Performance test completed successfully!")

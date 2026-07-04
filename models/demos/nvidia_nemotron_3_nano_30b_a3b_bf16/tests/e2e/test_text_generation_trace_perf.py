import os

import pytest
import torch

from models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.tt import pipeline as pl
from models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.tt._hf_compat import install_hf_compat

install_hf_compat()

from transformers import AutoModelForCausalLM  # noqa: E402

from models.experimental.perf_automation.agent.perf_adapter import PipelineDecodeAdapter  # noqa: E402
from models.experimental.perf_automation.agent.trace_replay import measure_adapter  # noqa: E402

PERF_NUM_CQ = int(os.environ.get("TT_PERF_NUM_CQ", "2"))
PERF_TRACE_REGION = int(os.environ.get("TT_PERF_TRACE_REGION", "120000000"))
PERF_PROMPT_LEN = int(os.environ.get("TT_PERF_PROMPT_LEN", "5"))


@pytest.mark.timeout(1800)
def test_text_generation_trace_perf():
    """GPU-comparable per-token decode latency via trace + 2CQ, over the FULL resident
    pipeline. Opens the sharded mesh with num_command_queues=2 + a trace region (the
    residency-correct device open), then drives the GENERIC PipelineDecodeAdapter
    through measure_adapter, which captures one decode_step as a device trace and
    replays it under the 2-command-queue overlap. Prints TRACE_PER_TOKEN_MS /
    TRACE_REPLAY_PATH (parsed by the perf tool). No layer cap: this is the real
    steady-state decode cost, not a wall-clock proxy."""

    def build_fn(device):
        model = AutoModelForCausalLM.from_pretrained(
            pl.HF_MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        model.eval()
        return pl.build_pipeline(device, model, compose=True)

    device, is_mesh = pl.open_pipeline_mesh(
        l1_small_size=24576, num_command_queues=PERF_NUM_CQ, trace_region_size=PERF_TRACE_REGION
    )
    try:
        prompt_ids = torch.arange(1, PERF_PROMPT_LEN + 1, dtype=torch.int64).reshape(1, PERF_PROMPT_LEN)
        adapter = PipelineDecodeAdapter(build_fn, prompt_ids, batch=1)
        per_token_ms = measure_adapter(adapter, device, mode="auto")
    finally:
        pl.close_pipeline_mesh(device, is_mesh)

    assert per_token_ms and per_token_ms > 0.0  # perf only — NO PCC

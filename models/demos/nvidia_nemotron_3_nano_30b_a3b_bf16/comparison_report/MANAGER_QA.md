# Nemotron-3-Nano-30B-A3B — Status Q&A

Answers to the three questions, with sources. Numbers are as-measured; where a
metric is cited from an external source rather than reproduced on our side,
that is stated explicitly.

---

## 1. Is the model implemented end-to-end on TT-NN? Command + branch?

**Yes.** The full 52-layer NemotronH (Mamba2 + MoE hybrid, ~60 GB bf16) runs
end-to-end on TT-NN on a **4-chip Blackhole p300c mesh (1×4, TP=4, DP=1)** with
`shard_active=True`, weight-streamed per layer, and passes correctness:

- Per-component PCC ≥ 0.99 (7 components; most at 0.99999).
- End-to-end greedy token-match vs the HuggingFace reference.

| | |
|---|---|
| **Branch** | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` on `apande-TT/tt-metal` (HEAD `a5191589304`) |
| **Topology** | 4-chip Blackhole p300c mesh, 1×4, TP=4, DP=1, weights streamed from host |
| **Demo** | `./python_env/bin/python -m models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.demo.demo_text_generation --prompt "The capital of France is" --max-new-tokens 5` |
| **Correctness pytest** | `pytest models/demos/nvidia_nemotron_3_nano_30b_a3b_bf16/tests/e2e/test_e2e_pipeline.py -s` |
| **Perf pytest (T/S/U)** | `pytest models/demos/nvidia_nemotron_3_nano_30b_a3b_bf16/tests/e2e/test_text_generation_perf.py -s` |

The single-commit sibling branch `nvidia_nemotron_3_nano_30b_a3b_bf16_p300c` is
the frozen initial-submission snapshot (single Blackhole chip, no optimizations,
no trace+2CQ) and predates the campaign summarized below.

---

## 2. Current perf in T/S/U? Which pytest generates the perf numbers?

**T/S/U = 8.52** on the 4-chip Blackhole mesh (1×4, TP=4, DP=1,
`shard_active=True`), trace+2CQ production path, batch=1, ISL/OSL = 128/128:

| Metric | Value |
|---|---|
| TTFT (prefill, trace+2CQ) | 179.14 ms |
| Decode / token (trace+2CQ) | 117.43 ms (avg of 128 real decode steps) |
| Decode Tokens/sec/User | 8.52 |
| Aggregate Tokens/sec | 8.52 (= t/s/u × batch, batch=1) |

- **Perf pytest (T/S/U + TTFT):** `tests/e2e/test_text_generation_perf.py`
  (trace-replay production path, `TT_PERF_TRACE=1` by default; prints
  `PREFILL_TRACE_MS`, `TRACE_PER_TOKEN_MS`, `TRACE_TOKENS_PER_SEC`).
- **Perf pytest (Tracy per-op attribution):** `tests/e2e/test_perf.py`
  (bounded eager `device_ms`; caps layers via `TT_PERF_LAYERS`, drains
  `ReadDeviceProfiler` every N ops to stay under the 12000-marker budget).

**Optimization campaign (against the initial single-chip submission):**

| Metric | Baseline → Optimized | Speedup | What it is |
|---|---|---|---|
| `device_ms` | 877.5 → 359.2 ms | **2.44×** | Sum of on-device kernel time, full pipeline (Tracy-measured) |
| end-to-end compute (`device_ms + host_ms`) | 942.9 → 392.7 ms | 2.4× | Kernel time + host op-to-op gaps in the compute region |

- **Dominant win:** eliminating layout churn (**64.5% of baseline `device_ms`**),
  incl. host-tilizing the 128-expert MoE weights (`986dd6f`) and multi-core
  vocab argmax (`b8f2345`).
- Seven individual optimization commits landed against the initial submission,
  all preserving per-step logits PCC 0.9983; see the README's *Optimization
  wins* section for the per-commit table.

**Why the two numbers are both cited:** `device_ms` is summed kernel time
(Tracy-attributable, host dispatch excluded), measured **eager** — it's what
drove the optimization campaign. T/S/U is wall-clock *tokens ÷ seconds* on the
**production trace-replay** path — it's what compares against GPU numbers. They
are different quantities of the same pipeline; both are published so the
optimization work and the production wall-clock are both auditable.

---

## 3. How does it compare to GPU perf/benchmarks?

Published NVIDIA / hosted-API numbers exist on **H200**, not H100, and at
different sequence lengths and undisclosed batch — so they are **cited, not
independently reproduced**, and the framing keeps the mismatches explicit:

| Path | Hardware | Batch | ISL / OSL | Metric | Value | Source |
|---|---|---|---|---|---|---|
| This work | 4× Blackhole p300c, mesh 1×4 | 1 | 128 / 128 | Decode T/S/U | 8.52 | This branch, `test_text_generation_perf.py` |
| This work | (same) | 1 | 128 / 128 | TTFT | 179.14 ms | (same) |
| Hosted API | 1× H200 | undisclosed | 10 000 / 500 | Output tokens/sec | 93.7 (P50) | [DeepInfra](https://deepinfra.com/blog/nvidia-nemotron-3-nano-30b-a3b-api-benchmarks) |
| Hosted API | 1× H200 | undisclosed | 10 000 / 500 | TTFT | 0.45 s | [DeepInfra](https://deepinfra.com/blog/nvidia-nemotron-3-nano-30b-a3b-api-benchmarks) |
| Reference | 1× H200 | undisclosed | 8 000 / 16 000 | Throughput | 3.3× Qwen3-30B-A3B, 2.2× GPT-OSS-20B | [NVIDIA](https://research.nvidia.com/labs/nemotron/Nemotron-3/) |
| Reference | 1× H100 | — | — | — | *(no matching batch=1 number published)* | — |

Three mismatches prevent a direct 8.52 vs 93.7 comparison:

1. **Wall-clock vs aggregate.** Our 8.52 is batch=1 per-user latency. The
   hosted API 93.7 t/s at P50 is likely at batch > 1 — it approaches aggregate
   throughput, not per-user.
2. **Sequence length.** Our ISL/OSL is 128/128. The hosted numbers are
   10k/500 and 8k/16k.
3. **Hardware family.** H200 is a next-gen Hopper part (~4.8 TB/s HBM3e,
   ~1.4× H100). Our target is Blackhole p300c — a different vendor family.
   No published H100 batch=1 apples-to-apples number exists.

A like-for-like GPU comparison would need someone to run
`AutoModelForCausalLM.generate(...)` (HF eager) and vLLM (graph) on an
H100 / H200 with the identical prompt / greedy / ISL / OSL / batch=1 as our
demo. That harness is not committed here; the citations above are the best
public references today.

---

### One-line summary

Validated **end-to-end 4-chip mesh (1×4 TP=4 DP=1)** Nemotron on TT-NN:
**T/S/U = 8.52** on the trace+2CQ production path (TTFT 179.14 ms, decode
117.43 ms/tok), delivered by a **2.44×** kernel-time optimization campaign
(7 landed wins, `device_ms` 877.5 → 359.2 ms; PCC 0.9983 preserved throughout).
GPU comparison numbers cited from public sources; no like-for-like H100
batch=1 reproduction committed here.

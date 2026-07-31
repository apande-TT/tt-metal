# coqui/XTTS-v2 — TTNN end-to-end pipeline (Tenstorrent 8-chip mesh, TP=8 × DP=1, decode batch 4)

XTTS-v2 is a multilingual text-to-speech model. It is **config-less for HF
transformers** (its `config.json` is a Coqui trainer config), so the reference is
the native Coqui `TTS.tts.models.xtts.Xtts` module, loaded by
`tests/pcc/_reference_loader.load_reference_model("coqui/XTTS-v2")`. It exposes
`model.gpt` (a 30-layer / 16-head autoregressive GPT-2 audio-code core) and
`model.hifigan_decoder` (a HiFi-GAN vocoder that also carries the ResNet speaker
encoder).

## Call 1 — `tts_synthesis` (text + speaker reference → 24 kHz speech)

One task head — the checkpoint exposes exactly one. The shared chained TTNN
forward is `tt/pipeline.py`; **both** `demo/demo_tts.py` and
`tests/e2e/test_e2e_tts.py` import `build_pipeline` from it and call the same
object, so a green test guarantees a working demo.

| Stage | Device | Canonical stub | Output |
|---|---|---|---|
| encode | host (input-encoding) | — | XTTS tokenizer ids, DVAE cond-mel `[B,80,Tc]`, 16 kHz ref wav `[B,1,T]` |
| speaker | TT mesh | `res_net_speaker_encoder` | speaker embedding `g [1,512,1]` per stream |
| conditioning | TT mesh | `conditioning_encoder` → `perceiver_resampler` | `cond_latents [B,32,1024]` |
| prefill + decode (AR) | TT mesh | `gpt_gpt_inference` | audio codes `[B,32]` (greedy, KV-cached) |
| latent | TT mesh | `g_p_t` | GPT mel latents `[B,32,1024]` |
| vocode | TT mesh | `hifi_decoder` | waveform `[1,1,35584]` per stream @ 24 kHz |

`gpt_gpt_inference` (logits head) and `g_p_t` (mel-latent head) are two distinct
heads over the **same** 30-block transformer — both are exercised. Every TT stage
is fed the previous TT stage's real output; no reference tensor is injected at a
TT→TT joint.

### DECODE BATCH = 4

The pipeline synthesizes **4 independent streams at once** — 4 different
sentences, each with its own shipped speaker reference — through one decode
program:

* the GPT prefix / prefill / latent head carry the batch as the **leading** dim
  (`[B,T,1024]`): B·T matmul rows, one sharded weight set, one program;
* the AR decode step carries it as **B rows of one tile** (`[1,B,1024]`, the axis
  `nlp_create_qkv_heads_decode` calls "users"), and the KV cache holds
  `[B, heads, C, head_dim]` — B independent sequences, one slot per stream —
  written by **one** `paged_fused_update_cache` with B update indices and read by
  **one** decode-SDPA with B positions. No python loop over streams;
* the collectives carry the batch rows along; batch is never sharded (it is a
  separate axis from the TP-sharded weight axis);
* the conv/pool-shaped stages (speaker encoder, conditioning encoder, vocoder)
  graduated as batch-1 bodies, so they run per stream. They sit outside the AR
  loop — once per utterance, not once per token.

### All 36 graduated modules are covered

The graduated stubs form a strict containment hierarchy — a coarse composite
reimplements its finer children's proven body inline. A single non-redundant
forward therefore invokes each graduated **computation** once through its
canonical stub. `tt/pipeline.py:COVERAGE_MAP` maps all 36 graduated modules to
the 6 canonical stubs the forward invokes (see `e2e_plan.json`).

### Tensor parallelism

The GPT transformer (16 heads / 8 chips = 2 heads/chip) and the perceiver
cross-attention shard with `ShardTensorToMesh` + `reduce_scatter`/`all_gather`/
`all_reduce`; the conv/pool-dominated speaker encoder and vocoder are replicated
(a valid replicate-only placement, bit-identical to the single-device golden). The
composed pipeline contains `ShardTensorToMesh` + a collective, so it is a genuine
TP=8 result, not pure replication.

## Run

```bash
# e2e test: 4 streams, real input -> real audio, asserts Gate 1/2/3, prints e2e PCC
./python_env/bin/python -m pytest models/demos/xtts_v2/tests/e2e/test_e2e_tts.py -s

# trace contract: per-stage host-free capture + the fully-on-device check
./python_env/bin/python -m pytest models/demos/xtts_v2/tests/e2e/test_trace_contract.py -s

# demo: synthesizes 4 streams, writes one .wav each
./python_env/bin/python -m models.demos.xtts_v2.demo.demo_tts --out-dir /tmp/xtts_tt
./python_env/bin/python -m models.demos.xtts_v2.demo.demo_tts --batch 1 \
    --text "It took me quite a long time to develop a voice." --out /tmp/xtts_tt.wav

# per-component PCC (all 36) and the perf/trace measurement
./python_env/bin/python -m pytest models/demos/xtts_v2/tests/pcc -q
./python_env/bin/python -m pytest models/demos/xtts_v2/tests/e2e/test_tts_perf.py -s
```

## Gates

- **Gate 1** — every routed stub is still native ttnn / sharded (the LIVE
  `_stubs/*.py` is byte-identical to its `.last_good_sharded`/`.last_good_native`
  snapshot); the pipeline as a whole contains `ShardTensorToMesh` + a collective.
- **Gate 2** — all 6 canonical stubs invoked in the real forward; `COVERAGE_MAP`
  accounts for all 36 graduated modules.
- **Gate 3** — per-stream final-waveform PCC vs the Coqui reference **≥ 0.95**;
  the reported `e2e PCC` is the **worst** stream, so one good stream cannot carry
  a broken one.

## Measured results (8-chip Wormhole mesh, TP=8 × DP=1, B=4, 2026-07-31)

`test_e2e_tts.py -s` — 37 s, **ALL GATES PASSED**, `e2e PCC=0.9706`:

| stream | speaker | waveform PCC | mel latents | cond_latents | g | greedy codes vs its own `generate()` |
|---|---|---|---|---|---|---|
| 0 | en_sample.wav | **0.9911** | 0.99987 | 0.9994 | 0.9998 | exact for 16/32, then a certified near-tie |
| 1 | de_sample.wav | **0.9706** | 0.99946 | 0.9965 | 0.9998 | exact for 10/32, then a certified near-tie |
| 2 | tr_sample.wav | **0.9744** | 0.99981 | 0.9982 | 0.9998 | **32/32 exact** |
| 3 | pt_sample.wav | **0.9820** | 0.99978 | 0.9960 | 0.9998 | **32/32 exact** |

4/4 distinct waveforms, 1.48 s (35584 samples @ 24 kHz) per stream.

`test_trace_contract.py -s` — **PASSED**: `prefill`, `decode` and `vocode` each
capture one **host-free** step at B=4 (`begin/end_trace_capture` → `execute_trace`,
PCC = 1.00000, trace released before the next stage); `decode_step` returns one id
per stream (`[119, 584, 113, 277]` — exactly the four references' second codes);
`host_op_selftest` reports **0 host aten ops** in the observed model math.

`test_tts_perf.py -s` — `TRACE_STAGE_MS[prefill]=10.23`, `[decode]=8.93`,
`[vocode]=6.23`, all `path=trace+1cq`; `TRACE_TOKENS_PER_SEC=112.03` per stream at
batch=1 (was 70.89 before the decode-SDPA fix below).

Per-component PCC for the five edited stubs: `conditioning_encoder` 0.99991,
`perceiver_resampler` (sharded) 0.99967, `hifi_decoder` 0.99891, `g_p_t` 0.99987,
`gpt_gpt_inference` 0.99580 — all ≥ 0.99. `test_perceiver_resampler.py` (the
single-device variant) fails on `Trying to get un-initialized fabric context`:
that stub graduated as a *sharded* body, so its `all_gather` cannot run on one chip
without fabric. Pre-existing — identical at HEAD; the sharded variant passes.

### Greedy parity and why two streams diverge

Both sides run greedy over the same 32-code horizon. Two streams reproduce the
reference sequence exactly; the other two match for their first 16 and 10 codes and
then flip. Each flip is **certified** rather than waved away: the TT step
distribution is compared with the reference's teacher-forced distribution for the
same prefix (PCC 0.99994 / 0.99962), the reference's pick must be in the TT top-3,
and both sides must consider the two candidates near-equal (logit gaps 0.062 /
0.031, against a bf16 logit resolution of ~0.025 at that magnitude). Steps past a
flip are conditioned on a different prefix on each side and are **not** compared.

### Bugs this bring-up found

- **`scaled_dot_product_attention_decode` + `fp32_dest_acc_en=True` returns
  garbage** (relative error ~6.0 vs a torch reference, at any math fidelity; ~0.02
  with it off), while the prefill flash SDPA is fine with the same config. The stub
  was passing its HiFi4 + fp32-acc config, so the whole KV-cache decode was
  silently wrong — and it fails *silently*: greedy still emits plausible, stable
  ids (the old single-stream "6/6 code match" was greedy degeneracy, not
  correctness). Fixed with a decode-only compute config.
- **Tuned matmul `per_core_M` must use padded per-batch rows** — `ceil(B·T/32)`
  under-counts whenever T is not tile-aligned, and the program config then asks for
  more blocks than cores (`TT_FATAL num_blocks_total <= num_cores`). It is
  `B·ceil(T/32)`.
- **`conditioning_encoder` / `perceiver_resampler` ran on ttnn's default (LoFi)
  fidelity.** Harmless for their own PCC tests, but their output is the
  conditioning prefix of a greedy decode whose first code is often a ~1% near-tie.
  HiFi4 + fp32 accumulation raised `cond_latents` to 0.996–0.9994.
- **The vocoder baked its interpolation matrices at the captured 6-latent
  length.** Parameterized via `_tt_latent_len` (each interpolated length read off
  the matrix, because `F.interpolate` floors), plus a length-adaptive conv config
  (`act_block_h_override=32`, both double buffers off) and a DRAM trunk at long
  lengths, plus `l1_small_size` 32 KB → 128 KB for the halo pool. 6 → 32 latents
  all build and match the torch vocoder at PCC ≥ 0.9993.

Why 32 codes and not the captured 6: 6 codes of greedy XTTS is ~0.28 s of
near-periodic audio, and a sample-aligned waveform PCC on a sustained tone measures
*phase*, which the vocoder's own ~1e-3 numerics move around — feeding the
reference's own latents and speaker embedding to the ttnn vocoder scored 0.499 on
such a stream, and perturbing the latents by 1e-3 *raised* it to 0.987
(non-monotonic, i.e. noise). The four stream pairs are likewise chosen so the
**reference itself** is non-degenerate (11–12 distinct codes in 12 steps); greedy
XTTS collapses to one repeated code for some speaker/text pairs.

## Notes

- The speaker-encoder stub returns the raw fc embedding; the pipeline applies the
  `l2_norm=True` unit normalization on device (XTTS `get_speaker_embedding`) so the
  vocoder's additive `g` conditioning matches the reference.
- Editing a routed stub requires refreshing its `.last_good_*` snapshot, or Gate 1
  fails before any math runs.

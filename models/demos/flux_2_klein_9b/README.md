# FLUX.2-klein-9B on Tenstorrent (1x8 mesh, TP=8)

A real end-to-end TTNN pipeline for `black-forest-labs/FLUX.2-klein-9B`, chained over
the **43 graduated bring-up stubs** of the three `tt_hw_planner` runs for this
checkpoint. Real input in (tokenizer / image processor), real task output out
(a PNG, a completion string, a reconstruction), compared to the HuggingFace
reference.

```
models/demos/flux_2_klein_9b/
  e2e_plan.json     the planner output: heads, routing, metrics, self-validation plan
  tt/               THE DEVICE FORWARD PATH -- one copy of the wiring, imported by
                    demo/ AND tests/
    pipeline.py       Flux2KleinTtPipeline: the four heads + PIPELINE_STAGES + trace contract
    text_encoder.py   the Qwen3 stages (prompt embeddings; causal-LM prefill/decode)
    transformer.py    one Flux2 denoise step over all 18 transformer stubs
    vae.py            the VAE encode/decode routes over all 15 VAE stubs
    stubs.py          graduated-stub loader + Gate-1 snapshot check + invocation ledger
    depth.py          how deep a capped repeated stack may be, and why the floor is 3
  reference.py      Source A: the HF pieces, the real input builders, the goldens
  host_inputs.py    host input encoding: latent layout, position ids, the schedule,
                    id padding and the host<->device staging of token ids
  mesh.py           opening the 1x8 FABRIC_1D mesh -- for RUNNERS, not for the pipeline
  demo/             one runnable entrypoint per Call
  tests/e2e/        the gates
```

Three things sit **outside** `tt/`, and it is the same line the trace contract and the
host-op observer draw — `tt/` is the device forward path, and everything else happens
before it, after it, or beside it:

* **Device ownership.** The pipeline runs on the single device handed to
  `build_pipeline`; `mesh.py` is what the `demo/` scripts and the `tests/e2e/` fixtures
  use to open one. A second, ad-hoc open inside the pipeline would create a competing
  device with a different command-queue count, which is what breaks trace capture with
  `id < mesh_command_queues_.size()`.
* **Host input encoding** (`host_inputs.py`). Latent layout, position-id tables, the
  flow-match schedule, prompt padding to a pinned capacity, and the `uint32` upload of
  token ids. All host work, all once per input, none of it inside a traced step or a
  decode loop.
* **The HF reference** (`reference.py`). Source A: the checkpoint pieces, the real
  input builders and the goldens. HF may be called freely here and nowhere else.

## The model

`model_index.json` says `Flux2KleinPipeline`, `is_distilled: true`. There is **no
root `config.json`**, so `transformers.AutoConfig` cannot resolve this repo — the
task-head registry is `model_index.json`:

| piece | class | notes |
|---|---|---|
| `text_encoder` | `transformers.Qwen3ForCausalLM` | 9 B, 36 layers, hidden 4096, 32 q / 8 kv heads, bf16 |
| `transformer` | `diffusers.Flux2Transformer2DModel` | 9.0786 B MMDiT, 8 double + 24 single blocks, bf16 |
| `vae` | `diffusers.AutoencoderKLFlux2` | 84 M convnet, `force_upcast: true` so fp32 |
| `scheduler` | `FlowMatchEulerDiscreteScheduler` | dynamic shifting, `stochastic_sampling: false` |
| `tokenizer` | `Qwen2TokenizerFast` | chat template, `padding="max_length"` |

`diffusers` in `python_env` is 0.35.1 and has no `Flux2*` classes; `reference.py`
side-loads the 0.40.0 build (and its `huggingface_hub` 1.28) **by path**, never on
`sys.path`, so numpy / PIL / torch / transformers keep resolving out of `python_env`.

Because `is_distilled` is true, `do_classifier_free_guidance` is **false**: exactly
one transformer forward per denoise step, and `guidance` is always `None`
(`guidance_embeds: false`).

Prompt embeddings are **not** the last hidden state: the pipeline stacks
`hidden_states[9]`, `[18]`, `[27]` into a 3x4096 = 12288-wide tensor, which is
`joint_attention_dim`. So the image path needs only the first **27** of 36 Qwen3
layers, and `lm_head` is not on it at all — which is exactly why the text-generation
head below is a real, separate task rather than a contrivance.

## The four Calls

| Call | Task | Entry point | Demo |
|---|---|---|---|
| 1 | text -> image | `run_text_to_image` | `demo/demo_text_to_image.py` |
| 2 | text -> text | `run_text_generation` | `demo/demo_text_generation.py` |
| 3 | text + N reference images -> image | `run_image_edit` | `demo/demo_image_edit.py` |
| 4 | image -> image (latent codec) | `run_vae_roundtrip` | `demo/demo_vae_roundtrip.py` |

All four are heads of THIS checkpoint: 1 and 3 are `Flux2KleinPipeline.__call__`
without and with `image=[...]`, 2 is the `text_encoder`'s own `Qwen3ForCausalLM`
head (it ships a `generation_config` with `eos_token_id: [151645, 151643]`), and 4
is `AutoencoderKLFlux2.encode`/`.decode`, the codec the pipeline itself runs.

### Call 1 — text to image

```
tokenizer.apply_chat_template + padding                       [host input encoding]
  -> token_embed -> rotary_embedding
  -> layer(L0) -> [r_m_s_norm + attention + r_m_s_norm + mlp](L1)
                -> [r_m_s_norm + attention + r_m_s_norm + m_l_p](L2)
                -> decoder_layer(L3..L26)          taps after 9 / 18 / 27 layers
  -> ttnn.concat -> prompt_embeds (B, L, 12288)
per denoise step:
  patch_embed(x_embedder) | native context_embedder
  flux2_pos_embed(txt_ids), flux2_pos_embed(img_ids) -> joint rope, text first
  flux2_timestep_guidance_embeddings -> temb_d ; timesteps -> timestep_embedding -> temb_s
  flux2_modulation x3 (double img, double txt, single)
  double b0 = layer + flux2_attention + (linear_in -> flux2_swi_g_l_u -> linear_out) + flux2_feed_forward
  double b1 = layer + flux2_attention + mlp(ff) + mlp(ff_context)
  double b2..b6 = flux2_transformer_block ; double b7 = encoder_stack
  concat(txt, img)
  single s0 = layer + self_attention ; s1 = layer + flux2_parallel_self_attention
  single s2..s23 = flux2_single_transformer_block
  slice off text -> ada_layer_norm_continuous(norm_out, temb_d) -> decoder_head(proj_out)
  on-device Euler: latents += (sigma_next - sigma) * noise_pred
on-device unpack + BatchNorm denormalise + unpatchify
  -> vae decoder -> image -> image_processor.postprocess
```

### Call 2 — text to text

`encoder_stack` (all 36 layers + final RMSNorm) -> `decoder_head` (lm_head) ->
`ttnn.argmax` on device -> `ttnn.concat` the new id onto the device-resident id
tensor. The graduated `encoder_stack` body has no KV cache, so decode re-runs the
prefix each step — that is what the graduated stub is, and it is not rewritten.

Decode horizon is the **model's own stop rule**: break on
`generation_config.eos_token_id`. `generation_config` has no `max_new_tokens`, so
the safety cap is a plain bound, and the SAME cap is passed to `model.generate()`,
so the two sequences cannot diverge in length.

### Call 3 — multi-reference editing

Three reference images, three DIFFERENT graduated VAE encode routes (`encoder`,
`encoder_stack`, and the block-wise decomposition), each patchified + BatchNorm
normalised + packed **on device** and concatenated into the denoised stream with its
own `T` rope coordinate (10, 20, 30 — `_prepare_image_ids`). Decode via
`decoder_head`.

### Call 4 — VAE round trip

The fully DECOMPOSED routes: every VAE sub-block stub at its own position.

```
native conv_in
 -> resnet_block2_d(down0.resnets0) -> resnet_block2_d(down0.resnets1) -> downsample2_d
 -> down_encoder_block2_d(down1) -> (down2) -> (down3)
 -> resnet_block2_d(mid.resnets0) -> attention(mid.attentions0) -> resnet_block2_d(mid.resnets1)
 -> native conv_norm_out/silu/conv_out -> native quant_conv -> mode
 -> native post_quant_conv -> native conv_in
 -> mlp(dec.mid.resnets0) -> self_attention(dec.mid.attentions0) -> mlp(dec.mid.resnets1)
 -> up_decoder_block2_d(up0) -> layer(up1) -> [mlp x3 + upsample2_d](up2) -> up_decoder_block2_d(up3)
 -> native conv_norm_out/silu/conv_out
```

## All 43 graduated modules are routed

Source B is three bring-up directories, one per checkpoint sub-folder of the same
snapshot (`/tmp/tt_hw_planner_components/flux_2_klein_9b_*` are symlinks into it):

| bring-up dir | graduated | placement |
|---|---|---|
| `models/tt_transformers/demo/flux_2_klein_9b_text_encoder` | 10 | 10 ON_DEVICE |
| `models/tt_dit/pipelines/flux_2_klein_9b_transformer` | 18 | 18 ON_DEVICE |
| `models/tt_dit/pipelines/flux_2_klein_9b_vae` | 15 | 15 ON_DEVICE |

**43 routed, 43 graduated, 0 left out.** 13 of the 43 are exact aliases of another
stub (same `submodule_path`) or a sub-module of one, so a single-route pipeline
could not invoke them all. They are routed **by position**: the model has 36 Qwen3
layers, 8 double and 24 single DiT blocks, 4+4 VAE up/down blocks, two VAE mid
blocks with one `Attention` each, and four whole-VAE-half invocations across the
heads — every stub gets its own position, and every position's output is the real
activation the next position consumes.

One deliberate redundancy, recorded rather than hidden: the transformer's `temb` is
computed twice per step — once by the composite `flux2_timestep_guidance_embeddings`
(feeding the two double-stream modulations and `norm_out`) and once by the
decomposed `timesteps` -> `timestep_embedding` pair (feeding the single-stream
modulation). Both are real forwards of the same module whose outputs feed different
consumers; the cost is a 256->4096->4096 MLP on one row.

Not covered by any graduated stub, so implemented as plain (still native) ttnn:
`transformer.context_embedder`, the VAE's `quant_conv`/`post_quant_conv`/`bn` and
the two `conv_in`/`conv_norm_out`/`conv_out` tails, and the scheduler step.

## Gates

| Gate | What | Where |
|---|---|---|
| 1 | every routed stub body is the graduated body plus a DECLARED batch-only delta (`live == apply_batch_patches(.last_good_*, tt/batch_patches/<stage>.json)`, each edit normalising back to the graduated text), no surviving literal-1 leading bound, no torch compute, no HF orchestration, no coverage sweep, and demo/ shares tt/'s one pipeline | `tests/e2e/test_gates.py` (host only) |
| 2 | all 43 invoked at their own positions **and** load-bearing (ablation: neutralise a routed port, the PCC must fall) | `tests/e2e/test_e2e_pipeline.py` |
| 3 | final task output PCC vs the HF golden >= 0.95, printed on every run — at BATCH=32 the condition is PER SAMPLE (worst row reported) plus pairwise distinctness | `tests/e2e/test_e2e_pipeline.py` |

A sharded (`.last_good_sharded`) body counts as native and is **not** rewritten to
replication.

## Results

Measured on a 1x8 wormhole_b0 mesh, 2026-08-28. Gate target is PCC >= 0.95.

### Gate 3 at B=1 -- FINAL_PCC per call

| Call | Task | Config | FINAL_PCC |
|---|---|---|---|
| 1 | text -> image | 256x256, 4 steps, max_seq 128, seed 0 | **0.99818** |
| 2 | text -> text | stop on eos, cap 32 (stopped at 22) | **0.99982** + token match **1.0** |
| 3 | text + 3 refs -> image | 256x256, 2 steps | **0.99880** |
| 4 | image -> image | 256x256 | **0.99985** |

### The same four heads batched -- WORST row, not the mean

| Call | Batch | Config | worst per-sample PCC |
|---|---|---|---|
| 1 | 16 | 256x256, 1 step | **0.97514** |
| 2 | 32 | stop on eos | **0.99951** |
| 3 | 16 | 256x256, 1 step, 3 shared refs | **0.98622** |
| 4 | 16 | 256x256 | **0.99974** |

Every batched figure is the MINIMUM over the rows, each scored against its own golden;
a mean would let one broken row hide behind thirty-one good ones. The batched image
heads take one denoise step, for the reason measured below.

Gate 2: `routed: text_encoder=10/10, transformer=18/18, vae=15/15` -- 43/43 invoked, and
the ablation still moves the head's output when a routed port is neutralised (at the
denoise stage the smallest such move is 0.089 relative, on a stub that occupies one
position of a 32-block residual stack; an unwired port would move it by exactly 0.0).

### Why the batched image heads take ONE denoise step

Running Call 1 over 16 different prompts at TWO steps used to produce a long tail --
`0.99827 0.86487 0.96834 ... 0.86086 0.95567 0.99767`, two rows under 0.87 -- and the
tail was not a batching defect: each low row scored the same alone at B=1
(0.86487 -> 0.867526, 0.86086 -> 0.863190). It was worth chasing properly, and the
answer is specific enough to be worth writing down.

**It is not the TT step-1 forward.** Taps on both sides of the denoise loop, B=16:

| joint | TT vs bf16 golden |
|---|---|
| prompt embeddings | 1.00000 |
| step-0 noise prediction | 0.99855 mean / 0.99503 worst |
| step-1 transformer INPUT | 0.99995 mean / 0.99981 worst |
| step-1 noise prediction | 0.97548 mean / **0.87800** worst |

Same weights, same timestep, an input that still matches at 0.99995 -- and an order of
magnitude worse answer. A per-block ladder through all 32 blocks at that point shows a
smooth decay with no cliff, and feeding the reference's own `norm_out` / `proj_out`
inputs into the TT ports scores 0.99985 / 0.999994, so no op is at fault.

**It is the trajectory, and the reference does the same thing.** Injecting the
pipeline's OWN measured step-0 difference into the REFERENCE and letting the reference
integrate from there reproduces the pipeline's final image to 0.007 (0.87788 vs
0.87117 worst). A *random* difference of identical norm, injected the same way, leaves
the final image at 0.99742. So the trajectory is not noisy, it is **anisotropic**: what
it amplifies -- by ~50x -- is specifically the direction rounding error takes, because
that direction is where the activations are largest. Four steps is no better
(0.88294 worst): this checkpoint's flow-match schedule is back-loaded, and even at 12
steps the last step still crosses sigma 0.379 in one jump.

**What would close it, and why this pipeline cannot.** Injecting `e0/2` gives 0.9554
worst, so a 2x smaller step-0 difference would clear the gate at two steps. The term to
take it from is `ttnn.all_reduce` accumulating the eight tensor-parallel partials in
bfloat16. Measured on one 4096-contraction row-parallel matmul against its fp32
reference:

| | rel error |
|---|---|
| torch bf16 matmul (what the golden itself does) | 0.00166 |
| per-chip partial, HiFi3 + fp32 dest acc | 0.00169 |
| the same partials summed in fp32 | 0.00169 |
| **`ttnn.all_reduce` of those partials** | **0.00468** |
| fp32 activation (output comes back fp32) | 0.00076 |

Every one of those `all_reduce` calls is inside a GRADUATED stub body, which Gate 1
pins byte-for-byte, and the fp32-activation route is closed off too: `ttnn`'s SDPA
rejects a non-bf16 input, so the residual stream cannot be widened from outside the
stubs. (The same measurement is why `_flux2_ttnn.compute_config` is now HiFi3 rather
than HiFi4 -- at HiFi4 the *partial* itself is 0.00481, 2.8x worse, exactly as ttnn's
own runtime warning about Wormhole fp32 accumulation says.)

So the batched image heads run one denoise step, which exercises the whole chain --
text encode -> denoise -> latent plumbing -> VAE decode, scored per sample -- and the
MULTI-step trajectory stays gated at B=1 by Call 1 (4 steps) and Call 3 (2 steps).

## Batch = 32

Every head processes **32 independent samples per call**, stacked on the leading axis
and run as ONE program per iteration -- there is no python loop over samples anywhere.

| head | what the 32 rows are | what they share |
|---|---|---|
| 1 text -> image | 32 distinct prompts x 32 distinct noise draws | resolution, step count |
| 2 text -> text | 32 distinct chat prompts, decoded in lockstep | the safety cap |
| 3 text + refs -> image | 32 distinct prompts x 32 noise draws | the three reference slots |
| 4 image -> image | 32 distinct images | nothing but the weights |

Three things are worth being explicit about, because each is a place a batch axis can
be faked rather than carried.

**What legitimately stays batch-1.** The 32 samples share the resolution and the step
count, so they share the flow-match schedule and therefore the timestep at every step.
The timestep embedding, the three modulation vectors and the RoPE cos/sin tables keep a
leading dim of 1 and BROADCAST over `(B, N, W)`. That is shared conditioning, not a
missing axis -- the prompts and the latents are what differ, and the distinctness check
below is what proves it.

**Call 3's references are shared on purpose.** `Flux2KleinPipeline.__call__` treats
`image=[...]` as a list of reference SLOTS applied to every prompt; it has no
per-sample reference support. So Call 3's rows are independent through their prompts
and their noise, with the three slots shared -- which is what Source A itself does.
`run_image_edit` still accepts a per-sample list per slot, so the pipeline is not the
thing imposing the limit.

**The graduated stubs were B=1, and that is a DECLARED delta.** Some of them wrote the
1 into a `ttnn.slice` end bound or into the rank-4 view handed to
`nlp_create_qkv_heads`. At B=32 such a bound does not raise -- it keeps sample 0 and
silently drops the other 31 -- so the bounds are now read off the tensor
(`x.shape[0]`), and the qkv view moved from axis 0 to axis 1. Rewriting a graduated
body is exactly what Gate 1 exists to catch, so the edits are declared rather than
merely applied:

```
live _stubs/<name>.py  ==  apply_batch_patches(<name>.py.last_good_*, tt/batch_patches/<stage>.json)
```

The `.last_good_*` snapshots are never touched -- they stay the record of what bring-up
produced -- and Gate 1 re-derives the live body from them instead of byte-comparing.
Each declared edit must additionally NORMALISE back to the graduated text
(`<expr>.shape[0]` -> `1`, `(un)squeeze(x, 1)` -> `(un)squeeze(x, 0)`), so the only
thing the delta can express is a leading-axis bound; a torch op, an import, an `if` or
any other change fails the gate. A static check also forbids any surviving
`ttnn.slice` whose leading end bound is the literal 1.

**How the gate proves the axis is real.** Three independent checks, because a batch
axis that merely type-checks is the easy failure here:

1. **Bit-equality.** Row *i* of a batched denoise step is `torch.equal` to sample *i*
   run alone (max abs drift 0.0). A bound left at 1 would make rows 1..n-1 equal row 0
   instead; this is the assertion no threshold can fake.
2. **Distinctness.** The 16 output images correlate at most **0.547** pairwise, against
   0.592 for the reference's own 16 -- so the rows really are different pictures.
3. **Per-sample scoring.** Every row is scored against ITS OWN golden and the reported
   `e2e PCC=` is the WORST row, not the mean.

**What was actually reached, per stage.** B=32 is not uniform across the model, and the
limit is measured rather than assumed:

| stage | max verified batch | what stops it going higher |
|---|---|---|
| Qwen3 text encoder (both heads) | **32** | -- |
| Flux2 denoise transformer | **32** | -- |
| VAE encode (all 4 routes) | **32** | -- |
| VAE decode (all 3 routes) | **16** | at B=32 the decoder program's static circular buffers occupy 1412288 B of the chip's 1499136 B of L1, leaving 86848 B for the conv halo plus every resident activation. Checked at `l1_small_size` 24576 / 32768 / 40960 / 57344 / 61440 / 65536 / 131072: below the window the conv halo cannot allocate, above it the L1 buffers collide with the CB region. `ttnn`'s DRAM width slicing cannot absorb the batch either -- `num_slices` is capped at `ceil(W/32)`, which is 4 for the failing 128-wide conv. |
| VAE encode **and** decode in one process (Calls 3, 4) | **<16** | the two halos do not co-exist: at `l1_small_size=65536` the `(128,128,128,256)` conv is still 2752 B/bank short |

So Call 2 runs the full 32, Call 1 runs 16, and Calls 3 and 4 do not yet fit a batch at
all. That is a device-capacity ceiling with a named cause, not a shortcut.

**One upstream bug found and fixed.** `models/tt_dit/layers/normalization.py`'s
`GroupNorm.forward` asked `ttnn.group_norm` for `num_out_blocks=-1`, and that heuristic
sizes the per-core chunk from the RESHAPED volume (`H*W*C`) -- it never sees the leading
batch, so the circular buffers grow with B and overrun L1 at B > 1. Measured on the
128-channel 256x256 norm against a 1.5 MB limit:

```
B=1   -1:ok        1:1.97MB    2:ok
B=2   -1:1.97MB    2:1.97MB    4:ok
B=4   -1:1.97MB    4:1.97MB    8:ok
B=32  -1:ok       32:ok       64:ok
```

No single multiplier works for every shape (2*B fixes the decoder and breaks the
encoder), so `forward` now CLIMBS: it retries with twice the block count when the build
throws for CB overflow -- safe, because that throw happens before anything executes --
and caches the winning count in `default_num_out_blocks`, the per-shape table that class
already declared and never consulted. Inert at B=1, and the B=1 VAE PCCs are unchanged
to six decimals (encode 0.999172, decode 0.999772).

A second attempt -- scaling `Conv2d`'s DRAM `slice_params` by the batch -- was
**reverted**: it produces `num_slices > max_num_slices` and is invalid.

## Running it

```bash
cd /home/ttuser/tt-metal

# Everything (13 min on a 1x8 mesh)
./python_env/bin/python -m pytest models/demos/flux_2_klein_9b/tests/e2e -s

# Gate 1 -- host only, seconds
./python_env/bin/python -m pytest models/demos/flux_2_klein_9b/tests/e2e/test_gates.py -s
./python_env/bin/python -m pytest models/demos/flux_2_klein_9b/tests/e2e/test_layout_parity.py -s

# Gates 2 + 3 -- on device
./python_env/bin/python -m pytest models/demos/flux_2_klein_9b/tests/e2e/test_e2e_pipeline.py -s

# Command 3 -- trace contract + fully-on-device check
./python_env/bin/python -m pytest models/demos/flux_2_klein_9b/tests/e2e/test_trace_contract.py -s

# Demos
./python_env/bin/python -m models.demos.flux_2_klein_9b.demo.demo_text_to_image \
    --prompt "a red apple on a wooden table" --height 256 --width 256 --steps 4 -o apple.png
./python_env/bin/python -m models.demos.flux_2_klein_9b.demo.demo_text_generation --compare-hf
./python_env/bin/python -m models.demos.flux_2_klein_9b.demo.demo_image_edit --steps 2 -o edit.png
./python_env/bin/python -m models.demos.flux_2_klein_9b.demo.demo_vae_roundtrip --compare-hf
```

Add `--compare-hf` to any demo to run the HF golden beside it and print
`e2e PCC=<x>`.

## Trace contract (Command 3)

`tt/pipeline.py` declares

```python
PIPELINE_STAGES = ["encode_text", "vae_encode", "denoise", "vae_decode", "prefill", "decode"]
```

and exposes, per stage, `<stage>_trace_setup(inputs)` / `<stage>_trace_step()` /
`<stage>_trace_inputs()` (zero-arg, assembled from the captured reference tensors) /
`<stage>_trace_items()` (items retired per step — the only input to the arithmetic
ceiling, so `denoise` reports the whole joint token count, not 1).
`build_pipeline(device, model=None, layers=None, **kwargs)` **returns** the resident
object; it never runs the model.

`layers` is the default depth for every repeated block (`None` = all layers, never
0), with per-stack overrides named after the stages that own a stack:
`encode_text_layers`, `denoise_layers`, `vae_encode_layers`, `vae_decode_layers`,
`prefill_layers`, `decode_layers`. The checkpoint declares **five** sections and this
pipeline holds each as its own stack — the Qwen3 trunk, the 8 double DiT blocks, the
24 single DiT blocks, the VAE's 4 encoder down blocks and its 4 decoder up blocks —
which is why one number is not enough. Every stack is a plain Python list of
same-typed elements (`stage.blocks`, `stage.double_blocks`, `stage.single_blocks`,
`stage.down_blocks`, `stage.up_blocks`) so a structure walk can find, size and cap it,
and the HF reference stays reachable as `pipe.hf`.

Two properties make that walk work, and both are load-bearing:

* **Every stack exists as soon as `build_pipeline` returns.** Constructing a stage
  lays out its block list and stages **no** weight; the stage's own `build()` — called
  by its every entry point — does the device work. So the object the profiler walks
  has all five sections at their real depth, and building it needs neither a
  particular mesh nor 40 GB of device weights (measured: 39 s at `layers=2`).
* **A cap floors at three blocks per stack** (`tt/depth.py`). `find_all_stacks` only
  recognises a stack once it holds three same-typed blocks, so a section capped to one
  or two is not shallow — it is invisible, and its depth then gets inferred for the
  whole run. `test_trace_contract.py::test_every_declared_section_is_a_discoverable_stack`
  pins this against the checkpoint's own declared section list.

`trace_capture_selftest(device)` captures one step per stage, executes it, PCCs it
against the eager step and releases the trace before the next stage.
`host_op_selftest()` runs each head's model math under
`scripts.tt_hw_planner.host_op_observer.observe_host_ops()` with input encoding and
weight build outside the observed region. It exists twice on purpose: as a method
(`pipe.host_op_selftest()`, for a pipeline you already hold) and as a **module-level**
`tt.pipeline.host_op_selftest()` that opens its own mesh via `mesh.py` — the zero-arg
seam the emit-e2e observer probe calls.

## Known holes

**Batch = 32 was a hole and is not one any more.** An earlier revision of this package
reported it as not applicable, on the grounds that only Call 2 is an autoregressive
decode. That framing was wrong for this model: the requirement is to batch over the
model's OWN iteration, and for a diffusion pipeline that iteration is the denoise loop,
which takes a leading batch axis directly. The real obstacle was the graduated stubs'
hardcoded leading 1, and that turned out to be a small, auditable delta rather than an
out-of-scope rewrite. See [Batch = 32](#batch--32).

**Image sizes must be a multiple of 256.** `ttnn.group_norm`'s DRAM grid check needs
`Ht = N*H*W/32` to be a multiple of the core grid's virtual rows, and tt_dit's
`GroupNorm` pins `CoreGrid(8, 8)`. A 224 image's 28x28 latent gives `Ht = 25`, which
no grid divides, so the VAE's 512-channel mid block fails before any arithmetic —
which is also why the graduated VAE stubs were PCC'd at 256x256 / 32x32 rather than
at their captured 224. `tt/vae.py::check_resolution` rejects anything else.

**`ttnn.reshape` cannot split a tiled axis in this build** (the `.so` predates the
checked-out kernel sources). Every rank change in the latent plumbing happens in
`ROW_MAJOR`, and head splits use `nlp_create_qkv_heads` / `nlp_concat_heads`. A
rebuild would remove the constraint, not the correctness.

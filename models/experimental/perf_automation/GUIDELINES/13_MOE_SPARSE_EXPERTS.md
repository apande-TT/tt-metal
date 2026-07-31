# 13. Mixture of Experts — sparse matmul, router, and the expert tail

Levers for a Mixture-of-Experts decoder: `ttnn.sparse_matmul` expert projections, the
router (top-k + softmax + scatter), and the SwiGLU expert tail.

Every number below is measured on gpt-oss-20b, one Blackhole p150, batch 1, TP 1,
decode. The campaign moved 21.0 -> 73.0 tok/s/user. Shapes: 24 layers, hidden 2880,
Kt 90, 32 experts, top-4. Expert weights bfloat4_b, attention and lm_head bfloat8_b.

Read `07_METHODOLOGY.md` for the sweep and noise rules. Read `09_PROFILING_AND_OP_ANALYSIS.md`
to find the top bucket first. Do not apply anything here before the profile says the
expert or router ops are the bucket.

---

## 1. Do not trust an isolated sweep — it mispredicted 4 times {#moe-sweep-trap}
<!-- route
op_class: matmul
rank: time
lever_type: search
-->

An isolated single-op harness mispredicted the in-model result **four times** in one
campaign: o_proj layout, the first qkv attempt, expert activation dtype, and RMSNorm
sharding. Two causes, both mechanical:

1. **The harness rebuilt the descriptor per call.** Building `ShardSpec` or a
   `ProgramConfig` inside the timed loop adds 19 us at 8 cores and 97 us at 80. That
   swamps the effect and even inverts the ranking. Build the config once, then time.
2. **The in-model op uses a different program factory.** The model's o_proj takes an
   `L1_WIDTH_SHARDED` in0, so the sharded factory runs, not the interleaved one the
   sweep exercised. A sweep on interleaved inputs says nothing about it.

Before trusting a sweep delta, answer both questions:

- Can the swept parameter physically affect this op's bottleneck? A parameter that
  only touches L1 residency cannot help an op that is dispatch-bound.
- Does the harness build descriptors inside the timed region?

**`ttnn` silently discards an invalid program config and falls back to its default.**
An op whose Tracy time does not move was probably **rejected**, not merely ineffective.
Always confirm the op's **core count changed** in the profile. Three silent rejections
found in one campaign, all from integer division:

```python
per_core_N = Nt // num_cores          # WRONG: 90 // 12 = 7 -> 13 blocks > 12 cores
per_core_N = -(-Nt // num_cores)      # ceil

per_core_M = m // 32                  # WRONG: decode passes m=1 -> 0 -> invalid config
per_core_M = max(1, -(-m // 32))      # the tensor is tile-padded to 32 rows
```

A dataclass that re-declares a field as `None` in a subclass also silently disables it.

---

## 2. Core count follows bytes per core, not core count {#moe-cores-per-byte}
<!-- route
op_class: matmul
bound: dram
rank: time
grid: partial,tiny
lever_type: search
-->

More cores help **only** when each core holds enough work to starve the pipeline.
Measured both directions in the same model:

| op | change | MB/core before | result |
|---|---|---|---|
| gate+up `sparse_matmul` | 30 -> 45 cores | 1.24 | **20% faster** (4.142 -> 3.295 ms/tok) |
| qkv matmul | 40 -> 80 cores | 0.009 | **slower** (1.110 -> 1.396 ms/tok) |
| down `sparse_matmul` | 30 / 45 / 90 | 0.6 | flat, 1.6% spread |

Rule: above about **1 MB/core** more cores help; below about **0.83 MB/core** they do
nothing or they hurt. Compute `bytes_per_core = weight_bytes / num_cores` before you
sweep, and skip the sweep when the op is already past the knee.

Pick a core count that **divides the work exactly**. The SwiGLU kernel accepts 72 of
80 cores because 72 divides its 360-tile workload and fits a `min(8, NC)` grid on a
10-row device. Asking for 80 makes the divisibility loop silently decrement to 72, so
`NC=80` is a mirage.

---

## 3. Retune `in0_block_w` after every core-count change {#moe-in0-block-w-coupling}
<!-- route
op_class: matmul
bound: dram
rank: time
lever_type: walk
-->

Changing the core count changes `per_core_N`, which changes L1 pressure, which moves
the best `in0_block_w`. Re-walk it **in the model** after any core change. Three wins
came from this coupling. Measured walk for gate+up at 45 cores (ms/tok, whole decode):

| `in0_block_w` | 45 | 30 | 18 | 15 | 10 | 9 | 6 | 5 | **3** | 2 |
|---|---|---|---|---|---|---|---|---|---|---|
| decode ms/tok | 13.475 | 13.359 | 13.264 | 13.276 | 13.175 | 13.168 | 13.127 | 13.078 | **13.000** | 14.082 |

The best value is **per projection**. Do not share one value:

| projection | best `in0_block_w` |
|---|---|
| gate+up | 3 |
| down | 9 |
| qkv | 2 |

Apply the walk only when (a) the core count changed **and** (b) the op reads tens of
MB, so L1 residency matters. `Kt % in0_block_w == 0` is required; snap to the largest
divisor at or below your target.

`out_subblock_w = 1` is catastrophic everywhere — gate+up 21.038 vs 12.677 ms/tok
baseline, down 14.235, qkv 16.370. Wide subblocks carry the pipeline. But see §4
before raising it.

---

## 4. `out_subblock_w > 1` corrupts output silently — guard it {#moe-subblock-guard}
<!-- route
op_class: matmul
rank: time
lever_type: single-shot
-->

The compute kernel `bmm_large_block_zm_fused_bias_activation.cpp` splits the reduction
over K into chunks when `in0_block_w` does not cover the depth. It spills the running
total to `cb_intermed0` in L1 and reloads it with `reload_from_cb_to_dst`. **The reload
restores only the first tile of each subblock; later tiles are double counted.**

Nothing crashes and nothing warns. The wrong values feed the next layer, grow, and
reach infinity a few decode steps later, so the model writes nonsense.

Three conditions must hold at the same time:

1. `out_subblock_w > 1` — the subblock spans more than one tile
2. `num_blocks_w_dim > 1` — that is `per_core_N / out_block_w`
3. `num_blocks_inner_dim > 1` — that is `Kt / in0_block_w`

Measured: `osw=2` gives 45 of 90 tiles wrong, median ratio exactly 2.000, even tiles
bit-identical. `osw=3` gives 60 of 90 wrong, period 3. Single-variable proof through
`in0_block_w` alone: `ib=30` (spill) scores 0.6667 top-1, `ib=90` (no spill) scores
1.0000 on the same osw.

Remove any one condition:

```python
in0_block_w = Kt          # num_blocks_inner_dim = 1, no spill at all
out_block_w = per_core_N  # num_blocks_w_dim = 1
out_subblock_w = 1        # last resort, it is slow (see §3)
```

`out_block_w` must derive from `out_subblock_w`, never a hardcoded 1:
`in1_num_subblocks = out_block_w / out_subblock_w`, so a hardcoded 1 makes that **0**
for any `osw > 1` and the kernel deadlocks.

A second, separate limit: the destination register file holds **8** tiles, or **4**
with `fp32_dest_acc_en`. `out_subblock_h * out_subblock_w` above that also corrupts
the output (`osw=6` gives PCC 0.65).

Use `cc_optimize/matmul_guard.py` to reject a bad config before it reaches the device:

```python
from matmul_guard import check_matmul_program_config
check_matmul_program_config(
    name="gate_up", Kt=90, in0_block_w=3, per_core_N=4,
    out_block_w=4, out_subblock_w=4, out_subblock_h=1,
    fp32_dest_acc_en=False,
)   # raises ValueError naming all three loop counts and both escapes
```

Call it where the model builds its program config, after any snap-to-divisor logic, so
the checked values are the ones the kernel receives.

---

## 5. Sweep the MoE sparse matmul {#moe-sparse-sweep}
<!-- route
op_class: matmul
bound: dram,both
rank: time
regime: decode
lever_type: search
-->

`cc_optimize/moe_sparse_matmul_sweep.py` sweeps the sparse expert GEMM. It is
model-agnostic: pass shapes on the command line. Axes are `--cores`, `--in0-block-w`,
`--out-subblock-w`, `--obw-mult`, `--act-dtypes`, `--fidelities`, plus `--experts`,
`--active` and `--nnz`.

```bash
# gate+up: the fused [gate|up] projection, sparse operand is the weight
python cc_optimize/moe_sparse_matmul_sweep.py --proj gate_up \
    --K 2880 --N 5760 --experts 32 --active 4 --sparse-input b

# down: the activation is the sparse operand
python cc_optimize/moe_sparse_matmul_sweep.py --proj down \
    --K 2880 --N 2880 --experts 32 --active 4 --sparse-input a
```

Score **per tile**, against the **device** output at `out_subblock_w=1`, never against
a torch reference. A whole-tensor PCC hides a periodic error: the isolated harness
reports the corrupt config from §4 as bit-identical and passing.

Two axis results worth reusing. `out_block_w > out_subblock_w` won 21 of 21 controlled
groups on gate+up, but every winner landed on `out_block_w == per_core_N`, so it is the
`num_blocks_w_dim = 1` effect from §4, not an independent lever. And a static
`nnz = num_experts_per_tok` beat runtime inference: probing 1704 calls gave a non-zero
distribution of `{4: 1392, 32: 120, 128: 96, 1024: 48, 2048: 48}` with a minimum of 4,
so the static value cannot under-run.

**Know the floor before you sweep.** Scaling fit for the expert GEMM:
`ms = 0.0237 + 0.0431 * nnz`. Only 12% is fixed overhead and marginal bandwidth
asymptotes near 203 GB/s, 40% of a 512 GB/s peak, even at `nnz=8`. Both expert
projections already run at 51-53% of their weight-read roofline, so the remaining
headroom is small.

---

## 6. Fuse the router into one kernel {#moe-router-fusion}
<!-- route
op_class: reduction,eltwise
rank: count
bound: slow
regime: decode
lever_type: single-shot
-->

The router is top-k, softmax, then scatter into a sparsity vector. As separate ops it
is 5 launches per layer, each paying the dispatch floor. One custom kernel removed all
of them:

| op | before (ms/tok) | after |
|---|---|---|
| TopK | 0.356 | 0 |
| Softmax | 0.157 | 0 |
| FillPad | 0.190 | 0 |
| Pad | 0.049 | 0 |
| Scatter | 0.046 | 0 |

Operation count fell 1123 -> 835 per token and decode went 14.724 -> 14.128 ms/tok
(63 -> 66.7 tok/s/user). Two failure modes cost real debugging time:

- **Dtype.** The router matmul emitted `bfloat8_b` while the kernel read raw bfloat16
  halves, so ids came out `[0,1,2,8]` instead of `[26,4,22,17]`. Out-of-range writes
  wedged the NoC and hung the board. Force `dtype=ttnn.bfloat16` on the router linear
  and clamp the ids to `[0, num_experts)` inside the kernel.
- **Gating on the wrong predicate.** Gating on `is_decode` (true for any sequence at
  or under 128 tokens) zeroed 127 of 128 prefill rows and dropped accuracy to 0.0333.
  Gate on `router_logits.shape[0] == 1`.

---

## 7. Rank fusion by measured ms/tok, not by launch count {#moe-fusion-ranking}
<!-- route
op_class: eltwise,datamove
rank: count
bound: slow
lever_type: single-shot
-->

Launches are **not** uniformly priced. A 1-tile `ttnn.clone` costs **9.12 us** and a
1-tile typecast 10.49 us — that is the dispatch floor. But a 0.8 us reshard that
unlocks a 33 us saving is a large win. Rank candidates by measured net ms/tok.

Piecemeal fusion loses; one monolithic kernel wins. Merging a kernel's own reader and
writer changed nothing (15.1 us both ways) because the kernel's read cost is real work,
not overhead.

Two negative results worth not repeating:

- **Expert-tail pre-scale.** Folding the routing weight into the SwiGLU output before
  the down projection is valid for the matmul term by linearity, but the bias is added
  **after**, so correctness needs `bias * w` — the same `[1,32,1,2880]` multiply you
  removed. Net zero. Also blocked in the kernel: `mul_tiles_bcast_scalar` reads both
  operands from circular buffers, and there is no DST-times-scalar primitive.
- **L1 pinning of a small DRAM operand.** Pinning the down-projection bias in L1 cut
  DRAM operations 48 -> 24, but total time did not move (0.400 -> 0.398 ms/tok). The
  cost **relocated**. These adds run at 40-70 GB/s against a 512 GB/s peak, so they
  are dispatch-bound, not bandwidth-bound.

---

## 8. Write narrower, and stop computing padding {#moe-dtype-and-padding}
<!-- route
op_class: matmul
bound: dram
rank: time
memory: dram_interleaved,l1_interleaved
lever_type: walk
-->

Cheap, reliable wins on the expert path:

| change | ms/tok saved |
|---|---|
| gate+up `sparse_matmul` writes bfloat16 not bfloat8_b | 0.640 |
| down `sparse_matmul` writes bfloat16 | 0.343 |
| qkv output to L1 instead of DRAM | 0.581 |
| trim lm_head to the real vocabulary size | 0.547 |

The vocabulary trim: a padded power-of-two vocabulary (262144 against 201088 real)
makes the output projection compute about **23% zero columns every token**. The tail is
all zeros and the real vocabulary is a prefix, so trimming on device is exact and
greedy argmax returns the identical token.

Watch the direction of a dtype change. Writing wider can still win when it deletes a
later op: emitting bfloat16 logits cost +0.041 ms/tok in the projection but removed a
0.058 ms/tok typecast, because the greedy path needs bfloat16 for `ttnn.argmax`.

**`ttnn.as_tensor(memory_config=...)` is ignored on a cache-file hit.** Add an explicit
`to_memory_config` after load, or the weight silently stays where the cache put it.

---

## 9. Reject an accuracy regression even when it passes the gate {#moe-accuracy-discipline}
<!-- route
op_class: matmul,reduction
rank: time
lever_type: single-shot
-->

A change that drops accuracy while still clearing the threshold is a change that will
be blamed later. Two rejected in this campaign, both passing the gate:

- uint32 ROW_MAJOR router ids: 0.9667 -> 0.9333 top-1 for 0.05 ms/tok
- a down projection config at 30 cores: 0.9667 -> 0.9333 for +0.4 tok/s

Validate a layout or `memory_config` change on **both** harnesses. The accuracy check
runs with tracing disabled; the perf demo captures a trace. A config can pass accuracy
and then hit a `TT_FATAL` under trace.

Refuse a config that is known unsafe even if it might be fast. gate+up at
`out_subblock_w=2` was never measured because `per_core_N=4` gives
`num_blocks_w_dim=2` with `num_blocks_inner_dim=30`, which is all three §4 conditions
at once.

---

## 10. When the MoE decode path is finished {#moe-exhausted}
<!-- route
op_class: matmul
rank: time
regime: decode
lever_type: single-shot
-->

Signals that model-level tuning is done, from the gpt-oss campaign at 73.0 tok/s/user
and 12.664 ms/tok device:

- Two matmul groups hold **72%** of decode (Matmul 4.55, SparseMatmul 4.53 ms/tok) and
  both sit at **51-53%** of their measured weight-read roofline.
- Every remaining op is near the **9 us** dispatch floor, which a 1-tile clone also pays.
- The last four accepted steps returned 0.34, 0.16, 0.11 and 0.04 ms/tok. The final one
  is below wall-clock noise and was kept only on Tracy evidence.

Check the **wall minus device** gap before blaming the model. Measured 13.70 wall
against 12.664 device = 1.06 ms/tok, of which only 0.07 ms is device work outside the
captured trace. The rest is trace launch, the token read back, and the host loop —
harness cost, not model cost. A single int32 read back costs 0.097 ms of pure latency
and cannot be hidden, because the host must know the token before the next step.

Two items stay blocked from Python and need a change inside tt-metal:

- An unconditional FILL inside the sparse matmul C++ path, 0.612 ms/tok.
- A hard `TT_FATAL` "Sharded output not supported for GQA", which blocks removing an
  SDPA round trip through DRAM.

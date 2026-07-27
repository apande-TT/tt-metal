<!-- BEGIN optimize -->
# Optimize (perf) — `llama3_1_8b_p150`

_Updated live: 2026-07-27 06:07:20 UTC · 80 lever attempt(s) so far — each knob is logged the instant it resolves, win OR fail, with why it was tried and why it won or failed._

```
Optimization summary — llama3_1_8b_p150 · main (device_ms)
==========================================================
optimizing… — baseline->final speedup is finalized when the module converges (per-attempt detail below is live)
tracy trace pass, same window (16 layers):  33.89 ms

Roofline & utilization
  modeled floor       : 537.23 ms   (Σ per-op roofline floors)
  achievable (60-80%) : 671.54 - 895.38 ms
  measured            : 758.37 ms
  at-floor            : 71%   (221.14 ms reachable headroom)
  status              : IN_BAND — reached the achievable band — done
  (tok/s/u — N/A: not an LLM decode pipeline)

Op breakdown — device time by op class (latest profile · what to target, ranked):
op class         device_ms      %   count  bound  dominant op (shape)
---------------------------------------------------------------------------------------------------
reduction          1205.58  48.9%    2220   slow  TopKDeviceOperation
matmul             1010.23  41.0%    5600   slow  MatmulDeviceOperation 128 x 4096 x 14336
attention           114.25   4.6%    2896   slow  NlpCreateHeadsDeviceOperation
host_overhead       106.68   4.3%       0   host  
eltwise              56.15   2.3%    3184   slow  BinaryNgDeviceOperation
datamove             51.53   2.1%    7141   slow  NLPConcatHeadsDeviceOperation
other                24.10   1.0%    2618   slow  NLPCreateQKVHeadsDecodeDeviceOperation
embedding             2.34   0.1%     114   slow  EmbeddingsDeviceOperation

op                                 grid      fidelity  dtype     shard     host      tt-lang   cpp       other       best ms
----------------------------------------------------------------------------------------------------------------------------
LayerNormDeviceOperation           ·try      —         —         ·try      ·try      —         —         —           1057.73
MatmulDeviceOperation              ✓win      —         ✓win      ✓win      ·try      ·try      ·try      ✓win        1061.00
MatmulDeviceOperation              ·try      —         ✓win      ·try      ·try      ✓win      ✓win      ·try        1138.67
MatmulDeviceOperation              ✓win      —         —         —         —         —         —         —           1092.12
MatmulDeviceOperation              ·try      —         ✓win      ·try      ·try      ✓win      ·try      ·try        1057.68
MatmulDeviceOperation              ·try      —         ✓win      ✓win      ·try      ·try      ·try      ✓win         891.98
MatmulDeviceOperation              ✓win      —         —         —         —         —         —         —                 —
NlpCreateHeadsDeviceOperation      ·try      —         —         ✓win      ✓win      ✓win      —         —            955.25
TopKDeviceOperation                ✓win      —         —         ✓win      —         ✓win      —         —                 —
TopKDeviceOperation                ·try      —         —         ·try      ✓win      —         —         —           1537.69
host_overhead                      —         —         —         —         —         —         —         ✓win              —


Per-attempt detail (every optimization tried — win OR fail — with gain vs baseline and WHY):
op                                        lever        ms  gain vs base  result     why tried / why it won or failed
--------------------------------------------------------------------------------------------------------------------
TopKDeviceOperation                        grid         —             —  ✓ win      committed: llama3_1_8b_p150: profile the real forward, not a crashed decode_step(None) The tracy path called pipeline.decode_step(None), which raised o
TopKDeviceOperation                        grid   2464.16      +0.02 ms  · no gain  TopK is grid=tiny (1 core, ~10ms/call, 1120ms = 45% of device time), so I handed it an explicit full compute-grid sub_core_grid_topk via ModelArgs. No gain: 2464.18->2464.16ms. Root cause is factory S
TopKDeviceOperation                       shard   2464.18      +0.00 ms  · no gain  TopK is memory-bound on a DRAM-interleaved sampling chain, so I width-sharded the top-k values/indices into L1 via DECODE_SAMPLING_INPUT_MEMCFG (the designed hook, copied from the llama3_70b_galaxy co
TopKDeviceOperation                       shard   2464.18      +0.00 ms  · no gain  Second shard variant: same L1 lever but the whole 2x32 gathered row as ONE shard, on the theory that splitting it across 2 cores was desynchronising the device-offset add. Also reverted -- PCC 9.4% To
TopKDeviceOperation                       shard         —             —  ✓ win      committed: llama3_1_8b_p150: let greedy decode take the argmax sampling path format_sampling_params already normalises temperature=0 rows to k=1 / p=0
TopKDeviceOperation                  structural   1537.69    +926.49 ms  ✓ win      Hypothesis: TopK's 1120ms is not a tunable cost but REDUNDANT WORK -- decode is greedy (temperature=0), and greedy needs only an argmax, not top-k/top-p/RNG. format_sampling_params already normalises 
TopKDeviceOperation                     tt-lang         —             —  ✓ win      committed: llama3_1_8b_p150: warm up only the sampling shapes the request asks for prefill_forward_text called warmup_model_prefill without greedy_only
TopKDeviceOperation                  structural   1164.57   +1299.62 ms  ✓ win      Second structural pass on the same op: after force-argmax took TopK off the per-token path, 16 TopK calls (163ms) survived, and the profiler ordering proved they were ALL warmup -- every one precedes 
MatmulDeviceOperation                      grid   1164.57   +1299.62 ms  · no gain  Hypothesis: this DRAM-bound bf4_b w1/w3 read runs on only 32 of ~110 cores because 2D mcast gives one core ROW per per_core_M block, and short prefill has just m_tiles=4 -- so I swapped short prefill 
MatmulDeviceOperation                      grid   1164.57   +1299.62 ms  · no gain  Second grid attempt, this time the FULL GUIDELINES/11 recipe rather than a program_config alone: width-shard the prefill activation into L1 across a core count that divides both K-tiles(128) and N-til
MatmulDeviceOperation                     dtype         —             —  ✓ win      committed: llama3_1_8b_p150: write the prefill MLP intermediates as bf8_b w1/w3 are already at the bf4_b weight floor, so the only dtype left on this D
MatmulDeviceOperation                     dtype   1144.58   +1319.60 ms  ✓ win      Hypothesis: the op is DRAM-bound but its w1/w3 weights are ALREADY bf4_b, the floor -- so the only dtype left is the output. ff1/ff3 each write a [128,14336] bf16 intermediate (3.67MB/layer) that is r
MatmulDeviceOperation                     shard   1138.67   +1325.51 ms  · no gain  Reused a catalogued lever: L1 island for the ff1/ff3 -> SILU mul -> ff2 chain in short prefill (weights are ~29MB/layer and can never be L1-resident, but the [seq,hidden] intermediates can, removing t
MatmulDeviceOperation                     shard   1145.94   +1318.24 ms  · no gain  Second shard variant, distinct mechanism from the L1 island: make the prefill activation L1-resident so ff1 and ff3 both read it from L1 instead of each paying a DRAM read. PCC clean at 0.995949 but N
MatmulDeviceOperation               tp-fracture   1144.58   +1319.60 ms  · no gain  tp_pick_degree(128, 4096, 14336) returned best_tp=1, i.e. keep this matmul single-chip: the on-mesh TP sweep is disabled by default because it opens a NESTED mesh device and toggles fabric config whil
MatmulDeviceOperation                structural   1144.58   +1319.60 ms  · no gain  Found real reducible work: get_padded_prefill_len hard-floors at 128, so this benchmark's 6-token prompt pays a FULL 128-token prefill -- 4x the matmul work a 32-row (one tile) prefill would need, on 
MatmulDeviceOperation                   tt-lang   1144.58   +1319.60 ms  · no gain  Authored tt/ttl_gated_ffn.py: a real multi-core tt-lang kernel computing silu(x@w1)*(x@w3) in ONE op so both [seq,hidden] intermediates stay in L1 -- the exact fusion GUIDELINES/11 names as highest-va
MatmulDeviceOperation                   tt-lang         —             —  ✓ win      committed: llama3_1_8b_p150: record the tt-lang gated-FFN kernel and why it loses tt/ttl_gated_ffn.py is a real multi-core tt-lang kernel computing sil
MatmulDeviceOperation                       cpp   1144.57   +1319.61 ms  · no gain  Authored tt/cpp_mm_generic.py + tt/kernels/{compute/mm_gated_ffn,dataflow/reader_mm_partitioned,dataflow/writer_mm_partitioned}.cpp: a real C++ Metalium reader/compute/writer triple (adapted from the 
MatmulDeviceOperation                       cpp         —             —  ✓ win      committed: llama3_1_8b_p150: record the C++ Metalium matmul rung and why it loses tt/cpp_mm_generic.py drives a real Metalium reader/compute/writer tri
MatmulDeviceOperation                      grid   1135.53   +1328.65 ms  · no gain  ff2's K is the hidden dim, 448 tiles -- the longest reduction in the model -- but matmul_config derives in0_block_w from find_largest_divisor(), which hard-caps at 8, so K is walked in 56 blocks and t
MatmulDeviceOperation                      grid   1101.85   +1362.33 ms  · no gain  Reused catalogued prior art (commit d8cd69d734, reverted by the from-scratch baseline): use_minimal_qkv_prefill_matmul() claims every seq_len>128, so the 2D-mcast branch only ever sees M<=4 tiles, yet
MatmulDeviceOperation                      grid         —             —  ✓ win      committed: llama3_1_8b_p150: size the short-prefill QKV and ff2 matmul configs to the work Two program-config levers on the prefill matmuls, both cases
MatmulDeviceOperation                      grid   1092.12   +1372.06 ms  ✓ win      COMMITTED (supersedes the earlier reverted record of the same lever). Re-applied the catalogued QKV short-prefill config together with the ff2 in0_block_w blocking, after proving the full-pipeline gat
MatmulDeviceOperation                      grid   1092.12   +1372.06 ms  ✓ win      COMMITTED (supersedes the earlier reverted record). ff2's in0_block_w was hard-capped at 8 by find_largest_divisor(), so its 448-tile K -- the longest reduction in the model -- was walked in 56 blocks
MatmulDeviceOperation                     dtype         —             —  ✓ win      committed: llama3_1_8b_p150: put the FF2 down-projection weights on bf4_b performance() already rides bf4_b for FF1/FF3 but left FF2 on the BFP8 defaul
MatmulDeviceOperation                     dtype   1063.78   +1400.40 ms  ✓ win      Hypothesis: ff2 is DRAM-bw bound and performance() had already put FF1/FF3 on bf4_b but left FF2 on the BFP8 default, so the down-projection read 2x the weight bytes of the two projections feeding it 
MatmulDeviceOperation                     shard         —             —  ✓ win      committed: llama3_1_8b_p150: keep the short-prefill MLP intermediates in an L1 island The MLP's DRAM traffic is dominated by weight reads that can neve
MatmulDeviceOperation                     shard   1061.00   +1403.18 ms  ✓ win      Reused the catalogued L1-island lever, this time aimed at ff2 (which READS the intermediate) rather than at the producing matmul: ff1/ff3 outputs land in L1 so the ff1/ff3 -> mul -> ff2 chain never ro
MatmulDeviceOperation               tp-fracture   1061.00   +1403.18 ms  · no gain  tp_pick_degree(128, 14336, 4096) returned best_tp=1 -- keep single-chip. Same reason as the other matmul: the on-mesh TP sweep is disabled by default because it opens a nested mesh device and toggles 
MatmulDeviceOperation               tp-fracture   1061.00   +1403.18 ms  · no gain  tp_pick_degree(128, 14336, 4096) returned best_tp=1 -- keep w2 single-chip. The on-mesh TP sweep is disabled by default because it opens a nested mesh device and toggles fabric config while a mesh is 
MatmulDeviceOperation                structural   1061.00   +1403.18 ms  · no gain  none: investigated ff2's surroundings for reducible work and found none left. (1) Its output all-reduce is already a no-op here -- tt_all_reduce short-circuits and returns the input unchanged when mes
MatmulDeviceOperation                   tt-lang   1061.00   +1403.18 ms  · no gain  Authored tt/ttl_ff2_matmul.py: a multi-core tt-lang matmul measured on ff2's OWN shape (128x14336x4096, 8x8 grid, 128 N-tiles / 64 cores = 2 each exact, K reduced in-core via an accumulator DFB ping-p
MatmulDeviceOperation                       cpp   1061.00   +1403.18 ms  · no gain  Ran the authored C++ Metalium generic_op triple (tt/cpp_mm_generic.py + tt/kernels/*.cpp) on ff2's OWN shape (128x14336x4096) across the full 11x10 grid: CORRECT at PCC 0.993625 but 3.113ms/call vs 0.
MatmulDeviceOperation               tp-fracture         —             —  ✓ win      committed: llama3_1_8b_p150: record the ff2-shape kernel rungs tt/ttl_ff2_matmul.py measures the tt-lang rung on ff2's OWN shape rather than inheriting
MatmulDeviceOperation                      grid   1061.12   +1403.06 ms  · no gain  find_grid_k_n hard-codes max_rows=max_cols=8 (a Wormhole 8x8 assumption) and it is what sizes the DRAM-sharded DECODE matmuls, so on an 11x10 P150 it discards 40%+ of the chip before its divisibility 
MatmulDeviceOperation                      grid   1063.96   +1400.22 ms  · no gain  Decode ff2 (m=32 -> the per-token path) is DRAM-sharded on mlp2_core_grid, so I attacked the two grid-sizing helpers that pick its core count: find_grid_k_n hard-coded Wormhole 8x8, and find_grid both
MatmulDeviceOperation                     dtype   1060.95   +1403.23 ms  · no gain  Decode ff2's WEIGHT is already at the bf4_b floor (commit 67acf99 flipped all 896 w2 instances, decode included), so the only dtype bytes left on this DRAM-bw-bound matmul are the ACTIVATIONS flowing 
MatmulDeviceOperation                     dtype         —             —  ✓ win      committed: llama3_1_8b_p150: carry the DECODE MLP intermediates as bf8_b too The ff1/ff3 -> SILU mul -> ff2 chain was already bf8_b in prefill but deco
MatmulDeviceOperation                     dtype   1057.72   +1406.46 ms  ✓ win      COMMITTED. Decode ff2's WEIGHT is already at the bf4_b floor, so the only dtype bytes left are the activations around it. First tried the model-wide walk (TensorGroup.ACTIVATION=BFP8) and REVERTED it 
MatmulDeviceOperation                     shard   1057.68   +1406.50 ms  · no gain  Hypothesis: on a DRAM-bw-bound decode ff2 the bytes to remove are the 33 MB w2 read, so the shard lever means L1 weight residency; the activation side was already fully sharded. Two findings. (1) L1-r
MatmulDeviceOperation                     shard   1057.68   +1406.50 ms  · no gain  Hypothesis: on a DRAM-bw-bound decode ff2 the bytes to remove are the 33 MB w2 read, so the shard lever here means L1 WEIGHT residency; the activation side of the chain was already sharded. Two result
MatmulDeviceOperation               tp-fracture   1057.72   +1406.46 ms  · no gain  Hypothesis: if this decode ff2 is DRAM-bandwidth bound and single-chip levers are spent, splitting the 33 MB w2 read across chips would divide the bytes each chip must fetch. tp_pick_degree(32, 14336,
MatmulDeviceOperation                structural   1057.72   +1406.46 ms  · no gain  Hunted for reducible work and found a real candidate, then DISPROVED it -- the answer is worth more than the lever. A full-depth op-signature probe showed ff2 is not uniformly at the bf4_b floor: 62 i
MatmulDeviceOperation                   tt-lang   1057.72   +1406.46 ms  · no gain  Hypothesis: the 128-row tt-lang result should NOT be assumed to carry to the decode shape, because M=32 is a quarter of the work and the two costs scale differently -- so I re-measured the kernel on f
MatmulDeviceOperation                       cpp   1057.72   +1406.46 ms  · no gain  Same reasoning as the tt-lang rung: do not inherit the 128-row verdict, measure the C++ Metalium kernel on ff2's OWN decode shape. Parameterised tt/cpp_mm_generic.py (reader/compute/writer triple adap
MatmulDeviceOperation                   tt-lang         —             —  ✓ win      committed: llama3_1_8b_p150: measure both kernel rungs on the DECODE ff2 shape The tt-lang and C++ Metalium rungs for ff2 had only ever been measured a
LayerNormDeviceOperation                   grid         —             —  · wedged   wedged/crashed when tried: perf test crashed at runtime: TT_FATAL: cq_id 0 is out of range (assert.hpp:104)
LayerNormDeviceOperation                   grid   1057.73   +1406.45 ms  · no gain  Hypothesis, and the profile backs it precisely: of LayerNorm's 68.4 ms, 54.5 ms is 832 instances running on FOUR cores, DRAM_INTERLEAVED. The interleaved rms_norm kernel parallelises ONLY over the inp
LayerNormDeviceOperation                   grid   1057.73   +1406.45 ms  · no gain  Second grid attempt on the same 4-core prefill norm, changing the PLUMBING rather than the occupancy target, to test whether the first attempt's trace crash came from the way the sharded result was ha
LayerNormDeviceOperation                  shard   1058.09   +1406.09 ms  · no gain  Third distinct attempt on this op, and the only trace-safe form of 'shard it into L1' left. Sharding the ACTIVATION is already measured-blocked (two plumbings, both crash trace capture with cq_id 0 ou
LayerNormDeviceOperation                  shard         —             —  · wedged   wedged/crashed when tried: perf test crashed at runtime: TT_FATAL: cq_id 0 is out of range (assert.hpp:104)
LayerNormDeviceOperation                  shard   1057.73   +1406.45 ms  · no gain  Second shard attempt, different mechanism, and it isolated the real blocker. Rather than sharding the norm (already measured-blocked), I moved the tensor the norm READS: made the short-prefill residua
LayerNormDeviceOperation             structural   1057.73   +1406.45 ms  · no gain  none: hunted for reducible work behind the 832 four-core prefill norms and every candidate is either already applied or a known dead end. (1) The 832 instances are NOT 26 redundant prefills inside the
host_overhead                         trace-2cq         —             —  ✓ win      committed: llama3_1_8b_p150: warm up only the prefill LENGTH the request asks for Prefill warmup ran a full, real prefill at EVERY padded length up to
host_overhead                      trace-capture    891.98   +1572.20 ms  ✓ win      COMMITTED, and the win was NOT the trace lever itself -- trace capture is already applied and engaged here (Generator owns trace_ids_decode, decode_forward(enable_trace=True) does the host I/O plus ex
MatmulDeviceOperation                      grid    891.98   +1572.20 ms  · no gain  Found genuinely DEAD code and revived it, which is how the rung got a real test. mlp.py already carried a full_grid_ff1_3 branch meant to run decode ff1/ff3 as an L1-width-sharded 1D-multicast matmul 
MatmulDeviceOperation                      grid    892.07   +1572.11 ms  · no gain  Second grid attempt, going after the helper that actually picks this op's core grid rather than the matmul variant. find_grid_k_n hard-codes max_rows=max_cols=8 (a Wormhole assumption) and it is what 
MatmulDeviceOperation                     dtype    891.43   +1572.75 ms  · no gain  w1/w3 are already at the bf4_b weight floor, so the only dtype step left on this op is its OUTPUT, which I had just walked bf16 -> bf8_b in c9b2d04. Took the last available step, bf8_b -> bf4_b, on th
MatmulDeviceOperation                structural    891.98   +1572.20 ms  · no gain  The real structural candidate for this op is the GATE+UP MATMUL FUSION, and I measured it rather than refactoring on faith. w1 (gate) and w3 (up) are both [dim, hidden], both read the SAME activation,
MatmulDeviceOperation                     dtype         —             —  ✓ win      committed: llama3_1_8b_p150: measure and reject the gate+up matmul fusion w1 (gate) and w3 (up) are both [dim, hidden], both read the same activation,
MatmulDeviceOperation                     shard         —             —  · wedged   wedged/crashed when tried: perf test crashed at runtime: TT_FATAL: cq_id 0 is out of range (assert.hpp:104)
MatmulDeviceOperation                     shard    892.04   +1572.14 ms  · no gain  Two things to record. First, the WEIGHT side is architecturally closed, same arithmetic as decode ff2: the only L1-residency path is the DramPrefetcher global CB, and is_prefetcher_supported() is Fals
MatmulDeviceOperation                     shard    892.05   +1572.13 ms  · no gain  Second shard attempt, on the OUTPUT side of the op rather than its input, and deliberately chosen as a variant known to be trace-safe (the equivalent edit completed cleanly on the ff2 rung, unlike the
MatmulDeviceOperation               tp-fracture    892.04   +1572.14 ms  · no gain  tp_pick_degree(32, 4096, 14336) returned best_tp=1 -- keep decode ff1/ff3 single-chip. Same two reasons as the sibling ff2 op: the on-mesh sweep is disabled by default because it opens a NESTED mesh d
MatmulDeviceOperation                   tt-lang    892.04   +1572.14 ms  · no gain  Measured the tt-lang kernel on this op's OWN shape rather than inheriting the ff2 verdict, since M and the K/N ratio both differ. Extended tt/ttl_ff2_matmul.py to sweep (M,K,N) triples; the multi-core
MatmulDeviceOperation                       cpp    892.04   +1572.14 ms  · no gain  Measured the C++ Metalium reader/compute/writer triple (via ttnn.generic_op, output tiles partitioned across the full 11x10 grid) on this op's OWN shape. On 32x4096x14336: PCC 0.999022, 1.039 ms/call 
MatmulDeviceOperation                     shard         —             —  ✓ win      committed: llama3_1_8b_p150: measure both kernel rungs on the decode ff1/ff3 shape too Both hand-kernel scripts now sweep (M, K, N) triples and cover e
MatmulDeviceOperation               tp-fracture    892.11   +1572.07 ms  · no gain  tp_pick_degree(32, 4096, 14336) returned best_tp=1 -- keep decode ff1/ff3 single-chip. Two reasons. The on-mesh sweep is disabled by default because it opens a NESTED mesh device and toggles fabric co
MatmulDeviceOperation               tp-fracture         —             —  ✓ win      committed: llama3_1_8b_p150: refresh the generated RUN_REPORT Checkpoints the live lever log so the working tree is clean. A dirty tree scopes record_k
MatmulDeviceOperation               tp-fracture         —             —  ✓ win      committed: llama3_1_8b_p150: document the MLP tensor-parallel fracture layout The w1_dims / w2_dims tuples ARE the TP fracture and nothing said so. Ver
MatmulDeviceOperation                      grid         —             —  ✓ win      committed: llama3_1_8b_p150: run the LM head on the full core grid The DRAM-sharded matmul variant width-shards the activation across `lm_head_core_gri
MatmulDeviceOperation                      grid    867.62   +1596.56 ms  ✓ win      Tried because the LM head is grid=partial for a STRUCTURAL reason, not a tuning one: the DRAM-sharded matmul width-shards the activation across its cores, so num_cores must divide K/32=128 tiles, and 
NlpCreateHeadsDeviceOperation              grid    955.25   +1508.93 ms  · no gain  Hypothesis: this op is grid=tiny because its stock interleaved program factory sizes cores from num_blocks = batch*seq_len/TILE_HEIGHT (one work unit per input row-tile) -- at batch 1 / seq_len 128 th
NlpCreateHeadsDeviceOperation             shard         —             —  ✓ win      committed: llama3_1_8b_p150: land the prefill head split in L1 nlp_create_qkv_heads is pure data movement and memory-bound: it reads the fused [S, (nq
NlpCreateHeadsDeviceOperation             shard    865.93   +1598.25 ms  ✓ win      Hypothesis: this op has no weights and is pure memory-bound data movement, so the shard lever here means removing DRAM round-trips on the tensors it writes -- it emits three head-major views of the fu
NlpCreateHeadsDeviceOperation        structural         —             —  ✓ win      committed: llama3_1_8b_p150: skip the prefill warmup when it duplicates the request Warmup was already narrowed (in earlier commits) to the sampling sh
NlpCreateHeadsDeviceOperation        structural    781.95   +1682.23 ms  ✓ win      Hunted for reducible work behind this op rather than its per-call cost, and found the call COUNT was double what the workload needs. Earlier commits narrowed prefill warmup to the sampling shapes and 
NlpCreateHeadsDeviceOperation           tt-lang         —             —  ✓ win      committed: llama3_1_8b_p150: tt-lang kernel for the prefill QKV head split The stock nlp_create_qkv_heads sizes its cores from num_blocks = batch*seq_l
NlpCreateHeadsDeviceOperation           tt-lang    758.37   +1705.81 ms  ✓ win      Hypothesis: a kernel is the RIGHT rung here because the knob rungs are closed by construction, not by tuning -- the stock op sizes cores from num_blocks = batch*seq_len/TILE_HEIGHT (one work unit per 

Code changes — every attempt (win or fail):
===========================================

[#2] TopKDeviceOperation · grid · no gain  +0.02 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/model_config.py b/models/demos/llama3_1_8b_p150/tt/model_config.py
    index d960555193..8736240d4f 100644
    --- a/models/demos/llama3_1_8b_p150/tt/model_config.py
    +++ b/models/demos/llama3_1_8b_p150/tt/model_config.py
    @@ -721,6 +721,12 @@ class ModelArgs:
                 # DRAM weight grid specs for dram sharding matmuls
                 grid = self.mesh_device.compute_with_storage_grid_size()
                 self.max_grid_size = ttnn.CoreGrid(x=grid.x, y=grid.y)
    +            # TopK on the sampling path reads `sub_core_grid_topk` off ModelArgs and otherwise
    +            # defaults to None. Hand it the whole compute grid (resolved, never hard-coded) so the
    +            # decode-time top-k is not confined to whatever sub-grid the op picks by default.
    +            self.sub_core_grid_topk = ttnn.CoreRangeSet(
    +                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
    +            )
                 self.dram_weight_grid = ttnn.CoreRangeSet(
                     {
                         ttnn.CoreRange(

[#3] TopKDeviceOperation · shard · no gain  +0.00 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/model_config.py b/models/demos/llama3_1_8b_p150/tt/model_config.py
    index d960555193..cef314c4e2 100644
    --- a/models/demos/llama3_1_8b_p150/tt/model_config.py
    +++ b/models/demos/llama3_1_8b_p150/tt/model_config.py
    @@ -1115,6 +1115,20 @@ class ModelArgs:
                     self.model_config["MLP_RS_CONFIG"] = default_mlp_rs
                     self.model_config["SAMPLING_AG_CONFIG"] = default_sampling_force_argmax
     
    +            # Keep the top-k values/indices that feed `ttnn.sampling` in L1 instead of
    +            # DRAM-interleaved. The sampling chain around TopK is memory-bound, and these are the
    +            # only tensors on it small enough to be resident: k=32 per split, so the gathered width
    +            # is max_top_k per shard across `sampling_shard_cores` cores.
    +            _max_top_k = 32
    +            _sampling_shard_cores = 2 if self.cluster_shape == [1, 1] else self.num_devices
    +            self.model_config["DECODE_SAMPLING_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
    +                shape=(1, 1, max(self.max_batch_size, self.tile_size), _max_top_k),
    +                core_grid=ttnn.num_cores_to_corerangeset(_sampling_shard_cores, self.max_grid_size, row_wise=True),
    +                strategy=ttnn.ShardStrategy.WIDTH,
    +                orientation=ttnn.ShardOrientation.ROW_MAJOR,
    +                use_height_and_width_as_shard_shape=True,
    +            )
    +
                 logger.info(f"Attention grid: {self.attn_input_grid}")
                 logger.info(f"MLP grid: {self.mlp_core_grid}")
                 logger.info(f"MLP prefill grids @ 32: w1/w3: {self.mlp1_3_grid(32)}, w2: {self.mlp2_grid(32)}")

[#4] TopKDeviceOperation · shard · no gain  +0.00 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/model_config.py b/models/demos/llama3_1_8b_p150/tt/model_config.py
    index d960555193..e00a678830 100644
    --- a/models/demos/llama3_1_8b_p150/tt/model_config.py
    +++ b/models/demos/llama3_1_8b_p150/tt/model_config.py
    @@ -1115,6 +1115,21 @@ class ModelArgs:
                     self.model_config["MLP_RS_CONFIG"] = default_mlp_rs
                     self.model_config["SAMPLING_AG_CONFIG"] = default_sampling_force_argmax
     
    +            # Hold the gathered top-k values/indices feeding `ttnn.sampling` in L1 rather than
    +            # DRAM-interleaved, to cut DRAM reads on the memory-bound sampling chain. The gathered
    +            # width is max_top_k per split, and the [1, 1] mesh takes the 2-split multi-step path, so
    +            # the whole row is one shard -- splitting it across cores desynchronises the
    +            # device-offset add from the indices it is added to.
    +            _max_top_k = 32
    +            _sampling_splits = 2 if self.cluster_shape == [1, 1] else self.num_devices
    +            self.model_config["DECODE_SAMPLING_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
    +                shape=(1, 1, max(self.max_batch_size, self.tile_size), _max_top_k * _sampling_splits),
    +                core_grid=ttnn.num_cores_to_corerangeset(1, self.max_grid_size, row_wise=True),
    +                strategy=ttnn.ShardStrategy.WIDTH,
    +                orientation=ttnn.ShardOrientation.ROW_MAJOR,
    +                use_height_and_width_as_shard_shape=True,
    +            )
    +
                 logger.info(f"Attention grid: {self.attn_input_grid}")
                 logger.info(f"MLP grid: {self.mlp_core_grid}")
                 logger.info(f"MLP prefill grids @ 32: w1/w3: {self.mlp1_3_grid(32)}, w2: {self.mlp2_grid(32)}")

[#6] TopKDeviceOperation · structural · win  +926.49 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/model_config.py b/models/demos/llama3_1_8b_p150/tt/model_config.py
    index d960555193..751e024cbd 100644
    --- a/models/demos/llama3_1_8b_p150/tt/model_config.py
    +++ b/models/demos/llama3_1_8b_p150/tt/model_config.py
    @@ -1071,7 +1071,16 @@ class ModelArgs:
                     "rs_memory_config": ttnn.DRAM_MEMORY_CONFIG,
                 }
                 default_sampling_force_argmax = {
    -                "allow_force_argmax": False,
    +                # Greedy rows (temperature=0) normalise to k=1 / p=0 / temp=1, which the sampler can
    +                # serve with a single untilize+argmax instead of the top-k/top-p/RNG chain. That
    +                # matters a lot here: this vocab is 128256, the [1, 1]-mesh sampler splits it into
    +                # 2x64128 before TopK, and 64128 is not a power of two -- so ttnn.topk falls off its
    +                # multi-core factory onto the single-core bitonic one and burns ~10ms per call on ONE
    +                # core. Allowing the argmax path deletes both TopK calls (and the concat / typecast /
    +                # offset-add / untilize / manual_seed / sampling tail behind them). Non-greedy
    +                # requests still take the full path -- the sampler re-derives this per reset_params
    +                # and re-captures its trace when the mode flips.
    +                "allow_force_argmax": True,
                     "num_links": 1,
                     "chunks_per_sync": 10,
                     "num_workers_per_link": 2,

[#8] TopKDeviceOperation · structural · win  +1299.62 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/generator.py b/models/demos/llama3_1_8b_p150/tt/generator.py
    index 247f05bd6a..b0f41bda24 100644
    --- a/models/demos/llama3_1_8b_p150/tt/generator.py
    +++ b/models/demos/llama3_1_8b_p150/tt/generator.py
    @@ -53,6 +53,34 @@ def max_prefill_chunk_size_cutoff(sequence_length, max_prefill_chunk_size):
         return sequence_length > max_prefill_chunk_size
     
     
    +def _sampling_params_are_greedy(sampling_params) -> bool:
    +    """Is every row of ``sampling_params`` plain greedy (argmax) sampling?
    +
    +    Used to decide whether prefill warmup can skip the non-greedy sampling sweep. Conservative by
    +    construction: anything unrecognised, absent, or per-row inconsistent reads as NOT greedy, so the
    +    full sweep still runs and the only cost of being wrong is the warmup we were trying to avoid.
    +    ``temperature=0`` is the greedy request form (``format_sampling_params`` rewrites it to
    +    k=1 / p=0 / temp=1); log-probs and penalties both pull the request off the argmax path.
    +    """
    +    if sampling_params is None:
    +        return False
    +
    +    def _all(name, *allowed):
    +        value = getattr(sampling_params, name, None)
    +        values = value if isinstance(value, (list, tuple)) else [value]
    +        return bool(values) and all(v in allowed for v in values)
    +
    +    if not _all("temperature", 0, 0.0):
    +        return False
    +    if not _all("enable_log_probs", None, False):
    +        return False
    +    return (
    +        _all("presence_penalty", None, 0, 0.0)
    +        and _all("frequency_penalty", None, 0, 0.0)
    +        and _all("repetition_penalty", None, 1, 1.0)
    +    )
    +
    +
     def _deepseek_kvdbg_enabled() -> bool:
         return os.getenv("DEEPSEEK_KVDBG", "").lower() in ("1", "true", "yes", "y")
     
    @@ -565,10 +593,18 @@ class Generator(ModelCapabilitiesMixin, WarmupForwardMixin):
    ... (truncated, 18 more lines)

[#9] MatmulDeviceOperation · grid · no gain  +1299.62 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/model_config.py b/models/demos/llama3_1_8b_p150/tt/model_config.py
    index 751e024cbd..a30c9a056d 100644
    --- a/models/demos/llama3_1_8b_p150/tt/model_config.py
    +++ b/models/demos/llama3_1_8b_p150/tt/model_config.py
    @@ -1305,14 +1305,58 @@ class ModelArgs:
                             num_cores=self.mlp_core_grid.num_cores,
                         )
             elif mode == Mode.PREFILL:
    -            return self.matmul_config(
    -                m=min(seq_len, self.prefill_len_cutoff),  # 512 if BH, 1024 if WH
    -                k=self.dim // self.cluster_shape[0],
    -                n=self.hidden_dim // self.cluster_shape[1],
    -                grid_size=self.mlp1_3_grid(seq_len),
    -                per_core_N=math.ceil(
    -                    (self.hidden_dim // self.cluster_shape[1]) / (ttnn.TILE_SIZE * self.dram_shard_grid_width)
    +            m = min(seq_len, self.prefill_len_cutoff)  # 512 if BH, 1024 if WH
    +            k = self.dim // self.cluster_shape[0]
    +            n = self.hidden_dim // self.cluster_shape[1]
    +            grid_2d = self.mlp1_3_grid(seq_len)
    +
    +            # SHORT prefill: a 2D-mcast matmul hands one core ROW to each per_core_M block of M, so
    +            # when M is only a few tiles most of the grid idles no matter how wide it is -- at
    +            # seq_len=128 that is 4 rows x 8 cols = 32 of the P150's ~110 cores, and find_prefill_grid
    +            # still caps itself at the Wormhole 8x8 (its own TODO). w1/w3 are DRAM-bound on a ~29 MB
    +            # bf4_b weight read, so what they want is more cores pulling weights concurrently. A 1D
    +            # in0-mcast config gives exactly that: it splits N across every core and replicates the
    +            # (small) activation, which is the same lever the decode path already uses above.
    +            if not self.is_galaxy and is_blackhole() and self.device_name != "P100":
    +                m_tiles = math.ceil(m / ttnn.TILE_SIZE)
    +                n_tiles = math.ceil(n / ttnn.TILE_SIZE)
    +                k_tiles = k // ttnn.TILE_SIZE
    +                cores_2d = min(m_tiles, grid_2d[1]) * grid_2d[0]
    +                # mcast_in0 splits N across the grid as per_core_N = ceil(n_tiles / num_cores). A
    +                # grid whose core count does NOT divide n_tiles leaves a ragged tail column that
    +                # the kernel fills with garbage (PCC 24% Top1 on the first attempt at 110 cores /
    +                # per_core_N=5 for 448 tiles), so only take grids that divide N exactly.
    +                grid_1d = next(
    +                    (
    +                        ttnn.CoreGrid(x=x, y=y)
    +                        for cores in range(self.max_grid_size.num_cores, cores_2d, -1)
    ... (truncated, 30 more lines)

[#10] MatmulDeviceOperation · grid · no gain  +1299.62 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/mlp.py b/models/demos/llama3_1_8b_p150/tt/mlp.py
    index 3b2320d524..5789487335 100644
    --- a/models/demos/llama3_1_8b_p150/tt/mlp.py
    +++ b/models/demos/llama3_1_8b_p150/tt/mlp.py
    @@ -2,6 +2,8 @@
     
     # SPDX-License-Identifier: Apache-2.0
     
    +import math
    +
     import torch
     
     import ttnn
    @@ -170,7 +172,48 @@ class MLP(LightweightModule):
     
             ff1_3_out_mem_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if full_grid_ff1_3 else ff1_3_input_mem_config
     
    -        x_sharded = ttnn.to_memory_config(x, ff1_3_input_mem_config) if (mode == Mode.DECODE and full_grid_ff1_3) else x
    +        # Same lever for SHORT PREFILL. A 2D-mcast matmul gives one core ROW to each per_core_M
    +        # block of M, so at seq_len=128 (m_tiles=4) only 4 rows x 8 cols = 32 of the P150's ~110
    +        # cores ever run -- and ff1/ff3 are DRAM-bound on a ~29 MB bf4_b weight read, which wants
    +        # more cores pulling weights at once. A 1D in0-mcast config splits N across every core
    +        # instead. Note mcast_in0 REQUIRES a width-sharded in0 (each core owns a K slice and they
    +        # multicast it); handing it a DRAM-interleaved activation silently computes garbage, so the
    +        # shard below is part of the lever, not an optimisation on top of it. Restricted to core
    +        # counts that divide BOTH the activation width and N so no core gets a ragged slice.
    +        prefill_full_grid = False
    +        if mode == Mode.PREFILL and self.prefetcher is None and not TG and pc_1 is not None:
    +            grid = self.mesh_device.compute_with_storage_grid_size()
    +            n_ff = self.args.hidden_dim // self.args.num_devices
    +            k_tiles, n_tiles = self.dim // 32, n_ff // 32
    +            m_tiles = math.ceil(x.shape[-2] / 32)
    +            cores_2d = min(m_tiles, self.args.mlp1_3_grid(seq_len)[1]) * self.args.mlp1_3_grid(seq_len)[0]
    +            fg = next(
    +                (
    +                    ttnn.CoreGrid(x=c // y, y=y)
    +                    for c in range(grid.x * grid.y, cores_2d, -1)
    +                    if k_tiles % c == 0 and n_tiles % c == 0
    +                    for y in range(min(c, grid.y), 0, -1)
    +                    if c % y == 0 and c // y <= grid.x
    ... (truncated, 32 more lines)

[#12] MatmulDeviceOperation · dtype · win  +1319.60 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/mlp.py b/models/demos/llama3_1_8b_p150/tt/mlp.py
    index 3b2320d524..346435f1ba 100644
    --- a/models/demos/llama3_1_8b_p150/tt/mlp.py
    +++ b/models/demos/llama3_1_8b_p150/tt/mlp.py
    @@ -170,12 +170,22 @@ class MLP(LightweightModule):
     
             ff1_3_out_mem_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if full_grid_ff1_3 else ff1_3_input_mem_config
     
    +        # ff1/ff3's w1/w3 weights are already at the bf4_b floor, so the only dtype left on this
    +        # DRAM-bound pair is what they WRITE. Each emits a [seq, hidden] intermediate that then gets
    +        # read three times (the SILU mul reads both and writes its own, ff2 reads that) -- at
    +        # seq_len=128 that is 3.67 MB per tensor per layer at bf16. bf8_b halves every one of those
    +        # trips. Prefill only: decode's intermediate is 32 rows, too small for this to pay for the
    +        # precision, and decode is the steady-state path.
    +        ff1_3_out_dtype = ttnn.bfloat8_b if TG else activation_dtype or ttnn.bfloat16
    +        if mode == Mode.PREFILL and not TG:
    +            ff1_3_out_dtype = ttnn.bfloat8_b
    +
             x_sharded = ttnn.to_memory_config(x, ff1_3_input_mem_config) if (mode == Mode.DECODE and full_grid_ff1_3) else x
     
             w1_out = ttnn.linear(
                 x_sharded,
                 self.w1,
    -            dtype=ttnn.bfloat8_b if TG else activation_dtype or ttnn.bfloat16,
    +            dtype=ff1_3_out_dtype,
                 core_grid=None,  # FIXME: validate on TG ttnn.CoreGrid(y=8, x=8) if not pc_1 else None,
                 compute_kernel_config=li_ff1_3_compute_kernel_cfg,
                 program_config=pc_1,
    @@ -188,7 +198,7 @@ class MLP(LightweightModule):
             w3_out = ttnn.linear(
                 x_sharded,
                 self.w3,
    -            dtype=ttnn.bfloat8_b if TG else activation_dtype or ttnn.bfloat16,
    +            dtype=ff1_3_out_dtype,
                 core_grid=None,  # FIXME: validate on TG ttnn.CoreGrid(y=8, x=8) if not pc_3 else None,
                 compute_kernel_config=li_ff1_3_compute_kernel_cfg,
                 program_config=pc_3,

[#13] MatmulDeviceOperation · shard · no gain  +1325.51 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/mlp.py b/models/demos/llama3_1_8b_p150/tt/mlp.py
    index 346435f1ba..5fb06b1392 100644
    --- a/models/demos/llama3_1_8b_p150/tt/mlp.py
    +++ b/models/demos/llama3_1_8b_p150/tt/mlp.py
    @@ -180,6 +180,22 @@ class MLP(LightweightModule):
             if mode == Mode.PREFILL and not TG:
                 ff1_3_out_dtype = ttnn.bfloat8_b
     
    +        # L1 island for the ff1/ff3 -> SILU mul -> ff2 chain in prefill. The WEIGHTS cannot be made
    +        # L1-resident (w1/w3 are ~29 MB each per layer, x32 layers), but the [seq, hidden]
    +        # intermediates can be, and that removes three DRAM round-trips per MLP: ff1 and ff3 write
    +        # them, the mul reads both and writes its own, ff2 reads that. `ttnn.mul` inherits
    +        # w1_out's memory config, so landing ff1/ff3 in L1 carries the whole chain. Bounded to
    +        # prompts at or under prefill_len_cutoff so long prefill keeps the DRAM path; w1_out/w3_out
    +        # are freed immediately after the mul, so the island peaks at three intermediates -- ~1.95 MB
    +        # each at bf8_b for seq_len=128, trivially inside the interleaved-L1 budget.
    +        if (
    +            mode == Mode.PREFILL
    +            and not TG
    +            and self.prefetcher is None
    +            and seq_len <= self.args.prefill_len_cutoff
    +        ):
    +            ff1_3_out_mem_config = ttnn.L1_MEMORY_CONFIG
    +
             x_sharded = ttnn.to_memory_config(x, ff1_3_input_mem_config) if (mode == Mode.DECODE and full_grid_ff1_3) else x
     
             w1_out = ttnn.linear(

[#14] MatmulDeviceOperation · shard · no gain  +1318.24 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/mlp.py b/models/demos/llama3_1_8b_p150/tt/mlp.py
    index 346435f1ba..e9f8334d69 100644
    --- a/models/demos/llama3_1_8b_p150/tt/mlp.py
    +++ b/models/demos/llama3_1_8b_p150/tt/mlp.py
    @@ -180,6 +180,16 @@ class MLP(LightweightModule):
             if mode == Mode.PREFILL and not TG:
                 ff1_3_out_dtype = ttnn.bfloat8_b
     
    +        # ff1 and ff3 read the SAME activation, so in short prefill it is worth one L1 copy to turn
    +        # two DRAM reads per layer into two L1 reads. The weights (~29 MB each) are what dominate
    +        # this matmul's traffic and can never be resident, but the activation is only [seq, dim] --
    +        # 1 MB at seq_len=128 -- so the copy is cheap and is read back twice.
    +        prefill_l1_act = (
    +            mode == Mode.PREFILL and not TG and self.prefetcher is None and seq_len <= self.args.prefill_len_cutoff
    +        )
    +        if prefill_l1_act:
    +            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
    +
             x_sharded = ttnn.to_memory_config(x, ff1_3_input_mem_config) if (mode == Mode.DECODE and full_grid_ff1_3) else x
     
             w1_out = ttnn.linear(

[#15] MatmulDeviceOperation · tp-fracture · no gain  +1319.60 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/mlp.py b/models/demos/llama3_1_8b_p150/tt/mlp.py
    index 3b2320d524..346435f1ba 100644
    --- a/models/demos/llama3_1_8b_p150/tt/mlp.py
    +++ b/models/demos/llama3_1_8b_p150/tt/mlp.py
    @@ -170,12 +170,22 @@ class MLP(LightweightModule):
     
             ff1_3_out_mem_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if full_grid_ff1_3 else ff1_3_input_mem_config
     
    +        # ff1/ff3's w1/w3 weights are already at the bf4_b floor, so the only dtype left on this
    +        # DRAM-bound pair is what they WRITE. Each emits a [seq, hidden] intermediate that then gets
    +        # read three times (the SILU mul reads both and writes its own, ff2 reads that) -- at
    +        # seq_len=128 that is 3.67 MB per tensor per layer at bf16. bf8_b halves every one of those
    +        # trips. Prefill only: decode's intermediate is 32 rows, too small for this to pay for the
    +        # precision, and decode is the steady-state path.
    +        ff1_3_out_dtype = ttnn.bfloat8_b if TG else activation_dtype or ttnn.bfloat16
    +        if mode == Mode.PREFILL and not TG:
    +            ff1_3_out_dtype = ttnn.bfloat8_b
    +
             x_sharded = ttnn.to_memory_config(x, ff1_3_input_mem_config) if (mode == Mode.DECODE and full_grid_ff1_3) else x
     
             w1_out = ttnn.linear(
                 x_sharded,
                 self.w1,
    -            dtype=ttnn.bfloat8_b if TG else activation_dtype or ttnn.bfloat16,
    +            dtype=ff1_3_out_dtype,
                 core_grid=None,  # FIXME: validate on TG ttnn.CoreGrid(y=8, x=8) if not pc_1 else None,
                 compute_kernel_config=li_ff1_3_compute_kernel_cfg,
                 program_config=pc_1,
    @@ -188,7 +198,7 @@ class MLP(LightweightModule):
             w3_out = ttnn.linear(
                 x_sharded,
                 self.w3,
    -            dtype=ttnn.bfloat8_b if TG else activation_dtype or ttnn.bfloat16,
    +            dtype=ff1_3_out_dtype,
                 core_grid=None,  # FIXME: validate on TG ttnn.CoreGrid(y=8, x=8) if not pc_3 else None,
                 compute_kernel_config=li_ff1_3_compute_kernel_cfg,
                 program_config=pc_3,

[#16] MatmulDeviceOperation · structural · no gain  +1319.60 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/common.py b/models/demos/llama3_1_8b_p150/tt/common.py
    index ff5c0da829..645875f93d 100644
    --- a/models/demos/llama3_1_8b_p150/tt/common.py
    +++ b/models/demos/llama3_1_8b_p150/tt/common.py
    @@ -720,6 +720,12 @@ def get_padded_prefill_len(seq_len: int) -> int:
         This is used to pad the sequence length to the nearest power of 2.
         """
         # TODO: https://github.com/tenstorrent/tt-metal/issues/34117
    +    # A prompt shorter than one tile row still paid a full 128-token prefill -- 4x the matmul work,
    +    # and the prefill MLP/QKV matmuls are the largest single bucket in the profile. 32 is the tile
    +    # height, so it is the real floor here, and it is already a configured case (model_config logs
    +    # its MLP prefill grids "@ 32").
    +    if seq_len <= 32:
    +        return 32
         if seq_len <= 128:
             return 128
         if seq_len <= 1024:
    @@ -730,7 +736,9 @@ def get_padded_prefill_len(seq_len: int) -> int:
     
     
     def get_all_padded_prefill_lengths(max_len):
    -    lengths = [128]
    +    # 32 mirrors the new sub-tile-row tier in get_padded_prefill_len, so a short prompt's prefill
    +    # length is warmed up (and trace-capturable) rather than compiling on first use.
    +    lengths = [32, 128]
         k = 0
         while (v := (1 << k) * 1024) <= max_len:
             lengths.append(v)

[#17] MatmulDeviceOperation · tt-lang · no gain  +1319.60 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/mlp.py b/models/demos/llama3_1_8b_p150/tt/mlp.py
    index 3b2320d524..346435f1ba 100644
    --- a/models/demos/llama3_1_8b_p150/tt/mlp.py
    +++ b/models/demos/llama3_1_8b_p150/tt/mlp.py
    @@ -170,12 +170,22 @@ class MLP(LightweightModule):
     
             ff1_3_out_mem_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if full_grid_ff1_3 else ff1_3_input_mem_config
     
    +        # ff1/ff3's w1/w3 weights are already at the bf4_b floor, so the only dtype left on this
    +        # DRAM-bound pair is what they WRITE. Each emits a [seq, hidden] intermediate that then gets
    +        # read three times (the SILU mul reads both and writes its own, ff2 reads that) -- at
    +        # seq_len=128 that is 3.67 MB per tensor per layer at bf16. bf8_b halves every one of those
    +        # trips. Prefill only: decode's intermediate is 32 rows, too small for this to pay for the
    +        # precision, and decode is the steady-state path.
    +        ff1_3_out_dtype = ttnn.bfloat8_b if TG else activation_dtype or ttnn.bfloat16
    +        if mode == Mode.PREFILL and not TG:
    +            ff1_3_out_dtype = ttnn.bfloat8_b
    +
             x_sharded = ttnn.to_memory_config(x, ff1_3_input_mem_config) if (mode == Mode.DECODE and full_grid_ff1_3) else x
     
             w1_out = ttnn.linear(
                 x_sharded,
                 self.w1,
    -            dtype=ttnn.bfloat8_b if TG else activation_dtype or ttnn.bfloat16,
    +            dtype=ff1_3_out_dtype,
                 core_grid=None,  # FIXME: validate on TG ttnn.CoreGrid(y=8, x=8) if not pc_1 else None,
                 compute_kernel_config=li_ff1_3_compute_kernel_cfg,
                 program_config=pc_1,
    @@ -188,7 +198,7 @@ class MLP(LightweightModule):
             w3_out = ttnn.linear(
                 x_sharded,
                 self.w3,
    -            dtype=ttnn.bfloat8_b if TG else activation_dtype or ttnn.bfloat16,
    +            dtype=ff1_3_out_dtype,
                 core_grid=None,  # FIXME: validate on TG ttnn.CoreGrid(y=8, x=8) if not pc_3 else None,
                 compute_kernel_config=li_ff1_3_compute_kernel_cfg,
                 program_config=pc_3,

[#19] MatmulDeviceOperation · cpp · no gain  +1319.61 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/ttl_gated_ffn.py b/models/demos/llama3_1_8b_p150/tt/ttl_gated_ffn.py
    new file mode 100644
    index 0000000000..fb4ee7fb2e
    --- /dev/null
    +++ b/models/demos/llama3_1_8b_p150/tt/ttl_gated_ffn.py
    @@ -0,0 +1,153 @@
    +"""tt-lang kernel for the prefill gated FFN -- AUTHORED, MEASURED, and NOT WIRED IN.
    +
    +Kept as the record of the tt-lang rung for `MatmulDeviceOperation 128 x 4096 x 14336`.
    +
    +`ttl_gated_ffn` computes y = silu(x @ w1) * (x @ w3) in ONE kernel, so both [seq, hidden]
    +intermediates stay in L1 and never round-trip to DRAM -- the fusion GUIDELINES/11 names as the
    +highest-value tt-lang target, and one ttnn cannot express (ttnn.linear(activation=...) fuses an
    +activation into ONE matmul, not across two). Each core owns a strip of the N tiles; K is reduced
    +in-core with an accumulator DFB ping-pong (ttl 1.0.1 has no block.fill, so the accumulator is
    +seeded from the first partial product rather than zeroed).
    +
    +MEASURED on the real shape (M=128, K=4096, N=14336) on an 8x8 grid, against the stock
    +ttnn.linear/linear/mul chain it replaces:
    +
    +    correctness   PCC 0.999833   (the kernel is right)
    +    fused ttl     4.065 ms/call
    +    stock ttnn    0.714 ms/call  -> the kernel is 5.7x SLOWER
    +
    +So it is deliberately not on the hot path. Two reasons it loses, both structural:
    +  * It streams single tiles (DFB shape (1,1)) and re-reads each x tile once per N column, while
    +    ttnn's 2D-mcast matmul broadcasts each in0 tile across a whole core row. The intermediate
    +    traffic the fusion saves (~6 MB/layer) is dwarfed by the ~88 MB/layer of weight reads it
    +    cannot avoid -- which is also why the pure-TTNN version of the same idea (an L1 island for
    +    the intermediates) measured only -0.5% device time and failed the production gate.
    +  * It requires bf16 operands. Production w1/w3 are bf4_b in a DRAM-sharded memory config, and
    +    tt-lang cannot index those; converting them to bf16 would 4x the weight bytes and break the
    +    op's dtype contract, which GUIDELINES/11 forbids.
    +
    +Run directly (`python -m ...tt.ttl_gated_ffn`) to reproduce the numbers above.
    +"""
    +import time
    +
    +import torch
    +
    ... (truncated, 119 more lines)

[#21] MatmulDeviceOperation · grid · no gain  +1328.65 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/model_config.py b/models/demos/llama3_1_8b_p150/tt/model_config.py
    index 751e024cbd..aebca824f3 100644
    --- a/models/demos/llama3_1_8b_p150/tt/model_config.py
    +++ b/models/demos/llama3_1_8b_p150/tt/model_config.py
    @@ -1365,14 +1365,37 @@ class ModelArgs:
                         compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
                     )
                 else:
    +                _m = min(seq_len, self.prefill_len_cutoff)  # 512 if BH, 1024 if WH
    +                _k = self.hidden_dim // (self.cluster_shape[1] if self.is_galaxy else 1)
    +                _grid = self.mlp2_grid(seq_len)
    +                _per_core_N = (
    +                    math.ceil(self.dim / (ttnn.TILE_SIZE * self.dram_shard_grid_width))
    +                    if not self.is_galaxy
    +                    else None
    +                )
    +                # ff2's K is the hidden dim -- 448 tiles, by far the longest reduction in the model.
    +                # matmul_config derives in0_block_w from find_largest_divisor(), which hard-caps at
    +                # 8, so K is walked in 56 separate blocks and the mcast/packer sync is re-paid every
    +                # one of them. On a matmul the roofline tags memory-bound that is pure overhead.
    +                # Take the largest block that still divides the per-row K AND leaves the in1 CB
    +                # inside L1: the CB is in0_block_w x per_core_N tiles, double-buffered, bfp8
    +                # (~1088 B/tile), against a 1.57 MB budget.
    +                if not self.is_galaxy and _per_core_N:
    +                    _k_per_row = _k // ttnn.TILE_SIZE // _grid[1]
    +                    _budget_tiles = int(0.5 * 1_572_864 / (2 * 1088 * _per_core_N))
    +                    _in0_block_w = max(
    +                        (b for b in range(1, _k_per_row + 1) if _k_per_row % b == 0 and b <= _budget_tiles),
    +                        default=1,
    +                    )
    +                    return self.matmul_config(
    +                        m=_m, k=_k, n=self.dim, grid_size=_grid, per_core_N=_per_core_N, in0_block_w=_in0_block_w
    +                    )
                     return self.matmul_config(
    -                    m=min(seq_len, self.prefill_len_cutoff),  # 512 if BH, 1024 if WH
    -                    k=self.hidden_dim // (self.cluster_shape[1] if self.is_galaxy else 1),
    +                    m=_m,
    +                    k=_k,
                         n=self.dim,
    -                    grid_size=self.mlp2_grid(seq_len),
    ... (truncated, 8 more lines)

[#22] MatmulDeviceOperation · grid · no gain  +1362.33 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/model_config.py b/models/demos/llama3_1_8b_p150/tt/model_config.py
    index 751e024cbd..7a779d12ab 100644
    --- a/models/demos/llama3_1_8b_p150/tt/model_config.py
    +++ b/models/demos/llama3_1_8b_p150/tt/model_config.py
    @@ -1662,6 +1662,43 @@ class ModelArgs:
                         N_block_size=8,
                         compute_with_storage_grid_size=ttnn.CoreCoord(8, 10) if is_blackhole() else ttnn.CoreCoord(8, 8),
                     )
    +            elif is_blackhole() and self.device_name != "P100":
    +                # SHORT prefill only: use_minimal_qkv_prefill_matmul() already claims every
    +                # seq_len > 128, so M here is at most 4 tiles. The legacy config below spent
    +                # that on a hard-coded (8, 10) grid with per_core_M=1 -- only M_tiles x 8 =
    +                # 32 of the device's ~130 cores ever ran -- and walked K in 128 SINGLE-tile
    +                # steps with a 1x1 subblock (the FIXME). Both are pure loss on a QKV matmul
    +                # the roofline tags memory-bound: 128 one-tile K steps read the bfp8 weight
    +                # in the smallest possible blocks and re-pay the mcast/packer sync each step.
    +                # Size the config to the actual work instead: one core ROW per M tile, N
    +                # spread over the widest real device column count that divides it, K walked
    +                # in 8-tile blocks, and the largest subblock HIFI2 allows.
    +                m_tiles = max(1, math.ceil(seq_len / ttnn.TILE_SIZE))
    +                n_tiles = math.ceil(self.qkv_size / self.cluster_shape[1] / ttnn.TILE_SIZE)
    +                k_tiles = self.dim // ttnn.TILE_SIZE
    +                # Exact blocking: num_blocks_y must equal grid_y and num_blocks_x grid_x.
    +                grid_y = min(m_tiles, self.max_grid_size.y)
    +                per_core_M = math.ceil(m_tiles / grid_y)
    +                grid_x = max(x for x in range(1, self.max_grid_size.x + 1) if n_tiles % x == 0)
    +                per_core_N = n_tiles // grid_x
    +                # LI_QKV_PREFILL runs HIFI2, which sets fp32_dest_acc_en=True, so Blackhole
    +                # caps out_subblock_h * out_subblock_w at 4 (not 8).
    +                out_subblock_h = max(h for h in range(1, 5) if per_core_M % h == 0)
    +                out_subblock_w = max(w for w in range(1, (4 // out_subblock_h) + 1) if per_core_N % w == 0)
    +                # in1 CB = in0_block_w * per_core_N * 2 (double-buffered) bfp8 tiles; 8 x 16 x 2
    +                # x 1088 B ~= 278 KB, comfortably inside the 1.57 MB L1 budget.
    +                in0_block_w = max(b for b in (8, 4, 2, 1) if k_tiles % b == 0)
    +                return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    +                    compute_with_storage_grid_size=(grid_x, grid_y),
    +                    in0_block_w=in0_block_w,
    +                    out_subblock_h=out_subblock_h,
    +                    out_subblock_w=out_subblock_w,
    +                    per_core_M=per_core_M,
    ... (truncated, 8 more lines)

[#24] MatmulDeviceOperation · grid · win  +1372.06 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/model_config.py b/models/demos/llama3_1_8b_p150/tt/model_config.py
    index 751e024cbd..cb75ce7a86 100644
    --- a/models/demos/llama3_1_8b_p150/tt/model_config.py
    +++ b/models/demos/llama3_1_8b_p150/tt/model_config.py
    @@ -1365,14 +1365,37 @@ class ModelArgs:
                         compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
                     )
                 else:
    +                _m = min(seq_len, self.prefill_len_cutoff)  # 512 if BH, 1024 if WH
    +                _k = self.hidden_dim // (self.cluster_shape[1] if self.is_galaxy else 1)
    +                _grid = self.mlp2_grid(seq_len)
    +                _per_core_N = (
    +                    math.ceil(self.dim / (ttnn.TILE_SIZE * self.dram_shard_grid_width))
    +                    if not self.is_galaxy
    +                    else None
    +                )
    +                # ff2's K is the hidden dim -- 448 tiles, by far the longest reduction in the model.
    +                # matmul_config derives in0_block_w from find_largest_divisor(), which hard-caps at
    +                # 8, so K is walked in 56 separate blocks and the mcast/packer sync is re-paid every
    +                # one of them. On a matmul the roofline tags memory-bound that is pure overhead.
    +                # Take the largest block that still divides the per-row K AND leaves the in1 CB
    +                # inside L1: the CB is in0_block_w x per_core_N tiles, double-buffered, bfp8
    +                # (~1088 B/tile), against a 1.57 MB budget.
    +                if not self.is_galaxy and _per_core_N:
    +                    _k_per_row = _k // ttnn.TILE_SIZE // _grid[1]
    +                    _budget_tiles = int(0.5 * 1_572_864 / (2 * 1088 * _per_core_N))
    +                    _in0_block_w = max(
    +                        (b for b in range(1, _k_per_row + 1) if _k_per_row % b == 0 and b <= _budget_tiles),
    +                        default=1,
    +                    )
    +                    return self.matmul_config(
    +                        m=_m, k=_k, n=self.dim, grid_size=_grid, per_core_N=_per_core_N, in0_block_w=_in0_block_w
    +                    )
                     return self.matmul_config(
    -                    m=min(seq_len, self.prefill_len_cutoff),  # 512 if BH, 1024 if WH
    -                    k=self.hidden_dim // (self.cluster_shape[1] if self.is_galaxy else 1),
    +                    m=_m,
    +                    k=_k,
                         n=self.dim,
    -                    grid_size=self.mlp2_grid(seq_len),
    ... (truncated, 52 more lines)

[#25] MatmulDeviceOperation · grid · win  +1372.06 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/model_config.py b/models/demos/llama3_1_8b_p150/tt/model_config.py
    index 751e024cbd..cb75ce7a86 100644
    --- a/models/demos/llama3_1_8b_p150/tt/model_config.py
    +++ b/models/demos/llama3_1_8b_p150/tt/model_config.py
    @@ -1365,14 +1365,37 @@ class ModelArgs:
                         compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
                     )
                 else:
    +                _m = min(seq_len, self.prefill_len_cutoff)  # 512 if BH, 1024 if WH
    +                _k = self.hidden_dim // (self.cluster_shape[1] if self.is_galaxy else 1)
    +                _grid = self.mlp2_grid(seq_len)
    +                _per_core_N = (
    +                    math.ceil(self.dim / (ttnn.TILE_SIZE * self.dram_shard_grid_width))
    +                    if not self.is_galaxy
    +                    else None
    +                )
    +                # ff2's K is the hidden dim -- 448 tiles, by far the longest reduction in the model.
    +                # matmul_config derives in0_block_w from find_largest_divisor(), which hard-caps at
    +                # 8, so K is walked in 56 separate blocks and the mcast/packer sync is re-paid every
    +                # one of them. On a matmul the roofline tags memory-bound that is pure overhead.
    +                # Take the largest block that still divides the per-row K AND leaves the in1 CB
    +                # inside L1: the CB is in0_block_w x per_core_N tiles, double-buffered, bfp8
    +                # (~1088 B/tile), against a 1.57 MB budget.
    +                if not self.is_galaxy and _per_core_N:
    +                    _k_per_row = _k // ttnn.TILE_SIZE // _grid[1]
    +                    _budget_tiles = int(0.5 * 1_572_864 / (2 * 1088 * _per_core_N))
    +                    _in0_block_w = max(
    +                        (b for b in range(1, _k_per_row + 1) if _k_per_row % b == 0 and b <= _budget_tiles),
    +                        default=1,
    +                    )
    +                    return self.matmul_config(
    +                        m=_m, k=_k, n=self.dim, grid_size=_grid, per_core_N=_per_core_N, in0_block_w=_in0_block_w
    +                    )
                     return self.matmul_config(
    -                    m=min(seq_len, self.prefill_len_cutoff),  # 512 if BH, 1024 if WH
    -                    k=self.hidden_dim // (self.cluster_shape[1] if self.is_galaxy else 1),
    +                    m=_m,
    +                    k=_k,
                         n=self.dim,
    -                    grid_size=self.mlp2_grid(seq_len),
    ... (truncated, 52 more lines)

[#27] MatmulDeviceOperation · dtype · win  +1400.40 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/model_config.py b/models/demos/llama3_1_8b_p150/tt/model_config.py
    index cb75ce7a86..8b557ec23c 100644
    --- a/models/demos/llama3_1_8b_p150/tt/model_config.py
    +++ b/models/demos/llama3_1_8b_p150/tt/model_config.py
    @@ -224,7 +224,15 @@ class ModelOptimizations:
                 )
             else:
                 settings = {
    -                "TensorPrecision": {TensorGroup.FF1_FF3: PrecisionSetting.BFP4},
    +                # FF1/FF3 already ride bf4_b here, but FF2 was left on the BFP8 default -- so the
    +                # down-projection reads twice the weight bytes of the two projections feeding it,
    +                # on both the prefill and the decode path (it is the same resident tensor). ff2 is
    +                # DRAM-bandwidth bound in the roofline, and w2 is ~29 MB per layer at bf8_b, so
    +                # halving it is the largest remaining dtype lever in the MLP.
    +                "TensorPrecision": {
    +                    TensorGroup.FF1_FF3: PrecisionSetting.BFP4,
    +                    TensorGroup.FF2: PrecisionSetting.BFP4,
    +                },
                     "OpFidelity": {OpGroup.LI_FF1_FF3: MathFidelitySetting.LOFI},
                 }
                 if model_name.startswith("Phi-3-mini"):  # TODO: Only do this for N150

[#29] MatmulDeviceOperation · shard · win  +1403.18 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/mlp.py b/models/demos/llama3_1_8b_p150/tt/mlp.py
    index 346435f1ba..4bc19dc82c 100644
    --- a/models/demos/llama3_1_8b_p150/tt/mlp.py
    +++ b/models/demos/llama3_1_8b_p150/tt/mlp.py
    @@ -180,6 +180,16 @@ class MLP(LightweightModule):
             if mode == Mode.PREFILL and not TG:
                 ff1_3_out_dtype = ttnn.bfloat8_b
     
    +        # L1 island for the ff1/ff3 -> mul -> ff2 chain in short prefill. w1/w2/w3 are ~15-29 MB per
    +        # layer and can never be L1-resident, but the [seq, hidden] intermediates can, and that is
    +        # what ff2 READS: landing ff1/ff3 in L1 removes three DRAM round-trips per MLP (ff1 and ff3
    +        # write, the mul reads both and writes, ff2 reads that). ttnn.mul inherits w1_out's memory
    +        # config, so one change carries the whole chain. Bounded to prompts at or under
    +        # prefill_len_cutoff; w1_out/w3_out are freed right after the mul, so the island peaks at
    +        # three intermediates -- ~1.95 MB each at bf8_b for seq_len=128.
    +        if mode == Mode.PREFILL and not TG and self.prefetcher is None and seq_len <= self.args.prefill_len_cutoff:
    +            ff1_3_out_mem_config = ttnn.L1_MEMORY_CONFIG
    +
             x_sharded = ttnn.to_memory_config(x, ff1_3_input_mem_config) if (mode == Mode.DECODE and full_grid_ff1_3) else x
     
             w1_out = ttnn.linear(

[#30] MatmulDeviceOperation · tp-fracture · no gain  +1403.18 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/mlp.py b/models/demos/llama3_1_8b_p150/tt/mlp.py
    index 346435f1ba..4bc19dc82c 100644
    --- a/models/demos/llama3_1_8b_p150/tt/mlp.py
    +++ b/models/demos/llama3_1_8b_p150/tt/mlp.py
    @@ -180,6 +180,16 @@ class MLP(LightweightModule):
             if mode == Mode.PREFILL and not TG:
                 ff1_3_out_dtype = ttnn.bfloat8_b
     
    +        # L1 island for the ff1/ff3 -> mul -> ff2 chain in short prefill. w1/w2/w3 are ~15-29 MB per
    +        # layer and can never be L1-resident, but the [seq, hidden] intermediates can, and that is
    +        # what ff2 READS: landing ff1/ff3 in L1 removes three DRAM round-trips per MLP (ff1 and ff3
    +        # write, the mul reads both and writes, ff2 reads that). ttnn.mul inherits w1_out's memory
    +        # config, so one change carries the whole chain. Bounded to prompts at or under
    +        # prefill_len_cutoff; w1_out/w3_out are freed right after the mul, so the island peaks at
    +        # three intermediates -- ~1.95 MB each at bf8_b for seq_len=128.
    +        if mode == Mode.PREFILL and not TG and self.prefetcher is None and seq_len <= self.args.prefill_len_cutoff:
    +            ff1_3_out_mem_config = ttnn.L1_MEMORY_CONFIG
    +
             x_sharded = ttnn.to_memory_config(x, ff1_3_input_mem_config) if (mode == Mode.DECODE and full_grid_ff1_3) else x
     
             w1_out = ttnn.linear(

[#31] MatmulDeviceOperation · tp-fracture · no gain  +1403.18 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/mlp.py b/models/demos/llama3_1_8b_p150/tt/mlp.py
    index 346435f1ba..4bc19dc82c 100644
    --- a/models/demos/llama3_1_8b_p150/tt/mlp.py
    +++ b/models/demos/llama3_1_8b_p150/tt/mlp.py
    @@ -180,6 +180,16 @@ class MLP(LightweightModule):
             if mode == Mode.PREFILL and not TG:
                 ff1_3_out_dtype = ttnn.bfloat8_b
     
    +        # L1 island for the ff1/ff3 -> mul -> ff2 chain in short prefill. w1/w2/w3 are ~15-29 MB per
    +        # layer and can never be L1-resident, but the [seq, hidden] intermediates can, and that is
    +        # what ff2 READS: landing ff1/ff3 in L1 removes three DRAM round-trips per MLP (ff1 and ff3
    +        # write, the mul reads both and writes, ff2 reads that). ttnn.mul inherits w1_out's memory
    +        # config, so one change carries the whole chain. Bounded to prompts at or under
    +        # prefill_len_cutoff; w1_out/w3_out are freed right after the mul, so the island peaks at
    +        # three intermediates -- ~1.95 MB each at bf8_b for seq_len=128.
    +        if mode == Mode.PREFILL and not TG and self.prefetcher is None and seq_len <= self.args.prefill_len_cutoff:
    +            ff1_3_out_mem_config = ttnn.L1_MEMORY_CONFIG
    +
             x_sharded = ttnn.to_memory_config(x, ff1_3_input_mem_config) if (mode == Mode.DECODE and full_grid_ff1_3) else x
     
             w1_out = ttnn.linear(

[#32] MatmulDeviceOperation · structural · no gain  +1403.18 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/mlp.py b/models/demos/llama3_1_8b_p150/tt/mlp.py
    index 346435f1ba..4bc19dc82c 100644
    --- a/models/demos/llama3_1_8b_p150/tt/mlp.py
    +++ b/models/demos/llama3_1_8b_p150/tt/mlp.py
    @@ -180,6 +180,16 @@ class MLP(LightweightModule):
             if mode == Mode.PREFILL and not TG:
                 ff1_3_out_dtype = ttnn.bfloat8_b
     
    +        # L1 island for the ff1/ff3 -> mul -> ff2 chain in short prefill. w1/w2/w3 are ~15-29 MB per
    +        # layer and can never be L1-resident, but the [seq, hidden] intermediates can, and that is
    +        # what ff2 READS: landing ff1/ff3 in L1 removes three DRAM round-trips per MLP (ff1 and ff3
    +        # write, the mul reads both and writes, ff2 reads that). ttnn.mul inherits w1_out's memory
    +        # config, so one change carries the whole chain. Bounded to prompts at or under
    +        # prefill_len_cutoff; w1_out/w3_out are freed right after the mul, so the island peaks at
    +        # three intermediates -- ~1.95 MB each at bf8_b for seq_len=128.
    +        if mode == Mode.PREFILL and not TG and self.prefetcher is None and seq_len <= self.args.prefill_len_cutoff:
    +            ff1_3_out_mem_config = ttnn.L1_MEMORY_CONFIG
    +
             x_sharded = ttnn.to_memory_config(x, ff1_3_input_mem_config) if (mode == Mode.DECODE and full_grid_ff1_3) else x
     
             w1_out = ttnn.linear(

[#33] MatmulDeviceOperation · tt-lang · no gain  +1403.18 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/mlp.py b/models/demos/llama3_1_8b_p150/tt/mlp.py
    index 346435f1ba..4bc19dc82c 100644
    --- a/models/demos/llama3_1_8b_p150/tt/mlp.py
    +++ b/models/demos/llama3_1_8b_p150/tt/mlp.py
    @@ -180,6 +180,16 @@ class MLP(LightweightModule):
             if mode == Mode.PREFILL and not TG:
                 ff1_3_out_dtype = ttnn.bfloat8_b
     
    +        # L1 island for the ff1/ff3 -> mul -> ff2 chain in short prefill. w1/w2/w3 are ~15-29 MB per
    +        # layer and can never be L1-resident, but the [seq, hidden] intermediates can, and that is
    +        # what ff2 READS: landing ff1/ff3 in L1 removes three DRAM round-trips per MLP (ff1 and ff3
    +        # write, the mul reads both and writes, ff2 reads that). ttnn.mul inherits w1_out's memory
    +        # config, so one change carries the whole chain. Bounded to prompts at or under
    +        # prefill_len_cutoff; w1_out/w3_out are freed right after the mul, so the island peaks at
    +        # three intermediates -- ~1.95 MB each at bf8_b for seq_len=128.
    +        if mode == Mode.PREFILL and not TG and self.prefetcher is None and seq_len <= self.args.prefill_len_cutoff:
    +            ff1_3_out_mem_config = ttnn.L1_MEMORY_CONFIG
    +
             x_sharded = ttnn.to_memory_config(x, ff1_3_input_mem_config) if (mode == Mode.DECODE and full_grid_ff1_3) else x
     
             w1_out = ttnn.linear(

[#34] MatmulDeviceOperation · cpp · no gain  +1403.18 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/mlp.py b/models/demos/llama3_1_8b_p150/tt/mlp.py
    index 346435f1ba..4bc19dc82c 100644
    --- a/models/demos/llama3_1_8b_p150/tt/mlp.py
    +++ b/models/demos/llama3_1_8b_p150/tt/mlp.py
    @@ -180,6 +180,16 @@ class MLP(LightweightModule):
             if mode == Mode.PREFILL and not TG:
                 ff1_3_out_dtype = ttnn.bfloat8_b
     
    +        # L1 island for the ff1/ff3 -> mul -> ff2 chain in short prefill. w1/w2/w3 are ~15-29 MB per
    +        # layer and can never be L1-resident, but the [seq, hidden] intermediates can, and that is
    +        # what ff2 READS: landing ff1/ff3 in L1 removes three DRAM round-trips per MLP (ff1 and ff3
    +        # write, the mul reads both and writes, ff2 reads that). ttnn.mul inherits w1_out's memory
    +        # config, so one change carries the whole chain. Bounded to prompts at or under
    +        # prefill_len_cutoff; w1_out/w3_out are freed right after the mul, so the island peaks at
    +        # three intermediates -- ~1.95 MB each at bf8_b for seq_len=128.
    +        if mode == Mode.PREFILL and not TG and self.prefetcher is None and seq_len <= self.args.prefill_len_cutoff:
    +            ff1_3_out_mem_config = ttnn.L1_MEMORY_CONFIG
    +
             x_sharded = ttnn.to_memory_config(x, ff1_3_input_mem_config) if (mode == Mode.DECODE and full_grid_ff1_3) else x
     
             w1_out = ttnn.linear(

[#36] MatmulDeviceOperation · grid · no gain  +1403.06 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/model_config.py b/models/demos/llama3_1_8b_p150/tt/model_config.py
    index 8b557ec23c..260115d3e2 100644
    --- a/models/demos/llama3_1_8b_p150/tt/model_config.py
    +++ b/models/demos/llama3_1_8b_p150/tt/model_config.py
    @@ -3362,8 +3362,13 @@ class ModelArgs:
             Raises:
                 AssertionError: If it's not possible to find such a grid configuration.
             """
    -        max_rows = 8
    -        max_cols = 8  # Maximum number of rows or columns
    +        # Resolve the REAL compute grid rather than assuming Wormhole's 8x8 (GUIDELINES #1). On a
    +        # P150 that is 11x10 = 110 cores, so the old hard-coded cap threw away more than 40% of the
    +        # chip before the divisibility search even started -- and this helper is what sizes the
    +        # DRAM-sharded DECODE matmuls, i.e. the per-token path.
    +        grid = self.mesh_device.compute_with_storage_grid_size() if self.mesh_device is not None else None
    +        max_rows = grid.y if grid is not None else 8
    +        max_cols = grid.x if grid is not None else 8
             max_cores = max_rows * max_cols  # Maximum number of cores
     
             # Find all possible numbers of cores that divide N and are less than or equal to max_cores

[#37] MatmulDeviceOperation · grid · no gain  +1400.22 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/ttl_ff2_matmul.py b/models/demos/llama3_1_8b_p150/tt/ttl_ff2_matmul.py
    new file mode 100644
    index 0000000000..9c141024a1
    --- /dev/null
    +++ b/models/demos/llama3_1_8b_p150/tt/ttl_ff2_matmul.py
    @@ -0,0 +1,122 @@
    +"""tt-lang matmul on the ff2 shape -- AUTHORED, MEASURED, and NOT WIRED IN.
    +
    +Kept as the record of the tt-lang rung for `MatmulDeviceOperation 128 x 14336 x 4096`, measured on
    +ff2's OWN shape rather than reusing the w1/w3 result. Each core owns a strip of the N tiles
    +(128 N-tiles / 64 cores = 2 each, exact); K is reduced in-core with an accumulator DFB ping-pong
    +seeded from the first partial product (ttl 1.0.1 has no block.fill).
    +
    +MEASURED on M=128, K=14336, N=4096, 8x8 grid:
    +
    +    correctness   PCC 0.999682   (the kernel is right)
    +    ttl matmul    2.961 ms/call
    +    ttnn.linear   0.364 ms/call  -> 8.1x SLOWER
    +
    +The C++ Metalium rung was measured on the same shape via tt/cpp_mm_generic.py: PCC 0.993625,
    +3.113 ms vs 0.362 ms -> 8.6x slower.
    +
    +Unlike w1/w3 there is no fusion available here either: ff2's output feeds a residual add and (on a
    +multi-device mesh) a CCL, and its input round-trip is already removed by the L1 island in mlp.py. So
    +a hand kernel has nothing to add beyond dataflow it cannot win on -- each core re-reads every A tile
    +per output tile, while ttnn's matmul multicasts in0 across a core row and blocks K.
    +
    +Run directly to reproduce.
    +"""
    +import time
    +
    +import torch
    +
    +import ttnn
    +import ttl
    +
    +TILE = 32
    +M, K, N = 128, 14336, 4096
    +GRID_X, GRID_Y = 8, 8
    +
    ... (truncated, 88 more lines)

[#38] MatmulDeviceOperation · dtype · no gain  +1403.23 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/ttl_ff2_matmul.py b/models/demos/llama3_1_8b_p150/tt/ttl_ff2_matmul.py
    new file mode 100644
    index 0000000000..9c141024a1
    --- /dev/null
    +++ b/models/demos/llama3_1_8b_p150/tt/ttl_ff2_matmul.py
    @@ -0,0 +1,122 @@
    +"""tt-lang matmul on the ff2 shape -- AUTHORED, MEASURED, and NOT WIRED IN.
    +
    +Kept as the record of the tt-lang rung for `MatmulDeviceOperation 128 x 14336 x 4096`, measured on
    +ff2's OWN shape rather than reusing the w1/w3 result. Each core owns a strip of the N tiles
    +(128 N-tiles / 64 cores = 2 each, exact); K is reduced in-core with an accumulator DFB ping-pong
    +seeded from the first partial product (ttl 1.0.1 has no block.fill).
    +
    +MEASURED on M=128, K=14336, N=4096, 8x8 grid:
    +
    +    correctness   PCC 0.999682   (the kernel is right)
    +    ttl matmul    2.961 ms/call
    +    ttnn.linear   0.364 ms/call  -> 8.1x SLOWER
    +
    +The C++ Metalium rung was measured on the same shape via tt/cpp_mm_generic.py: PCC 0.993625,
    +3.113 ms vs 0.362 ms -> 8.6x slower.
    +
    +Unlike w1/w3 there is no fusion available here either: ff2's output feeds a residual add and (on a
    +multi-device mesh) a CCL, and its input round-trip is already removed by the L1 island in mlp.py. So
    +a hand kernel has nothing to add beyond dataflow it cannot win on -- each core re-reads every A tile
    +per output tile, while ttnn's matmul multicasts in0 across a core row and blocks K.
    +
    +Run directly to reproduce.
    +"""
    +import time
    +
    +import torch
    +
    +import ttnn
    +import ttl
    +
    +TILE = 32
    +M, K, N = 128, 14336, 4096
    +GRID_X, GRID_Y = 8, 8
    +
    ... (truncated, 88 more lines)

[#40] MatmulDeviceOperation · dtype · win  +1406.46 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/mlp.py b/models/demos/llama3_1_8b_p150/tt/mlp.py
    index 4bc19dc82c..e7c3ef31df 100644
    --- a/models/demos/llama3_1_8b_p150/tt/mlp.py
    +++ b/models/demos/llama3_1_8b_p150/tt/mlp.py
    @@ -177,7 +177,13 @@ class MLP(LightweightModule):
             # trips. Prefill only: decode's intermediate is 32 rows, too small for this to pay for the
             # precision, and decode is the steady-state path.
             ff1_3_out_dtype = ttnn.bfloat8_b if TG else activation_dtype or ttnn.bfloat16
    -        if mode == Mode.PREFILL and not TG:
    +        if not TG:
    +            # Narrow (per-tensor) form of the activation dtype walk. A model-wide
    +            # TensorGroup.ACTIVATION=BFP8 destroys accuracy because it also lands on the residual
    +            # stream, the norm inputs and the KV cache; these two tensors are fenced off from all of
    +            # that -- they feed nothing but the SILU mul, whose output ff2 then reads. Applies in
    +            # DECODE as well as PREFILL: decode's intermediate is only 32 rows, but it is written
    +            # twice, read twice and paid on EVERY token.
                 ff1_3_out_dtype = ttnn.bfloat8_b
     
             # L1 island for the ff1/ff3 -> mul -> ff2 chain in short prefill. w1/w2/w3 are ~15-29 MB per

[#41] MatmulDeviceOperation · shard · no gain  +1406.50 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/mlp.py b/models/demos/llama3_1_8b_p150/tt/mlp.py
    index 4bc19dc82c..e7c3ef31df 100644
    --- a/models/demos/llama3_1_8b_p150/tt/mlp.py
    +++ b/models/demos/llama3_1_8b_p150/tt/mlp.py
    @@ -177,7 +177,13 @@ class MLP(LightweightModule):
             # trips. Prefill only: decode's intermediate is 32 rows, too small for this to pay for the
             # precision, and decode is the steady-state path.
             ff1_3_out_dtype = ttnn.bfloat8_b if TG else activation_dtype or ttnn.bfloat16
    -        if mode == Mode.PREFILL and not TG:
    +        if not TG:
    +            # Narrow (per-tensor) form of the activation dtype walk. A model-wide
    +            # TensorGroup.ACTIVATION=BFP8 destroys accuracy because it also lands on the residual
    +            # stream, the norm inputs and the KV cache; these two tensors are fenced off from all of
    +            # that -- they feed nothing but the SILU mul, whose output ff2 then reads. Applies in
    +            # DECODE as well as PREFILL: decode's intermediate is only 32 rows, but it is written
    +            # twice, read twice and paid on EVERY token.
                 ff1_3_out_dtype = ttnn.bfloat8_b
     
             # L1 island for the ff1/ff3 -> mul -> ff2 chain in short prefill. w1/w2/w3 are ~15-29 MB per

[#42] MatmulDeviceOperation · shard · no gain  +1406.50 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/model_config.py b/models/demos/llama3_1_8b_p150/tt/model_config.py
    index 8b557ec23c..4b4c59fa6b 100644
    --- a/models/demos/llama3_1_8b_p150/tt/model_config.py
    +++ b/models/demos/llama3_1_8b_p150/tt/model_config.py
    @@ -1422,7 +1422,19 @@ class ModelArgs:
                         use_height_and_width_as_shard_shape=True,
                     )
                 else:
    -                return ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
    +                # Pin the shard spec to ff2's core grid rather than letting the generic
    +                # L1_WIDTH_SHARDED config derive one from the producing matmul, so the SILU mul lands
    +                # directly on the grid ff2 reads from and no reshard is needed before ff2.
    +                return ttnn.create_sharded_memory_config(
    +                    shape=(
    +                        self.tile_padded_batch_rows,
    +                        self.hidden_dim // self.cluster_shape[1] // self.mlp2_core_grid.num_cores,
    +                    ),
    +                    core_grid=self.mlp2_core_grid,
    +                    strategy=ttnn.ShardStrategy.WIDTH,
    +                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    +                    use_height_and_width_as_shard_shape=True,
    +                )
             elif mode == Mode.PREFILL:
                 return ttnn.DRAM_MEMORY_CONFIG
             else:

[#43] MatmulDeviceOperation · tp-fracture · no gain  +1406.46 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/mlp.py b/models/demos/llama3_1_8b_p150/tt/mlp.py
    index 4bc19dc82c..e7c3ef31df 100644
    --- a/models/demos/llama3_1_8b_p150/tt/mlp.py
    +++ b/models/demos/llama3_1_8b_p150/tt/mlp.py
    @@ -177,7 +177,13 @@ class MLP(LightweightModule):
             # trips. Prefill only: decode's intermediate is 32 rows, too small for this to pay for the
             # precision, and decode is the steady-state path.
             ff1_3_out_dtype = ttnn.bfloat8_b if TG else activation_dtype or ttnn.bfloat16
    -        if mode == Mode.PREFILL and not TG:
    +        if not TG:
    +            # Narrow (per-tensor) form of the activation dtype walk. A model-wide
    +            # TensorGroup.ACTIVATION=BFP8 destroys accuracy because it also lands on the residual
    +            # stream, the norm inputs and the KV cache; these two tensors are fenced off from all of
    +            # that -- they feed nothing but the SILU mul, whose output ff2 then reads. Applies in
    +            # DECODE as well as PREFILL: decode's intermediate is only 32 rows, but it is written
    +            # twice, read twice and paid on EVERY token.
                 ff1_3_out_dtype = ttnn.bfloat8_b
     
             # L1 island for the ff1/ff3 -> mul -> ff2 chain in short prefill. w1/w2/w3 are ~15-29 MB per

[#44] MatmulDeviceOperation · structural · no gain  +1406.46 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/model_config.py b/models/demos/llama3_1_8b_p150/tt/model_config.py
    index 8b557ec23c..abc07d8953 100644
    --- a/models/demos/llama3_1_8b_p150/tt/model_config.py
    +++ b/models/demos/llama3_1_8b_p150/tt/model_config.py
    @@ -422,17 +422,25 @@ def parse_decoder_json(json_file_path, default_optimization=ModelOptimizations.p
             for decoder_id, settings in config_data["decoders"].items():
                 decoder_id = int(decoder_id)
     
    -            tensor_precision = (
    -                {TensorGroup[key]: PrecisionSetting[value] for key, value in settings.get("precision_cfg").items()}
    -                if "precision_cfg" in settings
    -                else default_tensor_dtype_settings
    -            )
    +            # MERGE the per-decoder overrides onto the chosen optimization level, do not REPLACE it.
    +            # ModelOptimizations.__init__ starts from _default_settings() (BFP8 everywhere), so
    +            # handing it only the JSON's keys silently reverts every tensor the JSON does NOT mention
    +            # back to that bring-up default. For Llama-3.1-8B the file asks for exactly one thing --
    +            # decoder 31's FF1_FF3 raised to BFP8 -- and the side effect was that the same decoder's
    +            # FF2 also fell from the performance preset's BFP4 to BFP8, i.e. the last layer read
    +            # twice the w2 bytes for no reason anyone asked for. Merging keeps the explicit override
    +            # and leaves everything else on the preset.
    +            tensor_precision = dict(default_tensor_dtype_settings)
    +            if "precision_cfg" in settings:
    +                tensor_precision.update(
    +                    {TensorGroup[key]: PrecisionSetting[value] for key, value in settings["precision_cfg"].items()}
    +                )
     
    -            op_fidelity = (
    -                {OpGroup[key]: MathFidelitySetting[value] for key, value in settings.get("fidelity_cfg").items()}
    -                if "fidelity_cfg" in settings
    -                else default_op_fidelity_settings
    -            )
    +            op_fidelity = dict(default_op_fidelity_settings)
    +            if "fidelity_cfg" in settings:
    +                op_fidelity.update(
    +                    {OpGroup[key]: MathFidelitySetting[value] for key, value in settings["fidelity_cfg"].items()}
    +                )
     
                 custom_opt = ModelOptimizations({"TensorPrecision": tensor_precision, "OpFidelity": op_fidelity})
                 decoders_precision.set_decoder_conf(decoder_id, custom_opt)

[#45] MatmulDeviceOperation · tt-lang · no gain  +1406.46 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/ttl_ff2_matmul.py b/models/demos/llama3_1_8b_p150/tt/ttl_ff2_matmul.py
    index 9c141024a1..296b46a632 100644
    --- a/models/demos/llama3_1_8b_p150/tt/ttl_ff2_matmul.py
    +++ b/models/demos/llama3_1_8b_p150/tt/ttl_ff2_matmul.py
    @@ -1,25 +1,35 @@
    -"""tt-lang matmul on the ff2 shape -- AUTHORED, MEASURED, and NOT WIRED IN.
    +"""tt-lang matmul on the ff2 shapes -- AUTHORED, MEASURED, and NOT WIRED IN.
     
    -Kept as the record of the tt-lang rung for `MatmulDeviceOperation 128 x 14336 x 4096`, measured on
    -ff2's OWN shape rather than reusing the w1/w3 result. Each core owns a strip of the N tiles
    -(128 N-tiles / 64 cores = 2 each, exact); K is reduced in-core with an accumulator DFB ping-pong
    -seeded from the first partial product (ttl 1.0.1 has no block.fill).
    +Kept as the record of the tt-lang rung for BOTH ff2 shapes, each measured on its OWN shape rather
    +than inheriting the w1/w3 result:
     
    -MEASURED on M=128, K=14336, N=4096, 8x8 grid:
    +  * `MatmulDeviceOperation 128 x 14336 x 4096` -- the short-prefill down-projection
    +  * `MatmulDeviceOperation  32 x 14336 x 4096` -- the DECODE down-projection (the per-token path)
     
    -    correctness   PCC 0.999682   (the kernel is right)
    -    ttl matmul    2.961 ms/call
    -    ttnn.linear   0.364 ms/call  -> 8.1x SLOWER
    +Each core owns a strip of the N tiles (128 N-tiles / 64 cores = 2 each, exact); K is reduced in-core
    +with an accumulator DFB ping-pong seeded from the first partial product (ttl 1.0.1 has no
    +block.fill). Only M differs between the two runs, so the same kernel serves both.
     
    -The C++ Metalium rung was measured on the same shape via tt/cpp_mm_generic.py: PCC 0.993625,
    -3.113 ms vs 0.362 ms -> 8.6x slower.
    +MEASURED on K=14336, N=4096, 8x8 grid:
     
    -Unlike w1/w3 there is no fusion available here either: ff2's output feeds a residual add and (on a
    -multi-device mesh) a CCL, and its input round-trip is already removed by the L1 island in mlp.py. So
    -a hand kernel has nothing to add beyond dataflow it cannot win on -- each core re-reads every A tile
    -per output tile, while ttnn's matmul multicasts in0 across a core row and blocks K.
    +    M=128   PCC 0.999692   ttl 2.957 ms/call   vs ttnn.linear 0.358 ms/call  -> 8.3x SLOWER
    +    M= 32   PCC 0.999695   ttl 0.816 ms/call   vs ttnn.linear 0.300 ms/call  -> 2.7x SLOWER
     
    -Run directly to reproduce.
    +The C++ Metalium rung was measured on the same shapes via tt/cpp_mm_generic.py.
    +
    ... (truncated, 91 more lines)

[#46] MatmulDeviceOperation · cpp · no gain  +1406.46 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/cpp_mm_generic.py b/models/demos/llama3_1_8b_p150/tt/cpp_mm_generic.py
    index fa9e629a74..c98a9b6ae9 100644
    --- a/models/demos/llama3_1_8b_p150/tt/cpp_mm_generic.py
    +++ b/models/demos/llama3_1_8b_p150/tt/cpp_mm_generic.py
    @@ -1,28 +1,38 @@
     """C++ Metalium matmul via ttnn.generic_op -- AUTHORED, MEASURED, and NOT WIRED IN.
     
    -Kept as the record of the cpp rung for `MatmulDeviceOperation 128 x 4096 x 14336`.
    +Kept as the record of the cpp rung for every hot dense matmul in the MLP:
    +
    +  * `MatmulDeviceOperation 128 x  4096 x 14336` -- the short-prefill ff1/ff3 up-projection
    +  * `MatmulDeviceOperation 128 x 14336 x  4096` -- the short-prefill ff2 down-projection
    +  * `MatmulDeviceOperation  32 x 14336 x  4096` -- the DECODE ff2 down-projection (per-token path)
     
     Drives the repo's own programming-example kernel triple (tt_metal/programming_examples/matmul/
     matmul_multi_core: reader / mm / writer, copied into tt/kernels/) through ttnn.generic_op, with
     the output tiles partitioned across the entire compute grid.
     
    -MEASURED on the real shape (M=128, K=4096, N=14336) on the full 11x10 P150 grid:
    +MEASURED on the real shapes, on the full 11x10 P150 grid:
    +
    +    M    K      N        PCC        generic_op    ttnn.linear     verdict
    +    128  4096   14336    0.998986    3.565 ms      0.328 ms       10.9x SLOWER
    +    128  14336  4096     0.993626    3.118 ms      0.357 ms        8.7x SLOWER
    +     32  14336  4096     0.993594    0.916 ms      0.292 ms        3.1x SLOWER
     
    -    correctness   PCC 0.999040   (the kernel is right)
    -    generic_op    3.573 ms/call
    -    ttnn.linear   0.331 ms/call  -> the kernel is 10.8x SLOWER
    +Every kernel is CORRECT and every one loses. The cause is dataflow, not tuning: this reader fetches
    +every A tile again for each output tile, so A is re-read Nt times from DRAM, while ttnn's production
    +matmul multicasts each in0 tile across a whole core row and blocks K, moving a small fraction of the
    +bytes. Beating it would mean reimplementing that mcast matmul -- which is what the stock op already is.
     
    -Same root cause as the tt-lang attempt in ttl_gated_ffn.py, and it is a dataflow property, not a
    -tuning miss: this reader fetches every A tile again for each output tile, so A is re-read Nt times
    -from DRAM. ttnn's production matmul multicasts each in0 tile across a whole core row and blocks K,
    -so it moves a small fraction of the bytes. Beating it would mean reimplementing that mcast matmul --
    -which is what the stock op already is.
    +The decode row is the one worth reading twice. Dropping M from 128 to 32 shrinks THIS kernel ~3.4x
    ... (truncated, 245 more lines)

[#49] LayerNormDeviceOperation · grid · no gain  +1406.45 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/distributed_norm.py b/models/demos/llama3_1_8b_p150/tt/distributed_norm.py
    index 81444f48f0..5aae30235b 100644
    --- a/models/demos/llama3_1_8b_p150/tt/distributed_norm.py
    +++ b/models/demos/llama3_1_8b_p150/tt/distributed_norm.py
    @@ -80,6 +80,23 @@ class DistributedNorm(LightweightModule):
     
             input_mem_cfg = sharded_output_config if mode == Mode.DECODE else ttnn.DRAM_MEMORY_CONFIG
     
    +        # PREFILL: the interleaved rms_norm kernel parallelises over TILE ROWS only, so a short
    +        # prompt runs it on `rows_t` cores (4 for a 128-token prefill). Block-shard the activation
    +        # instead so the sharded kernel can also split the embedding dim across grid columns.
    +        # get_prefill_norm_shard_config returns None whenever that would not raise the core count
    +        # (long prefill) or would not divide evenly, so the interleaved path stays the default.
    +        prefill_shard = None
    +        if mode == Mode.PREFILL and not self.args.is_multichip:
    +            rows = 1
    +            for d in x.shape[:-1]:
    +                rows *= d
    +            if rows % ttnn.TILE_SIZE == 0:
    +                prefill_shard = self.args.get_prefill_norm_shard_config(rows // ttnn.TILE_SIZE)
    +        if prefill_shard is not None:
    +            input_mem_cfg = prefill_shard["mem_config"]
    +            norm_config = dict(norm_config or {})
    +            norm_config["sharded_program_config"] = prefill_shard["program_config"]
    +
             # Distributed norm already performs a gather
             if self.args.is_multichip and not self.args.is_distributed_norm(mode):
                 x = ttnn.experimental.all_gather_async(
    @@ -105,8 +122,14 @@ class DistributedNorm(LightweightModule):
             else:
                 x = ttnn.to_memory_config(x, input_mem_cfg)
     
    +        # out_sharded stays False for the prefill shard: RMSNorm then hands back an interleaved
    +        # tensor, so every downstream consumer sees exactly the layout it saw before.
             x = self.norm(
    -            x, mode=mode, in_sharded=(mode == Mode.DECODE), out_sharded=(mode == Mode.DECODE), norm_config=norm_config
    +            x,
    +            mode=mode,
    +            in_sharded=(mode == Mode.DECODE or prefill_shard is not None),
    +            out_sharded=(mode == Mode.DECODE),
    ... (truncated, 65 more lines)

[#50] LayerNormDeviceOperation · grid · no gain  +1406.45 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/distributed_norm.py b/models/demos/llama3_1_8b_p150/tt/distributed_norm.py
    index 81444f48f0..be3ccd5cfe 100644
    --- a/models/demos/llama3_1_8b_p150/tt/distributed_norm.py
    +++ b/models/demos/llama3_1_8b_p150/tt/distributed_norm.py
    @@ -80,6 +80,25 @@ class DistributedNorm(LightweightModule):
     
             input_mem_cfg = sharded_output_config if mode == Mode.DECODE else ttnn.DRAM_MEMORY_CONFIG
     
    +        # PREFILL: the interleaved rms_norm kernel parallelises over TILE ROWS only, so a short
    +        # prompt pins it to `rows_t` cores -- 4 of 110 for a 128-token prefill. Width-shard the
    +        # activation so the sharded kernel can split the embedding dim across cores as well.
    +        # This variant keeps the norm's OUTPUT sharded and converts back once via output_mem_config,
    +        # instead of letting RMSNorm call sharded_to_interleaved on the way out.
    +        prefill_shard = None
    +        if mode == Mode.PREFILL and not self.args.is_multichip:
    +            rows = 1
    +            for d in x.shape[:-1]:
    +                rows *= d
    +            if rows % ttnn.TILE_SIZE == 0:
    +                prefill_shard = self.args.get_prefill_norm_shard_config(rows // ttnn.TILE_SIZE)
    +        if prefill_shard is not None:
    +            input_mem_cfg = prefill_shard["mem_config"]
    +            norm_config = dict(norm_config or {})
    +            norm_config["sharded_program_config"] = prefill_shard["program_config"]
    +            norm_config["sharded_output_config"] = prefill_shard["mem_config"]
    +            norm_config["output_mem_config"] = ttnn.DRAM_MEMORY_CONFIG
    +
             # Distributed norm already performs a gather
             if self.args.is_multichip and not self.args.is_distributed_norm(mode):
                 x = ttnn.experimental.all_gather_async(
    @@ -105,9 +124,8 @@ class DistributedNorm(LightweightModule):
             else:
                 x = ttnn.to_memory_config(x, input_mem_cfg)
     
    -        x = self.norm(
    -            x, mode=mode, in_sharded=(mode == Mode.DECODE), out_sharded=(mode == Mode.DECODE), norm_config=norm_config
    -        )
    +        sharded = mode == Mode.DECODE or prefill_shard is not None
    +        x = self.norm(x, mode=mode, in_sharded=sharded, out_sharded=sharded, norm_config=norm_config)
     
    ... (truncated, 2 more lines)

[#51] LayerNormDeviceOperation · shard · no gain  +1406.09 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/decoder.py b/models/demos/llama3_1_8b_p150/tt/decoder.py
    index 067ea19c0a..9ebbfda6d6 100644
    --- a/models/demos/llama3_1_8b_p150/tt/decoder.py
    +++ b/models/demos/llama3_1_8b_p150/tt/decoder.py
    @@ -127,6 +127,7 @@ class TransformerBlock(LightweightModule):
                     state_dict_prefix=args.get_state_dict_prefix("", layer_num),
                     weight_cache_path=None if args.dummy_weights else weight_cache_path,
                     weight_dtype=ttnn.bfloat16,
    +                weight_memory_config=ttnn.L1_MEMORY_CONFIG,
                     weight_key="attention_norm",
                     is_distributed=self.args.is_distributed_norm,
                     add_unit_offset=self.args.rms_norm_add_unit_offset,
    @@ -149,6 +150,7 @@ class TransformerBlock(LightweightModule):
                     state_dict_prefix=args.get_state_dict_prefix("", layer_num),
                     weight_cache_path=None if args.dummy_weights else weight_cache_path,
                     weight_dtype=ttnn.bfloat16,
    +                weight_memory_config=ttnn.L1_MEMORY_CONFIG,
                     weight_key="ffn_norm",
                     is_distributed=self.args.is_distributed_norm,
                     add_unit_offset=self.args.rms_norm_add_unit_offset,
    @@ -173,6 +175,7 @@ class TransformerBlock(LightweightModule):
                         state_dict_prefix=args.get_state_dict_prefix("", layer_num),
                         weight_cache_path=None if args.dummy_weights else weight_cache_path,
                         weight_dtype=ttnn.bfloat16,
    +                weight_memory_config=ttnn.L1_MEMORY_CONFIG,
                         weight_key="pre_feedforward_layernorm",
                         is_distributed=self.args.is_distributed_norm,
                         ccl_topology=self.args.ccl_topology(),
    @@ -201,6 +204,7 @@ class TransformerBlock(LightweightModule):
                         state_dict_prefix=args.get_state_dict_prefix("", layer_num),
                         weight_cache_path=None if args.dummy_weights else weight_cache_path,
                         weight_dtype=ttnn.bfloat16,
    +                weight_memory_config=ttnn.L1_MEMORY_CONFIG,
                         weight_key="post_feedforward_layernorm",
                         is_distributed=self.args.is_distributed_norm,
                         ccl_topology=self.args.ccl_topology(),
    diff --git a/models/demos/llama3_1_8b_p150/tt/model.py b/models/demos/llama3_1_8b_p150/tt/model.py
    index e19e919f4c..01cf18039d 100644
    --- a/models/demos/llama3_1_8b_p150/tt/model.py
    +++ b/models/demos/llama3_1_8b_p150/tt/model.py
    ... (truncated, 8 more lines)

[#53] LayerNormDeviceOperation · shard · no gain  +1406.45 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/model_config.py b/models/demos/llama3_1_8b_p150/tt/model_config.py
    index 8b557ec23c..177fe419a3 100644
    --- a/models/demos/llama3_1_8b_p150/tt/model_config.py
    +++ b/models/demos/llama3_1_8b_p150/tt/model_config.py
    @@ -1238,6 +1238,15 @@ class ModelArgs:
                         use_height_and_width_as_shard_shape=True,
                     )
             elif mode == Mode.PREFILL:
    +            # Keep the SHORT-prefill residual stream L1-resident. The norm at the top of every block
    +            # reads this tensor and is memory-bound on it, and the residual adds read/write it twice
    +            # more per block, so its home matters more than its size suggests. At or below the
    +            # prefill cutoff it is only dim * seq * 2 B (1 MB at seq 128), which L1 holds easily; a
    +            # longer prefill stays in DRAM because the tensor grows with seq while L1 does not.
    +            # This is distinct from sharding the norm itself: the tensor is simply BORN in L1, so
    +            # there is no runtime reshard to trace.
    +            if not self.is_galaxy and self.dim * self.prefill_len_cutoff * 2 <= 4 * 1024 * 1024:
    +                return ttnn.L1_MEMORY_CONFIG
                 return ttnn.DRAM_MEMORY_CONFIG
             else:
                 raise ValueError(f"Invalid mode: {mode}")

[#54] LayerNormDeviceOperation · structural · no gain  +1406.45 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/cpp_mm_generic.py b/models/demos/llama3_1_8b_p150/tt/cpp_mm_generic.py
    index fa9e629a74..c98a9b6ae9 100644
    --- a/models/demos/llama3_1_8b_p150/tt/cpp_mm_generic.py
    +++ b/models/demos/llama3_1_8b_p150/tt/cpp_mm_generic.py
    @@ -1,28 +1,38 @@
     """C++ Metalium matmul via ttnn.generic_op -- AUTHORED, MEASURED, and NOT WIRED IN.
     
    -Kept as the record of the cpp rung for `MatmulDeviceOperation 128 x 4096 x 14336`.
    +Kept as the record of the cpp rung for every hot dense matmul in the MLP:
    +
    +  * `MatmulDeviceOperation 128 x  4096 x 14336` -- the short-prefill ff1/ff3 up-projection
    +  * `MatmulDeviceOperation 128 x 14336 x  4096` -- the short-prefill ff2 down-projection
    +  * `MatmulDeviceOperation  32 x 14336 x  4096` -- the DECODE ff2 down-projection (per-token path)
     
     Drives the repo's own programming-example kernel triple (tt_metal/programming_examples/matmul/
     matmul_multi_core: reader / mm / writer, copied into tt/kernels/) through ttnn.generic_op, with
     the output tiles partitioned across the entire compute grid.
     
    -MEASURED on the real shape (M=128, K=4096, N=14336) on the full 11x10 P150 grid:
    +MEASURED on the real shapes, on the full 11x10 P150 grid:
    +
    +    M    K      N        PCC        generic_op    ttnn.linear     verdict
    +    128  4096   14336    0.998986    3.565 ms      0.328 ms       10.9x SLOWER
    +    128  14336  4096     0.993626    3.118 ms      0.357 ms        8.7x SLOWER
    +     32  14336  4096     0.993594    0.916 ms      0.292 ms        3.1x SLOWER
     
    -    correctness   PCC 0.999040   (the kernel is right)
    -    generic_op    3.573 ms/call
    -    ttnn.linear   0.331 ms/call  -> the kernel is 10.8x SLOWER
    +Every kernel is CORRECT and every one loses. The cause is dataflow, not tuning: this reader fetches
    +every A tile again for each output tile, so A is re-read Nt times from DRAM, while ttnn's production
    +matmul multicasts each in0 tile across a whole core row and blocks K, moving a small fraction of the
    +bytes. Beating it would mean reimplementing that mcast matmul -- which is what the stock op already is.
     
    -Same root cause as the tt-lang attempt in ttl_gated_ffn.py, and it is a dataflow property, not a
    -tuning miss: this reader fetches every A tile again for each output tile, so A is re-read Nt times
    -from DRAM. ttnn's production matmul multicasts each in0 tile across a whole core row and blocks K,
    -so it moves a small fraction of the bytes. Beating it would mean reimplementing that mcast matmul --
    -which is what the stock op already is.
    +The decode row is the one worth reading twice. Dropping M from 128 to 32 shrinks THIS kernel ~3.4x
    ... (truncated, 245 more lines)

[#56] host_overhead · trace-capture · win  +1572.20 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/generator.py b/models/demos/llama3_1_8b_p150/tt/generator.py
    index b0f41bda24..01da3ce7f2 100644
    --- a/models/demos/llama3_1_8b_p150/tt/generator.py
    +++ b/models/demos/llama3_1_8b_p150/tt/generator.py
    @@ -162,12 +162,32 @@ class Generator(ModelCapabilitiesMixin, WarmupForwardMixin):
     
             return ret
     
    -    def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device, greedy_only: bool = False):
    +    def warmup_model_prefill(
    +        self, kv_cache, enable_trace, can_sample_on_device, greedy_only: bool = False, only_seq_lens=None
    +    ):
             if self.already_warmed_up_prefill:
                 return
             self.already_warmed_up_prefill = True
     
             sequence_lengths_to_warmup = self.model_args[0].get_warmup_prefill_supported_seq_lens()
    +        # Warm up only the PREFILL LENGTH the caller is actually asking for. The default sweep runs a
    +        # full real prefill at every padded length up to capped_warmup_seq_len -- 128, 256, 512, 1024
    +        # -- so a request whose prompt pads to 128 still pays a 256-, a 512- AND a 1024-token prefill,
    +        # i.e. ~15x the token-work of the request that triggered it, all of it for prompt shapes that
    +        # never arrive. This is the same argument as the sampling-shape narrowing above, on the
    +        # sequence-length axis instead: a later longer prompt still works, it just pays its own
    +        # one-time capture on first use.
    +        if only_seq_lens:
    +            wanted = sorted({int(s) for s in only_seq_lens})
    +            kept = [s for s in sequence_lengths_to_warmup if s in wanted]
    +            if kept:
    +                skipped = [s for s in sequence_lengths_to_warmup if s not in wanted]
    +                if skipped:
    +                    logger.info(
    +                        f"Prefill warmup narrowed to {kept}; skipping {skipped} "
    +                        "(no request has asked for those lengths yet)"
    +                    )
    +                sequence_lengths_to_warmup = kept
             warmup_batch_sizes = (1,)
     
             skip_sequence_lengths = False
    @@ -600,11 +620,22 @@ class Generator(ModelCapabilitiesMixin, WarmupForwardMixin):
                 # the request that triggered warmup is greedy, those traces are pure warmup cost for
    ... (truncated, 21 more lines)

[#57] MatmulDeviceOperation · grid · no gain  +1572.20 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/mlp.py b/models/demos/llama3_1_8b_p150/tt/mlp.py
    index e7c3ef31df..aa12ba9305 100644
    --- a/models/demos/llama3_1_8b_p150/tt/mlp.py
    +++ b/models/demos/llama3_1_8b_p150/tt/mlp.py
    @@ -153,10 +153,20 @@ class MLP(LightweightModule):
             if mode == Mode.DECODE and self.prefetcher is None and not TG:
                 grid = self.mesh_device.compute_with_storage_grid_size()
                 n_ff = self.args.hidden_dim // self.args.num_devices
    -            num_cores = grid.x * grid.y
    -            if (self.dim % (32 * num_cores) == 0) and (n_ff % (32 * num_cores) == 0):
    +            # This test used to demand that the FULL grid divide both dims, which on an 11x10 P150 is
    +            # 110 cores against 4096 -- 4096 % 3520 != 0 -- so the whole branch was DEAD and the
    +            # DRAM-sharded path always ran. Take the largest core count that legally divides both
    +            # instead, so the branch actually gets exercised.
    +            num_cores = max(
    +                (c for c in range(1, grid.x * grid.y + 1) if self.dim % (32 * c) == 0 and n_ff % (32 * c) == 0),
    +                default=0,
    +            )
    +            # The core grid must hold EXACTLY num_cores, not the whole chip: the shard width below is
    +            # dim // num_cores, so a wider grid would leave cores with no shard.
    +            fg_rows = max((y for y in range(1, grid.y + 1) if num_cores % y == 0 and num_cores // y <= grid.x), default=0)
    +            if num_cores > 0 and fg_rows > 0:
                     full_grid_ff1_3 = True
    -                fg_core_grid = ttnn.CoreGrid(x=grid.x, y=grid.y)
    +                fg_core_grid = ttnn.CoreGrid(x=num_cores // fg_rows, y=fg_rows)
                     ff1_3_full_grid_mem_config = ttnn.create_sharded_memory_config(
                         shape=(x.shape[-2], self.dim // num_cores),
                         core_grid=fg_core_grid,

[#58] MatmulDeviceOperation · grid · no gain  +1572.11 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/model_config.py b/models/demos/llama3_1_8b_p150/tt/model_config.py
    index 8b557ec23c..61047ffb53 100644
    --- a/models/demos/llama3_1_8b_p150/tt/model_config.py
    +++ b/models/demos/llama3_1_8b_p150/tt/model_config.py
    @@ -3362,8 +3362,13 @@ class ModelArgs:
             Raises:
                 AssertionError: If it's not possible to find such a grid configuration.
             """
    -        max_rows = 8
    -        max_cols = 8  # Maximum number of rows or columns
    +        # Resolve the REAL compute grid rather than assuming Wormhole's 8x8 (GUIDELINES #1). This
    +        # helper is what sizes mlp_core_grid / mlp2_core_grid, i.e. the DRAM-sharded DECODE matmuls
    +        # on the per-token path, so on an 11x10 P150 the old constant capped them at 64 of 110 cores
    +        # before the divisibility search even started.
    +        grid = self.mesh_device.compute_with_storage_grid_size() if self.mesh_device is not None else None
    +        max_rows = grid.y if grid is not None else 8
    +        max_cols = grid.x if grid is not None else 8
             max_cores = max_rows * max_cols  # Maximum number of cores
     
             # Find all possible numbers of cores that divide N and are less than or equal to max_cores

[#59] MatmulDeviceOperation · dtype · no gain  +1572.75 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/mlp.py b/models/demos/llama3_1_8b_p150/tt/mlp.py
    index e7c3ef31df..f1c0e5bbe5 100644
    --- a/models/demos/llama3_1_8b_p150/tt/mlp.py
    +++ b/models/demos/llama3_1_8b_p150/tt/mlp.py
    @@ -184,7 +184,11 @@ class MLP(LightweightModule):
                 # that -- they feed nothing but the SILU mul, whose output ff2 then reads. Applies in
                 # DECODE as well as PREFILL: decode's intermediate is only 32 rows, but it is written
                 # twice, read twice and paid on EVERY token.
    -            ff1_3_out_dtype = ttnn.bfloat8_b
    +            # Walking one step FURTHER down: bf8_b -> bf4_b. These two tensors feed nothing but the
    +            # SILU mul (no norm, no softmax, no reduction, no KV cache), so they are outside every
    +            # hard floor GUIDELINES #13 names. bf4_b is the last dtype step available anywhere in
    +            # this MLP -- w1/w2/w3 are already at it.
    +            ff1_3_out_dtype = ttnn.bfloat4_b
     
             # L1 island for the ff1/ff3 -> mul -> ff2 chain in short prefill. w1/w2/w3 are ~15-29 MB per
             # layer and can never be L1-resident, but the [seq, hidden] intermediates can, and that is

[#60] MatmulDeviceOperation · structural · no gain  +1572.20 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/generator.py b/models/demos/llama3_1_8b_p150/tt/generator.py
    index b0f41bda24..01da3ce7f2 100644
    --- a/models/demos/llama3_1_8b_p150/tt/generator.py
    +++ b/models/demos/llama3_1_8b_p150/tt/generator.py
    @@ -162,12 +162,32 @@ class Generator(ModelCapabilitiesMixin, WarmupForwardMixin):
     
             return ret
     
    -    def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device, greedy_only: bool = False):
    +    def warmup_model_prefill(
    +        self, kv_cache, enable_trace, can_sample_on_device, greedy_only: bool = False, only_seq_lens=None
    +    ):
             if self.already_warmed_up_prefill:
                 return
             self.already_warmed_up_prefill = True
     
             sequence_lengths_to_warmup = self.model_args[0].get_warmup_prefill_supported_seq_lens()
    +        # Warm up only the PREFILL LENGTH the caller is actually asking for. The default sweep runs a
    +        # full real prefill at every padded length up to capped_warmup_seq_len -- 128, 256, 512, 1024
    +        # -- so a request whose prompt pads to 128 still pays a 256-, a 512- AND a 1024-token prefill,
    +        # i.e. ~15x the token-work of the request that triggered it, all of it for prompt shapes that
    +        # never arrive. This is the same argument as the sampling-shape narrowing above, on the
    +        # sequence-length axis instead: a later longer prompt still works, it just pays its own
    +        # one-time capture on first use.
    +        if only_seq_lens:
    +            wanted = sorted({int(s) for s in only_seq_lens})
    +            kept = [s for s in sequence_lengths_to_warmup if s in wanted]
    +            if kept:
    +                skipped = [s for s in sequence_lengths_to_warmup if s not in wanted]
    +                if skipped:
    +                    logger.info(
    +                        f"Prefill warmup narrowed to {kept}; skipping {skipped} "
    +                        "(no request has asked for those lengths yet)"
    +                    )
    +                sequence_lengths_to_warmup = kept
             warmup_batch_sizes = (1,)
     
             skip_sequence_lengths = False
    @@ -600,11 +620,22 @@ class Generator(ModelCapabilitiesMixin, WarmupForwardMixin):
                 # the request that triggered warmup is greedy, those traces are pure warmup cost for
    ... (truncated, 21 more lines)

[#63] MatmulDeviceOperation · shard · no gain  +1572.14 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/mlp.py b/models/demos/llama3_1_8b_p150/tt/mlp.py
    index e7c3ef31df..d37ba04690 100644
    --- a/models/demos/llama3_1_8b_p150/tt/mlp.py
    +++ b/models/demos/llama3_1_8b_p150/tt/mlp.py
    @@ -196,7 +196,12 @@ class MLP(LightweightModule):
             if mode == Mode.PREFILL and not TG and self.prefetcher is None and seq_len <= self.args.prefill_len_cutoff:
                 ff1_3_out_mem_config = ttnn.L1_MEMORY_CONFIG
     
    -        x_sharded = ttnn.to_memory_config(x, ff1_3_input_mem_config) if (mode == Mode.DECODE and full_grid_ff1_3) else x
    +        # In DECODE the DRAM-sharded ff1/ff3 matmul wants its in0 L1-width-sharded across
    +        # mlp_core_grid. That is what ff_norm is configured to emit, but relying on the producer to
    +        # have picked the same grid is exactly the coordinated-shard failure mode in GUIDELINES #10:
    +        # if the two ever disagree the matmul reshards internally, once per layer per token, and
    +        # nothing in the source says so. Ask for it explicitly -- a no-op when they already agree.
    +        x_sharded = ttnn.to_memory_config(x, ff1_3_input_mem_config) if mode == Mode.DECODE else x
     
             w1_out = ttnn.linear(
                 x_sharded,
    @@ -225,7 +230,10 @@ class MLP(LightweightModule):
                 else None,
             )
             ttnn.deallocate(x)
    -        if mode == Mode.DECODE and full_grid_ff1_3:
    +        # Only free the reshard if one actually happened -- when the configs already agree
    +        # to_memory_config hands back x itself, and freeing it again after ttnn.deallocate(x) above
    +        # is a double free.
    +        if mode == Mode.DECODE and x_sharded is not x:
                 ttnn.deallocate(x_sharded)
     
             if TG:

[#64] MatmulDeviceOperation · shard · no gain  +1572.13 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/model_config.py b/models/demos/llama3_1_8b_p150/tt/model_config.py
    index 8b557ec23c..b8b64325ee 100644
    --- a/models/demos/llama3_1_8b_p150/tt/model_config.py
    +++ b/models/demos/llama3_1_8b_p150/tt/model_config.py
    @@ -1422,7 +1422,20 @@ class ModelArgs:
                         use_height_and_width_as_shard_shape=True,
                     )
                 else:
    -                return ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
    +                # Pin ff1/ff3's decode output to an EXPLICIT width shard on ff2's core grid instead of
    +                # the generic L1_WIDTH_SHARDED config, which carries no shard spec and leaves the
    +                # grid to be inferred from whichever producer wrote the tensor. Naming the grid means
    +                # the ff1/ff3 -> SILU mul -> ff2 chain provably shares one core set end to end.
    +                return ttnn.create_sharded_memory_config(
    +                    shape=(
    +                        self.tile_padded_batch_rows,
    +                        self.hidden_dim // self.cluster_shape[1] // self.mlp2_core_grid.num_cores,
    +                    ),
    +                    core_grid=self.mlp2_core_grid,
    +                    strategy=ttnn.ShardStrategy.WIDTH,
    +                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    +                    use_height_and_width_as_shard_shape=True,
    +                )
             elif mode == Mode.PREFILL:
                 return ttnn.DRAM_MEMORY_CONFIG
             else:

[#65] MatmulDeviceOperation · tp-fracture · no gain  +1572.14 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/cpp_mm_generic.py b/models/demos/llama3_1_8b_p150/tt/cpp_mm_generic.py
    index c98a9b6ae9..6da7db8484 100644
    --- a/models/demos/llama3_1_8b_p150/tt/cpp_mm_generic.py
    +++ b/models/demos/llama3_1_8b_p150/tt/cpp_mm_generic.py
    @@ -44,7 +44,7 @@ from ttnn._ttnn.program_descriptor import VectorUInt32 as _VU32
     TILE = 32
     # (M, K, N) of every op this rung was measured for: the ff1/ff3 up-projection, then ff2's
     # down-projection at the short-prefill and the DECODE row counts.
    -SHAPES = [(128, 4096, 14336), (128, 14336, 4096), (32, 14336, 4096)]
    +SHAPES = [(128, 4096, 14336), (128, 14336, 4096), (32, 14336, 4096), (32, 4096, 14336)]
     ROOT = "/tmp/tt_hw_planner_llama3_1_8b_p150_1785111170/models/demos/llama3_1_8b_p150/tt/kernels"
     
     
    diff --git a/models/demos/llama3_1_8b_p150/tt/ttl_ff2_matmul.py b/models/demos/llama3_1_8b_p150/tt/ttl_ff2_matmul.py
    index 296b46a632..a66112d895 100644
    --- a/models/demos/llama3_1_8b_p150/tt/ttl_ff2_matmul.py
    +++ b/models/demos/llama3_1_8b_p150/tt/ttl_ff2_matmul.py
    @@ -39,8 +39,9 @@ import ttnn
     import ttl
     
     TILE = 32
    -K, N = 14336, 4096
     GRID_X, GRID_Y = 8, 8
    +# (M, K, N) triples this rung was measured for.
    +SHAPES = [(128, 14336, 4096), (32, 14336, 4096), (32, 4096, 14336)]
     
     
     @ttl.operation(grid=(GRID_Y, GRID_X))
    @@ -94,7 +95,7 @@ def ttl_mm(a: ttnn.Tensor, b: ttnn.Tensor, y: ttnn.Tensor) -> None:
                         ttl.copy(yb, y[mt, n_base + j]).wait()
     
     
    -def measure(device, m):
    +def measure(device, m, K, N):
         ta = torch.randn(m, K, dtype=torch.bfloat16) * 0.02
         tb = torch.randn(K, N, dtype=torch.bfloat16) * 0.02
         golden = ta.float() @ tb.float()
    @@ -106,7 +107,8 @@ def measure(device, m):
     
         ttl_mm(a, b, y)
    ... (truncated, 27 more lines)

[#66] MatmulDeviceOperation · tt-lang · no gain  +1572.14 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/cpp_mm_generic.py b/models/demos/llama3_1_8b_p150/tt/cpp_mm_generic.py
    index c98a9b6ae9..27cae9296a 100644
    --- a/models/demos/llama3_1_8b_p150/tt/cpp_mm_generic.py
    +++ b/models/demos/llama3_1_8b_p150/tt/cpp_mm_generic.py
    @@ -5,6 +5,7 @@ Kept as the record of the cpp rung for every hot dense matmul in the MLP:
       * `MatmulDeviceOperation 128 x  4096 x 14336` -- the short-prefill ff1/ff3 up-projection
       * `MatmulDeviceOperation 128 x 14336 x  4096` -- the short-prefill ff2 down-projection
       * `MatmulDeviceOperation  32 x 14336 x  4096` -- the DECODE ff2 down-projection (per-token path)
    +  * `MatmulDeviceOperation  32 x  4096 x 14336` -- the DECODE ff1/ff3 up-projection (per-token path)
     
     Drives the repo's own programming-example kernel triple (tt_metal/programming_examples/matmul/
     matmul_multi_core: reader / mm / writer, copied into tt/kernels/) through ttnn.generic_op, with
    @@ -13,9 +14,10 @@ the output tiles partitioned across the entire compute grid.
     MEASURED on the real shapes, on the full 11x10 P150 grid:
     
         M    K      N        PCC        generic_op    ttnn.linear     verdict
    -    128  4096   14336    0.998986    3.565 ms      0.328 ms       10.9x SLOWER
    -    128  14336  4096     0.993626    3.118 ms      0.357 ms        8.7x SLOWER
    -     32  14336  4096     0.993594    0.916 ms      0.292 ms        3.1x SLOWER
    +    128  4096   14336    0.999039    3.561 ms      0.332 ms       10.7x SLOWER
    +    128  14336  4096     0.993591    3.119 ms      0.358 ms        8.7x SLOWER
    +     32  14336  4096     0.993562    0.919 ms      0.296 ms        3.1x SLOWER
    +     32  4096   14336    0.999022    1.039 ms      0.309 ms        3.4x SLOWER
     
     Every kernel is CORRECT and every one loses. The cause is dataflow, not tuning: this reader fetches
     every A tile again for each output tile, so A is re-read Nt times from DRAM, while ttnn's production
    @@ -44,7 +46,7 @@ from ttnn._ttnn.program_descriptor import VectorUInt32 as _VU32
     TILE = 32
     # (M, K, N) of every op this rung was measured for: the ff1/ff3 up-projection, then ff2's
     # down-projection at the short-prefill and the DECODE row counts.
    -SHAPES = [(128, 4096, 14336), (128, 14336, 4096), (32, 14336, 4096)]
    +SHAPES = [(128, 4096, 14336), (128, 14336, 4096), (32, 14336, 4096), (32, 4096, 14336)]
     ROOT = "/tmp/tt_hw_planner_llama3_1_8b_p150_1785111170/models/demos/llama3_1_8b_p150/tt/kernels"
     
     
    diff --git a/models/demos/llama3_1_8b_p150/tt/ttl_ff2_matmul.py b/models/demos/llama3_1_8b_p150/tt/ttl_ff2_matmul.py
    index 296b46a632..d126d0c5d2 100644
    --- a/models/demos/llama3_1_8b_p150/tt/ttl_ff2_matmul.py
    +++ b/models/demos/llama3_1_8b_p150/tt/ttl_ff2_matmul.py
    @@ -1,35 +1,40 @@
    ... (truncated, 114 more lines)

[#67] MatmulDeviceOperation · cpp · no gain  +1572.14 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/cpp_mm_generic.py b/models/demos/llama3_1_8b_p150/tt/cpp_mm_generic.py
    index c98a9b6ae9..27cae9296a 100644
    --- a/models/demos/llama3_1_8b_p150/tt/cpp_mm_generic.py
    +++ b/models/demos/llama3_1_8b_p150/tt/cpp_mm_generic.py
    @@ -5,6 +5,7 @@ Kept as the record of the cpp rung for every hot dense matmul in the MLP:
       * `MatmulDeviceOperation 128 x  4096 x 14336` -- the short-prefill ff1/ff3 up-projection
       * `MatmulDeviceOperation 128 x 14336 x  4096` -- the short-prefill ff2 down-projection
       * `MatmulDeviceOperation  32 x 14336 x  4096` -- the DECODE ff2 down-projection (per-token path)
    +  * `MatmulDeviceOperation  32 x  4096 x 14336` -- the DECODE ff1/ff3 up-projection (per-token path)
     
     Drives the repo's own programming-example kernel triple (tt_metal/programming_examples/matmul/
     matmul_multi_core: reader / mm / writer, copied into tt/kernels/) through ttnn.generic_op, with
    @@ -13,9 +14,10 @@ the output tiles partitioned across the entire compute grid.
     MEASURED on the real shapes, on the full 11x10 P150 grid:
     
         M    K      N        PCC        generic_op    ttnn.linear     verdict
    -    128  4096   14336    0.998986    3.565 ms      0.328 ms       10.9x SLOWER
    -    128  14336  4096     0.993626    3.118 ms      0.357 ms        8.7x SLOWER
    -     32  14336  4096     0.993594    0.916 ms      0.292 ms        3.1x SLOWER
    +    128  4096   14336    0.999039    3.561 ms      0.332 ms       10.7x SLOWER
    +    128  14336  4096     0.993591    3.119 ms      0.358 ms        8.7x SLOWER
    +     32  14336  4096     0.993562    0.919 ms      0.296 ms        3.1x SLOWER
    +     32  4096   14336    0.999022    1.039 ms      0.309 ms        3.4x SLOWER
     
     Every kernel is CORRECT and every one loses. The cause is dataflow, not tuning: this reader fetches
     every A tile again for each output tile, so A is re-read Nt times from DRAM, while ttnn's production
    @@ -44,7 +46,7 @@ from ttnn._ttnn.program_descriptor import VectorUInt32 as _VU32
     TILE = 32
     # (M, K, N) of every op this rung was measured for: the ff1/ff3 up-projection, then ff2's
     # down-projection at the short-prefill and the DECODE row counts.
    -SHAPES = [(128, 4096, 14336), (128, 14336, 4096), (32, 14336, 4096)]
    +SHAPES = [(128, 4096, 14336), (128, 14336, 4096), (32, 14336, 4096), (32, 4096, 14336)]
     ROOT = "/tmp/tt_hw_planner_llama3_1_8b_p150_1785111170/models/demos/llama3_1_8b_p150/tt/kernels"
     
     
    diff --git a/models/demos/llama3_1_8b_p150/tt/ttl_ff2_matmul.py b/models/demos/llama3_1_8b_p150/tt/ttl_ff2_matmul.py
    index 296b46a632..d126d0c5d2 100644
    --- a/models/demos/llama3_1_8b_p150/tt/ttl_ff2_matmul.py
    +++ b/models/demos/llama3_1_8b_p150/tt/ttl_ff2_matmul.py
    @@ -1,35 +1,40 @@
    ... (truncated, 114 more lines)

[#69] MatmulDeviceOperation · tp-fracture · no gain  +1572.07 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/cpp_mm_generic.py b/models/demos/llama3_1_8b_p150/tt/cpp_mm_generic.py
    index c98a9b6ae9..27cae9296a 100644
    --- a/models/demos/llama3_1_8b_p150/tt/cpp_mm_generic.py
    +++ b/models/demos/llama3_1_8b_p150/tt/cpp_mm_generic.py
    @@ -5,6 +5,7 @@ Kept as the record of the cpp rung for every hot dense matmul in the MLP:
       * `MatmulDeviceOperation 128 x  4096 x 14336` -- the short-prefill ff1/ff3 up-projection
       * `MatmulDeviceOperation 128 x 14336 x  4096` -- the short-prefill ff2 down-projection
       * `MatmulDeviceOperation  32 x 14336 x  4096` -- the DECODE ff2 down-projection (per-token path)
    +  * `MatmulDeviceOperation  32 x  4096 x 14336` -- the DECODE ff1/ff3 up-projection (per-token path)
     
     Drives the repo's own programming-example kernel triple (tt_metal/programming_examples/matmul/
     matmul_multi_core: reader / mm / writer, copied into tt/kernels/) through ttnn.generic_op, with
    @@ -13,9 +14,10 @@ the output tiles partitioned across the entire compute grid.
     MEASURED on the real shapes, on the full 11x10 P150 grid:
     
         M    K      N        PCC        generic_op    ttnn.linear     verdict
    -    128  4096   14336    0.998986    3.565 ms      0.328 ms       10.9x SLOWER
    -    128  14336  4096     0.993626    3.118 ms      0.357 ms        8.7x SLOWER
    -     32  14336  4096     0.993594    0.916 ms      0.292 ms        3.1x SLOWER
    +    128  4096   14336    0.999039    3.561 ms      0.332 ms       10.7x SLOWER
    +    128  14336  4096     0.993591    3.119 ms      0.358 ms        8.7x SLOWER
    +     32  14336  4096     0.993562    0.919 ms      0.296 ms        3.1x SLOWER
    +     32  4096   14336    0.999022    1.039 ms      0.309 ms        3.4x SLOWER
     
     Every kernel is CORRECT and every one loses. The cause is dataflow, not tuning: this reader fetches
     every A tile again for each output tile, so A is re-read Nt times from DRAM, while ttnn's production
    @@ -44,7 +46,7 @@ from ttnn._ttnn.program_descriptor import VectorUInt32 as _VU32
     TILE = 32
     # (M, K, N) of every op this rung was measured for: the ff1/ff3 up-projection, then ff2's
     # down-projection at the short-prefill and the DECODE row counts.
    -SHAPES = [(128, 4096, 14336), (128, 14336, 4096), (32, 14336, 4096)]
    +SHAPES = [(128, 4096, 14336), (128, 14336, 4096), (32, 14336, 4096), (32, 4096, 14336)]
     ROOT = "/tmp/tt_hw_planner_llama3_1_8b_p150_1785111170/models/demos/llama3_1_8b_p150/tt/kernels"
     
     
    diff --git a/models/demos/llama3_1_8b_p150/tt/ttl_ff2_matmul.py b/models/demos/llama3_1_8b_p150/tt/ttl_ff2_matmul.py
    index 296b46a632..d126d0c5d2 100644
    --- a/models/demos/llama3_1_8b_p150/tt/ttl_ff2_matmul.py
    +++ b/models/demos/llama3_1_8b_p150/tt/ttl_ff2_matmul.py
    @@ -1,35 +1,40 @@
    ... (truncated, 114 more lines)

[#73] MatmulDeviceOperation · grid · win  +1596.56 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/lm_head.py b/models/demos/llama3_1_8b_p150/tt/lm_head.py
    index ce12a6def3..eeb42ce095 100644
    --- a/models/demos/llama3_1_8b_p150/tt/lm_head.py
    +++ b/models/demos/llama3_1_8b_p150/tt/lm_head.py
    @@ -71,6 +71,18 @@ class LMHead(LightweightModule):
             self.output_weights_dram_sharded = []
             self.output_weights_ring_mm = []
     
    +        # FULL-GRID LM HEAD. The DRAM-sharded matmul variant width-shards the activation across
    +        # `lm_head_core_grid` cores, so its core count must divide K/32 = dim/32 = 128 tiles. On a
    +        # P150 (11x10 = 110 cores) the largest such divisor is 64 -- the head therefore leaves ~46
    +        # cores idle on every one of its 8 vocab splits, which is exactly the grid=partial tag.
    +        # There is no way to widen it while keeping that variant, so take the other variant: hold
    +        # the weights DRAM-INTERLEAVED and let ttnn.linear auto-route, which splits the 501 output
    +        # tiles per split over the whole grid. Trades DRAM-sharded read bandwidth for occupancy;
    +        # which one wins is a measurement, not a derivation.
    +        self.full_grid = (
    +            self.prefetcher is None and not args.is_galaxy and self.num_devices == 1
    +        )
    +
             self.split_sizes = [self.split_sizes_dram_sharded]
             if self.prefetcher is not None:
                 self.split_sizes.append(self.split_sizes_ring_mm)
    @@ -87,11 +99,15 @@ class LMHead(LightweightModule):
                     # Concatenate the splits from all devices
                     combined_split = torch.cat(device_splits, dim=-1)
     
    +                # The cache key must encode the memory config: a cached DRAM-width-sharded weight
    +                # reloaded for the interleaved path (or vice versa) silently feeds the matmul an
    +                # operand whose shard spec its program config does not expect.
    +                _layout_tag = "_ilv" if (mode == 0 and self.full_grid) else ""
                     cache_file_name = (
                         None
                         if args.dummy_weights
                         else weight_cache_path
    -                    / f"output_lm_head_{len(split_sizes)}_split_shard_{i}_{combined_split.shape[-1]}_mode_{mode}"
    +                    / f"output_lm_head_{len(split_sizes)}_split_shard_{i}_{combined_split.shape[-1]}_mode_{mode}{_layout_tag}"
                     )
     
                     def pad_to_power_of_2(n):
    ... (truncated, 50 more lines)

[#74] NlpCreateHeadsDeviceOperation · grid · no gain  +1508.93 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/lm_head.py b/models/demos/llama3_1_8b_p150/tt/lm_head.py
    index ce12a6def3..eeb42ce095 100644
    --- a/models/demos/llama3_1_8b_p150/tt/lm_head.py
    +++ b/models/demos/llama3_1_8b_p150/tt/lm_head.py
    @@ -71,6 +71,18 @@ class LMHead(LightweightModule):
             self.output_weights_dram_sharded = []
             self.output_weights_ring_mm = []
     
    +        # FULL-GRID LM HEAD. The DRAM-sharded matmul variant width-shards the activation across
    +        # `lm_head_core_grid` cores, so its core count must divide K/32 = dim/32 = 128 tiles. On a
    +        # P150 (11x10 = 110 cores) the largest such divisor is 64 -- the head therefore leaves ~46
    +        # cores idle on every one of its 8 vocab splits, which is exactly the grid=partial tag.
    +        # There is no way to widen it while keeping that variant, so take the other variant: hold
    +        # the weights DRAM-INTERLEAVED and let ttnn.linear auto-route, which splits the 501 output
    +        # tiles per split over the whole grid. Trades DRAM-sharded read bandwidth for occupancy;
    +        # which one wins is a measurement, not a derivation.
    +        self.full_grid = (
    +            self.prefetcher is None and not args.is_galaxy and self.num_devices == 1
    +        )
    +
             self.split_sizes = [self.split_sizes_dram_sharded]
             if self.prefetcher is not None:
                 self.split_sizes.append(self.split_sizes_ring_mm)
    @@ -87,11 +99,15 @@ class LMHead(LightweightModule):
                     # Concatenate the splits from all devices
                     combined_split = torch.cat(device_splits, dim=-1)
     
    +                # The cache key must encode the memory config: a cached DRAM-width-sharded weight
    +                # reloaded for the interleaved path (or vice versa) silently feeds the matmul an
    +                # operand whose shard spec its program config does not expect.
    +                _layout_tag = "_ilv" if (mode == 0 and self.full_grid) else ""
                     cache_file_name = (
                         None
                         if args.dummy_weights
                         else weight_cache_path
    -                    / f"output_lm_head_{len(split_sizes)}_split_shard_{i}_{combined_split.shape[-1]}_mode_{mode}"
    +                    / f"output_lm_head_{len(split_sizes)}_split_shard_{i}_{combined_split.shape[-1]}_mode_{mode}{_layout_tag}"
                     )
     
                     def pad_to_power_of_2(n):
    ... (truncated, 50 more lines)

[#76] NlpCreateHeadsDeviceOperation · shard · win  +1598.25 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/attention.py b/models/demos/llama3_1_8b_p150/tt/attention.py
    index 69bba753ac..e4e19d7461 100644
    --- a/models/demos/llama3_1_8b_p150/tt/attention.py
    +++ b/models/demos/llama3_1_8b_p150/tt/attention.py
    @@ -963,6 +963,20 @@ class Attention(LightweightModule):
             ttnn.deallocate(x_11SH)
     
             # split qkv into heads
    +        # L1 island for the head split. This op is pure data movement and memory-bound: it reads
    +        # the fused [S, (nq + 2*nkv) * head_dim] activation and writes three head-major views of
    +        # it, and at short prefill every one of those trips is to DRAM. Q/K/V total
    +        # S * 6144 * 2 B (1.5 MB at S=128), so the whole set fits interleaved L1 comfortably, and
    +        # the immediate consumers -- q_norm/k_norm, then rotary_embedding -- read it straight back.
    +        # The op's non-sharded factory requires an INTERLEAVED output config, so this is L1
    +        # interleaved rather than a shard spec; the sharded factory is not reachable here (its
    +        # output shard spec is fixed at {TILE_HEIGHT, head_dim}, i.e. seq_len == 32 only).
    +        # Bounded to short prefill so long prompts keep the DRAM path.
    +        create_heads_mem_config = (
    +            ttnn.L1_MEMORY_CONFIG
    +            if (not self.TG and self.prefetcher is None and seq_len <= self.args.prefill_len_cutoff)
    +            else ttnn.DRAM_MEMORY_CONFIG
    +        )
             (
                 q_heads_1QSD_pre_rot,
                 k_heads_1KSD_pre_rot,
    @@ -972,7 +986,7 @@ class Attention(LightweightModule):
                 num_heads=self.n_local_heads,
                 num_kv_heads=self.n_local_kv_heads,
                 transpose_k_heads=False,
    -            memory_config=ttnn.DRAM_MEMORY_CONFIG,
    +            memory_config=create_heads_mem_config,
             )
     
             norm_config = self.args.get_norm_config("attn", Mode.PREFILL, None)

[#78] NlpCreateHeadsDeviceOperation · structural · win  +1682.23 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/generator.py b/models/demos/llama3_1_8b_p150/tt/generator.py
    index 01da3ce7f2..19e57ec6e8 100644
    --- a/models/demos/llama3_1_8b_p150/tt/generator.py
    +++ b/models/demos/llama3_1_8b_p150/tt/generator.py
    @@ -630,13 +630,45 @@ class Generator(ModelCapabilitiesMixin, WarmupForwardMixin):
                     _warm_lens = {get_padded_prefill_len(int(n)) for n in _req_lens}
                 except Exception:  # noqa: BLE001 -- never let the narrowing break a real prefill
                     _warm_lens = None
    -            self.warmup_model_prefill(
    -                kv_cache=kv_cache,
    -                enable_trace=enable_trace,
    -                can_sample_on_device=on_device_sampling_enabled,
    -                greedy_only=_sampling_params_are_greedy(sampling_params),
    -                only_seq_lens=_warm_lens,
    +            # ...and once that narrowing leaves ONLY the length this very call is about to run,
    +            # the warmup prefill is a straight duplicate of the request behind it: same padded
    +            # length, same layers, same ops, and its mock-token KV writes are immediately
    +            # overwritten by the real prefill. Every prefill op therefore ran TWICE.
    +            #
    +            # Nothing needs it. Prefill trace capture is lazy -- `_easy_trace_text_prefill` does
    +            # `if self.trace_id_prefill[trace_key] is None: self._capture_trace_prefill(...)` -- and
    +            # the sampling trace captures the same way, so the real call compiles and captures for
    +            # itself. Skipping only trades a one-time HOST compile onto the first request; the
    +            # DEVICE work it removes is pure redundancy.
    +            #
    +            # Only skip the exact-duplicate case. A warmup covering any length this request will
    +            # not run still has something to pre-build, so it still runs.
    +            _this_call_lens = None
    +            try:
    +                _this_call_lens = {get_padded_prefill_len(int(tokens.shape[1]))}
    +            except Exception:  # noqa: BLE001 -- never let the narrowing break a real prefill
    +                _this_call_lens = None
    +            _warmup_is_duplicate = (
    +                _warm_lens is not None
    +                and _this_call_lens is not None
    +                and _warm_lens == _this_call_lens
    +                and int(tokens.shape[0]) == 1
                 )
    +            if _warmup_is_duplicate:
    +                logger.info(
    ... (truncated, 16 more lines)

[#80] NlpCreateHeadsDeviceOperation · tt-lang · win  +1705.81 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/attention.py b/models/demos/llama3_1_8b_p150/tt/attention.py
    index e4e19d7461..c4ed5954c9 100644
    --- a/models/demos/llama3_1_8b_p150/tt/attention.py
    +++ b/models/demos/llama3_1_8b_p150/tt/attention.py
    @@ -7,6 +7,7 @@ import math
     import torch
     
     import ttnn
    +from models.demos.llama3_1_8b_p150.tt import ttl_create_qkv_heads
     from models.common.lightweightmodule import LightweightModule
     from models.common.rmsnorm import RMSNorm
     from models.common.utility_functions import nearest_32
    @@ -977,17 +978,35 @@ class Attention(LightweightModule):
                 if (not self.TG and self.prefetcher is None and seq_len <= self.args.prefill_len_cutoff)
                 else ttnn.DRAM_MEMORY_CONFIG
             )
    -        (
    -            q_heads_1QSD_pre_rot,
    -            k_heads_1KSD_pre_rot,
    -            v_heads_1VSD,
    -        ) = ttnn.experimental.nlp_create_qkv_heads(
    -            xqkv_fused,
    -            num_heads=self.n_local_heads,
    -            num_kv_heads=self.n_local_kv_heads,
    -            transpose_k_heads=False,
    -            memory_config=create_heads_mem_config,
    +        # tt-lang rung: the stock op's core count is baked into its factory (one work unit per
    +        # input row-tile -> 4 cores here), so the only way past it is a kernel whose work
    +        # decomposition differs. ttl_create_qkv_heads parallelises over (seq_tile x head) and
    +        # measures 0.0555 ms/call vs the stock op's 0.0810 at this shape, PCC 1.000000.
    +        _use_ttl_heads = (
    +            not self.TG
    +            and self.prefetcher is None
    +            and ttl_create_qkv_heads.supports(
    +                xqkv_fused, self.n_local_heads, self.n_local_kv_heads, self.head_dim
    +            )
             )
    +        if _use_ttl_heads:
    +            (
    +                q_heads_1QSD_pre_rot,
    ... (truncated, 219 more lines)

Limitations / suggested manual next steps:
- 1 op(s) tried but no lever beat baseline: LayerNormDeviceOperation
  -> inspect the per-op device report and consider a hand-written kernel or a structural change.

Reproduce:
  trace+1CQ perf:  python -m pytest models/demos/llama3_1_8b_p150/tests/e2e/test_main_perf.py::test_main_perf -svv
  full-model e2e PCC:  python -m pytest models/demos/llama3_1_8b_p150/tests/e2e/test_pcc.py -svv

levels: grid -> fidelity -> dtype -> shard -> host -> tt-lang -> cpp   |   ✓win = beat baseline, ·try = measured no-gain, ·wedge = wedged/crashed when tried, — = not attempted
```
<!-- END optimize -->

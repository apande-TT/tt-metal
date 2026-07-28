<!-- BEGIN optimize -->
# Optimize (perf) — `llama3_1_8b_p150`

_Updated live: 2026-07-28 08:27:43 UTC · 150 lever attempt(s) so far — each knob is logged the instant it resolves, win OR fail, with why it was tried and why it won or failed._

```
Optimization summary — llama3_1_8b_p150 · main (device_ms)
==========================================================
optimizing… — baseline->final speedup is finalized when the module converges (per-attempt detail below is live)
tracy trace pass, BASELINE, same window (16 layers):  2.63 ms
trace+1CQ full-pipeline e2e (all layers):  48.38 ms  ->  22.79 ms   (+52.9%, 2.12x)

Roofline & utilization
  modeled floor       : 537.23 ms   (Σ per-op roofline floors)
  achievable (60-80%) : 671.54 - 895.38 ms
  measured            : 648.17 ms
  at-floor            : 83%   (110.94 ms reachable headroom)
  status              : IN_BAND — reached the achievable band — done
  (tok/s/u — N/A: not an LLM decode pipeline)

Op breakdown — device time by op class (BASELINE profile · what to target, ranked):
op class         device_ms      %   count  bound  dominant op (shape)
---------------------------------------------------------------------------------------------------
matmul              108.16  85.0%     886   slow  MatmulDeviceOperation 32 x 4096 x 16032
host_overhead        32.15  25.3%       0   host  
reduction             8.86   7.0%     292   slow  LayerNormDeviceOperation
datamove              3.73   2.9%    1013   slow  UntilizeDeviceOperation
other                 2.31   1.8%     398   slow  GenericOpDeviceOperation
eltwise               2.01   1.6%     306   slow  BinaryNgDeviceOperation
embedding             1.37   1.1%     103   slow  EmbeddingsDeviceOperation
attention             0.74   0.6%     102   slow  SDPAOperation

Block-level timing (per-stage trace) — latest lever on BinaryNgDeviceOperation:
  MatmulDeviceOperation (all shapes)    547.69 ms  ###################### · True  <- hottest
  LayerNormDeviceOperation     52.02 ms  ##....................
  BinaryNg [128,14336] bf8_b gate mul (kernel-blocked: bf8_b pack)      8.23 ms  ......................
  BinaryNg [32,14336] bf8_b decode gate mul (kernel-blocked: bf8_b pack)      3.50 ms  ......................
  BinaryNg [128,4096] post-MLP add, stock 3.67us/call @110 cores      1.29 ms  ......................
    same add, C++ Metalium 3.87us/call @110 cores (parity, reverted)      1.36 ms  ......................
    same add, tt-lang 4.87us/call @64 cores (1.32x, reverted)      1.71 ms  ......................
  BinaryNg [128,4096] post-attn add (kernel-blocked: mixed dtype)      1.29 ms  ......................
  BinaryNg [32,4096] decode residual adds      1.03 ms  ......................
  GenericOpDeviceOperation (tt-lang split+rope + concat)      6.83 ms  ......................
  SDPAOperation (prefill)      4.80 ms  ......................

Block-level timing (per-stage trace) — latest lever on BinaryNgDeviceOperation:
  MatmulDeviceOperation (all shapes)    547.69 ms  ###################### · True  <- hottest
  LayerNormDeviceOperation     52.02 ms  ##....................
  BinaryNg [128,14336] bf8_b gate mul (kernel-blocked: bf8_b pack)      8.23 ms  ......................
  BinaryNg [32,14336] bf8_b decode gate mul (kernel-blocked: bf8_b pack)      3.50 ms  ......................
  BinaryNg [128,4096] post-MLP add, stock 3.67us/call @110 cores      1.29 ms  ......................
    same add, C++ Metalium 3.87us/call @110 cores (parity, reverted)      1.36 ms  ......................
    same add, tt-lang 4.87us/call @64 cores (1.32x, reverted)      1.71 ms  ......................
  BinaryNg [128,4096] post-attn add (kernel-blocked: mixed dtype)      1.29 ms  ......................
  BinaryNg [32,4096] decode residual adds      1.03 ms  ......................
  GenericOpDeviceOperation (tt-lang split+rope + concat)      6.83 ms  ......................
  SDPAOperation (prefill)      4.80 ms  ......................

op                                 grid      fidelity  dtype     shard     host      tt-lang   cpp       other       best ms
----------------------------------------------------------------------------------------------------------------------------
ArgMaxDeviceOperation              ✓win      —         —         ✓win      ✓win      —         —         —                 —
BinaryNgDeviceOperation            ·try      —         —         ✓win      ·try      ✓win      ✓win      —            648.17
GenericOpDeviceOperation           ✓win      —         —         —         —         —         —         —                 —
LayerNormDeviceOperation           ·try      —         —         ·try      ·try      —         —         —           1057.73
MatmulDeviceOperation              ✓win      —         ✓win      ✓win      ·try      ·try      ·try      ✓win        1061.00
MatmulDeviceOperation              ·try      —         ✓win      ·try      ·try      ✓win      ✓win      ·try        1138.67
MatmulDeviceOperation              ✓win      —         ✓win      —         —         —         —         —           1092.12
MatmulDeviceOperation              ·try      —         ✓win      ·try      ·try      ✓win      ·try      ·try        1057.68
MatmulDeviceOperation              ·try      —         ✓win      ✓win      ·try      ·try      ·try      ✓win         891.98
MatmulDeviceOperation              ✓win      —         ✓win      ·try      ·try      ·try      ·try      ✓win         662.92
MatmulDeviceOperation              ·try      —         ✓win      ✓win      ·try      ✓win      ·try      ✓win         745.01
NLPConcatHeadsDeviceOperation      ·try      —         —         ✓win      ✓win      ✓win      —         —            714.94
NlpCreateHeadsDeviceOperation      ·try      —         —         ✓win      ✓win      ✓win      —         —            955.25
RotaryEmbeddingLlamaDeviceOperatio ✓win      —         —         —         ✓win      —         —         —                 —
SDPAOperation                      ✓win      —         —         —         ·try      —         —         —            655.73
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
NlpCreateHeadsDeviceOperation           tt-lang         —             —  ✓ win      committed: llama3_1_8b_p150: refresh the generated RUN_REPORT Checkpoints the live lever log so the tree is clean. A dirty tree scopes record_kernel_at
NlpCreateHeadsDeviceOperation           tt-lang    758.37   +1705.81 ms  ✓ win      Re-recorded against a clean tree so the kernel-marker scan sees tt/ttl_create_qkv_heads.py (the first record was diff-scoped to attention.py and flagged UNSUPPORTED). Hypothesis: a kernel is the RIGHT
MatmulDeviceOperation                     dtype         —             —  ✓ win      committed: llama3_1_8b_p150: put the fused QKV weight on bf4_b WQKV was still on the BFP8 default while the whole MLP already rides bf4_b. The fused QK
MatmulDeviceOperation                     dtype    749.85   +1714.33 ms  ✓ win      Hypothesis: this projection is DRAM-bandwidth bound and WQKV was the largest weight in the model still on the BFP8 default -- the MLP already rides bf4_b end to end, but the fused QKV weight [dim, (nq
MatmulDeviceOperation                      grid    754.36   +1709.82 ms  · no gain  Hypothesis: decode QKV is grid=partial because find_grid() sorts candidate core counts by distance from a hard-coded target=32 -- the right answer on a 64-core Wormhole, but on a 110-core P150 it pick
MatmulDeviceOperation                      grid    754.36   +1709.82 ms  · no gain  Hypothesis: decode QKV is grid=partial because find_grid() sorts candidate core counts by distance from a hard-coded target=32 -- correct on a 64-core Wormhole, but on a 110-core P150 it picks 32 core
MatmulDeviceOperation                     dtype         —             —  ✓ win      committed: llama3_1_8b_p150: write the decode QKV output as bf8_b wqkv reached the bf4_b weight floor in the previous commit, so the only dtype bytes l
MatmulDeviceOperation                     dtype    749.94   +1714.24 ms  ✓ win      Hypothesis via the catalogued matmul-coherence lever: wqkv hit the bf4_b weight floor in the previous commit, so the only dtype bytes left on this DRAM-bw-bound matmul are the ones it WRITES. Found th
MatmulDeviceOperation                     shard         —             —  ✓ win      committed: llama3_1_8b_p150: keep the decode QKV output sharded into the head split Both operands of this matmul were already L1 width-sharded and its
MatmulDeviceOperation                     shard    745.01   +1719.17 ms  ✓ win      Hypothesis: the usual shard targets were already spent on this op -- both operands are L1 width-sharded and the 13 MB/layer bf4_b weight can never be L1-resident (the only residency path is the DramPr
MatmulDeviceOperation               tp-fracture    745.01   +1719.17 ms  · no gain  Hypothesis: if decode QKV is still DRAM-bandwidth bound after every single-chip lever, splitting the wqkv read across chips would divide the bytes each chip fetches. tp_pick_degree(32, 4096, 6144) ret
MatmulDeviceOperation               tp-fracture         —             —  ✓ win      committed: llama3_1_8b_p150: refresh the generated RUN_REPORT Checkpoints the live lever log. Also needed for the marker scan: with a dirty tree, recor
MatmulDeviceOperation               tp-fracture    745.01   +1719.17 ms  · no gain  Re-recorded against a clean tree so the evidence scan sees the model-wide ShardTensorToMesh + CCL plumbing. Hypothesis: if decode QKV is still DRAM-bandwidth bound after every single-chip lever, split
MatmulDeviceOperation                structural    745.01   +1719.17 ms  · no gain  none: walked the whole decode QKV chain hunting reducible work and every candidate is already applied or structurally absent. (1) RECOMPUTE -> CACHE is already done: decode is not repeat_prefill, it r
MatmulDeviceOperation                   tt-lang    745.01   +1719.17 ms  · no gain  Measured the tt-lang matmul on THIS op's own shape rather than inheriting the sibling MLP verdicts, since M and the K/N ratio both differ -- extended tt/ttl_ff2_matmul.py's (M,K,N) sweep with (32, 409
MatmulDeviceOperation                   tt-lang         —             —  ✓ win      committed: llama3_1_8b_p150: measure both kernel rungs on the decode QKV shape The tt-lang and C++ Metalium rungs had only ever been measured on the ML
MatmulDeviceOperation                       cpp    745.01   +1719.17 ms  · no gain  Same discipline as the tt-lang rung: do not inherit the MLP verdicts, measure the C++ Metalium reader/compute/writer triple (tt/cpp_mm_generic.py + tt/kernels/*.cpp, driven through ttnn.generic_op wit
RotaryEmbeddingLlamaDeviceOperatio         grid         —             —  ✓ win      committed: llama3_1_8b_p150: fold heads into the batch dim for the prefill rope The prefill rope program factory parallelises over batch x seq-tiles ON
RotaryEmbeddingLlamaDeviceOperatio         grid    729.90   +1734.28 ms  ✓ win      Hypothesis: read the program factory instead of guessing at a program_config (this op exposes none). It parallelises over batch x seq-tiles ONLY -- batch_parallel_factor = min(batch, cores), seq_paral
NLPConcatHeadsDeviceOperation              grid    729.90   +1734.28 ms  · no gain  Hypothesis from reading the program factory: the INTERLEAVED path derives cores from num_blocks = batch * seq_len / TILE_HEIGHT, so heads (dim 1) never enter the split and at batch 1 / seq_len 128 it 
NLPConcatHeadsDeviceOperation              grid    729.90   +1734.28 ms  · no gain  Third and final grid variant on this op; the sharded factory is measured-blocked at this shape. Recap: the interleaved factory sets cores from num_blocks = batch * seq_len / TILE_HEIGHT, so heads neve
NLPConcatHeadsDeviceOperation             shard    729.99   +1734.19 ms  · no gain  Hypothesis: the INPUT side of this op is measured-unsafe (three head-sharded variants at the grid rung either scrambled the output or hung), but the OUTPUT side is free -- the interleaved factory only
NLPConcatHeadsDeviceOperation             shard         —             —  ✓ win      committed: llama3_1_8b_p150: refresh the generated RUN_REPORT
NLPConcatHeadsDeviceOperation             shard    729.99   +1734.19 ms  · no gain  Hypothesis: the INPUT side of this op is measured-unsafe (three head-sharded variants at the grid rung either scrambled the output to 25.6% top-1 or hung the gate), but the OUTPUT side is free -- the 
NLPConcatHeadsDeviceOperation        structural    729.90   +1734.28 ms  · no gain  none: hunted for reducible work around the prefill concat and found none left. (1) The op is NOT removable: SDPA emits head-major [1, H, S, D] and the wo projection requires heads contiguous in the WI
NLPConcatHeadsDeviceOperation        structural         —             —  ✓ win      committed: llama3_1_8b_p150: tt-lang kernel for the prefill concat-heads The mirror of tt/ttl_create_qkv_heads.py at the other end of attention: that k
NLPConcatHeadsDeviceOperation           tt-lang    714.94   +1749.24 ms  ✓ win      Hypothesis: a kernel is the right rung because the knob rungs proved the parallelisation is baked into the op, not merely untuned -- the interleaved factory sizes cores from num_blocks = batch*seq_len
ArgMaxDeviceOperation                      grid         —             —  · wedged   wedged/crashed when tried: perf test crashed at runtime: E RuntimeError: Read 0xffffffff over PCIe ID 3: the board should be reset.
ArgMaxDeviceOperation                      grid         —             —  ✓ win      committed: llama3_1_8b_p150: document why sub_core_grids is left undefined TTSampling reads getattr(args, "sub_core_grids", None), and leaving it None
ArgMaxDeviceOperation                      grid    714.94   +1749.24 ms  · no gain  Read the factory before reaching for a knob, and the grid turns out to be ALREADY full. ttnn.argmax takes the multi-core factory here (input is row-major after untilize, dim == rank-1), and with sub_c
ArgMaxDeviceOperation                      grid         —             —  ✓ win      committed: llama3_1_8b_p150: refresh the generated RUN_REPORT Final checkpoint of the live lever log for this round. Device hit a board-level PCIe faul
ArgMaxDeviceOperation                      grid         —             —  · wedged   wedged: round killed (UNPRODUCTIVE 10800s — agent watchdog judged the round stuck (no real progress))
ArgMaxDeviceOperation                     shard         —             —  ✓ win      committed: llama3_1_8b_p150: keep the prefill logits in L1 for the argmax `_apply_norm_and_lm_head` pushed the lm_head output straight back to DRAM, so
ArgMaxDeviceOperation                     shard    713.16   +1751.02 ms  ✓ win      Read the op before reaching for a shard spec, and it decided the whole rung: argmax_device_operation.cpp TT_FATALs unless the input memory_layout is INTERLEAVED, so a width/height shard is architectur
ArgMaxDeviceOperation                structural         —             —  · wedged   wedged/crashed when tried: perf test crashed at runtime: TT_FATAL: cq_id 0 is out of range (assert.hpp:104)
ArgMaxDeviceOperation                structural         —             —  ✓ win      committed: llama3_1_8b_p150: argmax only the live batch rows, not the padding TTSampling rounds the request batch up to a 32-row tile because the LOGIT
ArgMaxDeviceOperation                structural    682.47   +1781.71 ms  ✓ win      Hypothesis: the reducible work is PADDING, not per-element cost. TTSampling.__init__ does max_batch_size = max(32, roundup(raw_batch,32)) because the LOGITS are tile-layout, but the force-argmax path 
MatmulDeviceOperation                     dtype         —             —  ✓ win      committed: llama3_1_8b_p150: put the LM head weight on bf4_b The output projection is the single largest weight in the model ([dim, padded_vocab] = 409
MatmulDeviceOperation                     dtype    662.92   +1801.26 ms  ✓ win      Hypothesis: this op is DRAM-bw bound (profiler tags it DRAM, 101 cores, 171 us/call) and the LM head is the single largest weight in the model -- [dim, padded_vocab] = 4096 x 128256, ~70 MB at bf8_b. 
MatmulDeviceOperation                     shard         —             —  · wedged   wedged/crashed when tried: perf test crashed at runtime: E RuntimeError: NOC0 is hung on PCIe device ID 1.
MatmulDeviceOperation                     shard    662.92   +1801.26 ms  · no gain  Hypothesis and the arithmetic that bounds it: this op is DRAM-bw bound, so the only shard that can help is one that REMOVES BYTES, i.e. L1 residency -- and the measured rate says there is almost nothi
MatmulDeviceOperation                     shard    662.92   +1801.26 ms  · no gain  Hypothesis bounded by arithmetic first: this op is DRAM-bw bound, so only a BYTE-REMOVING shard (L1 residency) can help -- and each vocab split already reads 4096 x 16032 bf4_b = 36.9 MB in 121 us = ~
MatmulDeviceOperation               tp-fracture    662.92   +1801.26 ms  · no gain  Hypothesis: if the LM head is still DRAM-bandwidth bound after every single-chip lever, splitting the 295 MB output-projection weight across chips would divide the bytes each chip fetches. tp_pick_deg
MatmulDeviceOperation               tp-fracture         —             —  ✓ win      committed: llama3_1_8b_p150: refresh the generated RUN_REPORT Checkpoints the live lever log so the tree is clean. A dirty tree scopes record_kernel_at
MatmulDeviceOperation               tp-fracture    662.92   +1801.26 ms  · no gain  Re-recorded against a clean tree so the evidence scan sees the model-wide ShardTensorToMesh + CCL plumbing (the first record was diff-scoped and flagged UNSUPPORTED). Hypothesis: if the LM head is sti
MatmulDeviceOperation               tp-fracture    662.92   +1801.26 ms  · no gain  Re-recorded against a genuinely clean model dir so the evidence scan runs whole-model-dir and sees the ShardTensorToMesh + CCL plumbing (the first two records were diff-scoped -- the generated RUN_REP
MatmulDeviceOperation                structural   1283.73   +1180.45 ms  · no gain  Hunted the whole LM-head chain for reducible work and found exactly one candidate, then DISPROVED it -- the answer is worth more than the lever. Ruled out first, from source: (a) prefill WARMUP is alr
MatmulDeviceOperation                   tt-lang         —             —  · wedged   wedged/crashed when tried: perf test crashed at runtime: TT_FATAL: cq_id 0 is out of range (assert.hpp:104) custom generic_op/ttl kernels ARE trace-capturable on this build — verified on device: a cac
MatmulDeviceOperation                   tt-lang    662.92   +1801.26 ms  · no gain  Measured the tt-lang rung on the LM head's OWN shape rather than inheriting the MLP verdicts, by extending tt/ttl_ff2_matmul.py with ttl_mm_lmhead + LmHeadSplitTTL and routing vocab split 0 through it
MatmulDeviceOperation                       cpp    662.92   +1801.26 ms  · no gain  Authored and wired the C++ Metalium rung on the LM head's OWN shape: extended tt/cpp_mm_generic.py with LmHeadSplitCpp (the repo's reader/mm/writer triple via ttnn.generic_op) and routed vocab split 0
MatmulDeviceOperation                   tt-lang    662.92   +1801.26 ms  · no gain  CORRECTION to the earlier tt-lang record for this op, which claimed the kernel "is so slow the 225 s correctness run TIMES OUT". That inference is CONFOUNDED and I withdraw it: the device wedged durin
SDPAOperation                              grid         —             —  ✓ win      committed: llama3_1_8b_p150: size the prefill SDPA chunk+grid to the work, not to 8x8 The prefill SDPA factory flattens attention into B*n_local_heads*
SDPAOperation                              grid    661.00   +1803.18 ms  ✓ win      Hypothesis from READING the prefill SDPA program factory rather than guessing a grid: it flattens work into B*n_local_heads*ceil(seq/q_chunk) Q-chunks and pair-distributes them for causal attention, s
RotaryEmbeddingLlamaDeviceOperatio   structural         —             —  ✓ win      committed: llama3_1_8b_p150: fold the prefill rope into the tt-lang head-split kernel The prefill rope was DISPATCH bound, not compute bound: 736 launc
RotaryEmbeddingLlamaDeviceOperatio   structural    656.91   +1807.27 ms  ✓ win      Found real reducible work: this op is DISPATCH bound (736 prefill launches x 9.55 us vs a 1.21 us roofline -- a [32,1,128,128] rotation is only ~8 tiles/core, so nearly all of it is fixed launch cost 
GenericOpDeviceOperation                   grid         —             —  ✓ win      committed: llama3_1_8b_p150: widen both tt-lang kernels from 32 to 64 cores Both kernels were pinned at 32 of the P150's 110 cores, and not by tuning.
GenericOpDeviceOperation                   grid    653.69   +1810.49 ms  ✓ win      Hypothesis: this op is the model's own two tt-lang kernels (fused QKV split+rope, and concat-heads) and both were tagged grid=partial for a STRUCTURAL reason, not a tuning one -- their work is three-d
SDPAOperation                        structural    655.73   +1808.45 ms  · no gain  Hunted reducible work around this dispatch-bound SDPA (368 calls x 13.14us vs a 1.33us roofline, 64 cores, seq 128) and measured TWO candidates, both rejected -- the evidence is the value here. (1) FU
BinaryNgDeviceOperation                    grid    660.60   +1803.58 ms  · no gain  Read the program factory and the profile before reaching for a knob, and together they decided the rung: for an INTERLEAVED operand binary_ng calls split_work_to_cores over the whole device grid, so t
BinaryNgDeviceOperation                    grid    671.92   +1792.26 ms  · no gain  Second grid geometry on the same dominant SILU-gate multiply, chosen because BLOCK sharding splits the HEIGHT as well as the width and so can in principle address cores a width split cannot. It cannot
BinaryNgDeviceOperation                   shard         —             —  ✓ win      committed: llama3_1_8b_p150: keep the short-prefill residual stream in L1 The shard rung for BinaryNgDeviceOperation. The two residual adds were the mo
BinaryNgDeviceOperation                   shard    648.17   +1816.01 ms  ✓ win      Hypothesis from the per-instance profile rather than the op aggregate: five BinaryNg shapes exist and only ONE class is DRAM-resident -- the two residual adds ([128,4096], 352+352 calls, 6.23 ms at ~8
BinaryNgDeviceOperation              structural    648.17   +1816.01 ms  · no gain  Hunted reducible work behind this eltwise and found the single largest one in the whole benchmark, then proved it is CORRECTNESS-blocked -- which corrects an earlier round's record of the same lever a
BinaryNgDeviceOperation                 tt-lang         —             —  ✓ win      committed: llama3_1_8b_p150: record the tt-lang residual-add kernel and why it loses tt/ttl_residual_add.py is a real multi-core tt-lang kernel for the
BinaryNgDeviceOperation                 tt-lang    648.35   +1815.83 ms  · no gain  Authored tt/ttl_residual_add.py, a real multi-core tt-lang kernel, and wired it into the model rather than measuring it standalone. Instance choice first, because four of this op's five shapes are clo
BinaryNgDeviceOperation                 tt-lang         —             —  ✓ win      committed: llama3_1_8b_p150: refresh the generated RUN_REPORT Checkpoints the live lever log. Also required for the kernel-marker scan: with a dirty tr
BinaryNgDeviceOperation                 tt-lang    648.35   +1815.83 ms  · no gain  Re-recorded against a clean tree so the evidence scan runs whole-model-dir and sees tt/ttl_residual_add.py (the first record was diff-scoped by a dirty RUN_REPORT and flagged UNSUPPORTED). Authored tt
BinaryNgDeviceOperation                 tt-lang    648.35   +1815.83 ms  · no gain  Re-recorded against a genuinely clean tree so the evidence scan runs whole-model-dir and sees tt/ttl_residual_add.py (the first two records were diff-scoped and flagged UNSUPPORTED -- a committed kern
BinaryNgDeviceOperation                     cpp         —             —  ✓ win      committed: llama3_1_8b_p150: record the C++ Metalium eltwise-add rung and why it ties tt/cpp_add_generic.py + tt/kernels/{dataflow/reader_add_partition
BinaryNgDeviceOperation                     cpp    647.82   +1816.36 ms  · no gain  Authored tt/cpp_add_generic.py + tt/kernels/{dataflow/reader_add_partitioned,compute/add_tiles_stream}.cpp -- a real Metalium reader/compute/writer triple through ttnn.generic_op, adapted from the rep

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

[#84] MatmulDeviceOperation · dtype · win  +1714.33 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/model_config.py b/models/demos/llama3_1_8b_p150/tt/model_config.py
    index 8b557ec23c..61d172b9df 100644
    --- a/models/demos/llama3_1_8b_p150/tt/model_config.py
    +++ b/models/demos/llama3_1_8b_p150/tt/model_config.py
    @@ -232,6 +232,13 @@ class ModelOptimizations:
                     "TensorPrecision": {
                         TensorGroup.FF1_FF3: PrecisionSetting.BFP4,
                         TensorGroup.FF2: PrecisionSetting.BFP4,
    +                    # WQKV was still on the BFP8 default while the whole MLP rides bf4_b. The
    +                    # fused QKV weight is [dim, (nq + 2*nkv)*head_dim] = 25M params, ~26 MB per
    +                    # layer at bf8_b, and the projection is DRAM-bandwidth bound in the roofline
    +                    # -- so this is the largest weight left that has a dtype step available. It is
    +                    # one resident tensor shared by the prefill AND decode QKV matmuls, so the
    +                    # halving lands on both paths.
    +                    TensorGroup.WQKV: PrecisionSetting.BFP4,
                     },
                     "OpFidelity": {OpGroup.LI_FF1_FF3: MathFidelitySetting.LOFI},
                 }
    @@ -422,6 +429,12 @@ def parse_decoder_json(json_file_path, default_optimization=ModelOptimizations.p
             for decoder_id, settings in config_data["decoders"].items():
                 decoder_id = int(decoder_id)
     
    +            # A decoder entry REPLACES the optimization level's settings rather than merging onto
    +            # them, so a decoder named here falls back to the BFP8/HIFI2 defaults for every tensor
    +            # it does not mention. That looks like a bug and is load-bearing: it is what keeps
    +            # decoder 31 -- the last layer, which this file exists to protect -- entirely at BFP8.
    +            # Merging instead (so model-wide dtype levers reach it) was measured and REJECTED:
    +            # letting FF2/WQKV go bf4_b on layer 31 drops top-1 from 99% to 23.8%.
                 tensor_precision = (
                     {TensorGroup[key]: PrecisionSetting[value] for key, value in settings.get("precision_cfg").items()}
                     if "precision_cfg" in settings

[#85] MatmulDeviceOperation · grid · no gain  +1709.82 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/model_config.py b/models/demos/llama3_1_8b_p150/tt/model_config.py
    index 61d172b9df..8239ba34de 100644
    --- a/models/demos/llama3_1_8b_p150/tt/model_config.py
    +++ b/models/demos/llama3_1_8b_p150/tt/model_config.py
    @@ -773,7 +773,16 @@ class ModelArgs:
     
                 # For maximum performance, set the prefill grid row to 8, even if it can fit in a smaller grid
                 self.prefill_rows = 8
    -            self.attn_input_grid = self.dram_shard_core_grid_for_k(self.dim)
    +            # One grid drives three coupled things here: the width-shard of the attention input,
    +            # the DECODE QKV matmul's core count, and the attention norm's sharded program config.
    +            # find_grid() aims at 32 cores, so all three ran on 32 of the P150's 110 -- while 64
    +            # is equally legal (the only rule is that the core count divide dim/32 = 128 tiles).
    +            # Widen coherently so the norm still hands the matmul a shard it can consume.
    +            self.attn_input_grid = (
    +                self.widest_dram_shard_core_grid_for_k(self.dim)
    +                if not self.is_galaxy
    +                else None
    +            ) or self.dram_shard_core_grid_for_k(self.dim)
                 self.mlp1_3_grid = lambda seq_len: (
                     (8, min(min(seq_len, 1024) // 32, 4))
                     if self.is_galaxy
    @@ -3292,6 +3301,26 @@ class ModelArgs:
             rows, cols = self.find_grid(k // ttnn.TILE_SIZE)
             return ttnn.CoreGrid(x=cols, y=rows)
     
    +    def widest_dram_shard_core_grid_for_k(self, k: int):
    +        """The WIDEST core grid a DRAM-sharded matmul on this K can legally use.
    +
    +        ``find_grid`` sorts candidate core counts by distance from a hard-coded ``target = 32``,
    +        which was the right answer on a 64-core Wormhole and leaves most of a 110-core Blackhole
    +        idle. The actual constraint is only that the core count divide ``k / TILE`` (the
    +        activation is width-sharded, one K-slice per core) and fit the device grid -- so take the
    +        largest such divisor instead of the one nearest 32. Returns None when nothing beats
    +        ``find_grid``'s choice.
    +        """
    +        tiles = k // ttnn.TILE_SIZE
    +        gx, gy = self.max_grid_size.x, self.max_grid_size.y
    +        for cores in range(gx * gy, 0, -1):
    +            if tiles % cores:
    ... (truncated, 9 more lines)

[#86] MatmulDeviceOperation · grid · no gain  +1709.82 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/model_config.py b/models/demos/llama3_1_8b_p150/tt/model_config.py
    index 61d172b9df..fac6368d6e 100644
    --- a/models/demos/llama3_1_8b_p150/tt/model_config.py
    +++ b/models/demos/llama3_1_8b_p150/tt/model_config.py
    @@ -773,7 +773,11 @@ class ModelArgs:
     
                 # For maximum performance, set the prefill grid row to 8, even if it can fit in a smaller grid
                 self.prefill_rows = 8
    -            self.attn_input_grid = self.dram_shard_core_grid_for_k(self.dim)
    +            self.attn_input_grid = (
    +                self.widest_dram_shard_core_grid_for_k(self.dim)
    +                if not self.is_galaxy
    +                else None
    +            ) or self.dram_shard_core_grid_for_k(self.dim)
                 self.mlp1_3_grid = lambda seq_len: (
                     (8, min(min(seq_len, 1024) // 32, 4))
                     if self.is_galaxy
    @@ -3292,6 +3296,27 @@ class ModelArgs:
             rows, cols = self.find_grid(k // ttnn.TILE_SIZE)
             return ttnn.CoreGrid(x=cols, y=rows)
     
    +    def widest_dram_shard_core_grid_for_k(self, k: int):
    +        """The WIDEST core grid a DRAM-sharded matmul on this K can legally use.
    +
    +        MEASURED AND REJECTED for attn_input_grid -- kept as the record of the grid rung on the
    +        decode QKV matmul. ``find_grid`` sorts candidates by distance from a hard-coded
    +        ``target = 32``, which looks like a 64-core Wormhole leftover on a 110-core Blackhole:
    +        the real constraint is only that the core count divide ``k / TILE``. Taking the largest
    +        such divisor (64 instead of 32 at dim=4096) measured 749.85 -> 754.36 ms. The op is
    +        DRAM-bandwidth bound, so extra cores buy no bandwidth while halving each core's K-slice
    +        to 2 tiles and per_core_N from 6 to 3 -- target=32 is load-bearing at this shape.
    +        """
    +        tiles = k // ttnn.TILE_SIZE
    +        gx, gy = self.max_grid_size.x, self.max_grid_size.y
    +        for cores in range(gx * gy, 0, -1):
    +            if tiles % cores:
    +                continue
    +            for rows in range(min(cores, gy), 0, -1):
    +                if cores % rows == 0 and cores // rows <= gx:
    +                    return ttnn.CoreGrid(x=cores // rows, y=rows)
    ... (truncated, 5 more lines)

[#88] MatmulDeviceOperation · dtype · win  +1714.24 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/attention.py b/models/demos/llama3_1_8b_p150/tt/attention.py
    index c4ed5954c9..76b944f93e 100644
    --- a/models/demos/llama3_1_8b_p150/tt/attention.py
    +++ b/models/demos/llama3_1_8b_p150/tt/attention.py
    @@ -594,13 +594,24 @@ class Attention(LightweightModule):
             qkv_input_mem_config = self.args.get_attn_qkv_mm_mem_config(Mode.DECODE, self.prefetcher)
             x_sharded = ttnn.to_memory_config(x, qkv_input_mem_config) if self.prefetcher is not None else x
     
    +        # wqkv is already at the bf4_b weight floor, so the only dtype bytes left on this
    +        # DRAM-bw-bound matmul are what it WRITES. The fused [32, qkv_size] output is consumed
    +        # only by the (single-chip no-op) all-reduce and then by sharded_to_interleaved, which
    +        # re-emits bf16 regardless because nlp_create_qkv_heads_decode requires it -- so writing
    +        # bf8_b here halves the store and the reshard's read without changing the dtype contract
    +        # the head split sees. bf8_b is the floor: Q/K/V feed the KV cache and SDPA, and pushing
    +        # attention tensors below bf8_b compounds over depth.
    +        _qkv_decode_out_dtype = self.ccl_dtype if self.TG else self.activation_dtype or ttnn.bfloat16
    +        if not self.TG and self.prefetcher is None and self.activation_dtype is None:
    +            _qkv_decode_out_dtype = ttnn.bfloat8_b
    +
             xqkv_fused_sharded = ttnn.linear(
                 x_sharded,
                 self.wqkv,
                 memory_config=qkv_input_mem_config,
                 program_config=self.args.get_attn_qkv_program_config(Mode.DECODE, 1, self.prefetcher),
                 compute_kernel_config=self.li_qkv_decode_compute_kernel_cfg,
    -            dtype=self.ccl_dtype if self.TG else self.activation_dtype or ttnn.bfloat16,
    +            dtype=_qkv_decode_out_dtype,
                 global_cb=self.prefetcher.global_cb if self.prefetcher is not None else None,
                 sub_device_id=self.prefetcher.worker_sub_device_id if self.prefetcher is not None else None,
             )

[#90] MatmulDeviceOperation · shard · win  +1719.17 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/attention.py b/models/demos/llama3_1_8b_p150/tt/attention.py
    index 76b944f93e..aa99bba560 100644
    --- a/models/demos/llama3_1_8b_p150/tt/attention.py
    +++ b/models/demos/llama3_1_8b_p150/tt/attention.py
    @@ -605,6 +605,17 @@ class Attention(LightweightModule):
             if not self.TG and self.prefetcher is None and self.activation_dtype is None:
                 _qkv_decode_out_dtype = ttnn.bfloat8_b
     
    +        # SHARD RUNG (A/B against the bf8_b store above). Both operands of this matmul are
    +        # already L1 width-sharded and the 13 MB/layer bf4_b weight can never be L1-resident, so
    +        # the only sharding left to change is the HANDOFF: today the output is width-sharded in
    +        # L1 and then sharded_to_interleaved'd before the head split. Keeping it sharded deletes
    +        # that reshard from the per-token chain -- but nlp_create_qkv_heads_decode requires bf16,
    +        # and the reshard is currently what does the bf8_b -> bf16 upcast, so this trades the
    +        # store saving for the reshard. Which side wins is a measurement.
    +        _keep_qkv_sharded = not self.TG and self.prefetcher is None
    +        if _keep_qkv_sharded:
    +            _qkv_decode_out_dtype = ttnn.bfloat16
    +
             xqkv_fused_sharded = ttnn.linear(
                 x_sharded,
                 self.wqkv,
    @@ -649,7 +660,10 @@ class Attention(LightweightModule):
                 )
             else:
                 # bfloat16 is required by nlp_create_qkv_heads_decode
    -            if self.prefetcher is None:
    +            if _keep_qkv_sharded:
    +                # Already bf16 and L1 width-sharded -- hand it straight to the head split.
    +                xqkv_fused = xqkv_fused_sharded
    +            elif self.prefetcher is None:
                     xqkv_fused = ttnn.sharded_to_interleaved(xqkv_fused_sharded, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)
                     ttnn.deallocate(xqkv_fused_sharded)
                 else:

[#91] MatmulDeviceOperation · tp-fracture · no gain  +1719.17 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/attention.py b/models/demos/llama3_1_8b_p150/tt/attention.py
    index 76b944f93e..aa99bba560 100644
    --- a/models/demos/llama3_1_8b_p150/tt/attention.py
    +++ b/models/demos/llama3_1_8b_p150/tt/attention.py
    @@ -605,6 +605,17 @@ class Attention(LightweightModule):
             if not self.TG and self.prefetcher is None and self.activation_dtype is None:
                 _qkv_decode_out_dtype = ttnn.bfloat8_b
     
    +        # SHARD RUNG (A/B against the bf8_b store above). Both operands of this matmul are
    +        # already L1 width-sharded and the 13 MB/layer bf4_b weight can never be L1-resident, so
    +        # the only sharding left to change is the HANDOFF: today the output is width-sharded in
    +        # L1 and then sharded_to_interleaved'd before the head split. Keeping it sharded deletes
    +        # that reshard from the per-token chain -- but nlp_create_qkv_heads_decode requires bf16,
    +        # and the reshard is currently what does the bf8_b -> bf16 upcast, so this trades the
    +        # store saving for the reshard. Which side wins is a measurement.
    +        _keep_qkv_sharded = not self.TG and self.prefetcher is None
    +        if _keep_qkv_sharded:
    +            _qkv_decode_out_dtype = ttnn.bfloat16
    +
             xqkv_fused_sharded = ttnn.linear(
                 x_sharded,
                 self.wqkv,
    @@ -649,7 +660,10 @@ class Attention(LightweightModule):
                 )
             else:
                 # bfloat16 is required by nlp_create_qkv_heads_decode
    -            if self.prefetcher is None:
    +            if _keep_qkv_sharded:
    +                # Already bf16 and L1 width-sharded -- hand it straight to the head split.
    +                xqkv_fused = xqkv_fused_sharded
    +            elif self.prefetcher is None:
                     xqkv_fused = ttnn.sharded_to_interleaved(xqkv_fused_sharded, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)
                     ttnn.deallocate(xqkv_fused_sharded)
                 else:

[#95] MatmulDeviceOperation · tt-lang · no gain  +1719.17 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/ttl_ff2_matmul.py b/models/demos/llama3_1_8b_p150/tt/ttl_ff2_matmul.py
    index d126d0c5d2..cc46767225 100644
    --- a/models/demos/llama3_1_8b_p150/tt/ttl_ff2_matmul.py
    +++ b/models/demos/llama3_1_8b_p150/tt/ttl_ff2_matmul.py
    @@ -6,6 +6,7 @@ a sibling's result:
       * `MatmulDeviceOperation 128 x 14336 x 4096` -- short-prefill ff2 down-projection
       * `MatmulDeviceOperation  32 x 14336 x 4096` -- DECODE ff2 down-projection (per-token path)
       * `MatmulDeviceOperation  32 x  4096 x 14336` -- DECODE ff1/ff3 up-projection (per-token path)
    +  * `MatmulDeviceOperation  32 x  4096 x  6144` -- DECODE fused QKV projection (per-token path)
     
     Each core owns a strip of the N tiles; K is reduced in-core with an accumulator DFB ping-pong seeded
     from the first partial product (ttl 1.0.1 has no block.fill). The same kernel serves every shape.
    @@ -13,9 +14,10 @@ from the first partial product (ttl 1.0.1 has no block.fill). The same kernel se
     MEASURED on an 8x8 grid:
     
         M    K      N        PCC         ttl        ttnn.linear    verdict
    -    128  14336  4096     0.999691    2.974 ms    0.359 ms      8.3x SLOWER
    -     32  14336  4096     0.999692    0.816 ms    0.295 ms      2.8x SLOWER
    -     32   4096  14336    0.999929    0.687 ms    0.316 ms      2.2x SLOWER
    +    128  14336  4096     0.999706    2.991 ms    0.360 ms      8.3x SLOWER
    +     32  14336  4096     0.999689    0.813 ms    0.293 ms      2.8x SLOWER
    +     32   4096  14336    0.999918    0.680 ms    0.309 ms      2.2x SLOWER
    +     32   4096   6144    0.999913    0.324 ms    0.145 ms      2.2x SLOWER
     
     The C++ Metalium rung was measured on the same shapes via tt/cpp_mm_generic.py.
     
    @@ -46,7 +48,7 @@ import ttl
     TILE = 32
     GRID_X, GRID_Y = 8, 8
     # (M, K, N) triples this rung was measured for.
    -SHAPES = [(128, 14336, 4096), (32, 14336, 4096), (32, 4096, 14336)]
    +SHAPES = [(128, 14336, 4096), (32, 14336, 4096), (32, 4096, 14336), (32, 4096, 6144)]
     
     
     @ttl.operation(grid=(GRID_Y, GRID_X))

[#97] MatmulDeviceOperation · cpp · no gain  +1719.17 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/cpp_mm_generic.py b/models/demos/llama3_1_8b_p150/tt/cpp_mm_generic.py
    index 27cae9296a..e2c1fbd8cb 100644
    --- a/models/demos/llama3_1_8b_p150/tt/cpp_mm_generic.py
    +++ b/models/demos/llama3_1_8b_p150/tt/cpp_mm_generic.py
    @@ -6,6 +6,7 @@ Kept as the record of the cpp rung for every hot dense matmul in the MLP:
       * `MatmulDeviceOperation 128 x 14336 x  4096` -- the short-prefill ff2 down-projection
       * `MatmulDeviceOperation  32 x 14336 x  4096` -- the DECODE ff2 down-projection (per-token path)
       * `MatmulDeviceOperation  32 x  4096 x 14336` -- the DECODE ff1/ff3 up-projection (per-token path)
    +  * `MatmulDeviceOperation  32 x  4096 x  6144` -- the DECODE fused QKV projection (per-token path)
     
     Drives the repo's own programming-example kernel triple (tt_metal/programming_examples/matmul/
     matmul_multi_core: reader / mm / writer, copied into tt/kernels/) through ttnn.generic_op, with
    @@ -14,10 +15,11 @@ the output tiles partitioned across the entire compute grid.
     MEASURED on the real shapes, on the full 11x10 P150 grid:
     
         M    K      N        PCC        generic_op    ttnn.linear     verdict
    -    128  4096   14336    0.999039    3.561 ms      0.332 ms       10.7x SLOWER
    -    128  14336  4096     0.993591    3.119 ms      0.358 ms        8.7x SLOWER
    -     32  14336  4096     0.993562    0.919 ms      0.296 ms        3.1x SLOWER
    -     32  4096   14336    0.999022    1.039 ms      0.309 ms        3.4x SLOWER
    +    128  4096   14336    0.999015    3.562 ms      0.333 ms       10.7x SLOWER
    +    128  14336  4096     0.993626    3.113 ms      0.358 ms        8.7x SLOWER
    +     32  14336  4096     0.993630    0.916 ms      0.294 ms        3.1x SLOWER
    +     32  4096   14336    0.999003    1.041 ms      0.309 ms        3.4x SLOWER
    +     32  4096    6144    0.999014    0.344 ms      0.139 ms        2.5x SLOWER
     
     Every kernel is CORRECT and every one loses. The cause is dataflow, not tuning: this reader fetches
     every A tile again for each output tile, so A is re-read Nt times from DRAM, while ttnn's production
    @@ -46,7 +48,7 @@ from ttnn._ttnn.program_descriptor import VectorUInt32 as _VU32
     TILE = 32
     # (M, K, N) of every op this rung was measured for: the ff1/ff3 up-projection, then ff2's
     # down-projection at the short-prefill and the DECODE row counts.
    -SHAPES = [(128, 4096, 14336), (128, 14336, 4096), (32, 14336, 4096), (32, 4096, 14336)]
    +SHAPES = [(128, 4096, 14336), (128, 14336, 4096), (32, 14336, 4096), (32, 4096, 14336), (32, 4096, 6144)]
     ROOT = "/tmp/tt_hw_planner_llama3_1_8b_p150_1785111170/models/demos/llama3_1_8b_p150/tt/kernels"
     
     
    diff --git a/models/demos/llama3_1_8b_p150/tt/ttl_ff2_matmul.py b/models/demos/llama3_1_8b_p150/tt/ttl_ff2_matmul.py
    index d126d0c5d2..cc46767225 100644
    --- a/models/demos/llama3_1_8b_p150/tt/ttl_ff2_matmul.py
    ... (truncated, 32 more lines)

[#99] RotaryEmbeddingLlamaDeviceOperatio · grid · win  +1734.28 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/attention.py b/models/demos/llama3_1_8b_p150/tt/attention.py
    index aa99bba560..b6b453fdc9 100644
    --- a/models/demos/llama3_1_8b_p150/tt/attention.py
    +++ b/models/demos/llama3_1_8b_p150/tt/attention.py
    @@ -540,23 +540,46 @@ class Attention(LightweightModule):
             return q_heads_1BQD, k_heads_1BKD
     
         def _mllama_rope_prefill(self, q_heads_1QSD_pre_rot, k_heads_1KSD_pre_rot, rot_mats):
    -        q_heads_1QSD = ttnn.experimental.rotary_embedding_llama(
    -            q_heads_1QSD_pre_rot,
    -            rot_mats[0],
    -            rot_mats[1],
    -            self.transformation_mats["prefill"],
    -            is_decode_mode=False,
    -        )
    -
    -        k_heads_1KSD = ttnn.experimental.rotary_embedding_llama(
    -            k_heads_1KSD_pre_rot,
    -            rot_mats[0],
    -            rot_mats[1],
    -            self.transformation_mats["prefill"],
    -            is_decode_mode=False,
    -        )
    +        # HEADS-AS-BATCH. The prefill rope factory parallelises over batch x seq-tiles ONLY and
    +        # walks n_heads INSIDE each core (num_rows_per_core = sin_cos_rows_per_core * n_heads),
    +        # so at batch 1 it pins itself to min(cores, seq_len/32) cores -- four of the P150's 110
    +        # at seq_len 128, no matter how many heads there are. A [1, H, S, D] tiled tensor and a
    +        # [H, 1, S, D] one have identical physical tile ordering, so folding heads into the BATCH
    +        # dim is a metadata-only reshape that hands the factory H x min(cores/H, S_t) work units
    +        # instead of S_t. cos/sin are head-broadcast ([1, 1, S, D]), which the op still accepts
    +        # (it requires cos.shape[1] == input.shape[1] or 1, and both are 1 after the reshape).
    +        def _rope(x):
    +            heads = x.shape[1]
    +            as_batch = ttnn.reshape(x, [heads, 1, x.shape[2], x.shape[3]])
    +            out = ttnn.experimental.rotary_embedding_llama(
    +                as_batch,
    +                rot_mats[0],
    +                rot_mats[1],
    +                self.transformation_mats["prefill"],
    +                is_decode_mode=False,
    ... (truncated, 27 more lines)

[#100] NLPConcatHeadsDeviceOperation · grid · no gain  +1734.28 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/attention.py b/models/demos/llama3_1_8b_p150/tt/attention.py
    index b6b453fdc9..4b6a1a6418 100644
    --- a/models/demos/llama3_1_8b_p150/tt/attention.py
    +++ b/models/demos/llama3_1_8b_p150/tt/attention.py
    @@ -1217,6 +1217,42 @@ class Attention(LightweightModule):
             ###
             # Output matmul
             ###
    +        # GRID. nlp_concat_heads' INTERLEAVED factory derives its core count from
    +        # num_blocks = batch * seq_len / TILE_HEIGHT -- heads (dim 1) never enter the split, so at
    +        # batch 1 / seq_len 128 it runs on four cores while 32 heads of independent work sit idle.
    +        # Its SHARDED factory instead adopts the input's own shard grid
    +        # (num_cores = shard grid, blocks_per_core = shard_height / seq_len), so height-sharding
    +        # the SDPA output at ONE HEAD PER CORE hands it 32 cores. Validate wants full-width shards
    +        # (shard_w == head_dim), a shard height that is a multiple of seq_len, and a non-height-
    +        # sharded output, all of which this satisfies.
    +        _concat_cores = None
    +        if not self.TG and self.prefetcher is None and batch_size == 1:
    +            _rows = self.n_local_heads * seq_len
    +            _shard_h = seq_len  # one head per core
    +            _n_cores = _rows // _shard_h
    +            # RECTANGULAR core range only. num_cores_to_corerangeset would give a ragged set on an
    +            # 11-wide grid (two full rows plus a partial), and the sharded factory mixes
    +            # corerange_to_cores with grid_to_cores when assigning runtime args -- a non-rectangular
    +            # set makes those two orderings disagree and the output comes back scrambled
    +            # (measured: 25.6% top-1).
    +            _cols = next((c for c in range(self.args.max_grid_size.x, 0, -1) if _n_cores % c == 0), 0)
    +            _rows_of_cores = _n_cores // _cols if _cols else 0
    +            if _cols and _rows_of_cores <= self.args.max_grid_size.y:
    +                _concat_cores = ttnn.MemoryConfig(
    +                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    +                    ttnn.BufferType.L1,
    +                    ttnn.ShardSpec(
    +                        ttnn.CoreRangeSet(
    +                            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(_cols - 1, _rows_of_cores - 1))}
    +                        ),
    +                        [_shard_h, self.head_dim],
    +                        ttnn.ShardOrientation.ROW_MAJOR,
    +                    ),
    +                )
    ... (truncated, 7 more lines)

[#101] NLPConcatHeadsDeviceOperation · grid · no gain  +1734.28 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/attention.py b/models/demos/llama3_1_8b_p150/tt/attention.py
    index b6b453fdc9..d5d840d0e6 100644
    --- a/models/demos/llama3_1_8b_p150/tt/attention.py
    +++ b/models/demos/llama3_1_8b_p150/tt/attention.py
    @@ -1217,6 +1217,42 @@ class Attention(LightweightModule):
             ###
             # Output matmul
             ###
    +        # GRID. nlp_concat_heads' INTERLEAVED factory derives its core count from
    +        # num_blocks = batch * seq_len / TILE_HEIGHT -- heads (dim 1) never enter the split, so at
    +        # batch 1 / seq_len 128 it runs on four cores while 32 heads of independent work sit idle.
    +        # Its SHARDED factory instead adopts the input's own shard grid, with
    +        # blocks_per_core = shard_height / seq_len, so height-sharding the SDPA output by HEAD
    +        # widens it.
    +        #
    +        # TWO heads per core, not one, and a RECTANGULAR range. Both are load-bearing:
    +        #   * one head per core makes num_blocks_per_core = 1, and the writer splits that across
    +        #     two RISCs as div_up(1,2)=1 and 1-1=0 -- the zero-work RISC hangs the op.
    +        #   * a ragged core set (num_cores_to_corerangeset on an 11-wide grid) makes the factory's
    +        #     corerange_to_cores and grid_to_cores orderings disagree, scrambling the output
    +        #     (measured: 25.6% top-1).
    +        _concat_cores = None
    +        if not self.TG and self.prefetcher is None and batch_size == 1 and self.n_local_heads % 2 == 0:
    +            _heads_per_core = 2
    +            _shard_h = _heads_per_core * seq_len
    +            _n_cores = self.n_local_heads // _heads_per_core
    +            _cols = next((c for c in range(self.args.max_grid_size.x, 0, -1) if _n_cores % c == 0), 0)
    +            _core_rows = _n_cores // _cols if _cols else 0
    +            if _cols and 0 < _core_rows <= self.args.max_grid_size.y:
    +                _concat_cores = ttnn.MemoryConfig(
    +                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    +                    ttnn.BufferType.L1,
    +                    ttnn.ShardSpec(
    +                        ttnn.CoreRangeSet(
    +                            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(_cols - 1, _core_rows - 1))}
    +                        ),
    +                        [_shard_h, self.head_dim],
    +                        ttnn.ShardOrientation.ROW_MAJOR,
    +                    ),
    +                )
    ... (truncated, 7 more lines)

[#102] NLPConcatHeadsDeviceOperation · shard · no gain  +1734.19 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/attention.py b/models/demos/llama3_1_8b_p150/tt/attention.py
    index b6b453fdc9..f08dcfe873 100644
    --- a/models/demos/llama3_1_8b_p150/tt/attention.py
    +++ b/models/demos/llama3_1_8b_p150/tt/attention.py
    @@ -1217,9 +1217,21 @@ class Attention(LightweightModule):
             ###
             # Output matmul
             ###
    +        # SHARD. The INPUT side of this op is measured-unsafe -- feeding its sharded program
    +        # factory a head-sharded [1, H, S, D] either scrambles the output or hangs (three variants
    +        # tried at the grid rung). The OUTPUT side is free: the interleaved factory only requires
    +        # an INTERLEAVED output config, and L1 interleaved qualifies. The concat result is
    +        # [1, 1, S, H*D] -- 1 MB at seq_len 128 -- and its consumer is the wo matmul right below,
    +        # so keeping it in L1 removes a DRAM write plus the read back. Same lever that paid off on
    +        # the head split at the other end of attention. Bounded to short prefill.
    +        _concat_out_mem_config = (
    +            ttnn.L1_MEMORY_CONFIG
    +            if (not self.TG and self.prefetcher is None and seq_len <= self.args.prefill_len_cutoff)
    +            else ttnn.DRAM_MEMORY_CONFIG
    +        )
             attn_output_11SH = ttnn.experimental.nlp_concat_heads(
                 attn_output_1QSD,
    -            memory_config=ttnn.DRAM_MEMORY_CONFIG,
    +            memory_config=_concat_out_mem_config,
             )
             ttnn.deallocate(attn_output_1QSD)

[#107] NLPConcatHeadsDeviceOperation · tt-lang · win  +1749.24 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/attention.py b/models/demos/llama3_1_8b_p150/tt/attention.py
    index b6b453fdc9..29cd6c48a9 100644
    --- a/models/demos/llama3_1_8b_p150/tt/attention.py
    +++ b/models/demos/llama3_1_8b_p150/tt/attention.py
    @@ -7,7 +7,7 @@ import math
     import torch
     
     import ttnn
    -from models.demos.llama3_1_8b_p150.tt import ttl_create_qkv_heads
    +from models.demos.llama3_1_8b_p150.tt import ttl_concat_heads, ttl_create_qkv_heads
     from models.common.lightweightmodule import LightweightModule
     from models.common.rmsnorm import RMSNorm
     from models.common.utility_functions import nearest_32
    @@ -1217,10 +1217,22 @@ class Attention(LightweightModule):
             ###
             # Output matmul
             ###
    -        attn_output_11SH = ttnn.experimental.nlp_concat_heads(
    -            attn_output_1QSD,
    -            memory_config=ttnn.DRAM_MEMORY_CONFIG,
    -        )
    +        # tt-lang rung, and the mirror of the head-split kernel at the other end of attention.
    +        # The stock op's core count is baked into its factory (num_blocks = batch * seq_len / 32,
    +        # heads never enter the split -> 4 cores here), and its sharded factory is measured-unsafe
    +        # at this shape. ttl_concat_heads parallelises over (seq_tile x head) instead and measures
    +        # 0.0430 ms/call vs the stock op's 0.0572 at this shape, PCC 1.000000.
    +        if (
    +            not self.TG
    +            and self.prefetcher is None
    +            and ttl_concat_heads.supports(attn_output_1QSD, self.n_local_heads, self.head_dim)
    +        ):
    +            attn_output_11SH = ttl_concat_heads.concat_heads_ttl(attn_output_1QSD, ttnn.DRAM_MEMORY_CONFIG)
    +        else:
    +            attn_output_11SH = ttnn.experimental.nlp_concat_heads(
    +                attn_output_1QSD,
    +                memory_config=ttnn.DRAM_MEMORY_CONFIG,
    +            )
             ttnn.deallocate(attn_output_1QSD)
     
             # For batched prefill, reshape to concatenate batch dimension into sequence
    ... (truncated, 153 more lines)

[#110] ArgMaxDeviceOperation · grid · no gain  +1749.24 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/model_config.py b/models/demos/llama3_1_8b_p150/tt/model_config.py
    index 61d172b9df..9e0a02dd52 100644
    --- a/models/demos/llama3_1_8b_p150/tt/model_config.py
    +++ b/models/demos/llama3_1_8b_p150/tt/model_config.py
    @@ -1101,6 +1101,12 @@ class ModelArgs:
                     # offset-add / untilize / manual_seed / sampling tail behind them). Non-greedy
                     # requests still take the full path -- the sampler re-derives this per reset_params
                     # and re-captures its trace when the mode flips.
    +                #
    +                # NOTE on the argmax core grid: ModelArgs deliberately does NOT define
    +                # `sub_core_grids`. TTSampling reads it with getattr(args, "sub_core_grids", None),
    +                # and leaving it None is what lets ttnn.argmax fall through to
    +                # split_work_to_cores over the FULL compute grid. Defining it here would NARROW
    +                # the sampling ops, not widen them.
                     "allow_force_argmax": True,
                     "num_links": 1,
                     "chunks_per_sync": 10,

[#114] ArgMaxDeviceOperation · shard · win  +1751.02 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/model.py b/models/demos/llama3_1_8b_p150/tt/model.py
    index e19e919f4c..0d4df8171b 100644
    --- a/models/demos/llama3_1_8b_p150/tt/model.py
    +++ b/models/demos/llama3_1_8b_p150/tt/model.py
    @@ -3,6 +3,8 @@
     # SPDX-License-Identifier: Apache-2.0
     
     
    +import math
    +
     import torch
     from tqdm import tqdm
     
    @@ -261,7 +263,20 @@ class Transformer(LightweightModule):
             if lm_head_input_mem_cfg.is_sharded():
                 x = ttnn.interleaved_to_sharded(x, lm_head_input_mem_cfg)
             logits = self.lm_head(x)
    -        logits = ttnn.to_memory_config(logits, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    +        # Keep the prefill logits where lm_head already put them (L1 interleaved) instead of
    +        # spilling them to DRAM. The consumer is on-device sampling, whose force-argmax path is
    +        # `untilize -> ttnn.argmax`, and BOTH of those read this tensor straight through: the
    +        # profile shows untilize at 47.8 us/call from DRAM vs 31.4 us from L1, and argmax at
    +        # 737 us vs 715 us, on top of the 4.4 MB L1->DRAM copy this line itself pays. argmax
    +        # FATALs on any non-INTERLEAVED input, so L1-interleaved is the only shard the op will
    +        # accept -- a width/height shard is architecturally closed here.
    +        # [1, 1, 32, vocab] at bf8_b is ~4.4 MB, comfortably inside the interleaved-L1 budget;
    +        # fall back to DRAM if a wider vocab or a batched-prefill variant makes it big.
    +        logits_elems = math.prod(int(d) for d in logits.shape)
    +        if logits_elems * 2 > 16 * 1024 * 1024:  # bf16-equivalent bytes; bf8_b is ~half this
    +            logits = ttnn.to_memory_config(logits, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    +        elif logits.memory_config() != ttnn.L1_MEMORY_CONFIG:
    +            logits = ttnn.to_memory_config(logits, memory_config=ttnn.L1_MEMORY_CONFIG)
             return logits
     
         def process_hidden_states_after_prefill_trace(self, hidden_states, last_token_idx):

[#117] ArgMaxDeviceOperation · structural · win  +1781.71 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/model.py b/models/demos/llama3_1_8b_p150/tt/model.py
    index 0d4df8171b..b2d1ee38fc 100644
    --- a/models/demos/llama3_1_8b_p150/tt/model.py
    +++ b/models/demos/llama3_1_8b_p150/tt/model.py
    @@ -19,6 +19,7 @@ from models.demos.llama3_1_8b_p150.tt.distributed_norm import DistributedNorm
     from models.demos.llama3_1_8b_p150.tt.embedding import Embedding, ScaledEmbedding
     from models.demos.llama3_1_8b_p150.tt.lm_head import LMHead
     from models.demos.llama3_1_8b_p150.tt.model_config import TensorGroup
    +from models.demos.llama3_1_8b_p150.tt import sampling_unpadded_argmax as unpadded_argmax
     from models.demos.llama3_1_8b_p150.tt.rope import HfRotarySetup, RotarySetup
     
     
    @@ -163,6 +164,10 @@ class Transformer(LightweightModule):
                     mesh_device=mesh_device,
                     tt_ccl=self.tt_ccl,
                 )
    +            # TTSampling rounds the request batch up to a 32-row tile because the LOGITS are
    +            # tile-layout. Its force-argmax path untilizes first, though, and ROW_MAJOR has no
    +            # such constraint -- so the argmax reduction can skip the padding rows entirely.
    +            unpadded_argmax.install(self.sampling, args)
             else:
                 self.sampling = None
     
    diff --git a/models/demos/llama3_1_8b_p150/tt/sampling_unpadded_argmax.py b/models/demos/llama3_1_8b_p150/tt/sampling_unpadded_argmax.py
    new file mode 100644
    index 0000000000..4d628251cd
    --- /dev/null
    +++ b/models/demos/llama3_1_8b_p150/tt/sampling_unpadded_argmax.py
    @@ -0,0 +1,118 @@
    +"""Drop the tile-padding rows before the force-argmax sampling reduction.
    +
    +THE REDUCIBLE WORK. `TTSampling.__init__` rounds the request batch up to a tile:
    +
    +    raw_batch = getattr(args, "max_batch_size", 32)
    +    self.max_batch_size = max(32, ((raw_batch + 31) // 32) * 32)
    +
    +That round-up is REQUIRED for the logits themselves -- they are a TILE-layout
    +`[1, 1, 32, vocab]` tensor and a tile is 32 rows tall, so the norm / LM-head chain has to
    +compute all 32. It is NOT required for the argmax, because the force-argmax path untilizes
    +first and a ROW_MAJOR tensor has no such constraint. The profile shows the op paying for the
    ... (truncated, 107 more lines)

[#119] MatmulDeviceOperation · dtype · win  +1801.26 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/lm_head.py b/models/demos/llama3_1_8b_p150/tt/lm_head.py
    index eeb42ce095..7f72b1c0b6 100644
    --- a/models/demos/llama3_1_8b_p150/tt/lm_head.py
    +++ b/models/demos/llama3_1_8b_p150/tt/lm_head.py
    @@ -101,13 +101,17 @@ class LMHead(LightweightModule):
     
                     # The cache key must encode the memory config: a cached DRAM-width-sharded weight
                     # reloaded for the interleaved path (or vice versa) silently feeds the matmul an
    -                # operand whose shard spec its program config does not expect.
    +                # operand whose shard spec its program config does not expect. It must encode the
    +                # DTYPE for the same reason -- `as_tensor` returns the cached tensor as it was
    +                # stored, so without this a bf8_b cache is reloaded unchanged and a dtype change
    +                # here is a silent no-op.
                     _layout_tag = "_ilv" if (mode == 0 and self.full_grid) else ""
    +                _dtype_tag = f"_{str(dtype).rsplit('.', 1)[-1].lower()}"
                     cache_file_name = (
                         None
                         if args.dummy_weights
                         else weight_cache_path
    -                    / f"output_lm_head_{len(split_sizes)}_split_shard_{i}_{combined_split.shape[-1]}_mode_{mode}{_layout_tag}"
    +                    / f"output_lm_head_{len(split_sizes)}_split_shard_{i}_{combined_split.shape[-1]}_mode_{mode}{_layout_tag}{_dtype_tag}"
                     )
     
                     def pad_to_power_of_2(n):
    diff --git a/models/demos/llama3_1_8b_p150/tt/model.py b/models/demos/llama3_1_8b_p150/tt/model.py
    index b2d1ee38fc..3f5dfc99ff 100644
    --- a/models/demos/llama3_1_8b_p150/tt/model.py
    +++ b/models/demos/llama3_1_8b_p150/tt/model.py
    @@ -142,11 +142,18 @@ class Transformer(LightweightModule):
                 TG=args.is_galaxy,
             )
     
    +        # The output projection is the single largest weight in the model
    +        # ([dim, padded_vocab] = 4096 x 128256, ~70 MB at bf8_b) and its matmul is
    +        # DRAM-bandwidth bound, so the bytes it reads ARE its runtime. Every decoder weight
    +        # already rides bf4_b under `performance()`; the LM head was left on the model-wide
    +        # bf8_b default purely because it is built outside the decoder precision config.
    +        # Overridable via ModelArgs for accuracy-first configs.
    +        lm_head_weight_dtype = getattr(args, "lm_head_weight_dtype", None) or ttnn.bfloat4_b
             self.lm_head = LMHead(
    ... (truncated, 8 more lines)

[#121] MatmulDeviceOperation · shard · no gain  +1801.26 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/lm_head.py b/models/demos/llama3_1_8b_p150/tt/lm_head.py
    index eeb42ce095..7f72b1c0b6 100644
    --- a/models/demos/llama3_1_8b_p150/tt/lm_head.py
    +++ b/models/demos/llama3_1_8b_p150/tt/lm_head.py
    @@ -101,13 +101,17 @@ class LMHead(LightweightModule):
     
                     # The cache key must encode the memory config: a cached DRAM-width-sharded weight
                     # reloaded for the interleaved path (or vice versa) silently feeds the matmul an
    -                # operand whose shard spec its program config does not expect.
    +                # operand whose shard spec its program config does not expect. It must encode the
    +                # DTYPE for the same reason -- `as_tensor` returns the cached tensor as it was
    +                # stored, so without this a bf8_b cache is reloaded unchanged and a dtype change
    +                # here is a silent no-op.
                     _layout_tag = "_ilv" if (mode == 0 and self.full_grid) else ""
    +                _dtype_tag = f"_{str(dtype).rsplit('.', 1)[-1].lower()}"
                     cache_file_name = (
                         None
                         if args.dummy_weights
                         else weight_cache_path
    -                    / f"output_lm_head_{len(split_sizes)}_split_shard_{i}_{combined_split.shape[-1]}_mode_{mode}{_layout_tag}"
    +                    / f"output_lm_head_{len(split_sizes)}_split_shard_{i}_{combined_split.shape[-1]}_mode_{mode}{_layout_tag}{_dtype_tag}"
                     )
     
                     def pad_to_power_of_2(n):
    diff --git a/models/demos/llama3_1_8b_p150/tt/model.py b/models/demos/llama3_1_8b_p150/tt/model.py
    index b2d1ee38fc..3f5dfc99ff 100644
    --- a/models/demos/llama3_1_8b_p150/tt/model.py
    +++ b/models/demos/llama3_1_8b_p150/tt/model.py
    @@ -142,11 +142,18 @@ class Transformer(LightweightModule):
                 TG=args.is_galaxy,
             )
     
    +        # The output projection is the single largest weight in the model
    +        # ([dim, padded_vocab] = 4096 x 128256, ~70 MB at bf8_b) and its matmul is
    +        # DRAM-bandwidth bound, so the bytes it reads ARE its runtime. Every decoder weight
    +        # already rides bf4_b under `performance()`; the LM head was left on the model-wide
    +        # bf8_b default purely because it is built outside the decoder precision config.
    +        # Overridable via ModelArgs for accuracy-first configs.
    +        lm_head_weight_dtype = getattr(args, "lm_head_weight_dtype", None) or ttnn.bfloat4_b
             self.lm_head = LMHead(
    ... (truncated, 8 more lines)

[#122] MatmulDeviceOperation · shard · no gain  +1801.26 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/lm_head.py b/models/demos/llama3_1_8b_p150/tt/lm_head.py
    index eeb42ce095..7f72b1c0b6 100644
    --- a/models/demos/llama3_1_8b_p150/tt/lm_head.py
    +++ b/models/demos/llama3_1_8b_p150/tt/lm_head.py
    @@ -101,13 +101,17 @@ class LMHead(LightweightModule):
     
                     # The cache key must encode the memory config: a cached DRAM-width-sharded weight
                     # reloaded for the interleaved path (or vice versa) silently feeds the matmul an
    -                # operand whose shard spec its program config does not expect.
    +                # operand whose shard spec its program config does not expect. It must encode the
    +                # DTYPE for the same reason -- `as_tensor` returns the cached tensor as it was
    +                # stored, so without this a bf8_b cache is reloaded unchanged and a dtype change
    +                # here is a silent no-op.
                     _layout_tag = "_ilv" if (mode == 0 and self.full_grid) else ""
    +                _dtype_tag = f"_{str(dtype).rsplit('.', 1)[-1].lower()}"
                     cache_file_name = (
                         None
                         if args.dummy_weights
                         else weight_cache_path
    -                    / f"output_lm_head_{len(split_sizes)}_split_shard_{i}_{combined_split.shape[-1]}_mode_{mode}{_layout_tag}"
    +                    / f"output_lm_head_{len(split_sizes)}_split_shard_{i}_{combined_split.shape[-1]}_mode_{mode}{_layout_tag}{_dtype_tag}"
                     )
     
                     def pad_to_power_of_2(n):
    diff --git a/models/demos/llama3_1_8b_p150/tt/model.py b/models/demos/llama3_1_8b_p150/tt/model.py
    index b2d1ee38fc..3f5dfc99ff 100644
    --- a/models/demos/llama3_1_8b_p150/tt/model.py
    +++ b/models/demos/llama3_1_8b_p150/tt/model.py
    @@ -142,11 +142,18 @@ class Transformer(LightweightModule):
                 TG=args.is_galaxy,
             )
     
    +        # The output projection is the single largest weight in the model
    +        # ([dim, padded_vocab] = 4096 x 128256, ~70 MB at bf8_b) and its matmul is
    +        # DRAM-bandwidth bound, so the bytes it reads ARE its runtime. Every decoder weight
    +        # already rides bf4_b under `performance()`; the LM head was left on the model-wide
    +        # bf8_b default purely because it is built outside the decoder precision config.
    +        # Overridable via ModelArgs for accuracy-first configs.
    +        lm_head_weight_dtype = getattr(args, "lm_head_weight_dtype", None) or ttnn.bfloat4_b
             self.lm_head = LMHead(
    ... (truncated, 8 more lines)

[#123] MatmulDeviceOperation · tp-fracture · no gain  +1801.26 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/lm_head.py b/models/demos/llama3_1_8b_p150/tt/lm_head.py
    index eeb42ce095..7f72b1c0b6 100644
    --- a/models/demos/llama3_1_8b_p150/tt/lm_head.py
    +++ b/models/demos/llama3_1_8b_p150/tt/lm_head.py
    @@ -101,13 +101,17 @@ class LMHead(LightweightModule):
     
                     # The cache key must encode the memory config: a cached DRAM-width-sharded weight
                     # reloaded for the interleaved path (or vice versa) silently feeds the matmul an
    -                # operand whose shard spec its program config does not expect.
    +                # operand whose shard spec its program config does not expect. It must encode the
    +                # DTYPE for the same reason -- `as_tensor` returns the cached tensor as it was
    +                # stored, so without this a bf8_b cache is reloaded unchanged and a dtype change
    +                # here is a silent no-op.
                     _layout_tag = "_ilv" if (mode == 0 and self.full_grid) else ""
    +                _dtype_tag = f"_{str(dtype).rsplit('.', 1)[-1].lower()}"
                     cache_file_name = (
                         None
                         if args.dummy_weights
                         else weight_cache_path
    -                    / f"output_lm_head_{len(split_sizes)}_split_shard_{i}_{combined_split.shape[-1]}_mode_{mode}{_layout_tag}"
    +                    / f"output_lm_head_{len(split_sizes)}_split_shard_{i}_{combined_split.shape[-1]}_mode_{mode}{_layout_tag}{_dtype_tag}"
                     )
     
                     def pad_to_power_of_2(n):
    diff --git a/models/demos/llama3_1_8b_p150/tt/model.py b/models/demos/llama3_1_8b_p150/tt/model.py
    index b2d1ee38fc..3f5dfc99ff 100644
    --- a/models/demos/llama3_1_8b_p150/tt/model.py
    +++ b/models/demos/llama3_1_8b_p150/tt/model.py
    @@ -142,11 +142,18 @@ class Transformer(LightweightModule):
                 TG=args.is_galaxy,
             )
     
    +        # The output projection is the single largest weight in the model
    +        # ([dim, padded_vocab] = 4096 x 128256, ~70 MB at bf8_b) and its matmul is
    +        # DRAM-bandwidth bound, so the bytes it reads ARE its runtime. Every decoder weight
    +        # already rides bf4_b under `performance()`; the LM head was left on the model-wide
    +        # bf8_b default purely because it is built outside the decoder precision config.
    +        # Overridable via ModelArgs for accuracy-first configs.
    +        lm_head_weight_dtype = getattr(args, "lm_head_weight_dtype", None) or ttnn.bfloat4_b
             self.lm_head = LMHead(
    ... (truncated, 8 more lines)

[#133] SDPAOperation · grid · win  +1803.18 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/model_config.py b/models/demos/llama3_1_8b_p150/tt/model_config.py
    index 9e0a02dd52..0f4453f745 100644
    --- a/models/demos/llama3_1_8b_p150/tt/model_config.py
    +++ b/models/demos/llama3_1_8b_p150/tt/model_config.py
    @@ -1598,8 +1598,51 @@ class ModelArgs:
                 if seq_len >= 2048
                 else min(64, chunk_start_idx & -chunk_start_idx)
             )
    +        # Occupancy. The prefill SDPA factory flattens the work into
    +        #     total_q_chunks = B * n_local_heads * ceil(seq_len / q_chunk)
    +        # and, for causal attention with an even chunk count, hands out PAIRS (one light +
    +        # one heavy chunk per core) so the triangular load balances. The number of cores it
    +        # can keep busy is therefore capped by that chunk/pair count, NOT by the grid it is
    +        # handed: at seq_len=128 the 64-token default gives 2 chunks/head, so 32 heads
    +        # produce 64 chunks = 32 pairs = 32 busy cores however wide the grid is. That is why
    +        # the profiler tags this op grid=partial, and no program_config grid alone can fix it.
    +        #
    +        # Two coupled corrections, and they must move together:
    +        #   1. Halve q_chunk (never below one tile) until the busy-core cap covers the grid --
    +        #      this is what creates work for the idle half of the chip.
    +        #   2. Size the grid to that cap instead of a hard-coded Wormhole 8x8. Surplus cores
    +        #      are not free (they are still launched and still join the op's barriers), and
    +        #      long prefill is the opposite case -- there the 8x8 was leaving 46 of the P150's
    +        #      110 cores unused.
    +        grid = getattr(self, "max_grid_size", None)
    +        if grid is None:  # no device (e.g. test_torch.py) -- keep the historical 8x8
    +            grid = ttnn.CoreGrid(y=8, x=8)
    +        sdpa_grid = (min(8, grid.x), min(8, grid.y))
    +        if chunk_start_idx is None or chunk_start_idx == 0:
    +            heads = max(1, getattr(self, "n_local_heads", self.n_heads))
    +            max_cores = grid.x * grid.y
    +
    +            def _busy_cap(qc):
    +                q_num_chunks = max(1, -(-seq_len // qc))
    +                total = heads * q_num_chunks
    +                return max(1, total // 2 if q_num_chunks % 2 == 0 else total)
    +
    +            while (
    +                q_chunk > ttnn.TILE_SIZE
    +                and q_chunk % (2 * ttnn.TILE_SIZE) == 0
    ... (truncated, 17 more lines)

[#135] RotaryEmbeddingLlamaDeviceOperatio · structural · win  +1807.27 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/attention.py b/models/demos/llama3_1_8b_p150/tt/attention.py
    index 29cd6c48a9..99f360cb07 100644
    --- a/models/demos/llama3_1_8b_p150/tt/attention.py
    +++ b/models/demos/llama3_1_8b_p150/tt/attention.py
    @@ -317,6 +317,13 @@ class Attention(LightweightModule):
             else:
                 self.k_norm = lambda x, mode, norm_config: x
     
    +        # Whether q_norm/k_norm are identity. The fused split+rope kernel does the rotation inside
    +        # the head split, i.e. BEFORE where these norms sit in the chain, so it is only a legal
    +        # substitution when there are no norms to reorder past (llama has none; qwen3 does).
    +        self.qk_norm_is_identity = (
    +            f"{q_norm_str}.weight" not in state_dict and f"{k_norm_str}.weight" not in state_dict
    +        )
    +
             # For ring topology we can use all gather matmul for wo
             self.use_fused_all_gather_matmul = self.args.use_fused_all_gather_matmul
             pt_wo = state_dict[f"{wo_str}.weight"].transpose(-1, -2).unsqueeze(0).unsqueeze(0)
    @@ -1037,39 +1044,79 @@ class Attention(LightweightModule):
                     xqkv_fused, self.n_local_heads, self.n_local_kv_heads, self.head_dim
                 )
             )
    -        if _use_ttl_heads:
    -            (
    -                q_heads_1QSD_pre_rot,
    -                k_heads_1KSD_pre_rot,
    -                v_heads_1VSD,
    -            ) = ttl_create_qkv_heads.create_qkv_heads_ttl(xqkv_fused, create_heads_mem_config)
    -        else:
    +        # structural rung for RotaryEmbeddingLlamaDeviceOperation: the prefill rope is DISPATCH
    +        # bound (736 launches x 9.55 us against a 1.21 us roofline -- a [32,1,128,128] rotation is
    +        # only ~8 tiles per core, so almost all of the 9.55 us is fixed launch cost). No knob
    +        # removes a fixed cost, so remove the LAUNCHES: the rotation is tile-local (x*cos +
    +        # (x @ trans_mat)*sin, one 32x32 trans tile per tile, cos/sin head-broadcast), and the
    +        # head-split kernel already streams every Q/K tile through its compute stage -- so the
    +        # rotate rides along in the slot the split already pays for and the two rope dispatches
    +        # disappear. Only legal while q_norm/k_norm are identity (see qk_norm_is_identity).
    +        _fused_rope = (
    +            _use_ttl_heads
    +            and self.qk_norm_is_identity
    ... (truncated, 335 more lines)

[#137] GenericOpDeviceOperation · grid · win  +1810.49 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/ttl_concat_heads.py b/models/demos/llama3_1_8b_p150/tt/ttl_concat_heads.py
    index 31e6d95100..81c0fcecb2 100644
    --- a/models/demos/llama3_1_8b_p150/tt/ttl_concat_heads.py
    +++ b/models/demos/llama3_1_8b_p150/tt/ttl_concat_heads.py
    @@ -31,12 +31,21 @@ import ttnn
     import ttl
     
     TILE = 32
    -# seq_len 128 -> 4 seq tiles, one per grid row; 32 heads / 8 grid columns -> 4 heads per column.
    -GRID_X, GRID_Y = 8, 4
    -N_HEADS, HEAD_DIM = 32, 128
    +# GRID RUNG. The work has THREE dimensions -- head (32) x seq tile (4) x dim tile (4) -- but
    +# `ttl.node` is 2-D, so a division-free mapping can carry only two of them and every division-free
    +# choice lands on 32 of the P150's 110 cores (head groups must divide 32 and fit in <=11 columns,
    +# i.e. 8; the other axis is then seq(4) or dim(4)). Getting past 32 therefore needs ONE coordinate
    +# to carry two work dimensions, which needs a constant divide on the node coord. So: y carries
    +# (seq tile, dim half) as cy = dim_half * SEQ_T + s, giving 8 rows and 64 cores at 8 tiles each.
    +# If the compiler will not lower `//`/`%` on a node coord this falls back to the 4-row form.
     SEQ_LEN = 128
    -
    +N_HEADS, HEAD_DIM = 32, 128
     HD_T = HEAD_DIM // TILE
    +SEQ_T = SEQ_LEN // TILE
    +DIM_HALVES = 2
    +HALF_T = HD_T // DIM_HALVES
    +
    +GRID_X, GRID_Y = 8, SEQ_T * DIM_HALVES
     HEADS_PER_COL = N_HEADS // GRID_X
     
     
    @@ -51,15 +60,17 @@ def ttl_concat_heads(x: ttnn.Tensor, y: ttnn.Tensor) -> None:
         in_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
         out_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)
     
    -    per_core_tiles = HEADS_PER_COL * HD_T
    +    per_core_tiles = HEADS_PER_COL * HALF_T
     
         @ttl.datamovement()
         def read():
             cx, cy = ttl.node(dims=2)
    ... (truncated, 182 more lines)

[#138] SDPAOperation · structural · no gain  +1808.45 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/ttl_concat_heads.py b/models/demos/llama3_1_8b_p150/tt/ttl_concat_heads.py
    index 31e6d95100..81c0fcecb2 100644
    --- a/models/demos/llama3_1_8b_p150/tt/ttl_concat_heads.py
    +++ b/models/demos/llama3_1_8b_p150/tt/ttl_concat_heads.py
    @@ -31,12 +31,21 @@ import ttnn
     import ttl
     
     TILE = 32
    -# seq_len 128 -> 4 seq tiles, one per grid row; 32 heads / 8 grid columns -> 4 heads per column.
    -GRID_X, GRID_Y = 8, 4
    -N_HEADS, HEAD_DIM = 32, 128
    +# GRID RUNG. The work has THREE dimensions -- head (32) x seq tile (4) x dim tile (4) -- but
    +# `ttl.node` is 2-D, so a division-free mapping can carry only two of them and every division-free
    +# choice lands on 32 of the P150's 110 cores (head groups must divide 32 and fit in <=11 columns,
    +# i.e. 8; the other axis is then seq(4) or dim(4)). Getting past 32 therefore needs ONE coordinate
    +# to carry two work dimensions, which needs a constant divide on the node coord. So: y carries
    +# (seq tile, dim half) as cy = dim_half * SEQ_T + s, giving 8 rows and 64 cores at 8 tiles each.
    +# If the compiler will not lower `//`/`%` on a node coord this falls back to the 4-row form.
     SEQ_LEN = 128
    -
    +N_HEADS, HEAD_DIM = 32, 128
     HD_T = HEAD_DIM // TILE
    +SEQ_T = SEQ_LEN // TILE
    +DIM_HALVES = 2
    +HALF_T = HD_T // DIM_HALVES
    +
    +GRID_X, GRID_Y = 8, SEQ_T * DIM_HALVES
     HEADS_PER_COL = N_HEADS // GRID_X
     
     
    @@ -51,15 +60,17 @@ def ttl_concat_heads(x: ttnn.Tensor, y: ttnn.Tensor) -> None:
         in_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
         out_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)
     
    -    per_core_tiles = HEADS_PER_COL * HD_T
    +    per_core_tiles = HEADS_PER_COL * HALF_T
     
         @ttl.datamovement()
         def read():
             cx, cy = ttl.node(dims=2)
    ... (truncated, 182 more lines)

[#139] BinaryNgDeviceOperation · grid · no gain  +1803.58 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/ttl_concat_heads.py b/models/demos/llama3_1_8b_p150/tt/ttl_concat_heads.py
    index 31e6d95100..81c0fcecb2 100644
    --- a/models/demos/llama3_1_8b_p150/tt/ttl_concat_heads.py
    +++ b/models/demos/llama3_1_8b_p150/tt/ttl_concat_heads.py
    @@ -31,12 +31,21 @@ import ttnn
     import ttl
     
     TILE = 32
    -# seq_len 128 -> 4 seq tiles, one per grid row; 32 heads / 8 grid columns -> 4 heads per column.
    -GRID_X, GRID_Y = 8, 4
    -N_HEADS, HEAD_DIM = 32, 128
    +# GRID RUNG. The work has THREE dimensions -- head (32) x seq tile (4) x dim tile (4) -- but
    +# `ttl.node` is 2-D, so a division-free mapping can carry only two of them and every division-free
    +# choice lands on 32 of the P150's 110 cores (head groups must divide 32 and fit in <=11 columns,
    +# i.e. 8; the other axis is then seq(4) or dim(4)). Getting past 32 therefore needs ONE coordinate
    +# to carry two work dimensions, which needs a constant divide on the node coord. So: y carries
    +# (seq tile, dim half) as cy = dim_half * SEQ_T + s, giving 8 rows and 64 cores at 8 tiles each.
    +# If the compiler will not lower `//`/`%` on a node coord this falls back to the 4-row form.
     SEQ_LEN = 128
    -
    +N_HEADS, HEAD_DIM = 32, 128
     HD_T = HEAD_DIM // TILE
    +SEQ_T = SEQ_LEN // TILE
    +DIM_HALVES = 2
    +HALF_T = HD_T // DIM_HALVES
    +
    +GRID_X, GRID_Y = 8, SEQ_T * DIM_HALVES
     HEADS_PER_COL = N_HEADS // GRID_X
     
     
    @@ -51,15 +60,17 @@ def ttl_concat_heads(x: ttnn.Tensor, y: ttnn.Tensor) -> None:
         in_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
         out_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)
     
    -    per_core_tiles = HEADS_PER_COL * HD_T
    +    per_core_tiles = HEADS_PER_COL * HALF_T
     
         @ttl.datamovement()
         def read():
             cx, cy = ttl.node(dims=2)
    ... (truncated, 182 more lines)

[#140] BinaryNgDeviceOperation · grid · no gain  +1792.26 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/ttl_concat_heads.py b/models/demos/llama3_1_8b_p150/tt/ttl_concat_heads.py
    index 31e6d95100..81c0fcecb2 100644
    --- a/models/demos/llama3_1_8b_p150/tt/ttl_concat_heads.py
    +++ b/models/demos/llama3_1_8b_p150/tt/ttl_concat_heads.py
    @@ -31,12 +31,21 @@ import ttnn
     import ttl
     
     TILE = 32
    -# seq_len 128 -> 4 seq tiles, one per grid row; 32 heads / 8 grid columns -> 4 heads per column.
    -GRID_X, GRID_Y = 8, 4
    -N_HEADS, HEAD_DIM = 32, 128
    +# GRID RUNG. The work has THREE dimensions -- head (32) x seq tile (4) x dim tile (4) -- but
    +# `ttl.node` is 2-D, so a division-free mapping can carry only two of them and every division-free
    +# choice lands on 32 of the P150's 110 cores (head groups must divide 32 and fit in <=11 columns,
    +# i.e. 8; the other axis is then seq(4) or dim(4)). Getting past 32 therefore needs ONE coordinate
    +# to carry two work dimensions, which needs a constant divide on the node coord. So: y carries
    +# (seq tile, dim half) as cy = dim_half * SEQ_T + s, giving 8 rows and 64 cores at 8 tiles each.
    +# If the compiler will not lower `//`/`%` on a node coord this falls back to the 4-row form.
     SEQ_LEN = 128
    -
    +N_HEADS, HEAD_DIM = 32, 128
     HD_T = HEAD_DIM // TILE
    +SEQ_T = SEQ_LEN // TILE
    +DIM_HALVES = 2
    +HALF_T = HD_T // DIM_HALVES
    +
    +GRID_X, GRID_Y = 8, SEQ_T * DIM_HALVES
     HEADS_PER_COL = N_HEADS // GRID_X
     
     
    @@ -51,15 +60,17 @@ def ttl_concat_heads(x: ttnn.Tensor, y: ttnn.Tensor) -> None:
         in_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
         out_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)
     
    -    per_core_tiles = HEADS_PER_COL * HD_T
    +    per_core_tiles = HEADS_PER_COL * HALF_T
     
         @ttl.datamovement()
         def read():
             cx, cy = ttl.node(dims=2)
    ... (truncated, 182 more lines)

[#142] BinaryNgDeviceOperation · shard · win  +1816.01 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/attention.py b/models/demos/llama3_1_8b_p150/tt/attention.py
    index 99f360cb07..dc088f0e05 100644
    --- a/models/demos/llama3_1_8b_p150/tt/attention.py
    +++ b/models/demos/llama3_1_8b_p150/tt/attention.py
    @@ -1315,12 +1315,18 @@ class Attention(LightweightModule):
                 else attn_output_11SH
             )
     
    +        # wo writes straight into the residual stream's memory space. At short prefill that stream
    +        # lives in L1, and landing this output in DRAM instead would make the residual add copy it
    +        # back in -- which measured as 1149 CopyDeviceOperation calls / +4.20 ms, more than the
    +        # 3.03 ms the L1 residual saved. The residual config is the single source of truth.
    +        wo_out_mem_config = self.args.get_residual_mem_config(Mode.PREFILL, self.prefetcher, int(seq_len))
    +
             output_11SH = ttnn.linear(
                 attn_output_11SH_sharded,
                 self.wo,
                 compute_kernel_config=self.li_o_prefill_compute_kernel_cfg,
                 dtype=self.activation_dtype or ttnn.bfloat8_b,
    -            memory_config=wo_prefill_output_mem_config,
    +            memory_config=wo_out_mem_config,
                 program_config=self.args.get_attn_wo_program_config(Mode.PREFILL, seq_len, None),
             )
     
    diff --git a/models/demos/llama3_1_8b_p150/tt/decoder.py b/models/demos/llama3_1_8b_p150/tt/decoder.py
    index 067ea19c0a..84b99613b9 100644
    --- a/models/demos/llama3_1_8b_p150/tt/decoder.py
    +++ b/models/demos/llama3_1_8b_p150/tt/decoder.py
    @@ -234,7 +234,7 @@ class TransformerBlock(LightweightModule):
             residual = x
     
             # x is fractured across devices and interleaved in DRAM (for prefill) and sharded in L1 (for decode)
    -        skip_mem_cfg = self.args.get_residual_mem_config(mode, self.prefetcher)
    +        skip_mem_cfg = self.args.get_residual_mem_config(mode, self.prefetcher, int(x.shape[-2]))
     
             assert (
                 x.memory_config() == skip_mem_cfg
    diff --git a/models/demos/llama3_1_8b_p150/tt/distributed_norm.py b/models/demos/llama3_1_8b_p150/tt/distributed_norm.py
    index 81444f48f0..7cfaf32cf7 100644
    --- a/models/demos/llama3_1_8b_p150/tt/distributed_norm.py
    ... (truncated, 89 more lines)

[#143] BinaryNgDeviceOperation · structural · no gain  +1816.01 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/attention.py b/models/demos/llama3_1_8b_p150/tt/attention.py
    index 99f360cb07..dc088f0e05 100644
    --- a/models/demos/llama3_1_8b_p150/tt/attention.py
    +++ b/models/demos/llama3_1_8b_p150/tt/attention.py
    @@ -1315,12 +1315,18 @@ class Attention(LightweightModule):
                 else attn_output_11SH
             )
     
    +        # wo writes straight into the residual stream's memory space. At short prefill that stream
    +        # lives in L1, and landing this output in DRAM instead would make the residual add copy it
    +        # back in -- which measured as 1149 CopyDeviceOperation calls / +4.20 ms, more than the
    +        # 3.03 ms the L1 residual saved. The residual config is the single source of truth.
    +        wo_out_mem_config = self.args.get_residual_mem_config(Mode.PREFILL, self.prefetcher, int(seq_len))
    +
             output_11SH = ttnn.linear(
                 attn_output_11SH_sharded,
                 self.wo,
                 compute_kernel_config=self.li_o_prefill_compute_kernel_cfg,
                 dtype=self.activation_dtype or ttnn.bfloat8_b,
    -            memory_config=wo_prefill_output_mem_config,
    +            memory_config=wo_out_mem_config,
                 program_config=self.args.get_attn_wo_program_config(Mode.PREFILL, seq_len, None),
             )
     
    diff --git a/models/demos/llama3_1_8b_p150/tt/decoder.py b/models/demos/llama3_1_8b_p150/tt/decoder.py
    index 067ea19c0a..84b99613b9 100644
    --- a/models/demos/llama3_1_8b_p150/tt/decoder.py
    +++ b/models/demos/llama3_1_8b_p150/tt/decoder.py
    @@ -234,7 +234,7 @@ class TransformerBlock(LightweightModule):
             residual = x
     
             # x is fractured across devices and interleaved in DRAM (for prefill) and sharded in L1 (for decode)
    -        skip_mem_cfg = self.args.get_residual_mem_config(mode, self.prefetcher)
    +        skip_mem_cfg = self.args.get_residual_mem_config(mode, self.prefetcher, int(x.shape[-2]))
     
             assert (
                 x.memory_config() == skip_mem_cfg
    diff --git a/models/demos/llama3_1_8b_p150/tt/distributed_norm.py b/models/demos/llama3_1_8b_p150/tt/distributed_norm.py
    index 81444f48f0..7cfaf32cf7 100644
    --- a/models/demos/llama3_1_8b_p150/tt/distributed_norm.py
    ... (truncated, 89 more lines)

[#145] BinaryNgDeviceOperation · tt-lang · no gain  +1815.83 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/ttl_residual_add.py b/models/demos/llama3_1_8b_p150/tt/ttl_residual_add.py
    new file mode 100644
    index 0000000000..2d72b4599d
    --- /dev/null
    +++ b/models/demos/llama3_1_8b_p150/tt/ttl_residual_add.py
    @@ -0,0 +1,156 @@
    +"""tt-lang residual add on the short-prefill shape -- AUTHORED, MEASURED, and NOT WIRED IN.
    +
    +Kept as the record of the tt-lang rung for `BinaryNgDeviceOperation`.
    +
    +WHICH instance, and why this one. The op code covers five shapes. Only ONE is a same-dtype bf16
    +add that a ttl kernel can legally replace -- the post-MLP residual add. The others are closed:
    +  * the post-attention residual add is MIXED dtype (bf16 residual + bf8_b wo output), which the
    +    kernel's single-dtype dataflow buffers cannot express;
    +  * both SILU gate multiplies (the largest instances, 8.23 ms and 3.50 ms) are bf8_b, a BLOCK float
    +    format whose shared-exponent pack the tt-lang packer does not produce correctly on this build --
    +    measured on this model, having a ttl kernel emit bf8_b drove the e2e gate to a degenerate 28.917
    +    (a "PCC" above 1), so a kernel is not available for them at any grid.
    +
    +WORK DECOMPOSITION. [128, 4096] is 4 height tiles x 128 width tiles = 512 independent output tiles
    +and an add has no reduction, so every tile may go anywhere. `ttl.node` is 2-D, and naively that caps
    +the grid: a width split must divide 128 and fit 11 columns (so 8), leaving height (4) on y = 32
    +cores. As with the head-split kernels the way past it is to let ONE coordinate carry two work
    +dimensions through a constant divide, which the compiler does lower:
    +
    +    cy = w_half * H_T + h_tile        ->  8 rows, 64 cores, 8 tiles each
    +
    +MEASURED in the model, against the stock ttnn.add it replaces (352 calls each, same shape/dtype):
    +
    +    correctness   e2e PCC 0.985099, unchanged bit-for-bit from the stock op
    +    ttl kernel    4.87 us/call   on 64 cores
    +    stock ttnn    3.70 us/call   on 110 cores
    +    whole model   648.17 -> 648.35 ms  (BinaryNg -1.44 ms, GenericOp +1.74 ms)
    +
    +So it is 1.3x SLOWER and deliberately not on the hot path. The reason is structural and is the same
    +one that closed this op's grid rung: an interleaved binary_ng gets ALL 110 cores from
    +split_work_to_cores, while any tt-lang decomposition of a 4 x 128 tile grid tops out at 64 (the
    +width split needs a divisor of 128 that fits 11 columns, and y can only reach 8 rows before
    +H_T * W_HALVES exceeds the board's 10). A lone eltwise op moves a fixed number of bytes, so losing
    +42% of the cores cannot be bought back -- exactly the case GUIDELINES/11 names when it warns that a
    ... (truncated, 122 more lines)

[#150] BinaryNgDeviceOperation · cpp · no gain  +1816.36 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/cpp_add_generic.py b/models/demos/llama3_1_8b_p150/tt/cpp_add_generic.py
    new file mode 100644
    index 0000000000..a3ed3bd601
    --- /dev/null
    +++ b/models/demos/llama3_1_8b_p150/tt/cpp_add_generic.py
    @@ -0,0 +1,176 @@
    +"""C++ Metalium eltwise add via ttnn.generic_op -- AUTHORED, MEASURED, and NOT WIRED IN.
    +
    +Kept as the record of the cpp rung for `BinaryNgDeviceOperation`.
    +
    +WHICH instance. Same reasoning as the tt-lang rung (tt/ttl_residual_add.py): of this op's five
    +shapes only the post-MLP residual add, [128, 4096] bf16, is available to a hand kernel. The two
    +SILU gate multiplies -- the LARGEST at 8.23 ms and 3.50 ms -- are bf8_b, and the post-attention add
    +is mixed bf16/bf8_b; a generic_op CB carries one data_format per buffer index, so neither is
    +expressible without changing the op's dtype contract, which GUIDELINES/12 forbids.
    +
    +Drives a real reader/compute/writer triple through ttnn.generic_op, adapted from the repo's own
    +tt_metal/programming_examples/eltwise_binary (kernels copied into tt/kernels/ and generalised from
    +"one core walks page 0..n" to "each core owns a [start, start+n) slice"), with the output tiles
    +partitioned across the entire 11x10 compute grid -- i.e. the SAME 110 cores the stock op gets, which
    +is the one thing the tt-lang rung could not have (its 2-D node grid topped out at 64).
    +
    +MEASURED in the model against the stock ttnn.add it replaces (352 calls each, same shape/dtype and
    +the same 110 cores):
    +
    +    correctness   e2e PCC 0.985099, unchanged from the stock op
    +    cpp kernel    3.87 us/call
    +    stock ttnn    3.67 us/call
    +    whole model   648.17 -> 647.82 ms (BinaryNg -1.47 ms, GenericOp +1.37 ms)
    +
    +So it lands at PARITY -- 5% slower per call, and a whole-model delta of 0.05% that is inside
    +run-to-run noise. Not wired in, because the per-call number at equal core count is the honest signal
    +and it does not beat the stock op.
    +
    +Worth recording WHY this is the interesting result of the two kernel rungs. The tt-lang attempt on
    +the same add was 4.87 us/call, and it was slower for a specific reason: `ttl.node` is 2-D, so its
    +decomposition of a 4x128 tile grid could reach only 64 of the board's 110 cores. generic_op has no
    +such restriction -- the host hands each core an explicit tile slice -- so this kernel runs the same
    +110 cores as the stock op, and the gap duly closes from 1.32x to 1.05x. That isolates the cause: the
    +tt-lang loss was OCCUPANCY, not code quality, and once occupancy is equalised a hand-written
    ... (truncated, 240 more lines)

Limitations / suggested manual next steps:
- 1 op(s) tried but no lever beat baseline: LayerNormDeviceOperation
  -> inspect the per-op device report and consider a hand-written kernel or a structural change.

Reproduce:
  trace+1CQ perf:  python -m pytest models/demos/llama3_1_8b_p150/tests/e2e/test_main_perf.py::test_main_perf -svv
  full-model e2e PCC:  python -m pytest models/demos/llama3_1_8b_p150/tests/e2e/test_pcc.py -svv

levels: grid -> fidelity -> dtype -> shard -> host -> tt-lang -> cpp   |   ✓win = beat baseline, ·try = measured no-gain, ·wedge = wedged/crashed when tried, — = not attempted
```
<!-- END optimize -->

<!-- BEGIN bringup -->
# Bring-up run report — `llama3_1_8b_p150`

_Generated: 2026-07-27 20:39:19 UTC_

## Outcome

**Converged** after bring-up.

## Placement summary

- **ON_DEVICE** (0): graduated, native ttnn, PCC verified
- **KERNEL_MISSING** (0): on CPU temporarily — TTNN op gap
- **PENDING** (0): retry next run
- **CPU_REUSE** (0): REUSE/ADAPT tag NOT wired to a ttnn module — runs on CPU (eager runner), not verified on device

## Module placement (all components)

| Module | Status | Placement | Detail | Per-module PCC test |
|---|---|---|---|---|

## Reproduce

Run from the repo root. Per-component PCC (on device):
```bash
```

End-to-end / demo:
```bash
python -m pytest models/demos/llama3_1_8b_p150/tests/e2e/test_main_perf.py -svv
python -m pytest models/demos/llama3_1_8b_p150/tests/e2e/test_pcc.py -svv
python -m pytest models/demos/llama3_1_8b_p150/tests/e2e/test_pcc_hf.py -svv
python -m pytest models/demos/llama3_1_8b_p150/demo/conftest.py::test_demo -svv
python -m pytest models/demos/llama3_1_8b_p150/demo/demo.py::test_demo -svv
python -m pytest models/demos/llama3_1_8b_p150/demo/simple_text_demo.py::test_demo -svv
```

## Next steps
<!-- END bringup -->


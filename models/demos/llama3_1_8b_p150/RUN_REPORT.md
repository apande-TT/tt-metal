<!-- BEGIN optimize -->
# Optimize (perf) — `llama3_1_8b_p150`

_Updated live: 2026-07-27 02:41:44 UTC · 25 lever attempt(s) so far — each knob is logged the instant it resolves, win OR fail, with why it was tried and why it won or failed._

```
Optimization summary — llama3_1_8b_p150 · main (device_ms)
==========================================================
optimizing… — baseline->final speedup is finalized when the module converges (per-attempt detail below is live)
tracy trace pass, same window (16 layers):  33.89 ms

Roofline & utilization
  modeled floor       : 537.23 ms   (Σ per-op roofline floors)
  achievable (60-80%) : 671.54 - 895.38 ms
  measured            : 1092.12 ms
  at-floor            : 49%   (554.89 ms reachable headroom)
  status              : BELOW_BAND — keep optimizing
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
MatmulDeviceOperation              ✓win      —         —         —         —         —         —         —                 —
MatmulDeviceOperation              ·try      —         ✓win      ·try      ·try      ✓win      ✓win      ·try        1138.67
MatmulDeviceOperation              ✓win      —         —         —         —         —         —         —           1092.12
TopKDeviceOperation                ✓win      —         —         ✓win      —         ✓win      —         —                 —
TopKDeviceOperation                ·try      —         —         ·try      ✓win      —         —         —           1537.69


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

Limitations / suggested manual next steps:
- (none flagged automatically — see the per-op device report for remaining headroom.)

Reproduce:
  trace+1CQ perf:  python -m pytest models/demos/llama3_1_8b_p150/tests/e2e/test_main_perf.py::test_main_perf -svv
  full-model e2e PCC:  python -m pytest models/demos/llama3_1_8b_p150/tests/e2e/test_pcc.py -svv

levels: grid -> fidelity -> dtype -> shard -> host -> tt-lang -> cpp   |   ✓win = beat baseline, ·try = measured no-gain, ·wedge = wedged/crashed when tried, — = not attempted
```
<!-- END optimize -->

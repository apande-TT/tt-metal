<!-- BEGIN optimize -->
# Optimize (perf) — `llama3_1_8b_p150`

_Updated live: 2026-07-27 01:08:55 UTC · 6 lever attempt(s) so far — each knob is logged the instant it resolves, win OR fail, with why it was tried and why it won or failed._

```
Optimization summary — llama3_1_8b_p150 · main (device_ms)
==========================================================
optimizing… — baseline->final speedup is finalized when the module converges (per-attempt detail below is live)
tracy trace pass, same window (16 layers):  33.89 ms

Roofline & utilization
  modeled floor       : 537.23 ms   (Σ per-op roofline floors)
  achievable (60-80%) : 671.54 - 895.38 ms
  measured            : 1537.69 ms
  at-floor            : 35%   (1000.46 ms reachable headroom)
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

op                                 grid      fidelity  dtype     shard     host      tt-lang   cpp         best ms
------------------------------------------------------------------------------------------------------------------
TopKDeviceOperation                ✓win      —         —         ✓win      —         —         —                 —
TopKDeviceOperation                ·try      —         —         ·try      ✓win      —         —           1537.69


Per-attempt detail (every optimization tried — win OR fail — with gain vs baseline and WHY):
op                                        lever        ms  gain vs base  result     why tried / why it won or failed
--------------------------------------------------------------------------------------------------------------------
TopKDeviceOperation                        grid         —             —  ✓ win      committed: llama3_1_8b_p150: profile the real forward, not a crashed decode_step(None) The tracy path called pipeline.decode_step(None), which raised o
TopKDeviceOperation                        grid   2464.16      +0.02 ms  · no gain  TopK is grid=tiny (1 core, ~10ms/call, 1120ms = 45% of device time), so I handed it an explicit full compute-grid sub_core_grid_topk via ModelArgs. No gain: 2464.18->2464.16ms. Root cause is factory S
TopKDeviceOperation                       shard   2464.18      +0.00 ms  · no gain  TopK is memory-bound on a DRAM-interleaved sampling chain, so I width-sharded the top-k values/indices into L1 via DECODE_SAMPLING_INPUT_MEMCFG (the designed hook, copied from the llama3_70b_galaxy co
TopKDeviceOperation                       shard   2464.18      +0.00 ms  · no gain  Second shard variant: same L1 lever but the whole 2x32 gathered row as ONE shard, on the theory that splitting it across 2 cores was desynchronising the device-offset add. Also reverted -- PCC 9.4% To
TopKDeviceOperation                       shard         —             —  ✓ win      committed: llama3_1_8b_p150: let greedy decode take the argmax sampling path format_sampling_params already normalises temperature=0 rows to k=1 / p=0
TopKDeviceOperation                  structural   1537.69    +926.49 ms  ✓ win      Hypothesis: TopK's 1120ms is not a tunable cost but REDUNDANT WORK -- decode is greedy (temperature=0), and greedy needs only an argmax, not top-k/top-p/RNG. format_sampling_params already normalises 

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

Limitations / suggested manual next steps:
- (none flagged automatically — see the per-op device report for remaining headroom.)

Reproduce:
  trace+1CQ perf:  python -m pytest models/demos/llama3_1_8b_p150/tests/e2e/test_main_perf.py::test_main_perf -svv
  full-model e2e PCC:  python -m pytest models/demos/llama3_1_8b_p150/tests/e2e/test_pcc.py -svv

levels: grid -> fidelity -> dtype -> shard -> host -> tt-lang -> cpp   |   ✓win = beat baseline, ·try = measured no-gain, ·wedge = wedged/crashed when tried, — = not attempted
```
<!-- END optimize -->

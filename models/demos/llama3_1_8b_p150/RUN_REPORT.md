<!-- BEGIN optimize -->
# Optimize (perf) — `llama3_1_8b_p150`

_Updated live: 2026-07-26 16:02:19 UTC · 3 lever attempt(s) so far — each knob is logged the instant it resolves, win OR fail, with why it was tried and why it won or failed._

```
Optimization summary — llama3_1_8b_p150 · main (device_ms)
==========================================================
optimizing… — baseline->final speedup is finalized when the module converges (per-attempt detail below is live)

Roofline & utilization
  modeled floor       : 475.11 ms   (Σ per-op roofline floors)
  achievable (60-80%) : 593.88 - 791.84 ms
  measured            : 1168.12 ms
  at-floor            : 41%   (693.01 ms reachable headroom)
  status              : BELOW_BAND — keep optimizing
  (tok/s/u — N/A: not an LLM decode pipeline)

Op breakdown — device time by op class (latest profile · what to target, ranked):
op class         device_ms      %   count  bound  dominant op (shape)
---------------------------------------------------------------------------------------------------
reduction          1089.08  50.7%    1979   slow  TopKDeviceOperation
matmul              860.96  40.1%    4984   slow  MatmulDeviceOperation 128 x 4096 x 14336
attention            90.64   4.2%    2448   slow  NlpCreateHeadsDeviceOperation
host_overhead        69.49   3.2%       0   host  
eltwise              43.68   2.0%    2815   slow  BinaryNgDeviceOperation
datamove             42.21   2.0%    6532   slow  NLPConcatHeadsDeviceOperation
other                21.07   1.0%    2380   slow  NLPCreateQKVHeadsDecodeDeviceOperation
embedding             1.88   0.1%     107   slow  EmbeddingsDeviceOperation

op                                 grid      fidelity  dtype     shard     host      tt-lang   cpp         best ms
------------------------------------------------------------------------------------------------------------------
MatmulDeviceOperation              ✓win      —         —         —         —         —         —                 —
TopKDeviceOperation                ✓win      —         —         —         —         —         —           1168.12


Per-attempt detail (every optimization tried — win OR fail — with gain vs baseline and WHY):
op                                        lever        ms  gain vs base  result     why tried / why it won or failed
--------------------------------------------------------------------------------------------------------------------
MatmulDeviceOperation                      grid         —             —  ✓ win      committed: llama3_1_8b_p150: power-of-2 chunked TopK so on-device sampling runs MULTI-CORE The single-device sampling path (multi_step_reduction, mesh
TopKDeviceOperation                        grid   1168.12    +981.39 ms  ✓ win      Tried the full-grid knob on TopK because grid=tiny meant the whole 64128-wide vocab reduction ran on ONE core; select_program_factory only picks the multi-core factory when the reduced width is a powe
MatmulDeviceOperation                      grid   2149.51      +0.00 ms  · no gain  Tried full-grid occupancy on prefill ff1/ff3 because only 32 of ~130 cores are busy: M_tiles=4 caps the 2D-mcast at 4 row-blocks and per_core_N=56 is pinned to the 8 DRAM weight shards, so 4x8=32. Coo

Code changes — every attempt (win or fail):
===========================================

[#2] TopKDeviceOperation · grid · win  +981.39 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/model.py b/models/demos/llama3_1_8b_p150/tt/model.py
    index e19e919f4c..7343dd1574 100644
    --- a/models/demos/llama3_1_8b_p150/tt/model.py
    +++ b/models/demos/llama3_1_8b_p150/tt/model.py
    @@ -9,7 +9,6 @@ from tqdm import tqdm
     import ttnn
     from models.common.lightweightmodule import LightweightModule
     from models.common.rmsnorm import RMSNorm
    -from models.common.sampling.generator import SamplingGenerator
     from models.demos.llama3_1_8b_p150.tt.ccl import TT_CCL
     from models.demos.llama3_1_8b_p150.tt.common import Mode, copy_host_to_device
     from models.demos.llama3_1_8b_p150.tt.decoder import TransformerBlock
    @@ -18,6 +17,7 @@ from models.demos.llama3_1_8b_p150.tt.embedding import Embedding, ScaledEmbeddin
     from models.demos.llama3_1_8b_p150.tt.lm_head import LMHead
     from models.demos.llama3_1_8b_p150.tt.model_config import TensorGroup
     from models.demos.llama3_1_8b_p150.tt.rope import HfRotarySetup, RotarySetup
    +from models.demos.llama3_1_8b_p150.tt.sampling_multicore_topk import SamplingGenerator
     
     
     class Transformer(LightweightModule):
    diff --git a/models/demos/llama3_1_8b_p150/tt/sampling_multicore_topk.py b/models/demos/llama3_1_8b_p150/tt/sampling_multicore_topk.py
    new file mode 100644
    index 0000000000..2600fbd1d0
    --- /dev/null
    +++ b/models/demos/llama3_1_8b_p150/tt/sampling_multicore_topk.py
    @@ -0,0 +1,227 @@
    +# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
    +# SPDX-License-Identifier: Apache-2.0
    +"""Power-of-two chunked multi-step TopK so ``ttnn.topk`` takes the MULTI-CORE path.
    +
    +``TopKDeviceOperation::select_program_factory`` only picks ``TopKMultiCoreProgramFactory``
    +when *all* of these hold for the reduced width W:
    +
    +    W >= 8192 (multi_core_min_width)   W < 65535 (multi-core indices are UInt16)
    +    is_power_of_two(W)                 k <= 64
    +    verify_multi_core_cost(...) -> width % split_size == 0 for some power-of-2 split
    +
    +The stock single-device path (``multi_step_reduction``, mesh == [1, 1]) splits the
    +128256-wide logits into two 64128-wide halves. 64128 is not a power of two, so *both*
    +TopK calls fall back to ``TopKSingleCoreProgramFactory`` -- the whole vocabulary is
    ... (truncated, 213 more lines)

[#3] MatmulDeviceOperation · grid · no gain  +0.00 ms
    diff --git a/models/demos/llama3_1_8b_p150/tt/model.py b/models/demos/llama3_1_8b_p150/tt/model.py
    index e19e919f4c..7343dd1574 100644
    --- a/models/demos/llama3_1_8b_p150/tt/model.py
    +++ b/models/demos/llama3_1_8b_p150/tt/model.py
    @@ -9,7 +9,6 @@ from tqdm import tqdm
     import ttnn
     from models.common.lightweightmodule import LightweightModule
     from models.common.rmsnorm import RMSNorm
    -from models.common.sampling.generator import SamplingGenerator
     from models.demos.llama3_1_8b_p150.tt.ccl import TT_CCL
     from models.demos.llama3_1_8b_p150.tt.common import Mode, copy_host_to_device
     from models.demos.llama3_1_8b_p150.tt.decoder import TransformerBlock
    @@ -18,6 +17,7 @@ from models.demos.llama3_1_8b_p150.tt.embedding import Embedding, ScaledEmbeddin
     from models.demos.llama3_1_8b_p150.tt.lm_head import LMHead
     from models.demos.llama3_1_8b_p150.tt.model_config import TensorGroup
     from models.demos.llama3_1_8b_p150.tt.rope import HfRotarySetup, RotarySetup
    +from models.demos.llama3_1_8b_p150.tt.sampling_multicore_topk import SamplingGenerator
     
     
     class Transformer(LightweightModule):
    diff --git a/models/demos/llama3_1_8b_p150/tt/sampling_multicore_topk.py b/models/demos/llama3_1_8b_p150/tt/sampling_multicore_topk.py
    new file mode 100644
    index 0000000000..2600fbd1d0
    --- /dev/null
    +++ b/models/demos/llama3_1_8b_p150/tt/sampling_multicore_topk.py
    @@ -0,0 +1,227 @@
    +# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
    +# SPDX-License-Identifier: Apache-2.0
    +"""Power-of-two chunked multi-step TopK so ``ttnn.topk`` takes the MULTI-CORE path.
    +
    +``TopKDeviceOperation::select_program_factory`` only picks ``TopKMultiCoreProgramFactory``
    +when *all* of these hold for the reduced width W:
    +
    +    W >= 8192 (multi_core_min_width)   W < 65535 (multi-core indices are UInt16)
    +    is_power_of_two(W)                 k <= 64
    +    verify_multi_core_cost(...) -> width % split_size == 0 for some power-of-2 split
    +
    +The stock single-device path (``multi_step_reduction``, mesh == [1, 1]) splits the
    +128256-wide logits into two 64128-wide halves. 64128 is not a power of two, so *both*
    +TopK calls fall back to ``TopKSingleCoreProgramFactory`` -- the whole vocabulary is
    ... (truncated, 213 more lines)

Limitations / suggested manual next steps:
- (none flagged automatically — see the per-op device report for remaining headroom.)

Reproduce:
  trace+1CQ perf:  python -m pytest models/demos/llama3_1_8b_p150/tests/e2e/test_main_perf.py::test_main_perf -svv
  full-model e2e PCC:  python -m pytest models/demos/llama3_1_8b_p150/tests/e2e/test_pcc.py -svv

levels: grid -> fidelity -> dtype -> shard -> host -> tt-lang -> cpp   |   ✓win = beat baseline, ·try = measured no-gain, ·wedge = wedged/crashed when tried, — = not attempted
```
<!-- END optimize -->

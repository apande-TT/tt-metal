# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `decoder_head` of FLUX.2-klein-9B.

HF reference: `Flux2Transformer2DModel.proj_out` — the model's output head,
`nn.Linear(inner_dim=4096, patch_size**2 * out_channels=128, bias=False)`,
applied right after `norm_out` to turn hidden states back into latent patches.

(The scaffold seeded this file with a copy of `models/tt_transformers/tt/
lm_head.py`. That head is a vocabulary projection built around split column
shards and a weight cache; this one projects *down* to 128 channels, so the
column-split reasoning inverts — see below.)

Tensor-parallel scheme (TP=8)
-----------------------------
The head is ROW-parallel. Its 128 output features do not split into eight
tile-aligned (multiple-of-32) shards, but its 4096 input features do —
512 per chip = 16 tiles. So each chip holds `W[i*512:(i+1)*512, :]`, takes its
own slice of the replicated activation with `ttnn.mesh_partition` (a local
per-device slice, no fabric traffic), computes a partial product over those 512
input features, and one `ttnn.all_reduce` sums the eight partials. A matmul
over a concatenated contraction axis IS the sum of the per-block matmuls, so
the gathered result is the same math as the single-device head.
"""

from __future__ import annotations

import ttnn

HF_MODEL_ID = "/tmp/tt_hw_planner_components/flux_2_klein_9b_transformer"

_CANDIDATE_SUBMODULE_PATHS = ["proj_out", "norm_out"]


def _compute_config():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _mesh_width(device):
    """Number of chips this port is spread over (1 for a plain device)."""
    for attr in ("get_num_devices", "get_device_ids"):
        fn = getattr(device, attr, None)
        if fn is None:
            continue
        try:
            value = fn()
        except Exception:  # noqa: BLE001
            continue
        n = value if isinstance(value, int) else len(value)
        if n:
            return int(n)
    return 1


class TtDecoderHead:
    def __init__(self, device, torch_module):
        self.device = device
        self.cfg = _compute_config()
        self.tp = _mesh_width(device)

        # nn.Linear stores (out, in); matmul wants (in, out).
        wt = torch_module.weight.detach().float().t().contiguous()
        self.row_parallel = self.tp > 1 and wt.shape[0] % self.tp == 0
        if self.row_parallel:
            self.w = ttnn.from_torch(
                wt,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0),
            )
        else:
            self.w = ttnn.from_torch(wt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        b = torch_module.bias
        self.b = None
        if b is not None:
            # A row-parallel bias must be added ONCE, after the reduction —
            # folding it into the per-chip matmul would add it TP times.
            self.b = ttnn.from_torch(
                b.detach().float().reshape(1, -1).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

    def __call__(self, x, **kwargs):
        if self.row_parallel:
            local = ttnn.mesh_partition(x, dim=-1)
            out = ttnn.linear(local, self.w, compute_kernel_config=self.cfg)
            out = ttnn.all_reduce(out)
        else:
            out = ttnn.linear(x, self.w, compute_kernel_config=self.cfg)
        if self.b is not None:
            out = ttnn.add(out, self.b)
        return out


def build(device, torch_module):
    return TtDecoderHead(device, torch_module)


def decoder_head(device, torch_module, x, **kwargs):
    """Module-level entry point for callers that do not hold a built port."""
    return build(device, torch_module)(x, **kwargs)

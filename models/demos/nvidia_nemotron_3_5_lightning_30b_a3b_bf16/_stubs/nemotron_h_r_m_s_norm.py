# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# >>> MACHINE-GENERATED stub (ADAPT — canonical-wrapper path) <<<
"""Pure-TTNN NemotronH RMSNorm for `nemotron_h_r_m_s_norm` (`layers.N.norm`).

The canonical `RMSNorm` (models/common/rmsnorm.py) is structurally
inapplicable: its construction path goes through `ModelArgs(mesh_device=...)`,
which cannot build a config for this repo's custom NemotronH checkpoint (same
`ModelArgs` failure documented on the graduated `nemotron_h_attention`
sibling). So this stub computes the forward with native `ttnn.*` ops. The
canonical import is retained per the ADAPT requirement.

HF reference (`NemotronHRMSNorm.forward`, modeling_nemotron_h.py):

    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = x.float() * rsqrt(variance + eps)
    return weight * x.to(input_dtype)

A norm's weight is elementwise (not a matmul), so it stays REPLICATED under
tensor parallelism -- no sharding/collective needed here.
"""
from __future__ import annotations

import ttnn
from models.common.rmsnorm import RMSNorm  # kept per ADAPT requirement


class TtNemotronHRMSNorm:
    def __init__(self, device, torch_module) -> None:
        self.device = device
        self.eps = float(getattr(torch_module, "variance_epsilon", 1e-5))
        w = torch_module.weight.detach().float().reshape(1, 1, -1).contiguous()
        self._w = self._dev(w)
        self.ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def _is_mesh(self):
        try:
            if isinstance(self.device, ttnn.MeshDevice):
                return True
        except AttributeError:
            pass
        return hasattr(self.device, "get_device_ids") or hasattr(self.device, "get_devices")

    def _dev(self, torch_tensor, layout=ttnn.TILE_LAYOUT):
        if self._is_mesh():
            try:
                return ttnn.from_torch(
                    torch_tensor,
                    dtype=ttnn.float32,
                    layout=layout,
                    device=self.device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
                )
            except Exception:
                pass
        return ttnn.from_torch(torch_tensor, dtype=ttnn.float32, layout=layout, device=self.device)

    def _fp32(self, t):
        if isinstance(t, ttnn.Tensor):
            if t.dtype != ttnn.float32:
                return ttnn.typecast(t, ttnn.float32)
            return t
        return self._dev(t.float())

    def _row_shard(self, W):
        """(memory config, program config) for a width-sharded norm of ONE tile
        row: interleaved, a (32, W) input runs on a single core; sharded, the row
        is split over the largest core rectangle whose count divides W's tiles."""
        cfg = getattr(self, "_row_cfg", {}).get(W)
        if cfg is not None:
            return cfg
        g = self.device.compute_with_storage_grid_size()
        wt = W // 32
        best = max(
            ((x, y) for x in range(1, g.x + 1) for y in range(1, g.y + 1) if wt % (x * y) == 0),
            key=lambda c: (c[0] * c[1], c[0]),
        )
        bw = wt // (best[0] * best[1])
        mem = ttnn.create_sharded_memory_config(
            shape=(1, 1, 32, W), core_grid=ttnn.CoreGrid(y=best[1], x=best[0]), strategy=ttnn.ShardStrategy.WIDTH
        )
        pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=best,
            subblock_w=max(d for d in range(1, min(4, bw) + 1) if bw % d == 0),
            block_h=1,
            block_w=bw,
            inplace=False,
        )
        self._row_cfg = {**getattr(self, "_row_cfg", {}), W: (mem, pc)}
        return mem, pc

    def __call__(self, hidden_states, dtype=ttnn.bfloat16, **kwargs):
        """dtype: output dtype. A caller that upcasts to fp32 anyway asks for
        fp32 and skips a bf16 round trip (two passes, and the rounding)."""
        hs = self._fp32(hidden_states)
        if hs.layout != ttnn.TILE_LAYOUT:
            hs = ttnn.to_layout(hs, ttnn.TILE_LAYOUT)
        shape = [int(v) for v in hs.padded_shape]
        W = shape[-1]
        if shape[-2] == 32 and all(v == 1 for v in shape[:-2]) and W % 32 == 0:  # decode: one tile row
            mem, pc = self._row_shard(W)
            xs = ttnn.to_memory_config(hs, mem)
            ttnn.deallocate(hs)
            out = ttnn.rms_norm(
                xs,
                weight=self._w,
                epsilon=self.eps,
                program_config=pc,
                memory_config=mem,
                compute_kernel_config=self.ckc,
            )
            ttnn.deallocate(xs)
            out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        else:
            out = ttnn.rms_norm(hs, weight=self._w, epsilon=self.eps, compute_kernel_config=self.ckc)
            ttnn.deallocate(hs)
        return out if dtype == ttnn.float32 else ttnn.typecast(out, dtype)


# Module-level `build` — primary test entry point.
def build(device, torch_module=None):
    return TtNemotronHRMSNorm.build(device, torch_module)


# Module-level shim with the component's lowercase slug name. Kept for
# backward compatibility with legacy SMOKE/PCC tests that import the
# slug directly.
def nemotron_h_r_m_s_norm(device, torch_module=None):
    return TtNemotronHRMSNorm.build(device, torch_module)

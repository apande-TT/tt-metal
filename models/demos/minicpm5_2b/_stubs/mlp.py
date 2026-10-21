# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn port of `LlamaMLP` (model.layers.0.mlp) for `openbmb/MiniCPM5-2B`.

HF semantics: down(silu(gate(x)) * up(x)), 2048 -> 6144 -> 2048, no biases.
Self-contained: all intermediates and outputs live in DRAM interleaved.
"""
from __future__ import annotations

import torch

import ttnn
from models.demos.minicpm5_2b._stubs.decode_mm import decode_linear_config
from models.demos.minicpm5_2b._stubs.dram_mm import DramMatmul
from models.demos.minicpm5_2b._stubs.mm_swiglu import interleave_gate_up, mm_swiglu, mm_swiglu_supported

# Decode projections on the DRAM-bank-local generic_op (same arithmetic as the 1-D multicast matmul).
_USE_DRAM_MM = True
# ... with the post-attention RMSNorm fused into the gate/up program (bind_prenorm): its latency hides under the
# weight stream.
_USE_DRAM_MM_NORM = True
_NORM_AS_PROLOGUE = True
# silu(gate) * up as a prologue of the down + residual program (helper cores; DramMatmul silu_prologue).
_USE_SILU_PROLOGUE = True
# Prefill gate/up + silu * up as one program (kernels/mm_swiglu: minimal_matmul over the tile-pair interleaved
# [W_gate | W_up] with a SwiGLU epilogue bit-identical to the two minimal_matmuls + ttnn.multiply).
_PREFILL_MM_SWIGLU = True
# Prefill down on the C++ matmul (mm_swiglu's program, plain KSPLIT mode) with this K block.
_PREFILL_DOWN_CPP = True
_DOWN_K_BLOCK = 16
# Rows per M block: 13 rows per core as 7 + 6 (2 weight passes, not 4 + 4 + 4 + 1).
_DOWN_M_BLOCK = 7


def _compute_cfg(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


class TtMlp:
    def __init__(self, device, torch_module):
        self.device = device
        self.compute_cfg = _compute_cfg(device)
        self.mem = ttnn.DRAM_MEMORY_CONFIG

        def _w(linear):
            return ttnn.from_torch(
                linear.weight.detach().to(torch.float32).t().contiguous().unsqueeze(0).unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.w_gate = _w(torch_module.gate_proj)
        self.w_up = _w(torch_module.up_proj)
        self.w_down = _w(torch_module.down_proj)
        # Prefill only: [W_gate | W_up] tile-pair interleaved along N (tile 2p = gate tile p, 2p+1 = up tile p) for
        # mm_swiglu. The same fp32 values as w_gate / w_up, permuted by whole tiles, so the bf16 tiles are identical.
        self.w_gu_il = None
        if _PREFILL_MM_SWIGLU:
            w_il = interleave_gate_up(
                torch_module.gate_proj.weight.detach().to(torch.float32).t().contiguous(),
                torch_module.up_proj.weight.detach().to(torch.float32).t().contiguous(),
            )
            self.w_gu_il = ttnn.from_torch(
                w_il.contiguous().unsqueeze(0).unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        h, f = self.w_gate.shape[-2], self.w_gate.shape[-1]
        self.cfg_up = decode_linear_config(device, h, f)
        self.cfg_down = decode_linear_config(device, f, h, per_core_n=2)  # 64 tiles on 32 cores
        self.dram_mm = self.down_res = self.down_res_silu = None
        if _USE_DRAM_MM:

            def _t(linear):
                return linear.weight.detach().to(torch.float32).t().contiguous()

            fid = ttnn.MathFidelity.HiFi2  # as self.compute_cfg
            # gate and up as one program over [W_gate | W_up] (one activation read), written as two tensors.
            w_gu = torch.cat([_t(torch_module.gate_proj), _t(torch_module.up_proj)], dim=-1)
            self.dram_mm = (
                DramMatmul(device, w_gu, fid, split_n=f),
                DramMatmul(device, _t(torch_module.down_proj), fid),
            )
            # down + the layer's residual add in one program (ttnn.add's arithmetic on helper cores).
            dmm = DramMatmul(device, None, fid, prebuilt=self.dram_mm[1], residual_add=True)
            self.down_res = dmm if dmm.ok else None
            # ... and also silu(gate) * up computed in that program by the same helper cores, ahead of the stream
            # (a deep weight CB keeps the stream going meanwhile).
            dmm = DramMatmul(
                device, None, fid, prebuilt=self.dram_mm[1], residual_add=True, silu_prologue=True, cb_blocks=10
            )
            self.down_res_silu = dmm if (dmm.ok and dmm.silu_pro and _USE_SILU_PROLOGUE) else None

    def _mm(self, x, w, k_block=None, memory_config=None):
        """Many-row (prefill) gate/up projection on minimal_matmul's small CBs, which span the full grid under fp32
        accumulation. K_block is all of K, so each tile is one uninterrupted K-order accumulation as in ttnn.linear."""
        grid = self.device.compute_with_storage_grid_size()
        return ttnn.experimental.minimal_matmul(
            x,
            w,
            config=ttnn.MinimalMatmulConfig(
                M_block_size=2,
                K_block_size=k_block or w.shape[-2] // 32,
                N_block_size=2,
                subblock_h=2,
                subblock_w=2,
                compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
            ),
            memory_config=memory_config or self.mem,
            compute_kernel_config=self.compute_cfg,
        )

    def _silu_mul(self, gate, up, memory_config=None):
        # silu(gate) * up in one op: the SiLU runs on the bf16 gate inside the multiply, as the separate
        # unary did (ttnn.linear without a core grid applies its activation as a post-op on the bf16 output).
        return ttnn.multiply(
            gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=memory_config or self.mem
        )

    def bind_prenorm(self, norm):
        """Fuse `norm` (the post-attention RMSNorm) into the decode gate/up program."""
        self.dmm_gu_norm = None
        if _USE_DRAM_MM_NORM and self.dram_mm and self.dram_mm[0].ok and hasattr(norm, "consts"):
            gu = self.dram_mm[0]
            # The norm on helper cores ahead of the projection (prologue), or on its own cores with an all-gather.
            how = dict(norm_prologue=norm) if _NORM_AS_PROLOGUE else dict(norm=norm)
            dmm = DramMatmul(self.device, None, ttnn.MathFidelity.HiFi2, split_n=gu.split_n, prebuilt=gu, **how)
            self.dmm_gu_norm = dmm if dmm.ok else None
        return self.dmm_gu_norm is not None

    def __call__(self, x, *args, prenorm=None, residual=None, **kwargs):
        """prenorm: the RMSNorm still to be applied to x (inside the decode gate/up program when bound).
        residual: the layer's residual stream; when the decode down program adds it itself it returns
        residual + MLP and sets last_residual_added."""
        self.last_residual_added = False
        if not isinstance(x, ttnn.Tensor):
            x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        fused_norm = None
        if prenorm is not None:
            dmm = getattr(self, "dmm_gu_norm", None)
            bound = dmm is not None and (dmm.norm or dmm.norm_pro) is prenorm
            if bound and dmm.accepts(x) and prenorm._kernel_ok(x):
                fused_norm = dmm
            else:
                x = prenorm(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        in_shape = tuple(x.shape)
        if len(in_shape) != 4:
            # Fold every leading dim (batch included) into the matmul's M rows.
            rows = 1
            for d in in_shape[:-1]:
                rows *= d
            x = ttnn.reshape(x, (1, 1, rows, in_shape[-1]))
        if x.shape[-2] > 32 and x.shape[0] == x.shape[1] == 1:
            grid = self.device.compute_with_storage_grid_size()
            if (
                _PREFILL_MM_SWIGLU
                and self.w_gu_il is not None
                and mm_swiglu_supported(x, self.w_gu_il, grid)
            ):
                # silu(x @ W_gate) * (x @ W_up) in one program (bit-identical to the path below).
                h = mm_swiglu(
                    self.device, x, self.w_gu_il, compute_fidelity=ttnn.MathFidelity.HiFi2, memory_config=self.mem
                )
                if x.memory_config().buffer_type == ttnn.BufferType.L1:
                    ttnn.deallocate(x)  # an L1 norm output: free it so the down output can take its place in L1
            else:
                gate = self._mm(x, self.w_gate)
                up = self._mm(x, self.w_up)
                if x.memory_config().buffer_type == ttnn.BufferType.L1:
                    ttnn.deallocate(x)  # an L1 norm output: free it so the down output can take its place in L1
                h = self._silu_mul(gate, up)
                ttnn.deallocate(gate)
                ttnn.deallocate(up)
            # Output in L1 for the residual add.
            grid = self.device.compute_with_storage_grid_size()
            m_t, n_t = h.padded_shape[-2] // 32, self.w_down.padded_shape[-1] // 32
            n_core = -(-n_t // grid.x)
            if (
                _PREFILL_DOWN_CPP
                and n_core % 2 == 0
                and mm_swiglu_supported(h, self.w_down, grid, fuse_swiglu=False, k_block=_DOWN_K_BLOCK)
            ):
                # The full-grid C++ matmul in _DOWN_M_BLOCK-row x all-N blocks, 16-tile K blocks with exact fp32 partial reloads
                # (mm_swiglu's program, KSPLIT): bit-identical to the 2-D multicast linear below, with half the
                # partial spills.
                out = mm_swiglu(
                    self.device,
                    h,
                    self.w_down,
                    compute_fidelity=ttnn.MathFidelity.HiFi2,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    fuse_swiglu=False,
                    transpose=False,
                    blocks=(_DOWN_M_BLOCK, n_core, _DOWN_K_BLOCK),
                )
            else:
                # ttnn.linear's 2-D multicast over the full grid with 8-tile K blocks (24 K steps).
                out = ttnn.linear(
                    h,
                    self.w_down,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    compute_kernel_config=self.compute_cfg,
                    program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=(grid.x, grid.y),
                        in0_block_w=8,
                        out_subblock_h=1,
                        out_subblock_w=3,
                        per_core_M=-(-m_t // grid.y),
                        per_core_N=n_core,
                        transpose_mcast=False,
                        fused_activation=None,
                    ),
                )
        else:
            one_row = x.shape[0] == x.shape[1] == 1 and x.padded_shape[-2] == 32  # decode: one tile row
            cfg_up, cfg_down = (self.cfg_up, self.cfg_down) if one_row else (None, None)
            mid = ttnn.L1_MEMORY_CONFIG if one_row else self.mem  # decode intermediates fit in L1
            dmm = self.dram_mm if one_row and self.dram_mm and all(m.ok for m in self.dram_mm) else None
            if fused_norm is not None and one_row:
                # Post-attention RMSNorm + gate/up in one program, run through the norm's own call.
                gate, up = prenorm(x, memory_config=mid, fused=fused_norm)
            elif dmm and dmm[0].accepts(x):
                gate, up = dmm[0](x, mid)
            else:
                gate = ttnn.linear(
                    x, self.w_gate, memory_config=mid, compute_kernel_config=self.compute_cfg, program_config=cfg_up
                )
                up = ttnn.linear(
                    x, self.w_up, memory_config=mid, compute_kernel_config=self.compute_cfg, program_config=cfg_up
                )
            res = None
            if residual is not None and self.down_res is not None and one_row:
                res = ttnn.reshape(residual, tuple(gate.shape)[:-1] + (residual.shape[-1],))
            pro_ok = (
                res is not None
                and self.down_res_silu is not None
                and self.down_res_silu.accepts(gate)
                and list(up.padded_shape) == list(gate.padded_shape)
                and list(res.padded_shape) == list(gate.padded_shape)[:-1] + [res.padded_shape[-1]]
                and res.dtype == ttnn.bfloat16
                and not res.memory_config().is_sharded()
            )
            if pro_ok:
                # silu(gate) * up, down and the residual add as one program.
                out = self.down_res_silu(None, mid, residual=res, pro=(gate, up))
                ttnn.deallocate(gate)
                ttnn.deallocate(up)
                self.last_residual_added = True
                if len(in_shape) != 4:
                    out = ttnn.reshape(out, in_shape[:-1] + (out.shape[-1],))
                return out
            h = self._silu_mul(gate, up, memory_config=mid)
            ttnn.deallocate(gate)
            ttnn.deallocate(up)
            if (
                res is not None
                and self.down_res.accepts(h)
                and list(res.padded_shape) == list(h.padded_shape)[:-1] + [res.padded_shape[-1]]
                and res.dtype == ttnn.bfloat16
                and not res.memory_config().is_sharded()
            ):
                out = self.down_res(h, mid, residual=res)
                self.last_residual_added = True
            elif dmm and dmm[1].accepts(h):
                out = dmm[1](h, mid)
            else:
                out = ttnn.linear(
                    h, self.w_down, memory_config=mid, compute_kernel_config=self.compute_cfg, program_config=cfg_down
                )
        ttnn.deallocate(h)
        if len(in_shape) != 4:
            out = ttnn.reshape(out, in_shape[:-1] + (out.shape[-1],))
        return out


def build(device, torch_module=None):
    return TtMlp(device, torch_module)


def mlp(device, torch_module=None):
    return TtMlp(device, torch_module)

"""tt_hw_planner: native TTNN port of the Qwen-Image VAE mid-block attention.

Component: qwen_image_attention_block  (torch reference: diffusers QwenImageAttentionBlock,
`encoder.mid_block.attentions.0`)

QwenImageAttentionBlock is architecturally identical to the Wan2.1 VAE attention block (RMS norm,
1x1 to_qkv, single-head spatial SDPA per frame, 1x1 proj, residual), so this reuses tt_dit's native
`WanAttentionBlock`.

Tensor-parallel scheme (TP over the 1xN mesh): the block is single-head (no heads to split) and a
column split of the 384 -> 1152 qkv projection is not tile-aligned per chip, so -- as in
encoder_stack, which contains this block -- the parallel axis is SPATIAL. The activation arrives
W-partitioned across the mesh; the block all-gathers H/W for the (replicated) SDPA and re-partitions
its output, weights stay replicated, the residual is added per shard, and the output is
all-gathered along W. The gathered output equals the single-device result.
"""

from __future__ import annotations

import ttnn
from models.tt_dit.models.vae.vae_wan2_1 import WanAttentionBlock
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.qwen_image_edit_vae._stubs._resident import ResidentPort

# Mesh axis that carries the W partition; the other axis (size 1 on a 1xN mesh) carries H.
_H_AXIS = 0
_W_AXIS = 1


def _mesh_shape(device):
    try:
        shape = tuple(device.shape)
    except (AttributeError, TypeError):
        return (1, 1)
    return shape if len(shape) == 2 else (1, 1)


class TtQwenImageAttentionBlock(ResidentPort):
    BODY_ATTR = "block"  # inside encoder3d/decoder3d the port is entered via forward_sharded

    # precise (float32 bodies only; off by default = the graduated path): the body runs its q/k/v and
    # output projections without float32 accumulation and its SDPA on bf16 q/k/v. Here the same block
    # runs with float32 inputs carried as bf16 hi + lo, a 3-term split for QK^T and P @ V, and an explicit
    # float32 softmax. The gather / scatter around it is the body's.
    precise = False

    def forward_sharded(self, x_BTHWC, logical_h, logical_w=0):
        if not self.precise or x_BTHWC.dtype != ttnn.float32:
            return super().forward_sharded(x_BTHWC, logical_h, logical_w=logical_w)
        from models.tt_dit.pipelines.qwen_image_edit_transformer._stubs import _precise

        blk, pc = self.block, self.block.parallel_config
        if getattr(self, "_w32", None) is None:
            bf = lambda t: ttnn.typecast(t, ttnn.bfloat16)  # noqa: E731  (the checkpoint is bf16: exact)
            f32 = lambda t: ttnn.typecast(t, ttnn.float32)  # noqa: E731
            self._w32 = (
                bf(blk.to_qkv.weight.data),
                f32(blk.to_qkv.bias.data),
                bf(blk.proj.weight.data),
                f32(blk.proj.bias.data),
            )
            self._cfg32 = _precise.precise_config()
        w_qkv, b_qkv, w_proj, b_proj = self._w32
        residual = x_BTHWC
        x = x_BTHWC
        if pc.height_parallel.factor > 1:
            x = blk.ccl_manager.all_gather_persistent_buffer(x, dim=2, mesh_axis=pc.height_parallel.mesh_axis)
        if pc.width_parallel.factor > 1:
            x = blk.ccl_manager.all_gather_persistent_buffer(x, dim=3, mesh_axis=pc.width_parallel.mesh_axis)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        padded_h, padded_w = x.shape[2], x.shape[3]
        if padded_h > logical_h:
            x = x[:, :, :logical_h, :, :]
        if logical_w > 0 and padded_w > logical_w:
            x = x[:, :, :, :logical_w, :]
        B, T, H, W, C = x.shape
        x = ttnn.to_layout(ttnn.reshape(x, (B * T, H * W, C)), ttnn.TILE_LAYOUT)
        x = blk.norm(x, compute_kernel_config=self._cfg32)
        qkv = _precise.linear(x, w_qkv, bias=b_qkv)
        n = H * W
        q = ttnn.slice(qkv, [0, 0, 0], [B * T, n, C])
        k = ttnn.slice(qkv, [0, 0, C], [B * T, n, 2 * C])
        v = ttnn.slice(qkv, [0, 0, 2 * C], [B * T, n, 3 * C])
        sc = ttnn.multiply(_precise.matmul_bt(q, k), 1.0 / float(C) ** 0.5)
        e = ttnn.exp(ttnn.subtract(sc, ttnn.max(sc, dim=-1, keepdim=True)))
        prob = ttnn.divide(e, ttnn.sum(e, dim=-1, keepdim=True, compute_kernel_config=self._cfg32))
        out = _precise.linear(_precise.matmul(prob, v), w_proj, bias=b_proj)
        out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        if padded_h > logical_h or (logical_w > 0 and padded_w > logical_w):
            o4 = ttnn.reshape(out, (B * T, H, W, C))
            if logical_w > 0 and padded_w > logical_w:
                o4 = ttnn.pad(o4, [(0, 0), (0, 0), (0, padded_w - W), (0, 0)], value=0.0)
            if padded_h > logical_h:
                o4 = ttnn.pad(o4, [(0, 0), (0, padded_h - H), (0, 0), (0, 0)], value=0.0)
            out = ttnn.reshape(o4, (B, T, o4.shape[1], o4.shape[2], C))
        else:
            out = ttnn.reshape(out, (B, T, H, W, C))
        if pc.height_parallel.factor > 1:
            out = ttnn.mesh_partition(out, dim=2, cluster_axis=pc.height_parallel.mesh_axis)
        if pc.width_parallel.factor > 1:
            out = ttnn.mesh_partition(out, dim=3, cluster_axis=pc.width_parallel.mesh_axis)
        return ttnn.add(ttnn.to_layout(out, ttnn.TILE_LAYOUT), residual)

    def __init__(self, device, torch_module):
        self.device = device
        mesh_shape = _mesh_shape(device)
        self.parallel_config = VaeHWParallelConfig(
            height_parallel=ParallelFactor(factor=mesh_shape[_H_AXIS], mesh_axis=_H_AXIS),
            width_parallel=ParallelFactor(factor=mesh_shape[_W_AXIS], mesh_axis=_W_AXIS),
        )
        self.ccl_manager = CCLManager(device, topology=ttnn.Topology.Linear, num_links=1)

        self.block = WanAttentionBlock(
            dim=torch_module.dim,
            mesh_device=device,
            ccl_manager=self.ccl_manager,
            parallel_config=self.parallel_config,
            dtype=ttnn.bfloat16,
        )
        self.block.load_torch_state_dict(torch_module.state_dict())

    def __call__(self, x, **_ignored):
        # x: replicated TILE [B, C, T, H, W] (BCTHW, like the torch reference).
        B, C, T, H, W = x.shape
        pc = self.parallel_config
        assert (
            H % pc.height_parallel.factor == 0 and W % pc.width_parallel.factor == 0
        ), f"activation {H}x{W} must divide the {pc.height_parallel.factor}x{pc.width_parallel.factor} spatial mesh"

        x = ttnn.permute(x, (0, 2, 3, 4, 1))  # BTHWC
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        if pc.height_parallel.factor > 1:
            x = ttnn.mesh_partition(x, dim=2, cluster_axis=pc.height_parallel.mesh_axis)
        if pc.width_parallel.factor > 1:
            x = ttnn.mesh_partition(x, dim=3, cluster_axis=pc.width_parallel.mesh_axis)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        out = self.block(x, H, logical_w=W)

        out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        out = self.ccl_manager.all_gather(out, dim=3, mesh_axis=pc.width_parallel.mesh_axis, use_hyperparams=False)
        out = self.ccl_manager.all_gather(out, dim=2, mesh_axis=pc.height_parallel.mesh_axis, use_hyperparams=False)

        out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        return ttnn.permute(out, (0, 4, 1, 2, 3))  # BCTHW


def build(device, torch_module):
    return TtQwenImageAttentionBlock(device, torch_module)

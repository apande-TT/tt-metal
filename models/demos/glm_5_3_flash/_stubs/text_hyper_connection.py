# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn, tensor-parallel port of `Glm5NextTextHyperConnection` (GLM-5.3-Flash mHC).

Reference (fp32):
    flat  = UnweightedRMSNorm(streams.flatten(2))                    # [B,S,H*D]
    mix   = flat @ fn^T                                              # [B,S,(2+H)*H]
    pre   = sigmoid(mix[:H]   * s0 + b[:H]) + eps
    post  = 2 * sigmoid(mix[H:2H] * s1 + b[H:2H])
    comb  = Sinkhorn(softmax(mix[2H:].view(H,H) * s2 + b[2H:]) + eps)
    collapsed = sum_h pre[h] * streams[:, :, h]
    returns (post, comb, collapsed)

Tensor parallelism over N chips: the only matmul weight is `fn`. Its input (the streams) is
replicated, so it is COLUMN-parallel: the (2+H)*H mix outputs (scale folded in) are zero-padded to a
multiple of N and each chip computes its own output columns; an all_gather reassembles the full mix
logits on every chip. The norm, the Sinkhorn projection and the stream collapse are cheap,
elementwise / per-token, and stay REPLICATED. The gather payload is tiny, and all_gather deadlocks on
the 1x32 Galaxy line below ~64 tiles per chip (measured), so each chip's slice is zero-padded to
>= 64 tiles before the gather and the padding dropped afterwards.
"""
from __future__ import annotations

import torch

import ttnn


def _num_devices(device):
    try:
        return int(device.get_num_devices())
    except Exception:
        return 1


def _cluster_axis(device):
    try:
        shape = tuple(device.shape)
    except Exception:
        return None
    axes = [i for i, s in enumerate(shape) if s > 1]
    return axes[0] if len(axes) == 1 else None


def _rs(x, shape):
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(x, shape)
    return ttnn.to_layout(x, ttnn.TILE_LAYOUT)


class TtTextHyperConnection:
    def __init__(self, device, m):
        self.device = device
        self.n = _num_devices(device)
        self.axis = _cluster_axis(device)
        H = int(m.hc_mult)
        self.H = H
        self.iters = int(m.hc_sinkhorn_iters)
        self.hc_eps = float(m.hc_eps)
        self.norm_eps = float(m.input_norm.eps)
        from models.demos.glm_5_3_flash._stubs.text_unweighted_r_m_s_norm import TtTextUnweightedRMSNorm

        self.input_norm = TtTextUnweightedRMSNorm(device, m.input_norm)
        self.ckc = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        fn = m.fn.detach().to(torch.float32)  # [mix, H*D]
        s = m.scale.detach().to(torch.float32)
        mix = fn.shape[0]
        scales = torch.repeat_interleave(s[:3], torch.tensor([H, H, H * H]))  # pre (H) | post (H) | res (H*H)
        w = (fn * scales[:, None]).t()  # [H*D, mix], scale folded in
        self.mix = mix
        self.mix_pad = -(-mix // self.n) * self.n
        if self.mix_pad > mix:
            w = torch.nn.functional.pad(w, (0, self.mix_pad - mix))
        self.w_mix = self._put(w, shard_dim=-1)  # column-parallel: mix_pad / N columns per chip
        base = m.base.detach().to(torch.float32)
        self.b_pre = self._put(base[:H].reshape(1, 1, 1, H))
        self.b_post = self._put(base[H : 2 * H].reshape(1, 1, 1, H))
        self.b_comb = self._put(base[2 * H :].reshape(1, 1, 1, H * H))

    def _put(self, t, *, shard_dim=None):
        kw = dict(
            dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        if self.n > 1:
            kw["mesh_mapper"] = (
                ttnn.ShardTensorToMesh(self.device, dim=shard_dim)
                if shard_dim is not None
                else ttnn.ReplicateTensorToMesh(self.device)
            )
        return ttnn.from_torch(t.to(torch.float32).contiguous(), **kw)

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def _gather_cols(self, part, rows):
        """[1,1,rows,c] per chip -> [1,1,rows,N*c] in global column order (replicated). The partial logits
        travel as a bf16 hi/lo pair (~16 mantissa bits): Sinkhorn and the sigmoids amplify a bf16-rounded
        mix, and 90 hyper-connections in series compound it."""
        if self.n == 1:
            return part
        c = part.shape[-1]
        row_tiles = max(1, -(-rows // 32))
        cpad = -(-c // 32) * 32
        hi = ttnn.typecast(part, ttnn.bfloat16)
        lo = ttnn.typecast(ttnn.subtract(part, ttnn.typecast(hi, ttnn.float32)), ttnn.bfloat16)
        if cpad > c:
            hi = ttnn.pad(hi, [(0, 0), (0, 0), (0, 0), (0, cpad - c)], 0.0)
            lo = ttnn.pad(lo, [(0, 0), (0, 0), (0, 0), (0, cpad - c)], 0.0)
        pb = ttnn.concat([hi, lo], dim=-1)  # [1,1,rows,2*cpad]
        w = 2 * cpad
        wp = max(w, -(-64 // row_tiles) * 32)  # all_gather deadlocks below ~64 tiles/chip on the 1x32 line
        if wp > w:
            pb = ttnn.pad(pb, [(0, 0), (0, 0), (0, 0), (0, wp - w)], 0.0)
        g = ttnn.all_gather(pb, dim=1, cluster_axis=self.axis, topology=ttnn.Topology.Linear)  # [1,N,rows,wp]
        g = ttnn.to_layout(ttnn.typecast(g, ttnn.float32), ttnn.ROW_MAJOR_LAYOUT)
        g = ttnn.add(
            ttnn.slice(g, [0, 0, 0, 0], [1, self.n, rows, c]),
            ttnn.slice(g, [0, 0, 0, cpad], [1, self.n, rows, cpad + c]),
        )
        g = ttnn.permute(g, (0, 2, 1, 3))  # [1,rows,N,c]
        g = ttnn.reshape(g, [1, 1, rows, self.n * c])
        return ttnn.to_layout(g, ttnn.TILE_LAYOUT)

    def streams(self, Xs):
        """Stream-major entry used by the decoder stack. Xs [1,H,R,D] fp32 (stream h of row r at [0,h,r,:];
        rows on the tile axis, so nothing pads the H=4 stream axis to a 32-row tile). Returns
        post [1,1,R,H], comb [1,1,R,H*H] (row-major flattened comb[k][h]) and collapsed [1,1,R,D], all fp32."""
        H, R, D = Xs.shape[1], Xs.shape[2], Xs.shape[3]
        # flatten the streams per row ([1,1,R,H*D]) through ROW_MAJOR, then the graduated unweighted norm
        flat = ttnn.permute(ttnn.to_layout(Xs, ttnn.ROW_MAJOR_LAYOUT), (0, 2, 1, 3))  # [1,R,H,D]
        flat = ttnn.to_layout(ttnn.reshape(flat, [1, 1, R, H * D]), ttnn.TILE_LAYOUT)
        flat = self.input_norm(flat)  # graduated text_unweighted_r_m_s_norm stub
        part = ttnn.linear(flat, self.w_mix, dtype=ttnn.float32, compute_kernel_config=self.ckc)
        mix = self._gather_cols(part, R)  # [1,1,R,mix_pad]

        # split the mix logits (row-major slicing handles the non-tile-aligned boundaries)
        mrm = ttnn.to_layout(mix, ttnn.ROW_MAJOR_LAYOUT)
        cut = lambda a, b: ttnn.to_layout(ttnn.slice(mrm, [0, 0, 0, a], [1, 1, R, b]), ttnn.TILE_LAYOUT)
        pre = ttnn.add(ttnn.sigmoid(ttnn.add(cut(0, H), self.b_pre)), self.hc_eps)  # [1,1,R,H]
        post = ttnn.multiply(ttnn.sigmoid(ttnn.add(cut(H, 2 * H), self.b_post)), 2.0)  # [1,1,R,H]
        logits = _rs(ttnn.add(cut(2 * H, 2 * H + H * H), self.b_comb), [1, R, H, H])

        e = ttnn.exp(logits)
        comb = ttnn.add(ttnn.divide(e, ttnn.sum(e, dim=-1, keepdim=True)), self.hc_eps)
        comb = ttnn.divide(comb, ttnn.add(ttnn.sum(comb, dim=-2, keepdim=True), self.hc_eps))
        for _ in range(self.iters - 1):
            comb = ttnn.divide(comb, ttnn.add(ttnn.sum(comb, dim=-1, keepdim=True), self.hc_eps))
            comb = ttnn.divide(comb, ttnn.add(ttnn.sum(comb, dim=-2, keepdim=True), self.hc_eps))
        comb = _rs(comb, [1, 1, R, H * H])

        # stream collapse sum_h pre[h] * X[h] as exact fp32 broadcasts (a K = hc_mult matmul rounds to ~bf16)
        prm = ttnn.to_layout(pre, ttnn.ROW_MAJOR_LAYOUT)
        collapsed = None
        for h in range(H):
            ph = ttnn.to_layout(ttnn.slice(prm, [0, 0, 0, h], [1, 1, R, h + 1]), ttnn.TILE_LAYOUT)  # [1,1,R,1]
            term = ttnn.multiply(ph, ttnn.slice(Xs, [0, h, 0, 0], [1, h + 1, R, D]))
            collapsed = term if collapsed is None else ttnn.add(collapsed, term)
        return post, comb, collapsed

    def __call__(self, hidden_streams, *args, **kwargs):
        """HF layout entry: hidden_streams [B,S,H,D] -> post [B,S,H], comb [B,S,H,H], collapsed [B,S,D]."""
        X = hidden_streams
        if X.layout != ttnn.TILE_LAYOUT:
            X = ttnn.to_layout(X, ttnn.TILE_LAYOUT)
        X = ttnn.typecast(X, ttnn.float32)
        B, S, H, D = X.shape
        R = B * S
        Xs = ttnn.permute(ttnn.reshape(ttnn.to_layout(X, ttnn.ROW_MAJOR_LAYOUT), [1, R, H, D]), (0, 2, 1, 3))
        post, comb, collapsed = self.streams(ttnn.to_layout(Xs, ttnn.TILE_LAYOUT))
        rm = lambda t, shape: ttnn.reshape(ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT), shape)
        return rm(post, [B, S, H]), rm(comb, [B, S, H, H]), rm(collapsed, [B, S, D])


def build(device, torch_module=None):
    return TtTextHyperConnection.build(device, torch_module)


def text_hyper_connection(device, torch_module=None):
    return TtTextHyperConnection.build(device, torch_module)

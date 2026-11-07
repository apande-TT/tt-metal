# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn, tensor-parallel port of `Glm5NextTextDecoderLayer` (GLM-5.3-Flash), KDA + dense-MLP layers.

Layer math (HF):
    post, comb, x = attn_hc(H)                 # mHC: collapse hc_mult residual streams
    a  = LinearAttention(RMSNorm(x))           # Kimi delta attention (KDA)
    H  = post (x) a + comb^T @ H               # expand back into the streams
    post, comb, x = ffn_hc(H)
    m  = MLP(RMSNorm(x))                       # SwiGLU with swiglu_limit clamps
    H  = post (x) m + comb^T @ H

Tensor parallelism over N mesh chips:
  * hyper-connections, both RMSNorms, f_a_proj / g_a_proj, o_norm: REPLICATED (fp32).
  * KDA is HEAD-parallel: q/k/v_proj, the depthwise conv channels, f_b_proj, dt_bias, A_log,
    b_proj, g_b_proj are sharded so each chip owns num_heads/N whole heads; the recurrence runs
    unchanged within a head (never split over time). o_proj is ROW-parallel + all-reduce.
  * MLP: gate/up COLUMN-parallel, down ROW-parallel + all-reduce.
  * All-reduce = all_gather(partials) + on-device sum (ttnn.all_reduce deadlocks on the 1x32 line).

KDA (chunked, chunk C=64, fp32) per head, with exact reformulations that are device friendly:
    G = cumsum(g) within the chunk (ttnn.cumsum -- exact; a triangular matmul is not)
    dec[i,j,:] = exp(min(G_i - G_j, 0))           explicit [C, C, dk] tensor (no exp overflow)
    L = -(sum_d kb_i k_j dec) strictly-lower;  T = (I - L)^-1 = prod_{p<6} (I + L^(2^p))  (nilpotent)
    A = (sum_d q_i k_j dec) lower-incl.
    v' = T (v*beta);  kc = T (kb * e^G);  v_new = v' - kc S;  o = (q e^G) S + A v_new
    S <- S * e^{G_last}^T + (k e^{G_last - G})^T v_new
Verified against `chunk_kimi_delta_attention` in torch to 1e-7.
"""

from __future__ import annotations

import torch

import ttnn

CHUNK = 64


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


class _Ctx:
    def __init__(self, device):
        self.device = device
        self.n = _num_devices(device)
        self.axis = _cluster_axis(device)
        self.ckc = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def put(self, t, *, shard_dim=None, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        kw = dict(dtype=dtype, layout=layout, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.n > 1:
            kw["mesh_mapper"] = (
                ttnn.ShardTensorToMesh(self.device, dim=shard_dim)
                if shard_dim is not None
                else ttnn.ReplicateTensorToMesh(self.device)
            )
        return ttnn.from_torch(t.contiguous(), **kw)

    def f32(self, t, shard_dim=None):
        return self.put(t.to(torch.float32), shard_dim=shard_dim, dtype=ttnn.float32)

    def lin(self, x, w, dtype=ttnn.float32):
        return ttnn.linear(x, w, dtype=dtype, compute_kernel_config=self.ckc)

    @staticmethod
    def split(a):
        """fp32 -> (bf16 hi, bf16 lo) with hi + lo == a to ~16 mantissa bits."""
        ah = ttnn.typecast(a, ttnn.bfloat16)
        return ah, ttnn.typecast(ttnn.subtract(a, ttnn.typecast(ah, ttnn.float32)), ttnn.bfloat16)

    def lin_hp(self, x, w, dtype=ttnn.float32):
        """lin() with an fp32 activation kept to ~16 mantissa bits (hi*w + lo*w) instead of rounded to bf16;
        a non-fp32 activation goes through lin() unchanged."""
        if x.dtype != ttnn.float32:
            return self.lin(x, w, dtype)
        xh, xl = self.split(x)
        return ttnn.add(self.lin(xh, w, dtype), self.lin(xl, w, dtype))

    def mm(self, a, b):
        """fp32 matmul at ~16 mantissa bits. A plain fp32 ttnn.matmul consumes its operands at ~bf16/tf32
        precision (measured rel. err 1.1e-3 on Blackhole, HiFi4); the KDA chunk algebra (the nilpotent
        (I - L)^-1 product, the inter-chunk state) amplifies that into a wrong result on real
        activations. Split each operand into bf16 hi + lo and accumulate hi*hi + hi*lo + lo*hi in fp32."""
        ah, al = self.split(a)
        bh, bl = self.split(b)
        f = lambda x, y: ttnn.matmul(x, y, dtype=ttnn.float32, compute_kernel_config=self.ckc)
        return ttnn.add(ttnn.add(f(ah, bh), f(ah, bl)), f(al, bh))

    def all_reduce(self, x):
        if self.n == 1:
            return x
        g = ttnn.all_gather(x, dim=1, cluster_axis=self.axis, topology=ttnn.Topology.Linear)
        return ttnn.sum(g, dim=1, keepdim=True)


def _rs(x, shape):
    """Reshape through ROW_MAJOR (a pure view there), back to TILE."""
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(x, shape)
    return ttnn.to_layout(x, ttnn.TILE_LAYOUT)


def _wT(w):  # nn.Linear weight [out, in] -> [in, out]
    return w.detach().to(torch.float32).t().contiguous()


def _rms(ctx, x, w, eps):
    # fp32 RMSNorm over the last dim; w is a [1,1,1,D] fp32 tensor (or None).
    var = ttnn.mean(ttnn.multiply(x, x), dim=-1, keepdim=True)
    y = ttnn.multiply(x, ttnn.rsqrt(ttnn.add(var, eps)))
    return ttnn.multiply(y, w) if w is not None else y


# --------------------------------------------------------------------------- hyper-connection
class _HC:
    def __init__(self, ctx, hc, eps_norm):
        self.ctx = ctx
        H = int(hc.hc_mult)
        self.H = H
        self.iters = int(hc.hc_sinkhorn_iters)
        self.hc_eps = float(hc.hc_eps)
        self.norm_eps = float(eps_norm)
        fn = hc.fn.detach().to(torch.float32)
        base = hc.base.detach().to(torch.float32)
        s = hc.scale.detach().to(torch.float32)
        # Fold the per-output scale into the projection; split pre / post / comb so no slicing on device.
        self.w_pre = ctx.f32((fn[:H] * s[0]).t())
        self.w_post = ctx.f32((fn[H : 2 * H] * s[1]).t())
        self.w_comb = ctx.f32((fn[2 * H :] * s[2]).t())
        self.b_pre = ctx.f32(base[:H].reshape(1, 1, 1, H))
        self.b_post = ctx.f32(base[H : 2 * H].reshape(1, 1, 1, H))
        self.b_comb = ctx.f32(base[2 * H :].reshape(1, 1, 1, H * H))

    def __call__(self, X, S, D):
        """X: [1,S,H,D] fp32 streams -> post [1,S,H,1], comb [1,S,H,H], collapsed [1,1,S,D]."""
        c, H = self.ctx, self.H
        flat = _rms(c, _rs(X, [1, 1, S, H * D]), None, self.norm_eps)
        pre = ttnn.add(ttnn.sigmoid(ttnn.add(c.lin(flat, self.w_pre), self.b_pre)), self.hc_eps)
        post = ttnn.multiply(ttnn.sigmoid(ttnn.add(c.lin(flat, self.w_post), self.b_post)), 2.0)
        logits = _rs(ttnn.add(c.lin(flat, self.w_comb), self.b_comb), [1, S, H, H])
        e = ttnn.exp(logits)
        comb = ttnn.add(ttnn.divide(e, ttnn.sum(e, dim=-1, keepdim=True)), self.hc_eps)
        comb = ttnn.divide(comb, ttnn.add(ttnn.sum(comb, dim=-2, keepdim=True), self.hc_eps))
        for _ in range(self.iters - 1):
            comb = ttnn.divide(comb, ttnn.add(ttnn.sum(comb, dim=-1, keepdim=True), self.hc_eps))
            comb = ttnn.divide(comb, ttnn.add(ttnn.sum(comb, dim=-2, keepdim=True), self.hc_eps))
        collapsed = c.mm(_rs(pre, [1, S, 1, H]), X)  # [1,S,1,D]
        return _rs(post, [1, S, H, 1]), comb, _rs(collapsed, [1, 1, S, D])

    def expand(self, post, comb, sub, X, S, D):
        """post (x) sub + comb^T @ X  -> [1,S,H,D]."""
        c = self.ctx
        a = c.mm(post, _rs(sub, [1, S, 1, D]))
        b = c.mm(ttnn.transpose(comb, -2, -1), X)
        return ttnn.add(a, b)


# --------------------------------------------------------------------------- KDA linear attention
class _KDA:
    """Head-parallel KDA. Prefill = chunked delta rule over [B, S] (S a multiple of CHUNK, any number of
    chunks); decode = the exact one-token recurrence on the resident per-sample state.

    The forget gate is the graduated `text_forget_gate` stub (its chip-local, pre-gather half) and the
    output norm is the graduated `text_r_m_s_norm_gated` stub, so each exists once.
    """

    def __init__(self, ctx, m):
        from models.demos.glm_5_3_flash._stubs.text_forget_gate import TtTextForgetGate
        from models.demos.glm_5_3_flash._stubs.text_r_m_s_norm_gated import TtTextRMSNormGated

        self.ctx = ctx
        n = ctx.n
        self.nh = int(m.num_heads)
        self.dk = int(m.head_dim)
        if self.nh % n:
            raise ValueError(f"linear num_heads={self.nh} not divisible by TP={n}")
        self.hl = self.nh // n
        self.K = int(m.conv_kernel_size)
        if m.conv1d.bias is not None:
            raise NotImplementedError("conv1d bias not used by this checkpoint")
        if m.activation != "silu":
            raise NotImplementedError(m.activation)
        qkv = self.nh * self.dk
        self.w_q = ctx.put(_wT(m.q_proj.weight), shard_dim=-1)
        self.w_k = ctx.put(_wT(m.k_proj.weight), shard_dim=-1)
        self.w_v = ctx.put(_wT(m.v_proj.weight), shard_dim=-1)
        # depthwise causal conv weights [3*qkv, K] -> per-region taps, sharded along channels (heads)
        cw = m.conv1d.weight.detach().to(torch.float32).squeeze(1)  # [3*qkv, K]
        self.conv = []
        for r in range(3):
            reg = cw[r * qkv : (r + 1) * qkv]  # [qkv, K]
            # tap for shift s (y[t] += w[K-1-s] * x[t-s])
            self.conv.append(
                [ctx.f32(reg[:, self.K - 1 - s].reshape(1, 1, 1, qkv), shard_dim=-1) for s in range(self.K)]
            )
        self.forget_gate = TtTextForgetGate(ctx.device, m.forget_gate)  # graduated stub (head-sharded)
        wb = _wT(m.b_proj.weight)  # [D, nh]
        # one [D,1] column per local head (sharded via a [D, n*1] stack per local slot)
        self.w_b = []
        for j in range(self.hl):
            cols = torch.stack([wb[:, d * self.hl + j] for d in range(n)], dim=-1)  # [D, n]
            self.w_b.append(ctx.put(cols, shard_dim=-1))
        self.w_ga = ctx.put(_wT(m.g_a_proj.weight))
        self.w_gb = ctx.put(_wT(m.g_b_proj.weight), shard_dim=-1)
        self.o_norm = TtTextRMSNormGated(ctx.device, m.o_norm)  # graduated stub (replicated)
        self.w_o = ctx.put(_wT(m.o_proj.weight), shard_dim=0)
        # chunk constants
        C = CHUNK
        self.eye = ctx.f32(torch.eye(C).reshape(1, 1, C, C))
        self.tril_strict = ctx.f32(torch.tril(torch.ones(C, C), -1).reshape(1, 1, C, C))
        self.tril_incl = ctx.f32(torch.tril(torch.ones(C, C)).reshape(1, 1, C, C))
        # one-hot row selectors for the forward substitution in _unit_lower_inv
        self.row_sel = [ctx.f32(torch.eye(C)[i].reshape(1, 1, C, 1)) for i in range(C)]

    # ------------------------------------------------------------------ shape constants
    def prepare(self, B, S):
        """Kept for callers that prepare per-shape constants up front. The causal shift of the short conv
        is a device slice + concat (`_shift_rows`), so no [B,1,S,S] shift matrices are uploaded: those
        were B x S^2 fp32 per kernel tap per layer, enough to fill DRAM at full-size prompts."""
        return None

    @staticmethod
    def _shift_rows(x, s):
        """x [B,1,S,W] -> rows moved down by s, the first s rows zero (== diag(ones(S-s), -s) @ x, exact)."""
        B, _, S, W = x.shape
        r = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        head = ttnn.slice(r, [0, 0, 0, 0], [B, 1, s, W])
        zeros = ttnn.multiply(head, 0.0)
        body = ttnn.slice(r, [0, 0, 0, 0], [B, 1, S - s, W])
        return ttnn.to_layout(ttnn.concat([zeros, body], dim=2), ttnn.TILE_LAYOUT)

    def _heads(self, x, S):  # [B,1,S,hl*dk] -> [B,hl,S,dk]
        B = x.shape[0]
        hs = [ttnn.slice(x, [0, 0, 0, h * self.dk], [B, 1, S, (h + 1) * self.dk]) for h in range(self.hl)]
        return hs[0] if len(hs) == 1 else ttnn.concat(hs, dim=1)

    def _merge(self, x, S):  # [B,hl,S,dk] -> [B,1,S,hl*dk]
        B = x.shape[0]
        hs = [ttnn.slice(x, [0, h, 0, 0], [B, h + 1, S, self.dk]) for h in range(self.hl)]
        return hs[0] if len(hs) == 1 else ttnn.concat(hs, dim=-1)

    def _decay_dot(self, a, b, G):
        """sum_d a[i,d] b[j,d] exp(min(G_i[d] - G_j[d], 0)) per head -> [B,hl,C,C]."""
        C, dk = CHUNK, self.dk
        B = a.shape[0]
        outs = []
        for h in range(self.hl):
            sl = lambda t: ttnn.slice(t, [0, h, 0, 0], [B, h + 1, C, dk])  # [B,1,C,dk]
            Gh, ah, bh = sl(G), sl(a), sl(b)
            diff = ttnn.subtract(_rs(Gh, [B, C, 1, dk]), Gh)  # [B,C,C,dk]  G_i - G_j
            dec = ttnn.exp(ttnn.clamp(diff, max=0.0))
            prod = ttnn.multiply(ttnn.multiply(dec, _rs(ah, [B, C, 1, dk])), bh)
            outs.append(_rs(ttnn.sum(prod, dim=-1, keepdim=True), [B, 1, C, C]))
        return outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=1)

    def _unit_lower_inv(self, L):
        """T = (I - L)^-1 for strictly-lower L [B,hl,C,C], by row-wise forward substitution
        (T = I + L T, row i final once rows < i are), the same recurrence HF's chunk_kda uses.
        The previous nilpotent product prod_p (I + L^(2^p)) is the Neumann series: on real (highly
        correlated) keys |L| ~ 0.9 and its partial sums reach ~1e17 before cancelling to |T| <= 1,
        which no fp32 arithmetic survives."""
        c = self.ctx
        T = self.eye
        for i in range(1, CHUNK):
            T = ttnn.add(T, ttnn.multiply(self.row_sel[i], c.mm(L, T)))
        return T

    def _l2(self, x):
        return ttnn.multiply(x, ttnn.rsqrt(ttnn.add(ttnn.sum(ttnn.multiply(x, x), dim=-1, keepdim=True), 1e-6)))

    def _gates(self, hb, R):
        """Per-token projections shared by prefill and decode, rows R = B*S: q/k/v raw, g, beta, gate."""
        c = self.ctx
        proj = [c.lin(hb, w) for w in (self.w_q, self.w_k, self.w_v)]  # [1,1,R,hl*dk] fp32
        g = self.forget_gate.local(hb)  # [1,1,R,hl*dk] fp32 (graduated text_forget_gate, chip-local heads)
        beta = [ttnn.sigmoid(c.lin(hb, w)) for w in self.w_b]  # each [1,1,R,1]
        gate = c.lin(ttnn.typecast(c.lin(hb, self.w_ga), ttnn.bfloat16), self.w_gb)  # [1,1,R,hl*dk]
        return proj, g, beta, gate

    def _out(self, core, gate, B, S):
        """core [B,hl,S,dv] -> o_norm (gated, graduated stub) -> o_proj (row-parallel) + all-reduce."""
        c = self.ctx
        y = self.o_norm(core, self._heads(_rs(gate, [B, 1, S, self.hl * self.dk]), S))
        out = c.lin(
            ttnn.typecast(_rs(self._merge(y, S), [1, 1, B * S, self.hl * self.dk]), ttnn.bfloat16),
            self.w_o,
            dtype=ttnn.bfloat16,
        )
        return ttnn.typecast(c.all_reduce(out), ttnn.float32)

    # ------------------------------------------------------------------ prefill
    def prefill(self, h, B, S, valid=None, real_len=None):
        """h [1,1,B*S,D] fp32 (rows sample-major). valid [1,1,B*S,1] fp32 (1 real / 0 right-pad) keeps the
        recurrent state equal to the unpadded one (pad rows get beta = 0 and g = 0). Returns
        (out [1,1,B*S,D] fp32, state) where state = (S [B,hl,dk,dv], conv tails per q/k/v)."""
        c, C = self.ctx, CHUNK
        if S % C:
            raise NotImplementedError(f"S={S} must be a multiple of {C}")
        real_len = S if real_len is None else int(real_len)
        R = B * S
        W = self.hl * self.dk
        hb = ttnn.typecast(h, ttnn.bfloat16)
        proj, g, beta, gate = self._gates(hb, R)
        if valid is not None:
            g = ttnn.multiply(g, valid)
            beta = [ttnn.multiply(b_, valid) for b_ in beta]
        qkv, tails = [], []
        for r in range(3):
            x = _rs(proj[r], [B, 1, S, W])
            acc = None
            for s_, w in enumerate(self.conv[r]):
                xs = x if s_ == 0 else self._shift_rows(x, s_)
                term = ttnn.multiply(xs, w)
                acc = term if acc is None else ttnn.add(acc, term)
            qkv.append(ttnn.silu(acc))
            # conv tail for decode: raw inputs at t = real_len-1, real_len-2, ..., real_len-(K-1)
            tails.append(
                [ttnn.slice(x, [0, 0, real_len - s_, 0], [B, 1, real_len - s_ + 1, W]) for s_ in range(1, self.K)]
            )
        q = ttnn.multiply(self._l2(self._heads(qkv[0], S)), self.dk**-0.5)
        k = self._l2(self._heads(qkv[1], S))
        v = self._heads(qkv[2], S)
        gh = self._heads(_rs(g, [B, 1, S, W]), S)
        bt = [_rs(b_, [B, 1, S, 1]) for b_ in beta]
        bh = bt[0] if len(bt) == 1 else ttnn.concat(bt, dim=1)  # [B,hl,S,1]

        state, outs = None, []
        n_chunks = S // C
        for ci in range(n_chunks):
            rows = lambda t: (
                t
                if n_chunks == 1
                else ttnn.slice(t, [0, 0, ci * C, 0], [t.shape[0], t.shape[1], (ci + 1) * C, t.shape[3]])
            )
            qc, kc_, vc, gc, bc = rows(q), rows(k), rows(v), rows(gh), rows(bh)
            G = ttnn.cumsum(gc, dim=2)
            kb = ttnn.multiply(kc_, bc)
            L = ttnn.multiply(ttnn.neg(self._decay_dot(kb, kc_, G)), self.tril_strict)
            A = ttnn.multiply(self._decay_dot(qc, kc_, G), self.tril_incl)
            T = self._unit_lower_inv(L)
            vp = c.mm(T, ttnn.multiply(vc, bc))
            if state is None:
                vnew = vp
                o = c.mm(A, vnew)
            else:
                eG = ttnn.exp(G)
                kcd = c.mm(T, ttnn.multiply(kb, eG))
                vnew = ttnn.subtract(vp, c.mm(kcd, state))
                o = ttnn.add(c.mm(ttnn.multiply(qc, eG), state), c.mm(A, vnew))
            outs.append(o)
            Glast = ttnn.sum(gc, dim=2, keepdim=True)  # [B,hl,1,dk]
            kd = ttnn.multiply(kc_, ttnn.exp(ttnn.subtract(Glast, G)))
            upd = c.mm(ttnn.transpose(kd, -2, -1), vnew)  # [B,hl,dk,dv]
            state = (
                upd if state is None else ttnn.add(ttnn.multiply(state, ttnn.transpose(ttnn.exp(Glast), -2, -1)), upd)
            )
        core = outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=2)  # [B,hl,S,dv]
        return self._out(core, gate, B, S), (state, tails)

    # ------------------------------------------------------------------ decode
    def decode(self, h, st):
        """One token per sample. h [1,1,B,D] fp32; st = (S [B,hl,dk,dv], tails) from prefill / last step.
        Returns (out [1,1,B,D] fp32, new state)."""
        c = self.ctx
        state, tails = st
        B = h.shape[-2]
        W = self.hl * self.dk
        hb = ttnn.typecast(h, ttnn.bfloat16)
        proj, g, beta, gate = self._gates(hb, B)
        qkv, new_tails = [], []
        for r in range(3):
            x = _rs(proj[r], [B, 1, 1, W])
            acc = ttnn.multiply(x, self.conv[r][0])
            for s_ in range(1, self.K):
                acc = ttnn.add(acc, ttnn.multiply(tails[r][s_ - 1], self.conv[r][s_]))
            qkv.append(ttnn.silu(acc))
            new_tails.append([x] + tails[r][: self.K - 2])
        q = ttnn.multiply(self._l2(self._heads(qkv[0], 1)), self.dk**-0.5)  # [B,hl,1,dk]
        k = self._l2(self._heads(qkv[1], 1))
        v = self._heads(qkv[2], 1)
        gh = self._heads(_rs(g, [B, 1, 1, W]), 1)  # [B,hl,1,dk]
        bt = [_rs(b_, [B, 1, 1, 1]) for b_ in beta]
        bh = bt[0] if len(bt) == 1 else ttnn.concat(bt, dim=1)  # [B,hl,1,1]
        state = ttnn.multiply(state, ttnn.transpose(ttnn.exp(gh), -2, -1))  # decay per key channel
        kv_mem = c.mm(k, state)  # [B,hl,1,dv]
        delta = ttnn.multiply(ttnn.subtract(v, kv_mem), bh)
        state = ttnn.add(state, c.mm(ttnn.transpose(k, -2, -1), delta))
        o = c.mm(q, state)  # [B,hl,1,dv]
        return self._out(o, gate, B, 1), (state, new_tails)

    def __call__(self, h, S, mask=None):
        """Single-sequence prefill (the graduated entry point): h [1,1,S,D] fp32."""
        if mask is not None:
            h = ttnn.multiply(h, mask)
        out, _ = self.prefill(h, 1, S)
        return out


# --------------------------------------------------------------------------- dense MLP
class _MLP:
    def __init__(self, ctx, m):
        self.ctx = ctx
        if m.intermediate_size % ctx.n:
            raise ValueError("intermediate_size not divisible by TP")
        self.w_gate = ctx.put(_wT(m.gate_proj.weight), shard_dim=-1)
        self.w_up = ctx.put(_wT(m.up_proj.weight), shard_dim=-1)
        self.w_down = ctx.put(_wT(m.down_proj.weight), shard_dim=0)
        self.limit = float(m.swiglu_limit)

    def __call__(self, x):
        c = self.ctx
        xb = ttnn.typecast(x, ttnn.bfloat16)
        gate = ttnn.clamp(c.lin(xb, self.w_gate), max=self.limit)
        up = ttnn.clamp(c.lin(xb, self.w_up), min=-self.limit, max=self.limit)
        hmid = ttnn.typecast(ttnn.multiply(ttnn.silu(gate), up), ttnn.bfloat16)
        out = c.lin(hmid, self.w_down, dtype=ttnn.bfloat16)
        return ttnn.typecast(c.all_reduce(out), ttnn.float32)


# --------------------------------------------------------------------------- decoder layer
def _tile(x, shape):
    """ROW_MAJOR stub output -> TILE tensor of ``shape``."""
    return ttnn.to_layout(ttnn.reshape(x, shape), ttnn.TILE_LAYOUT)


class TtTextDecoderLayer:
    """One GLM-5.3-Flash decoder layer composed ONLY of the graduated leaf ports:

        post, comb, x = text_hyper_connection(H)          (input norm = text_unweighted_r_m_s_norm)
        a = text_linear_attention | text_attention(+text_indexer) ( text_r_m_s_norm(x) )
        H = post (x) a + comb^T @ H
        post, comb, x = text_hyper_connection(H)
        m = text_m_l_p | text_mo_e(text_topk_router -> text_experts, + text_m_l_p shared) ( text_r_m_s_norm(x) )
        H = post (x) m + comb^T @ H

    Residual streams are [1, R, hc_mult, D] fp32 with R = batch * tokens (sample-major rows): every
    token-wise op runs over all rows at once; only attention sees the [B, S] structure.
    """

    def __init__(self, ctx, m):
        from models.demos.glm_5_3_flash._stubs.text_attention import TtTextAttention
        from models.demos.glm_5_3_flash._stubs.text_hyper_connection import TtTextHyperConnection
        from models.demos.glm_5_3_flash._stubs.text_linear_attention import TtTextLinearAttention
        from models.demos.glm_5_3_flash._stubs.text_m_l_p import TtTextMLP
        from models.demos.glm_5_3_flash._stubs.text_mo_e import TtTextMoE
        from models.demos.glm_5_3_flash._stubs.text_r_m_s_norm import TtTextRMSNorm

        self.ctx = ctx
        dev = ctx.device
        self.kind = m.block_type
        self.D = int(m.hidden_size)
        self.attn_hc = TtTextHyperConnection(dev, m.attn_hc)
        self.ffn_hc = TtTextHyperConnection(dev, m.ffn_hc)
        self.input_layernorm = TtTextRMSNorm.build(dev, m.input_layernorm)
        self.post_attention_layernorm = TtTextRMSNorm.build(dev, m.post_attention_layernorm)
        if self.kind == "linear_attention":
            self.self_attn = TtTextLinearAttention(ctx, m.self_attn)
        else:
            if getattr(m.self_attn, "skip_topk", False):
                raise NotImplementedError("shared-indexer (skip_topk) DSA layers not used by this checkpoint")
            self.self_attn = TtTextAttention.build(dev, m.self_attn)
        self.is_moe = type(m.mlp).__name__ == "Glm5NextTextMoE"
        self.mlp = TtTextMoE.build(dev, m.mlp) if self.is_moe else TtTextMLP(ctx, m.mlp)

    @classmethod
    def build(cls, device, torch_module):
        return cls(_Ctx(device), torch_module)

    def _expand(self, post, comb, sub, Xs, R):
        """out[h] = post[h] * sub + sum_k comb[k][h] * X[k] -> [1,H,R,D] fp32 (stream-major), as exact fp32
        broadcasts (K = 1 and K = hc_mult are far too small for a matmul, which would also round the residual
        stream to ~bf16)."""
        H, D = Xs.shape[1], Xs.shape[3]
        prm = ttnn.to_layout(post, ttnn.ROW_MAJOR_LAYOUT)
        crm = ttnn.to_layout(comb, ttnn.ROW_MAJOR_LAYOUT)
        col = lambda t, j: ttnn.to_layout(ttnn.slice(t, [0, 0, 0, j], [1, 1, R, j + 1]), ttnn.TILE_LAYOUT)
        xs = [ttnn.slice(Xs, [0, k, 0, 0], [1, k + 1, R, D]) for k in range(H)]
        outs = []
        for h in range(H):
            o = ttnn.multiply(col(prm, h), sub)
            for k in range(H):
                o = ttnn.add(o, ttnn.multiply(col(crm, k * H + h), xs[k]))
            outs.append(o)
        return ttnn.concat(outs, dim=1)

    def forward(self, Xs, step):
        """Xs [1,H,R,D] fp32 stream-major residual streams. ``step`` describes the call: step.mode in
        {"prefill", "decode"}, step.B, step.S (tokens per sample this call), plus the layer's resident state in
        step.state[id(self)]."""
        R = Xs.shape[2]
        post, comb, x = self.attn_hc.streams(Xs)
        h = ttnn.typecast(self.input_layernorm(x), ttnn.float32)
        if self.kind == "linear_attention":
            if step.mode == "prefill":
                a, st = self.self_attn.prefill(h, step.B, step.S, step.valid, step.real_len)
            else:
                a, st = self.self_attn.decode(h, step.state[id(self)])
        else:
            hb = ttnn.typecast(h, ttnn.bfloat16)
            if step.mode == "prefill":
                a, st = self.self_attn.prefill(hb, step.B, step.S, step.real_len, step.capacity)
            else:
                a, st = self.self_attn.decode(hb, step.state[id(self)], step.pos)
            a = ttnn.typecast(a, ttnn.float32)
        if step.mode == "decode" and step.inplace:
            _copy_tree(st, step.state[id(self)])  # resident buffers keep their addresses (trace replay)
        else:
            step.state[id(self)] = st
        Xs = self._expand(post, comb, a, Xs, R)

        post, comb, x = self.ffn_hc.streams(Xs)
        h = ttnn.typecast(self.post_attention_layernorm(x), ttnn.float32)
        mo = self.mlp(h)
        return self._expand(post, comb, ttnn.typecast(mo, ttnn.float32), Xs, R)

    def __call__(self, hidden_states, attention_mask=None, past_key_values=None, **kwargs):
        """Graduated single-sequence entry: hidden_states [1,S,hc_mult,D] (no cache) -> same shape."""
        if past_key_values is not None:
            raise NotImplementedError("use forward(step) for the cached path")
        X = hidden_states
        if X.layout != ttnn.TILE_LAYOUT:
            X = ttnn.to_layout(X, ttnn.TILE_LAYOUT)
        X = ttnn.typecast(X, ttnn.float32)
        B, S, Hc, D = X.shape
        rm = lambda t: ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        Xs = ttnn.to_layout(ttnn.permute(ttnn.reshape(rm(X), [1, B * S, Hc, D]), (0, 2, 1, 3)), ttnn.TILE_LAYOUT)
        out = self.forward(Xs, Step("prefill", B, S))  # [1,Hc,B*S,D]
        return ttnn.to_layout(ttnn.reshape(ttnn.permute(rm(out), (0, 2, 1, 3)), [B, S, Hc, D]), ttnn.TILE_LAYOUT)


def _copy_tree(src, dst):
    """ttnn.copy every tensor of a (nested tuple/list) state into the matching resident tensor, back to front:
    the KDA conv tail is a shift register (new[i] = old[i-1]), so the copies must not overwrite a source first."""
    if isinstance(src, (tuple, list)):
        for a, b in reversed(list(zip(src, dst))):
            _copy_tree(a, b)
    else:
        ttnn.copy(src, dst)


class Step:
    """What one forward call is: prefill of B x S tokens or one decode token per sample. ``inplace`` decode
    writes the new KDA / DSA state into the existing resident tensors instead of rebinding them."""

    def __init__(self, mode, B, S, valid=None, real_len=None, capacity=None, pos=None, state=None, inplace=False):
        self.mode, self.B, self.S = mode, B, S
        self.inplace = inplace
        self.valid, self.real_len = valid, (S if real_len is None else real_len)
        self.capacity = capacity if capacity is not None else S
        self.pos = pos
        self.state = {} if state is None else state


def build(device, torch_module=None):
    return TtTextDecoderLayer.build(device, torch_module)


def text_decoder_layer(device, torch_module=None):
    return TtTextDecoderLayer.build(device, torch_module)

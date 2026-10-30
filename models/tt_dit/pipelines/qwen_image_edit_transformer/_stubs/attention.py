# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN, tensor-parallel port of QwenImage joint (dual-stream) attention
(`transformer_blocks.N.attn`, diffusers `QwenDoubleStreamAttnProcessor2_0`).

    img: q,k,v = to_q/k/v(hidden_states)            txt: q,k,v = add_q/k/v_proj(encoder_hidden_states)
    q,k        = RMSNorm_head(q,k)  -> RoPE(complex, interleaved pairs)
    joint      = SDPA(cat[txt, img] q/k/v)           (non-causal, optional key mask)
    img_out    = to_out[0](joint[img])               txt_out = to_add_out(joint[txt])

Tensor parallel (TP = mesh size, or the size of the mesh axis chosen with _ccl.tp_axis): heads are
split across chips.
  * q/k/v and add_q/k/v projections are COLUMN-parallel (output features = whole heads per chip).
  * to_out / to_add_out are ROW-parallel (input features = this chip's heads) followed by all_reduce;
    their biases are added once, after the reduce.
  * RMSNorm weights and rotary tables are replicated.

Numerics. The QK-RMSNorm weights of this checkpoint are large (up to ~65), so attention logits reach
~4e4 and softmax is nearly an argmax: rounding normalized q/k to bf16 alone drops PCC to ~0.93 and a
TF32 matmul to ~0.988. So everything from the q/k projection outputs through the logits stays in
float32 on the SFPU (elementwise), and QK^T is computed as a 3-term bf16 hi/lo split with float32
accumulation (qh.kh + qh.kl + ql.kh), which carries ~16 mantissa bits. P@V and the output
projections are insensitive and run as bf16 matmuls with float32 accumulation.

RoPE without a pair-swap: q/k projection output features are permuted per head from interleaved
(x0,x1,x2,x3,...) to half-split (x0,x2,...,x1,x3,...) order at build time (norm weights likewise).
The dot product q.k is invariant under a shared permutation, and in half-split order the complex
rotation is pure slicing + elementwise math. V is not permuted, so the attention output layout is
unchanged.

Rotary input convention (the harness stages complex tensors this way): each freqs tensor is a
float32 ttnn tensor [S, 2*(D/2)] = [cos | sin].
"""
from __future__ import annotations

import torch

import ttnn
from models.tt_dit.pipelines.qwen_image_edit_transformer._stubs import _ccl, _precise


def _is_mesh(device):
    return isinstance(device, ttnn.MeshDevice) and device.get_num_devices() > 1


def _replicated(t, device, dtype=ttnn.float32):
    kw = {"mesh_mapper": ttnn.ReplicateTensorToMesh(device)} if isinstance(device, ttnn.MeshDevice) else {}
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, **kw)


def _sharded(t, device, dim, dtype=ttnn.bfloat16):
    if _is_mesh(device):
        return ttnn.from_torch(
            t.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=_ccl.shard_mapper(device, dim),  # TP axis: see _ccl.tp_axis
        )
    return _replicated(t, device, dtype=dtype)


def _as_tt(t, device, dtype=ttnn.float32):
    """Inputs normally arrive on device already; stage a host tensor if one does not."""
    if t is None or isinstance(t, ttnn.Tensor):
        return t
    if t.is_complex():
        t = torch.cat([t.real, t.imag], dim=-1)
    return _replicated(t.to(torch.float32), device, dtype=dtype)


class TtQwenJointAttention:
    def __init__(self, device, torch_module):
        self.device = device
        m = torch_module
        self.n_heads = int(m.heads)
        inner = m.to_q.weight.shape[0]
        self.head_dim = inner // self.n_heads
        self.tp_axis = _ccl.get_tp_axis()  # mesh axis of the TP split (None = every device)
        self.tp = _ccl.tp_size(device, self.tp_axis)
        assert self.n_heads % self.tp == 0, f"{self.n_heads} heads not divisible by TP={self.tp}"
        self.local_heads = self.n_heads // self.tp
        self.half = self.head_dim // 2

        hd = self.head_dim
        in_head = torch.cat([torch.arange(0, hd, 2), torch.arange(1, hd, 2)])  # interleaved -> half-split
        perm = torch.cat([h * hd + in_head for h in range(self.n_heads)])

        def w_col(lin, permute):
            w = lin.weight.detach().to(torch.float32)
            b = lin.bias.detach().to(torch.float32) if lin.bias is not None else torch.zeros(w.shape[0])
            if permute:
                w, b = w[perm], b[perm]
            return _sharded(w.t(), device, dim=-1), _sharded(b.reshape(1, -1), device, dim=-1, dtype=ttnn.float32)

        def w_qkv(lq, lk, lv):
            # one column-parallel weight per chip: [this chip's q | k | v] columns, so a single (limb) matmul
            # reads the activation and the three weights once
            ws, bs = [], []
            for lin, permute in ((lq, True), (lk, True), (lv, False)):
                w = lin.weight.detach().to(torch.float32)
                b = lin.bias.detach().to(torch.float32) if lin.bias is not None else torch.zeros(w.shape[0])
                if permute:
                    w, b = w[perm], b[perm]
                ws.append(w.t().reshape(w.shape[1], self.tp, -1))
                bs.append(b.reshape(1, self.tp, -1))
            w = torch.stack(ws, dim=2).reshape(ws[0].shape[0], -1)
            b = torch.stack(bs, dim=2).reshape(1, -1)
            return _sharded(w, device, dim=-1), _sharded(b, device, dim=-1, dtype=ttnn.float32)

        self.fused_qkv = FUSED_QKV
        if self.fused_qkv:
            self.img_qkv = w_qkv(m.to_q, m.to_k, m.to_v)
            self.txt_qkv = w_qkv(m.add_q_proj, m.add_k_proj, m.add_v_proj)
        else:
            self.img_q = w_col(m.to_q, True)
            self.img_k = w_col(m.to_k, True)
            self.img_v = w_col(m.to_v, False)
            self.txt_q = w_col(m.add_q_proj, True)
            self.txt_k = w_col(m.add_k_proj, True)
            self.txt_v = w_col(m.add_v_proj, False)

        def w_row(lin):
            w = lin.weight.detach().to(torch.float32)
            b = lin.bias.detach().to(torch.float32) if lin.bias is not None else torch.zeros(w.shape[0])
            return _sharded(w.t(), device, dim=-2), _replicated(b.reshape(1, -1), device)

        self.img_out = w_row(m.to_out[0])
        self.txt_out = w_row(m.to_add_out)

        scale = 1.0 / (hd**0.5)

        def norm_w(norm, q_scale):
            if norm is None:
                w = torch.ones(hd)
            else:
                w = norm.weight.detach().to(torch.float32) if norm.weight is not None else torch.ones(hd)
            w = w[in_head] * q_scale  # softmax scale folded into the query norm weight
            self._nw_host.append(w)
            return _replicated(w.reshape(1, 1, 1, hd), device)

        self.eps = float(getattr(m.norm_q, "eps", 1e-6) or 1e-6)
        self._nw_host = []  # img q, img k, txt q, txt k: host copies for joint_norm_tables
        self.nw_img_q = norm_w(m.norm_q, scale)
        self.nw_img_k = norm_w(m.norm_k, 1.0)
        self.nw_txt_q = norm_w(m.norm_added_q, scale)
        self.nw_txt_k = norm_w(m.norm_added_k, 1.0)
        # (L, S) -> the joint-row (q, k) norm weights of JOINT_HEADS: on device, and built on host awaiting upload
        self._nw_joint = {}
        self._nw_joint_host = {}

        self.hifi = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    # ---- helpers -------------------------------------------------------------------------------
    def _proj(self, x, wb, dtype=ttnn.float32, parts=None):
        w, b = wb
        if _precise.ENABLED:
            return _precise.linear(x, w, bias=b, parts=parts)
        return ttnn.linear(x, w, bias=b, dtype=dtype, compute_kernel_config=self.hifi)

    def _heads(self, x, off=0):  # [B, S, Hl*D] (at column off of x) -> [B, Hl, S, D]
        B, S = x.shape[0], x.shape[1]
        if HEADS_BY_SLICES:  # one tile-aligned column block per head, stacked on dim 1 (no padded permute)
            D = self.head_dim
            parts = [
                ttnn.reshape(ttnn.slice(x, (0, 0, off + h * D), (B, S, off + (h + 1) * D)), (B, 1, S, D))
                for h in range(self.local_heads)
            ]
            return parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=1)
        if off or x.shape[-1] != self.local_heads * self.head_dim:
            x = ttnn.slice(x, (0, 0, off), (B, S, off + self.local_heads * self.head_dim))
        if HEADS_INPUT_L1:
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        x = ttnn.reshape(x, (B, S, self.local_heads, self.head_dim))
        return ttnn.permute(x, (0, 2, 1, 3))

    def _rms(self, x, w):  # float32, elementwise on SFPU
        if QK_NORM_L1:  # the same ops, x * x and x * r staged in L1 (freed as soon as they are read)
            l1 = ttnn.L1_MEMORY_CONFIG
            xx = ttnn.multiply(x, x, memory_config=l1)
            ms = ttnn.mean(xx, dim=-1, keepdim=True, compute_kernel_config=self.hifi, memory_config=l1)
            ttnn.deallocate(xx)
            r = ttnn.rsqrt(ttnn.add(ms, self.eps, memory_config=l1), memory_config=l1)
            xr = ttnn.multiply(x, r, memory_config=l1)
            out = ttnn.multiply(xr, w, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(xr)
            return out
        ms = ttnn.mean(ttnn.multiply(x, x), dim=-1, keepdim=True, compute_kernel_config=self.hifi)
        r = ttnn.rsqrt(ttnn.add(ms, self.eps))
        return ttnn.multiply(ttnn.multiply(x, r), w)

    def _rope(self, x, freqs):  # x [B, Hl, S, D] half-split; freqs [S, D] = [cos | sin]
        if freqs is None:
            return x
        if ROPE_SWAP:  # x * [cos | cos] + [x2 | x1] * [-sin | sin]: the same fp32 products and sums
            cc, ss = _rope_tables(freqs, self.half)
            B, H, S = x.shape[0], x.shape[1], x.shape[2]
            kw = {"memory_config": ttnn.L1_MEMORY_CONFIG} if QK_NORM_L1 else {}
            x1 = ttnn.slice(x, (0, 0, 0, 0), (B, H, S, self.half), **kw)
            x2 = ttnn.slice(x, (0, 0, 0, self.half), (B, H, S, 2 * self.half), **kw)
            sw = ttnn.concat([x2, x1], dim=-1, **kw)
            if QK_NORM_L1:  # the halves and their swap in L1, freed once read
                ttnn.deallocate(x1)
                ttnn.deallocate(x2)
                xc = ttnn.multiply(x, cc)
                sws = ttnn.multiply(sw, ss)
                ttnn.deallocate(sw)
                return ttnn.add(xc, sws)
            return ttnn.add(ttnn.multiply(x, cc), ttnn.multiply(sw, ss))
        S = freqs.shape[0]
        f = ttnn.reshape(freqs, (1, 1, S, 2 * self.half))
        cos = ttnn.slice(f, (0, 0, 0, 0), (1, 1, S, self.half))
        sin = ttnn.slice(f, (0, 0, 0, self.half), (1, 1, S, 2 * self.half))
        B, H = x.shape[0], x.shape[1]
        x1 = ttnn.slice(x, (0, 0, 0, 0), (B, H, S, self.half))
        x2 = ttnn.slice(x, (0, 0, 0, self.half), (B, H, S, 2 * self.half))
        o1 = ttnn.subtract(ttnn.multiply(x1, cos), ttnn.multiply(x2, sin))
        o2 = ttnn.add(ttnn.multiply(x1, sin), ttnn.multiply(x2, cos))
        return ttnn.concat([o1, o2], dim=-1)

    def joint_norm_tables(self, L, S):
        """Build (on host) JOINT_HEADS's (q, k) RMSNorm weights per joint row [1, 1, L + S, D] -- the text weight on
        the first L rows, the image weight on the rest -- for a (text, image) length pair known before the forward;
        the forward uploads them on first use."""
        if (L, S) not in self._nw_joint and (L, S) not in self._nw_joint_host:
            iq, ik, tq, tk = self._nw_host
            rows = lambda t, i: torch.cat([t.expand(L, -1), i.expand(S, -1)]).reshape(1, 1, L + S, -1).contiguous()
            self._nw_joint_host[(L, S)] = (rows(tq, iq), rows(tk, ik))

    def _joint_norm_w(self, L, S):
        """(q, k) RMSNorm weights per joint row, once per (L, S): joint_norm_tables' host tables uploaded, else built
        here on device (exact copies: 1.0 * w)."""
        key = (L, S)
        if key not in self._nw_joint and key in self._nw_joint_host:
            self._nw_joint[key] = tuple(_replicated(t, self.device) for t in self._nw_joint_host.pop(key))
        if key not in self._nw_joint:
            d, hd = self.device, self.head_dim
            ones = lambda n: ttnn.full((1, 1, n, hd), 1.0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=d)
            o_txt, o_img = ones(L), ones(S)
            self._nw_joint[key] = tuple(
                ttnn.concat([ttnn.multiply(o_txt, t), ttnn.multiply(o_img, i)], dim=2)
                for t, i in ((self.nw_txt_q, self.nw_img_q), (self.nw_txt_k, self.nw_img_k))
            )
        return self._nw_joint[key]

    @staticmethod
    def _split(x):  # float32 -> (hi, lo) bf16 with x ~= hi + lo
        hi = ttnn.typecast(x, ttnn.bfloat16)
        lo = ttnn.typecast(ttnn.subtract(x, ttnn.typecast(hi, ttnn.float32)), ttnn.bfloat16)
        return hi, lo

    def _qk(self, q, kh, kl):  # 3-term split QK^T, float32 out
        qh, ql = self._split(q)
        mm = lambda a, b: ttnn.matmul(a, b, transpose_b=True, dtype=ttnn.float32, compute_kernel_config=self.hifi)
        return ttnn.add(ttnn.add(mm(qh, kh), mm(qh, kl)), mm(ql, kh))

    def _attend(self, q, kh, kl, v, mask_add, shared=None):
        """shared (precise + SHARE_SPLIT): (K's folded operand or None, V's (hi, lo)), built once for
        the image and text queries that attend to the same joint K / V."""
        k_rep, v_parts = shared if shared is not None else (None, None)
        if _precise.ENABLED:  # kh = k (float32) in precise mode
            if _precise.EXACT_QK:
                s = _precise.exact_matmul_bt(q, kh, b_rep=k_rep)
            else:
                s = _precise.matmul_bt(q, kh)
        else:
            s = self._qk(q, kh, kl)
        # Explicit softmax: the fused ttnn.softmax loses ~0.2 abs on these ~1e4 logits when the key
        # length is not tile-aligned; max/exp/sum/divide on float32 stays within ~2e-3.
        if SOFTMAX_FN is not None and SOFTMAX_SPLIT and _precise.ENABLED:
            # P is only ever split into its bf16 limbs for P @ V: the softmax kernel writes them (L1 like
            # MATMUL_A_L1's split) instead of P, and hands them to _precise.matmul
            mem = ttnn.L1_MEMORY_CONFIG if _precise.MATMUL_A_L1 else ttnn.DRAM_MEMORY_CONFIG
            p, parts = SOFTMAX_FN(s, mask_add, ttnn.DRAM_MEMORY_CONFIG, split_mem=mem)
            _precise.register_split(p, parts)
            ttnn.deallocate(s)
        elif SOFTMAX_FN is not None:  # the same chain as one kernel
            p = SOFTMAX_FN(s, mask_add, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(s)
        elif SOFTMAX_L1:  # the same ops, the logits staged in L1 and the subtract / exp done in place there
            l1 = ttnn.L1_MEMORY_CONFIG
            s_in = s
            s = ttnn.to_memory_config(s, l1) if mask_add is None else ttnn.add(s, mask_add, memory_config=l1)
            ttnn.deallocate(s_in)
            mx = ttnn.max(s, dim=-1, keepdim=True, memory_config=l1)
            ttnn.subtract(s, mx, output_tensor=s)
            ttnn.deallocate(mx)
            ttnn.exp(s, output_tensor=s)
            den = ttnn.sum(s, dim=-1, keepdim=True, compute_kernel_config=self.hifi, memory_config=l1)
            p = ttnn.divide(s, den, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(den)
            ttnn.deallocate(s)
        else:
            if mask_add is not None:
                s = ttnn.add(s, mask_add)
            mx = ttnn.max(s, dim=-1, keepdim=True)
            e = ttnn.exp(ttnn.subtract(s, mx))
            p = ttnn.divide(e, ttnn.sum(e, dim=-1, keepdim=True, compute_kernel_config=self.hifi))
        if _precise.ENABLED:
            o = _precise.matmul(p, v, b_parts=v_parts)  # float32 [B, Hl, Sq, D]
        else:
            o = ttnn.matmul(
                ttnn.typecast(p, ttnn.bfloat16), v, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi
            )  # [B, Hl, Sq, D]
        B, Sq = o.shape[0], o.shape[2]
        if NLP_MERGE:  # heads side by side along the width in one tile-copy op
            o = ttnn.experimental.nlp_concat_heads(o, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            return ttnn.reshape(o, (B, Sq, self.local_heads * self.head_dim))
        if HEADS_BY_SLICES:  # heads side by side along the width by slices + concat (no padded permute)
            D = self.head_dim
            parts = [ttnn.slice(o, (0, h, 0, 0), (B, h + 1, Sq, D)) for h in range(self.local_heads)]
            o = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=-1)
            return ttnn.reshape(o, (B, Sq, self.local_heads * D))
        o = ttnn.permute(o, (0, 2, 1, 3))
        return ttnn.reshape(o, (B, Sq, self.local_heads * self.head_dim))

    def _out(self, x, wb):  # row-parallel + all_reduce, bias once
        w, b = wb
        if _precise.ENABLED:
            y = _precise.linear(x, w)
        else:
            y = ttnn.linear(x, w, dtype=ttnn.float32, compute_kernel_config=self.hifi)
        if self.tp > 1:
            return _ccl.all_reduce(y, self.device, axis=self.tp_axis, bias=b)
        return ttnn.add(y, b)

    # ---- forward -------------------------------------------------------------------------------
    def __call__(
        self,
        hidden_states,
        encoder_hidden_states=None,
        encoder_hidden_states_mask=None,
        attention_mask=None,
        image_rotary_emb=None,
        need_txt=True,
    ):
        """need_txt False: the text rows' output projection is skipped (txt returned as None)."""
        d = self.device
        hs = _as_tt(hidden_states, d, ttnn.float32 if _precise.ENABLED else ttnn.bfloat16)
        ehs = _as_tt(encoder_hidden_states, d, ttnn.float32 if _precise.ENABLED else ttnn.bfloat16)
        vdt = ttnn.float32 if _precise.ENABLED else ttnn.bfloat16
        img_f, txt_f = (None, None) if image_rotary_emb is None else image_rotary_emb
        img_f, txt_f = _as_tt(img_f, d), _as_tt(txt_f, d)

        share = _precise.ENABLED and _precise.SHARE_SPLIT
        hp = _precise.split_bf16(hs) if share else None  # q/k/v of one stream share its hi/lo split
        ep = _precise.split_bf16(ehs) if share else None
        joint = None
        if self.fused_qkv and JOINT_HEADS and JOINT_QUERIES:  # one [txt; img] row block, heads / norm / rope once
            n = self.local_heads * self.head_dim
            yi = self._proj(hs, self.img_qkv, parts=hp)
            yt = self._proj(ehs, self.txt_qkv, parts=ep)
            L, S_img = yt.shape[1], yi.shape[1]
            hd, H = self.head_dim, self.local_heads
            qkv = None
            if JOINT_HEADS_FN is not None:  # each of q / k / v straight from the two projections, no join
                qkv = [JOINT_HEADS_FN(yt, yi, o, H, hd, ttnn.DRAM_MEMORY_CONFIG) for o in (0, n, 2 * n)]
                qkv = None if any(t is None for t in qkv) else qkv
            y = None if qkv is not None else ttnn.concat([yt, yi], dim=1)
            ttnn.deallocate(yi)
            ttnn.deallocate(yt)
            wq, wk = self._joint_norm_w(L, S_img)
            fj = _joint_freqs(txt_f, img_f)
            if qkv is not None:
                q, k, v = qkv
            elif NLP_SPLIT:  # q/k/v heads in one tile-copy op ([q | k | v] column blocks per chip)
                B, Sj = y.shape[0], y.shape[1]
                q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                    ttnn.reshape(y, (B, 1, Sj, 3 * n)),
                    num_heads=self.local_heads,
                    num_kv_heads=self.local_heads,
                    transpose_k_heads=False,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            else:
                q, k, v = self._heads(y), self._heads(y, n), self._heads(y, 2 * n)
            q = self._rope(self._rms(q, wq), fj)
            k = self._rope(self._rms(k, wk), fj)
            if vdt != ttnn.float32:
                v = ttnn.typecast(v, vdt)
            if y is not None:
                ttnn.deallocate(y)
            joint = q
        elif self.fused_qkv:  # one projection per stream; q/k/v are column blocks of its float32 output
            n = self.local_heads * self.head_dim
            yi = self._proj(hs, self.img_qkv, parts=hp)
            yt = self._proj(ehs, self.txt_qkv, parts=ep)
            qi = self._rope(self._rms(self._heads(yi), self.nw_img_q), img_f)
            ki = self._rope(self._rms(self._heads(yi, n), self.nw_img_k), img_f)
            vi = self._heads(yi, 2 * n)
            qt = self._rope(self._rms(self._heads(yt), self.nw_txt_q), txt_f)
            kt = self._rope(self._rms(self._heads(yt, n), self.nw_txt_k), txt_f)
            vt = self._heads(yt, 2 * n)
            if vdt != ttnn.float32:
                vi, vt = ttnn.typecast(vi, vdt), ttnn.typecast(vt, vdt)
            ttnn.deallocate(yi)
            ttnn.deallocate(yt)
        else:
            qi = self._rope(self._rms(self._heads(self._proj(hs, self.img_q, parts=hp)), self.nw_img_q), img_f)
            ki = self._rope(self._rms(self._heads(self._proj(hs, self.img_k, parts=hp)), self.nw_img_k), img_f)
            vi = self._heads(self._proj(hs, self.img_v, vdt, parts=hp))
            qt = self._rope(self._rms(self._heads(self._proj(ehs, self.txt_q, parts=ep)), self.nw_txt_q), txt_f)
            kt = self._rope(self._rms(self._heads(self._proj(ehs, self.txt_k, parts=ep)), self.nw_txt_k), txt_f)
            vt = self._heads(self._proj(ehs, self.txt_v, vdt, parts=ep))

        # Joint sequence order is [text, image].
        if joint is None:
            k = ttnn.concat([kt, ki], dim=2)
            v = ttnn.concat([vt, vi], dim=2)
        kh, kl = (k, None) if _precise.ENABLED else self._split(k)

        mask_add = None
        if attention_mask is not None:
            m = _as_tt(attention_mask, d)  # 1 = attend, 0 = masked; [B, 1, 1, S_joint]
            mask_add = ttnn.multiply(ttnn.subtract(m, 1.0), 1e30)

        shared = None
        if share:  # the image and text queries share the joint K's folded operand and V's limbs
            fold = _precise.EXACT_QK and _precise.FOLD_LANES and _precise.EXACT_QK_FN is None
            k_rep = _precise.fold_b(kh) if fold else None
            shared = (k_rep, _precise.split_bf16(v))
        if JOINT_QUERIES:  # one attention over the joint [text, image] queries; rows split after P @ V
            if joint is None:
                L = qt.shape[2]
                joint = ttnn.concat([qt, qi], dim=2)
            o = self._attend(joint, kh, kl, v, mask_add, shared)
            B, S, W = o.shape[0], o.shape[1], o.shape[2]
            o_txt = ttnn.slice(o, (0, 0, 0), (B, L, W))
            o_img = ttnn.slice(o, (0, L, 0), (B, S, W))
            ttnn.deallocate(o)
            img = self._out(o_img, self.img_out)
            txt = self._out(o_txt, self.txt_out) if need_txt else None
            if not need_txt:
                ttnn.deallocate(o_txt)
        else:
            img = self._out(self._attend(qi, kh, kl, v, mask_add, shared), self.img_out)
            txt = self._out(self._attend(qt, kh, kl, v, mask_add, shared), self.txt_out) if need_txt else None
        if shared is not None and shared[0] is not None:
            ttnn.deallocate(shared[0])
        return img, txt


# the head split's input is moved to L1 before its reshape (False: read from DRAM)
HEADS_INPUT_L1 = False
# heads in/out by per-head slices + concat instead of reshape + permute (which pads the head count to a tile)
HEADS_BY_SLICES = False
# q/k/v (and add_q/k/v) projections as one column-parallel weight per stream (read at build: set before building)
FUSED_QKV = False
# the image and text queries attend as one joint query block (one QK^T / softmax / P @ V) instead of two
JOINT_QUERIES = False
# with FUSED_QKV + JOINT_QUERIES: the two streams' q/k/v outputs joined as [txt; img] rows once, then one head split,
# RMSNorm (per-row weights) and RoPE (joint tables) over the joint rows instead of per stream + three joins
JOINT_HEADS = False
# the joint [txt | img] rotary table of the last (img_f, txt_f) pair (the pair held so its ids stay valid)
_FREQS_JOINT = [None, None, None]


def _joint_freqs(txt_f, img_f):
    if txt_f is None or img_f is None:
        return None
    c = _FREQS_JOINT
    if c[0] is not txt_f or c[1] is not img_f:
        c[:] = [txt_f, img_f, ttnn.concat([txt_f, img_f], dim=0)]
    return c[2]


# heads split (JOINT_HEADS path only) by ttnn.experimental.nlp_create_qkv_heads, merged by nlp_concat_heads (plain
# tile copies) instead of per-head slices + concat
NLP_SPLIT = False
# JOINT_HEADS's q / k / v as fn(txt, img, col_off, heads, head_dim, mem) -> [B, heads, L + S, head_dim] (or None:
# the join + split above), each read straight from the two streams' projections
JOINT_HEADS_FN = None
NLP_MERGE = False
# RoPE as x * [cos | cos] + [x2 | x1] * [-sin | sin] on tables built once per rotary tensor (bit-identical: a - b is
# a + (-b), and the second half's sum is the same two products in the other order)
ROPE_SWAP = False
# rotary tensor id -> (the tensor, held so its id stays valid; [cos | cos]; [-sin | sin])
_ROPE_TABLES = {}


def _rope_tables(freqs, half):
    hit = _ROPE_TABLES.get(id(freqs))
    if hit is None or hit[0] is not freqs:
        if len(_ROPE_TABLES) >= 8:
            _ROPE_TABLES.clear()
        S = freqs.shape[0]
        f = ttnn.reshape(freqs, (1, 1, S, 2 * half))
        cos = ttnn.slice(f, (0, 0, 0, 0), (1, 1, S, half))
        sin = ttnn.slice(f, (0, 0, 0, half), (1, 1, S, 2 * half))
        hit = (freqs, ttnn.concat([cos, cos], dim=-1), ttnn.concat([ttnn.neg(sin), sin], dim=-1))
        _ROPE_TABLES[id(freqs)] = hit
    return hit[1], hit[2]


# the attention softmax's float32 logits in L1 (subtract / exp in place), only the probabilities written to DRAM
SOFTMAX_L1 = False
# the q / k RMSNorm's x * x and x * r and the RoPE's halves and their swap in L1 (False: DRAM)
QK_NORM_L1 = False
# the attention softmax (mask add .. divide) as one bit-identical kernel: f(s, mask_add or None, mem) -> p
SOFTMAX_FN = None
# SOFTMAX_FN(..., split_mem=) -> (p, (hi, lo)): the probabilities' bf16 limbs for the precise P @ V made in the
# softmax pass (P itself is then not written); False: SOFTMAX_FN writes P
SOFTMAX_SPLIT = False


def build(device, torch_module=None):
    return TtQwenJointAttention(device, torch_module)


def attention(device, torch_module=None):
    return TtQwenJointAttention(device, torch_module)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN, tensor-parallel port of QwenImage joint (dual-stream) attention
(`transformer_blocks.N.attn`, diffusers `QwenDoubleStreamAttnProcessor2_0`).

    img: q,k,v = to_q/k/v(hidden_states)            txt: q,k,v = add_q/k/v_proj(encoder_hidden_states)
    q,k        = RMSNorm_head(q,k)  -> RoPE(complex, interleaved pairs)
    joint      = SDPA(cat[txt, img] q/k/v)           (non-causal, optional key mask)
    img_out    = to_out[0](joint[img])               txt_out = to_add_out(joint[txt])

Tensor parallel (TP = mesh size): heads are split across chips.
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

# Optional kernel for the head split of one projection: HEAD_SPLIT(x, heads, None, parts=1) -> x [B, S, H * D] as
# [B, H, S, D] (the reshape + permute below), and with merge=True the inverse; None for a tensor it does not take.
HEAD_SPLIT = None
# The caller discards this call's text-stream output (the last block's: the transformer's head reads only the image
# stream): the text queries' attention and out-projection are skipped and None is returned for txt. The text keys and
# values still enter the joint attention.
SKIP_TXT_OUT = False


def _is_mesh(device):
    return isinstance(device, ttnn.MeshDevice) and device.get_num_devices() > 1


def _replicated(t, device, dtype=ttnn.float32):
    kw = {"mesh_mapper": ttnn.ReplicateTensorToMesh(device)} if isinstance(device, ttnn.MeshDevice) else {}
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, **kw)


# The column-parallel out-projections' two gathers axis 0 first (see TtQwenJointAttention.__init__)
OUT_GATHER_COL_MAJOR = True
# The column-parallel out-projections' bias + gated residual update on the chip's output columns before the gather.
ATTN_SCATTER_RESID = True


def _sharded(t, device, dim, dtype=ttnn.bfloat16):
    if _is_mesh(device):
        return ttnn.from_torch(
            t.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=dim),
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
        self.tp = device.get_num_devices() if _is_mesh(device) else 1
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

        self.img_q = w_col(m.to_q, True)
        self.img_k = w_col(m.to_k, True)
        self.img_v = w_col(m.to_v, False)
        self.txt_q = w_col(m.add_q_proj, True)
        self.txt_k = w_col(m.add_k_proj, True)
        self.txt_v = w_col(m.add_v_proj, False)
        # precise TP: each stream's q / k / v projections as ONE column-parallel linear (every chip's block is its
        # heads' [q | k | v] columns): one exact product per lane instead of three, on a weight wide enough to fill
        # the grid, split into heads by HEAD_SPLIT's three-part mode. Fused on device at the first call (the
        # separate weights freed), so the resident weights allocate as before (their layout moves the other
        # stages' buffers).
        self.qkv_fused = _precise.ENABLED and self.tp > 1
        self.img_qkv = self.txt_qkv = None

        def w_row(lin):
            w = lin.weight.detach().to(torch.float32)
            b = lin.bias.detach().to(torch.float32) if lin.bias is not None else torch.zeros(w.shape[0])
            return _sharded(w.t(), device, dim=-2), _replicated(b.reshape(1, -1), device)

        # precise TP: the out-projections COLUMN-parallel instead (the heads gathered, this chip's output features
        # from the whole input, then gathered): an exact row-parallel product writes a full-width float32 partial
        # per lane product and gathers every chip's partial for the exact sum
        self.out_cols = _precise.ENABLED and self.tp > 1

        # both of its gathers axis 0 first (the slower axis then carries the 1/C-wide shard): the heads' gather lands
        # device (r, c)'s heads at block c * R + r, so the weight's K rows are laid out in that order; and each chip
        # computes the output block that lands in natural order after the output's gather
        self.out_order = _ccl.col_major_order(device) if OUT_GATHER_COL_MAJOR and self.out_cols else None

        def w_col_out(lin):  # column-sharded weight; the bias whole and replicated, added after the gather
            w = lin.weight.detach().to(torch.float32)
            b = lin.bias.detach().to(torch.float32) if lin.bias is not None else torch.zeros(w.shape[0])
            wt = w.t()
            order = self.out_order
            if order is not None:
                n = len(order)
                kb, nb = wt.shape[0] // n, wt.shape[1] // n
                # K rows: gathered position p holds device order[p]'s heads
                wt = torch.cat([wt[d * kb : (d + 1) * kb] for d in order], 0)
                # N columns: device d computes the block that lands at its gathered position (inverse of order)
                pos = [0] * n
                for p, d in enumerate(order):
                    pos[d] = p
                wt = torch.cat([wt[:, pos[d] * nb : (pos[d] + 1) * nb] for d in range(n)], 1)
            return _sharded(wt, device, dim=-1), _replicated(b.reshape(1, -1), device)

        w_out = w_col_out if self.out_cols else w_row
        self.img_out = w_out(m.to_out[0])
        self.txt_out = w_out(m.to_add_out)

        scale = 1.0 / (hd**0.5)

        def norm_w(norm, q_scale):
            if norm is None:
                w = torch.ones(hd)
            else:
                w = norm.weight.detach().to(torch.float32) if norm.weight is not None else torch.ones(hd)
            w = w[in_head] * q_scale  # softmax scale folded into the query norm weight
            return _replicated(w.reshape(1, 1, 1, hd), device)

        self.eps = float(getattr(m.norm_q, "eps", 1e-6) or 1e-6)
        self.nw_img_q = norm_w(m.norm_q, scale)
        self.nw_img_k = norm_w(m.norm_k, 1.0)
        self.nw_txt_q = norm_w(m.norm_added_q, scale)
        self.nw_txt_k = norm_w(m.norm_added_k, 1.0)

        self.hifi = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    # ---- helpers -------------------------------------------------------------------------------
    def _proj(self, x, wb, dtype=ttnn.float32):
        w, b = wb
        if _precise.ENABLED:
            return _precise.linear(x, w, bias=b)
        return ttnn.linear(x, w, bias=b, dtype=dtype, compute_kernel_config=self.hifi)

    @staticmethod
    def _fuse(wbs):
        """Three column-parallel (weight, bias) -> one: each chip's shards concatenated, [q | k | v]; frees them."""
        w = ttnn.concat([w for w, _ in wbs], dim=-1)
        b = ttnn.concat([b for _, b in wbs], dim=-1)
        for wt, bt in wbs:
            ttnn.deallocate(wt)
            ttnn.deallocate(bt)
        return w, b

    def _qkv_heads(self, x, wb):
        """The fused projection (see __init__) -> q, k, v [B, Hl, S, D]."""
        w, b = wb
        y = _precise.linear(x, w, bias=b)  # [B, S, 3 * Hl * D]
        B, S, C = y.shape[0], y.shape[1], y.shape[2]
        heads = None if HEAD_SPLIT is None else HEAD_SPLIT(ttnn.reshape(y, (B, 1, S, C)), self.local_heads, None)
        if heads is not None:
            return heads
        c = C // 3
        return [self._heads(ttnn.slice(y, [0, 0, i * c], [B, S, (i + 1) * c])) for i in range(3)]

    def _projs(self, x, wbs, dtypes):
        """[self._proj(x, wb, dt) ...]: in precise mode the exact path's input-side work done once for all of them."""
        if _precise.ENABLED:
            return _precise.linears(x, [w for w, _ in wbs], [b for _, b in wbs])
        return [self._proj(x, wb, dt) for wb, dt in zip(wbs, dtypes)]

    def _heads(self, x):  # [B, S, Hl*D] -> [B, Hl, S, D]
        y = None if HEAD_SPLIT is None else HEAD_SPLIT(x, self.local_heads, None, parts=1)
        if y is not None:
            return y
        B, S = x.shape[0], x.shape[1]
        x = ttnn.reshape(x, (B, S, self.local_heads, self.head_dim))
        return ttnn.permute(x, (0, 2, 1, 3))

    def _rms(self, x, w):  # float32, elementwise on SFPU
        ms = ttnn.mean(ttnn.multiply(x, x), dim=-1, keepdim=True, compute_kernel_config=self.hifi)
        r = ttnn.rsqrt(ttnn.add(ms, self.eps))
        return ttnn.multiply(ttnn.multiply(x, r), w)

    def _rope(self, x, freqs):  # x [B, Hl, S, D] half-split; freqs [S, D] = [cos | sin]
        if freqs is None:
            return x
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

    @staticmethod
    def _split(x):  # float32 -> (hi, lo) bf16 with x ~= hi + lo
        hi = ttnn.typecast(x, ttnn.bfloat16)
        lo = ttnn.typecast(ttnn.subtract(x, ttnn.typecast(hi, ttnn.float32)), ttnn.bfloat16)
        return hi, lo

    def _qk(self, q, kh, kl):  # 3-term split QK^T, float32 out
        qh, ql = self._split(q)
        mm = lambda a, b: ttnn.matmul(a, b, transpose_b=True, dtype=ttnn.float32, compute_kernel_config=self.hifi)
        return ttnn.add(ttnn.add(mm(qh, kh), mm(qh, kl)), mm(ql, kh))

    def _attend(self, q, kh, kl, v, mask_add):
        g = _precise.ATTEND_ENTRIES
        e = q.shape[0] * q.shape[1]
        if _precise.ENABLED and g and e > g and e % g == 0:
            o = self._attend_entries(q, kh, v, mask_add, g)
        else:
            o = self._attend_core(q, kh, kl, v, mask_add)
        y = None if HEAD_SPLIT is None else HEAD_SPLIT(o, self.local_heads, None, parts=1, merge=True)
        if y is not None:
            return y
        B, Sq = o.shape[0], o.shape[2]
        o = ttnn.permute(o, (0, 2, 1, 3))
        return ttnn.reshape(o, (B, Sq, self.local_heads * self.head_dim))

    def _attend_entries(self, q, kh, v, mask_add, g):
        """_attend_core over the (sample, head) entries g at a time: each group's batched products fill the grid
        with one block per core (all 96 entries at once leave the guarded QK^T lanes on 6 cores). Every entry's
        values are the ones the whole-batch call computes."""
        B, H, Sq, D = q.shape
        Sk = kh.shape[2]
        e = B * H
        qe, ke, ve = (ttnn.reshape(t, (e, 1, t.shape[2], D)) for t in (q, kh, v))
        me = None
        if mask_add is not None:  # [B, 1, 1, Sk] -> one row per entry
            me = ttnn.reshape(ttnn.repeat(mask_add, ttnn.Shape([1, H, 1, 1])), (e, 1, 1, Sk))
        outs = []
        for g0 in range(0, e, g):
            sl = lambda t: ttnn.slice(t, [g0, 0, 0, 0], [g0 + g, 1, t.shape[2], t.shape[3]])  # noqa: E731
            outs.append(self._attend_core(sl(qe), sl(ke), None, sl(ve), None if me is None else sl(me)))
        o = ttnn.concat(outs, dim=0)
        for t in outs:
            ttnn.deallocate(t)
        return ttnn.reshape(o, (B, H, Sq, D))

    def _attend_core(self, q, kh, kl, v, mask_add):
        """The attention of one group of (sample, head) entries, in query-row chunks when the whole group's
        scores do not fit _precise.CHUNK_ROWS_BYTES.

        The scores are [entries, Sq, Sk] and the guarded exact product holds several at once. At the 256 x 256
        area the PCC gate runs the joint sequence is 686 tokens and _attend_entries hands this a group of 16,
        whose scores are ~32 MB -- one chunk, the pass this has always run. At the 1024 x 1024 area the
        pipeline targets the sequence is 9600 tokens and the group is the whole batch, so one score matrix is
        2.1 GB: a full-resolution denoise step died inside this product on a 943,718,400 B buffer with 662 MB
        free. A softmax row reads only its own scores, so the rows can be cut."""
        chunk = _precise.score_rows(kh.shape[-2])
        align = _precise.SCORE_ROW_ALIGN
        if align > _precise.TILE:  # query rows per chunk a multiple of `align` (a whole number of grid rows)
            per = 1
            for s in list(q.shape)[:-2]:
                per *= s
            step = chunk // per // align * align
            if step >= align:
                chunk = step * per
        if _precise.row_cuts(q.shape, chunk, dim=-2) is None:
            return self._attend_rows(q, kh, kl, v, mask_add)
        # mask_add is [B, 1, 1, Sk]: one value per KEY, so every row chunk reads the same one
        if not (_precise.ENABLED and _precise.EXACT_QK and _precise.B_SIDE_REUSE):
            return _precise._row_chunked(lambda qc: self._attend_rows(qc, kh, kl, v, mask_add), q, chunk, dim=-2)
        # the keys' side of every chunk's exact QK^T (limbs, column norms, K-repeated and K-concatenated limbs) is
        # the same: formed by the first chunk, reused by the rest, freed here
        from models.demos.qwen_image_edit_text_encoder._stubs import attention as _te

        prev, _te.B_SIDE_CACHE = _te.B_SIDE_CACHE, {}
        try:
            return _precise._row_chunked(lambda qc: self._attend_rows(qc, kh, kl, v, mask_add), q, chunk, dim=-2)
        finally:
            _te.free_b_side(_te.B_SIDE_CACHE)
            _te.B_SIDE_CACHE = prev

    def _attend_rows(self, q, kh, kl, v, mask_add):
        if _precise.ENABLED:  # kh = k (float32) in precise mode
            s = _precise.exact_matmul_bt(q, kh) if _precise.EXACT_QK else _precise.matmul_bt(q, kh)
        else:
            s = self._qk(q, kh, kl)
        if mask_add is not None:
            s = ttnn.add(s, mask_add)
        # Explicit softmax: the fused ttnn.softmax loses ~0.2 abs on these ~1e4 logits when the key
        # length is not tile-aligned; max/exp/sum/divide on float32 stays within ~2e-3.
        mx = ttnn.max(s, dim=-1, keepdim=True)
        if (
            _precise.ENABLED
            and _precise.EXP_SPLIT is not None
            and _precise.SOFTMAX_FUSED_EXP
            and _precise.SOFTMAX_DEFER_DIV
            and not _precise.EXACT_PV
        ):
            # exp(s - max) and its P @ V limbs in one pass: e is written once and only read back by the row sum
            res = (
                _precise.EXP_SPLIT(s, mx, _precise.SOFTMAX_EXP_APPROX, rowsum=True)
                if _precise.EXP_ROWSUM
                else _precise.EXP_SPLIT(s, mx, _precise.SOFTMAX_EXP_APPROX)
            )
            if res is not None:
                ttnn.deallocate(s)
                ttnn.deallocate(mx)
                e, eh, el = res  # (with EXP_ROWSUM, e = the partial row sums [.., Sq, S*32], not exp(s - max))
                sm = ttnn.sum(e, dim=-1, keepdim=True, compute_kernel_config=self.hifi)
                ttnn.deallocate(e)
                o = _precise.matmul_presplit(eh, el, v)
                ttnn.deallocate(eh)
                ttnn.deallocate(el)
                o2 = ttnn.divide(o, sm)
                ttnn.deallocate(o)
                ttnn.deallocate(sm)
                return o2
        if _precise.SOFTMAX_FUSED_EXP:  # exp(s - max) in the subtract's own pass: one score-sized write, not two
            e = ttnn.subtract(
                s,
                mx,
                activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, 1.0 if _precise.SOFTMAX_EXP_APPROX else 0.0)],
            )
        else:
            e = ttnn.exp(ttnn.subtract(s, mx))
        if _precise.ENABLED and _precise.SOFTMAX_DEFER_DIV:
            # (e @ v) / rowsum(e) == (e / rowsum(e)) @ v: the division on the [.., Sq, D] output, not the scores
            sm = ttnn.sum(e, dim=-1, keepdim=True, compute_kernel_config=self.hifi)
            o = _precise.matmul(e, v)
            ttnn.deallocate(e)
            o2 = ttnn.divide(o, sm)
            ttnn.deallocate(o)
            ttnn.deallocate(sm)
            return o2
        p = ttnn.divide(e, ttnn.sum(e, dim=-1, keepdim=True, compute_kernel_config=self.hifi))
        if _precise.ENABLED:
            o = _precise.matmul(p, v)  # float32 [B, Hl, Sq, D]
        else:
            o = ttnn.matmul(
                ttnn.typecast(p, ttnn.bfloat16), v, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi
            )  # [B, Hl, Sq, D]
        return o

    def _out(self, x, wb, resid=None, gate=None):
        """row-parallel + all_reduce, bias once (or column-parallel, see __init__); with resid / gate: the block's
        gated residual update resid + gate * out -- for the column-parallel out-projection on this chip's output
        columns before the gather (ATTN_SCATTER_RESID: the bias add and the addcmul 1/n wide, the same per-element
        ops; the gather carries the updated residual, the same bytes)."""
        w, b = wb
        if self.out_cols:
            col_major = self.out_order is not None
            gather = _ccl.all_gather_col_major if col_major else _ccl.all_gather
            y = _precise.linear(gather(x, self.device, dim=-1), w)
            if resid is not None and ATTN_SCATTER_RESID:
                b_s = _ccl.partition(b, self.device, col_major)
                g_s = _ccl.partition(gate, self.device, col_major)
                r_s = _ccl.partition(resid, self.device, col_major)
                yb = ttnn.add(y, b_s)
                ttnn.deallocate(y)
                out_s = ttnn.addcmul(r_s, g_s, yb)
                for t in (yb, r_s, g_s, b_s):
                    ttnn.deallocate(t)
                out = gather(out_s, self.device, dim=-1)
                ttnn.deallocate(out_s)
                return out
            out = ttnn.add(gather(y, self.device, dim=-1), b)
            return out if resid is None else ttnn.addcmul(resid, gate, out)
        if _precise.ENABLED:
            y = _precise.linear(x, w)
        else:
            y = ttnn.linear(x, w, dtype=ttnn.float32, compute_kernel_config=self.hifi)
        if self.tp > 1:
            y = _ccl.all_reduce(y, self.device)
        out = ttnn.add(y, b)
        return out if resid is None else ttnn.addcmul(resid, gate, out)

    # ---- forward -------------------------------------------------------------------------------
    def __call__(
        self,
        hidden_states,
        encoder_hidden_states=None,
        encoder_hidden_states_mask=None,
        attention_mask=None,
        image_rotary_emb=None,
        resid=None,
        gate=None,
    ):
        """(img, txt) attention outputs; with resid=(img_res, txt_res), gate=(img_gate, txt_gate): the gated residual
        updates res + gate * out instead (see _out)."""
        d = self.device
        r_img, r_txt = (None, None) if resid is None else resid
        g_img, g_txt = (None, None) if gate is None else gate
        hs = _as_tt(hidden_states, d, ttnn.float32 if _precise.ENABLED else ttnn.bfloat16)
        ehs = _as_tt(encoder_hidden_states, d, ttnn.float32 if _precise.ENABLED else ttnn.bfloat16)
        vdt = ttnn.float32 if _precise.ENABLED else ttnn.bfloat16
        img_f, txt_f = (None, None) if image_rotary_emb is None else image_rotary_emb
        img_f, txt_f = _as_tt(img_f, d), _as_tt(txt_f, d)

        f32 = ttnn.float32
        if self.qkv_fused and self.img_qkv is None:
            self.img_qkv = self._fuse((self.img_q, self.img_k, self.img_v))
            self.txt_qkv = self._fuse((self.txt_q, self.txt_k, self.txt_v))
            self.img_q = self.img_k = self.img_v = self.txt_q = self.txt_k = self.txt_v = None
        if self.qkv_fused:
            hqi, hki, vi = self._qkv_heads(hs, self.img_qkv)
            hqt, hkt, vt = self._qkv_heads(ehs, self.txt_qkv)
        else:
            pqi, pki, pvi = self._projs(hs, (self.img_q, self.img_k, self.img_v), (f32, f32, vdt))
            hqi, hki, vi = self._heads(pqi), self._heads(pki), self._heads(pvi)
            pqt, pkt, pvt = self._projs(ehs, (self.txt_q, self.txt_k, self.txt_v), (f32, f32, vdt))
            hqt, hkt, vt = self._heads(pqt), self._heads(pkt), self._heads(pvt)
        qi = self._rope(self._rms(hqi, self.nw_img_q), img_f)
        ki = self._rope(self._rms(hki, self.nw_img_k), img_f)
        qt = self._rope(self._rms(hqt, self.nw_txt_q), txt_f)
        kt = self._rope(self._rms(hkt, self.nw_txt_k), txt_f)

        # Joint sequence order is [text, image].
        k = ttnn.concat([kt, ki], dim=2)
        v = ttnn.concat([vt, vi], dim=2)
        kh, kl = (k, None) if _precise.ENABLED else self._split(k)

        mask_add = None
        if attention_mask is not None:
            m = _as_tt(attention_mask, d)  # 1 = attend, 0 = masked; [B, 1, 1, S_joint]
            mask_add = ttnn.multiply(ttnn.subtract(m, 1.0), 1e30)

        img = self._out(self._attend(qi, kh, kl, v, mask_add), self.img_out, r_img, g_img)
        if SKIP_TXT_OUT:
            return img, None
        txt = self._out(self._attend(qt, kh, kl, v, mask_add), self.txt_out, r_txt, g_txt)
        return img, txt


def build(device, torch_module=None):
    return TtQwenJointAttention(device, torch_module)


def attention(device, torch_module=None):
    return TtQwenJointAttention(device, torch_module)

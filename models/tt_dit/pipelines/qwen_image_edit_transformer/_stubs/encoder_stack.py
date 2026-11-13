# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN, tensor-parallel port of the QwenImage block stack (`transformer_blocks`, all 60
`QwenImageTransformerBlock`s applied in order). Per block (diffusers QwenImageTransformerBlock):

    img_mod = Linear(SiLU(temb)) -> (shift1, scale1, gate1, shift2, scale2, gate2)   (txt likewise)
    img_m   = LN(img) * (1 + scale1) + shift1                                        (txt likewise)
    a_img, a_txt = joint_attention(img_m, txt_m)
    img    += gate1 * a_img ;  img += gate2 * MLP(LN(img) * (1 + scale2) + shift2)   (txt likewise)

Returns cat([txt, img], dim=1), matching the test's stack reference.

Tensor parallel (TP = mesh size):
  * img_mod / txt_mod: COLUMN-parallel (output features split) + all_gather, so every chip holds the
    full modulation vector.
  * attention: the graduated TP attention stub (heads split, row-parallel out + all_reduce).
  * FeedForward: proj (3072 -> 12288) COLUMN-parallel, GELU(tanh) local, out (12288 -> 3072)
    ROW-parallel + all_reduce, bias added once after the reduce.
  * LayerNorm (no affine), biases of row-parallel layers and the residual streams are replicated.

Numerics: the residual streams, LayerNorm and modulation stay in float32; matmul inputs are bf16
with float32 accumulation.
"""

from __future__ import annotations

import torch

import ttnn
from models.tt_dit.pipelines.qwen_image_edit_transformer._stubs import _ccl, _precise
from models.tt_dit.pipelines.qwen_image_edit_transformer._stubs.attention import TtQwenJointAttention
from models.tt_dit.pipelines.qwen_image_edit_transformer._stubs.feed_forward import TtQwenFeedForward

# Optional fused kernel for the LayerNorm variance: SUMSQ(xc) -> sum(xc * xc, dim=-1, keepdim=True) in float32 from one
# read of xc, or None for a tensor it does not take.
SUMSQ = None
# Optional fused kernel for the modulated LayerNorm's tail: MODULATE(xc, r, s, h) -> ((xc * r) * s) + h with
# r = 1/std across a tile [B, S, 32] and s = 1 + scale, h = shift down a tile [B, 32, D]; None for shapes it does not
# take. One pass instead of the multiply, multiply, add passes over the [B, S, D] float32 stream.
MODULATE = None
_ONES = {}


def _ones(device):
    """(ones [1, 32], ones [32, 1]) float32, built once per device (no host upload inside an observed forward)."""
    if id(device) not in _ONES:
        _ONES[id(device)] = (
            ttnn.Tensor([1.0] * 32, [1, 32], ttnn.float32, ttnn.TILE_LAYOUT, device),
            ttnn.Tensor([1.0] * 32, [32, 1], ttnn.float32, ttnn.TILE_LAYOUT, device),
        )
    return _ONES[id(device)]


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
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=dim),
        )
    return _replicated(t, device, dtype=dtype)


def _as_tt(t, device, dtype=ttnn.float32):
    if t is None or isinstance(t, ttnn.Tensor):
        return t
    if t.is_complex():
        t = torch.cat([t.real, t.imag], dim=-1)
    return _replicated(t.to(torch.float32), device, dtype=dtype)


class _TtBlock:
    def __init__(self, device, block, hifi, tp):
        self.device = device
        self.hifi = hifi
        self.tp = tp
        self.dim = block.dim
        self.eps = float(block.img_norm1.eps)
        self.attn = TtQwenJointAttention(device, block.attn)

        def col(lin, dtype_b=ttnn.float32):
            w = lin.weight.detach().to(torch.float32)
            b = lin.bias.detach().to(torch.float32)
            return _sharded(w.t(), device, dim=-1), _sharded(b.reshape(1, -1), device, dim=-1, dtype=dtype_b)

        self.img_mod = col(block.img_mod[1])
        self.txt_mod = col(block.txt_mod[1])
        # The graduated FeedForward port (column/row-parallel + all_reduce, bias once) for both streams.
        self.img_ff = TtQwenFeedForward(device, block.img_mlp)
        self.txt_ff = TtQwenFeedForward(device, block.txt_mlp)

    def _gather(self, y):
        if self.tp > 1:
            y = _ccl.all_gather(y, self.device, dim=-1)
        return y

    def _mod_params(self, silu_temb, wb):
        """[B, 6*dim] float32 -> six [B, 1, dim] tensors (shift1, scale1, gate1, shift2, scale2, gate2)."""
        w, b = wb
        if _precise.ENABLED:
            y = _precise.linear(silu_temb, w, bias=b)
        else:
            y = ttnn.linear(silu_temb, w, bias=b, dtype=ttnn.float32, compute_kernel_config=self.hifi)
        y = self._gather(y)  # [B, 6*dim] (column shards concatenate in order)
        B = y.shape[0]
        d = self.dim
        return [ttnn.reshape(ttnn.slice(y, (0, i * d), (B, (i + 1) * d)), (B, 1, d)) for i in range(6)]

    def _ln(self, x):  # LayerNorm without affine, float32
        mu = ttnn.mean(x, dim=-1, keepdim=True, compute_kernel_config=self.hifi)
        xc = ttnn.subtract(x, mu)
        ss = None if SUMSQ is None else SUMSQ(xc)
        if ss is not None:  # the variance from one read of xc (the squares never written)
            var = ttnn.multiply(ss, 1.0 / xc.shape[-1])
        else:
            var = ttnn.mean(ttnn.multiply(xc, xc), dim=-1, keepdim=True, compute_kernel_config=self.hifi)
        return ttnn.multiply(xc, ttnn.rsqrt(ttnn.add(var, self.eps)))

    def _modulate(self, x, shift, scale):
        if MODULATE is not None and len(x.shape) == 3:
            mu = ttnn.mean(x, dim=-1, keepdim=True, compute_kernel_config=self.hifi)
            xc = ttnn.subtract(x, mu)
            ss = None if SUMSQ is None else SUMSQ(xc)
            if ss is not None:
                var = ttnn.multiply(ss, 1.0 / xc.shape[-1])
            else:
                var = ttnn.mean(ttnn.multiply(xc, xc), dim=-1, keepdim=True, compute_kernel_config=self.hifi)
            ones_r, ones_c = _ones(x.device())
            r = ttnn.multiply(ttnn.rsqrt(ttnn.add(var, self.eps)), ones_r)  # [B, S, 32]
            s = ttnn.multiply(ttnn.add(scale, 1.0), ones_c)  # [B, 32, D]
            h = ttnn.multiply(shift, ones_c)  # [B, 32, D]
            y = MODULATE(xc, r, s, h)
            for t in (xc, r, s, h):
                ttnn.deallocate(t)
            if y is not None:
                return y
        return ttnn.add(ttnn.multiply(self._ln(x), ttnn.add(scale, 1.0)), shift)

    def __call__(self, img, txt, silu_temb, rotary, mask):
        i_sh1, i_sc1, i_g1, i_sh2, i_sc2, i_g2 = self._mod_params(silu_temb, self.img_mod)
        t_sh1, t_sc1, t_g1, t_sh2, t_sc2, t_g2 = self._mod_params(silu_temb, self.txt_mod)

        mdt = ttnn.float32 if _precise.ENABLED else ttnn.bfloat16
        img_m = ttnn.typecast(self._modulate(img, i_sh1, i_sc1), mdt)
        txt_m = ttnn.typecast(self._modulate(txt, t_sh1, t_sc1), mdt)
        # (the attention's gated residual updates x + g * y inside its call: on the out-projections' output columns
        # before their gathers -- attention.ATTN_SCATTER_RESID)
        img, txt_new = self.attn(
            img_m, txt_m, attention_mask=mask, image_rotary_emb=rotary, resid=(img, txt), gate=(i_g1, t_g1)
        )

        # the gated residual updates x + g * y as one ternary pass each (the product, then the sum)
        # (the FF's gated residual update on the reduced output's column slice, then gathered: feed_forward's resid / gate)
        img = self.img_ff(self._modulate(img, i_sh2, i_sc2), resid=img, gate=i_g2)
        if txt_new is None:  # the text stream's output is discarded (attention.SKIP_TXT_OUT): it is not updated
            return img, txt
        txt = txt_new
        txt = self.txt_ff(self._modulate(txt, t_sh2, t_sc2), resid=txt, gate=t_g2)
        return img, txt


class TtQwenBlockStack:
    def __init__(self, device, torch_module):
        self.device = device
        blocks = torch_module.blocks if hasattr(torch_module, "blocks") else torch_module
        self.tp = device.get_num_devices() if _is_mesh(device) else 1
        self.hifi = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        for blk in blocks:
            assert not getattr(blk, "zero_cond_t", False), "zero_cond_t (modulate_index) path not ported"
        self.blocks = [_TtBlock(device, blk, self.hifi, self.tp) for blk in blocks]

    def __call__(self, *args, **kwargs):
        img, txt = self.run_streams(*args, **kwargs)
        return ttnn.concat([txt, img], dim=1)

    def run_streams(
        self,
        hidden_states,
        encoder_hidden_states=None,
        encoder_hidden_states_mask=None,
        temb=None,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
        modulate_index=None,
    ):
        """Apply every block; returns the (img, txt) float32 streams without concatenating them."""
        assert modulate_index is None, "modulate_index path not ported"
        d = self.device
        img = _as_tt(hidden_states, d)
        txt = _as_tt(encoder_hidden_states, d)
        if img.dtype != ttnn.float32:
            img = ttnn.typecast(img, ttnn.float32)
        if txt.dtype != ttnn.float32:
            txt = ttnn.typecast(txt, ttnn.float32)

        temb = _as_tt(temb, d)
        if temb.dtype != ttnn.float32:
            temb = ttnn.typecast(temb, ttnn.float32)
        silu_temb = ttnn.silu(temb) if _precise.ENABLED else ttnn.typecast(ttnn.silu(temb), ttnn.bfloat16)

        rotary = None
        if image_rotary_emb is not None:
            rotary = tuple(_as_tt(f, d) for f in image_rotary_emb)

        mask = (joint_attention_kwargs or {}).get("attention_mask")
        mask = _as_tt(mask, d)
        if mask is None and encoder_hidden_states_mask is not None:
            # Same joint mask QwenImageTransformer2DModel builds: [text_mask, ones for image].
            m = encoder_hidden_states_mask
            if isinstance(m, ttnn.Tensor):
                m = ttnn.to_torch(m, mesh_composer=ttnn.ConcatMeshToTensor(d, dim=0))[: m.shape[0]]
            B, S_img = hidden_states.shape[0], hidden_states.shape[1]
            m = torch.cat([m.to(torch.float32), torch.ones(B, S_img)], dim=1)[:, None, None, :]
            mask = _as_tt(m, d)

        for blk in self.blocks:
            img, txt = blk(img, txt, silu_temb, rotary, mask)
        return img, txt


def build(device, torch_module=None):
    return TtQwenBlockStack(device, torch_module)


def encoder_stack(device, torch_module=None):
    return TtQwenBlockStack(device, torch_module)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN, tensor-parallel port of the QwenImage block stack (`transformer_blocks`, all 60
`QwenImageTransformerBlock`s applied in order). Per block (diffusers QwenImageTransformerBlock):

    img_mod = Linear(SiLU(temb)) -> (shift1, scale1, gate1, shift2, scale2, gate2)   (txt likewise)
    img_m   = LN(img) * (1 + scale1) + shift1                                        (txt likewise)
    a_img, a_txt = joint_attention(img_m, txt_m)
    img    += gate1 * a_img ;  img += gate2 * MLP(LN(img) * (1 + scale2) + shift2)   (txt likewise)

Returns cat([txt, img], dim=1), matching the test's stack reference.

Tensor parallel (TP = mesh size, or one mesh axis via _ccl.tp_axis):
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
    if t is None or isinstance(t, ttnn.Tensor):
        return t
    if t.is_complex():
        t = torch.cat([t.real, t.imag], dim=-1)
    return _replicated(t.to(torch.float32), device, dtype=dtype)


# the blocks of one forward share temb's SiLU and its bf16 limbs (split once into L1) for their modulation linears
SHARE_TEMB_LIMBS = False
_TEMB_CACHE = {}
# skip the modulated activations' typecast when they already have the target dtype (precise: float32 ->
# float32, a full copy)
SKIP_SAME_CAST = False
# fn(x, 1 + scale, shift, eps) -> LN(x) * (1 + scale) + shift in one call (e.g. a fused kernel), or None to run the
# ops; None: the stock LayerNorm + modulation chain
MODULATE_FN = None


class _TtBlock:
    def __init__(self, device, block, hifi, tp):
        self.device = device
        self.tp_axis = _ccl.get_tp_axis()
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
            y = _ccl.all_gather(y, self.device, dim=-1, axis=self.tp_axis)
        return y

    def _mod_params(self, silu_temb, wb, parts=None):
        """[B, 6*dim] float32 -> six [B, 1, dim] tensors (shift1, scale1, gate1, shift2, scale2, gate2)."""
        w, b = wb
        if _precise.ENABLED:
            y = _precise.linear(silu_temb, w, bias=b, parts=parts)
        else:
            y = ttnn.linear(silu_temb, w, bias=b, dtype=ttnn.float32, compute_kernel_config=self.hifi)
        y = self._gather(y)  # [B, 6*dim] (column shards concatenate in order)
        B = y.shape[0]
        d = self.dim
        return [ttnn.reshape(ttnn.slice(y, (0, i * d), (B, (i + 1) * d)), (B, 1, d)) for i in range(6)]

    def _ln(self, x):  # LayerNorm without affine, float32
        mu = ttnn.mean(x, dim=-1, keepdim=True, compute_kernel_config=self.hifi)
        xc = ttnn.subtract(x, mu)
        var = ttnn.mean(ttnn.multiply(xc, xc), dim=-1, keepdim=True, compute_kernel_config=self.hifi)
        return ttnn.multiply(xc, ttnn.rsqrt(ttnn.add(var, self.eps)))

    def _modulate(self, x, shift, scale):
        if MODULATE_FN is not None:
            y = MODULATE_FN(x, ttnn.add(scale, 1.0), shift, self.eps)
            if y is not None:
                return y
        return ttnn.add(ttnn.multiply(self._ln(x), ttnn.add(scale, 1.0)), shift)

    def __call__(self, img, txt, silu_temb, rotary, mask, temb_parts=None):
        i_sh1, i_sc1, i_g1, i_sh2, i_sc2, i_g2 = self._mod_params(silu_temb, self.img_mod, temb_parts)
        t_sh1, t_sc1, t_g1, t_sh2, t_sc2, t_g2 = self._mod_params(silu_temb, self.txt_mod, temb_parts)

        mdt = ttnn.float32 if _precise.ENABLED else ttnn.bfloat16
        cast = lambda t: t if SKIP_SAME_CAST and t.dtype == mdt else ttnn.typecast(t, mdt)  # noqa: E731
        img_m = cast(self._modulate(img, i_sh1, i_sc1))
        txt_m = cast(self._modulate(txt, t_sh1, t_sc1))

        # residual += gate * branch; a TP reduce may apply it on its own slice (_ccl.gated_residual), else here
        def gated(r, g, branch, entry):
            return entry.out if entry is not None and entry.out is not None else ttnn.add(r, ttnn.multiply(g, branch))

        with _ccl.gated_residual((i_g1, img), (t_g1, txt)) as e:
            a_img, a_txt = self.attn(img_m, txt_m, attention_mask=mask, image_rotary_emb=rotary)
        img = gated(img, i_g1, a_img, e[0] if e else None)
        txt = gated(txt, t_g1, a_txt, e[1] if e else None)

        with _ccl.gated_residual((i_g2, img)) as e:
            f = self.img_ff(self._modulate(img, i_sh2, i_sc2))
        img = gated(img, i_g2, f, e[0] if e else None)
        with _ccl.gated_residual((t_g2, txt)) as e:
            f = self.txt_ff(self._modulate(txt, t_sh2, t_sc2))
        txt = gated(txt, t_g2, f, e[0] if e else None)
        return img, txt


class TtQwenBlockStack:
    def __init__(self, device, torch_module):
        self.device = device
        blocks = torch_module.blocks if hasattr(torch_module, "blocks") else torch_module
        self.tp_axis = _ccl.get_tp_axis()  # mesh axis of the TP split (None = every device)
        self.tp = _ccl.tp_size(device, self.tp_axis)
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

        temb_arg = temb
        temb = _as_tt(temb, d)
        if temb.dtype != ttnn.float32:
            temb = ttnn.typecast(temb, ttnn.float32)
        temb_parts = None
        share = SHARE_TEMB_LIMBS and _precise.ENABLED and _precise.SPLIT_FN is not None
        if share and isinstance(temb_arg, ttnn.Tensor):
            # every block of a forward gets the same temb: its SiLU and bf16 limbs (in L1) once, not per block
            if _TEMB_CACHE.get("temb") is not temb_arg:
                _TEMB_CACHE.clear()
                silu = ttnn.silu(temb)
                _TEMB_CACHE.update(temb=temb_arg, silu=silu, parts=_precise.SPLIT_FN(silu, ttnn.L1_MEMORY_CONFIG))
            silu_temb, temb_parts = _TEMB_CACHE["silu"], _TEMB_CACHE["parts"]
        else:
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
            img, txt = blk(img, txt, silu_temb, rotary, mask, temb_parts)
        return img, txt


def build(device, torch_module=None):
    return TtQwenBlockStack(device, torch_module)


def encoder_stack(device, torch_module=None):
    return TtQwenBlockStack(device, torch_module)

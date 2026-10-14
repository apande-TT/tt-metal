# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native tensor-parallel TTNN port of `ministral3_decoder_layer` (Ministral3DecoderLayer,
`language_model.layers.0`) of `mistralai/Devstral-Small-2-24B-Instruct-2512`.

    h   = x + self_attn(input_layernorm(x))
    out = h + mlp(post_attention_layernorm(h))          mlp = down(silu(gate(n)) * up(n))

Composed from the graduated child ports, each holding its own weights:
  * input_layernorm / post_attention_layernorm  `ministral3_r_m_s_norm.py` (replicated weight)
  * self_attn  `ministral3_attention.py` (q/k/v column-parallel by head with GQA groups kept local,
               o_proj row-parallel + all_reduce; prefill fills / decode appends the resident KV cache)
  * mlp        `ministral3_m_l_p.py` (gate/up column-parallel, down row-parallel + all_reduce)
so the residual between layers is full-width and replicated on every device.

Layouts: prefill x (B, 1, S, D) [or (B, S, D)], decode x (1, 1, B, D). The output keeps the input's rank.
"""
from __future__ import annotations

import torch

import ttnn
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs.ministral3_attention import (
    TtMinistral3Attention,
    tp_ccl_args,
)
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs.ministral3_m_l_p import TtMinistral3MLP
from models.tt_transformers.demo.devstral_small_2_24b_instruct_2512._stubs.ministral3_r_m_s_norm import (
    TtMinistral3RMSNorm,
)


class TtMinistral3DecoderLayer:
    # The residual stream is carried in fp32: bf16 residual adds round with an upward bias (each layer's output
    # ran ~0.1% high in level on fp32 input), which compounded to a +0.8% level drift by layer 36 and cost the
    # image-token rows most. Norms read a bf16 copy; attention / MLP outputs arrive in fp32 (their OUT_DTYPE).
    RESIDUAL_DTYPE = ttnn.float32
    # Prefill on a TP mesh carries the residual HIDDEN-SHARDED (each device its 1/TP slice of the hidden dim, the
    # layout attention's and the MLP's reduce-scatter already produce): the fp32 residual adds and the RMS
    # norms run on the shard (see _sharded_norm), so only each norm's bf16 output is all_gathered for the next
    # projections. The model partitions the embeddings into the shard and gathers the final residual. A full-width
    # input takes the unsharded path.
    SHARD_PREFILL_RESIDUAL = True
    # False: cast the shard to bf16, all_gather it and run the unsharded HiFi4 norm (bit-identical to the unsharded
    # path, but the norm then reads and writes the full-width activation on every device).
    SHARDED_PREFILL_NORM = True
    # With the sharded norm: each prefill residual add also produces the next norm's per-row sum of squares (one C++
    # pass over x and the delta, tt/add_stats.py), so the norm does not read the new residual back from DRAM. The MLP
    # residual's sums are handed to the next layer's input norm through _CARRY (only when that layer receives the
    # very same tensor object; anything else recomputes them).
    FUSED_ADD_STATS = True
    _CARRY = None  # (residual tensor, its sum-of-squares stats) from the last layer's fused add

    def __init__(self, device, torch_module, meta_rope=False) -> None:
        """meta_rope: see TtMinistral3Attention (the caller supplies rotary tables in the matching order)."""
        self.device = device
        # Each norm's gamma is folded into the projections that read it (q/k/v; gate/up), so the norms apply no
        # weight and their outputs feed the matmuls without a separate gamma multiply.
        ln_in, ln_post = torch_module.input_layernorm, torch_module.post_attention_layernorm
        self.self_attn = TtMinistral3Attention(
            device, torch_module.self_attn, in_scale=ln_in.weight, meta_rope=meta_rope
        )
        self.tp = self.self_attn.tp
        self.input_layernorm = TtMinistral3RMSNorm.build(device, ln_in, fold_gamma=True)
        self.post_attention_layernorm = TtMinistral3RMSNorm.build(device, ln_post, fold_gamma=True)
        self.mlp = TtMinistral3MLP(device, torch_module.mlp, in_scale=ln_post.weight)
        self.compute_cfg = self.self_attn.compute_cfg
        self.hidden = int(ln_in.weight.shape[0])
        self._stats_mask = None
        if self.tp > 1 and self.SHARDED_PREFILL_NORM:
            # Picks column 0 of each device's sum-of-squares tile out of the gathered (.., S, 32 * TP) stats.
            mask = torch.zeros(1, 1, 1, ttnn.TILE_SIZE * self.tp)
            mask[..., :: ttnn.TILE_SIZE] = 1.0
            self._stats_mask = ttnn.from_torch(
                mask,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )

    @property
    def attn(self):
        return self.self_attn

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        position_embeddings=None,
        kv_cache=None,
        cur_pos=None,
        mode="prefill",
        **kwargs,
    ):
        x = hidden_states
        in_shape = tuple(x.shape)
        if len(in_shape) == 3:
            x = ttnn.reshape(x, (in_shape[0], 1, in_shape[1], in_shape[2]))

        if self.RESIDUAL_DTYPE is not None and x.dtype != self.RESIDUAL_DTYPE:
            x = ttnn.typecast(x, self.RESIDUAL_DTYPE)
        if mode != "decode" and self.tp > 1 and int(x.shape[-1]) * self.tp == self.hidden:
            out = self._prefill_sharded(x, attention_mask, position_ids, position_embeddings, kv_cache)
            return ttnn.reshape(out, in_shape) if len(in_shape) == 3 else out
        if mode == "decode":
            n = self.input_layernorm.decode(x)  # sharded norm reads the fp32 residual directly
        else:
            n = self._bf16(self.input_layernorm(self._bf16(x)))  # rms_norm reads bf16 (fp32 input overflows L1 at 5120)
        a = self.self_attn(
            n,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
            cur_pos=cur_pos,
            mode=mode,
        )
        h = self._residual_add(x, a)
        if mode == "decode":
            n = self.post_attention_layernorm.decode(h)
        else:
            n = self._bf16(self.post_attention_layernorm(self._bf16(h)))
        m = self.mlp(n, mode=mode)
        out = self._residual_add(h, m)
        if len(in_shape) == 3:
            return ttnn.reshape(out, in_shape)
        return out

    def _prefill_sharded(self, x, attention_mask, position_ids, position_embeddings, kv_cache):
        """Prefill with the residual x as this device's (B, 1, S, D/TP) hidden shard (see SHARD_PREFILL_RESIDUAL)."""
        carry, TtMinistral3DecoderLayer._CARRY = TtMinistral3DecoderLayer._CARRY, None
        x_sq = carry[1] if carry is not None and carry[0] is x else None
        self.self_attn.prefill_gather = self.mlp.prefill_gather = False
        try:
            a = self.self_attn(
                self._prefill_norm(self.input_layernorm, x, sq=x_sq),
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
                mode="prefill",
            )
            h, h_sq = self._residual_add_stats(x, a)
            m = self.mlp(self._prefill_norm(self.post_attention_layernorm, h, sq=h_sq), mode="prefill")
        finally:
            self.self_attn.prefill_gather = self.mlp.prefill_gather = True
        out, out_sq = self._residual_add_stats(h, m)
        if out_sq is not None:
            TtMinistral3DecoderLayer._CARRY = (out, out_sq)
        return out

    def _prefill_norm(self, norm, x, sq=None):
        """Full-width bf16 norm output of the hidden-sharded fp32 residual x (sq: its sum-of-squares stats, if known)."""
        if self.SHARDED_PREFILL_NORM:
            return self._sharded_norm(norm, x, sq)
        return self._bf16(norm(ttnn.all_gather(self._bf16(x), dim=3, **tp_ccl_args())))

    def _sharded_norm(self, norm, x, sq=None):
        """Unit-gamma RMSNorm of the fp32 hidden shard x: each device's sum of squares, an all_gather of those sums,
        the shard scaled by rsqrt(mean + eps) in fp32, and an all_gather of the bf16 result.

        rms_norm_pre_all_gather's fp32 (SFPU) path puts the exact sum in column 0 of its stats tile but leaves
        nonzero partials in columns 1..31 (measured: up to 15% of the sum), and rms_norm_post_all_gather reduces
        all 32 columns (and, on bf16 input, scales by a 1/N truncated to bf16), which is why ttnn's distributed norm failed the e2e
        test here. So the gathered stats are masked to each device's column 0 before they are summed, and 1/N,
        eps and rsqrt are applied in fp32."""
        assert norm.gamma is None, "the sharded prefill norm expects its gamma folded into the next projections"
        ccl = tp_ccl_args()
        cfg = norm._impl.compute_kernel_config_hifi2  # HiFi4 + fp32 dest, exact: the fp32 input takes the SFPU path
        if sq is None:
            sq = ttnn.rms_norm_pre_all_gather(x, compute_kernel_config=cfg, dtype=ttnn.float32)  # sum(x^2) in column 0
        sq = ttnn.multiply(ttnn.all_gather(sq, dim=3, **ccl), self._stats_mask)
        sq = ttnn.sum(sq, dim=-1, keepdim=True, compute_kernel_config=cfg)
        r = ttnn.rsqrt(ttnn.add(ttnn.multiply(sq, 1.0 / self.hidden), norm._impl.eps))
        return ttnn.all_gather(ttnn.multiply(x, r, dtype=ttnn.bfloat16), dim=3, **ccl)

    def _residual_add_stats(self, x, delta):
        """(x + delta, the sum-of-squares stats of the sum or None) for the prefill residual (see FUSED_ADD_STATS)."""
        if self.FUSED_ADD_STATS and self.SHARDED_PREFILL_NORM:
            from models.demos.devstral_small_2_24b_instruct_2512.tt import add_stats

            if add_stats.supports(x, delta):
                return add_stats.apply(x, delta)
        return self._residual_add(x, delta), None

    @staticmethod
    def _residual_add(x, delta):
        # A bf16 attention / MLP output is added into the fp32 residual without a cast op: binary ADD runs on the
        # SFPU in fp32 for mixed dtypes, so the bf16 operand is widened exactly.
        return ttnn.add(x, delta) if delta.dtype == x.dtype else ttnn.add(x, delta, dtype=x.dtype)

    @staticmethod
    def _as(t, dtype):
        return t if t.dtype == dtype else ttnn.typecast(t, dtype)

    def _bf16(self, t):
        return self._as(t, ttnn.bfloat16)


def build(device, torch_module=None):
    return TtMinistral3DecoderLayer.build(device, torch_module)


def ministral3_decoder_layer(device, torch_module=None):
    return TtMinistral3DecoderLayer.build(device, torch_module)

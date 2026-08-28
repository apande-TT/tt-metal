# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN, tensor-parallel port of the Qwen3 decoder block.

Component `decoder_layer` of `/tmp/tt_hw_planner_components/flux_2_klein_9b_text_encoder`
(`model.layers.0`, `Qwen3DecoderLayer`):

    h = x  + attn(input_layernorm(x))
    y = h  + mlp(post_attention_layernorm(h))

Shapes: hidden=4096, head_dim=128, n_heads=32, n_kv_heads=8 (GQA group 4),
intermediate=12288, SwiGLU, no projection biases, per-head q_norm/k_norm.

Tensor-parallel scheme (TP = number of mesh devices; 8 here)
------------------------------------------------------------
Two column-then-row pairs, one collective each:

* Attention. q/k/v_proj are COLUMN-parallel -- their outputs feed per-head
  work (RMSNorm, RoPE, softmax), so each chip owns n_heads/TP = 4 query
  heads and n_kv_heads/TP = 1 kv head. Query head `j` attends to kv head
  `j // 4`, so the contiguous split puts query heads 4i..4i+3 and kv head i
  on the same chip i: every chip holds a whole GQA group and the attention
  core needs no cross-chip traffic. o_proj is ROW-parallel over that same
  head axis and yields a PARTIAL sum -> one `ttnn.all_reduce`.
* MLP. gate_proj/up_proj are COLUMN-parallel (their outputs feed the SiLU
  gate elementwise, 12288/8 = 1536 columns per chip); down_proj is
  ROW-parallel over the intermediate axis -> a second `ttnn.all_reduce`.

Replicated (never sharded): both RMSNorm gammas and the q_norm/k_norm
gammas (they act on axes that are not split), the RoPE cos/sin tables, and
the residual stream -- which stays full-width on every chip because each
all_reduce restores it before the residual add.

The k and v projections are packed per chip as [K_i | V_i] so
`nlp_create_qkv_heads` can do the head split tile-natively (this build's
row-major reshape kernel does not compile, so reshapes that split the last
dim are avoided entirely).

Causal masking: HF dispatches this block's attention through
`sdpa_attention_forward`, which sets `is_causal = q_len > 1 and
attention_mask is None`. With no mask supplied -- the case the harness and
the captured real inputs both exercise -- the reference is therefore
CAUSAL, so this port applies a causal additive mask when no mask is given
and uses the caller's mask when one is.

Placement changes; the math does not.
"""
from __future__ import annotations

import torch

import ttnn

TILE = 32

_ACT = {
    "silu": ttnn.silu,
    "swish": ttnn.silu,
    "gelu": ttnn.gelu,
    "relu": ttnn.relu,
}


def _num_devices(device) -> int:
    fn = getattr(device, "get_num_devices", None)
    if callable(fn):
        try:
            return int(fn())
        except Exception:
            pass
    ids = getattr(device, "get_device_ids", None)
    if callable(ids):
        try:
            return max(1, len(ids()))
        except Exception:
            pass
    return 1


def _is_mesh(device) -> bool:
    try:
        if isinstance(device, ttnn.MeshDevice):
            return True
    except AttributeError:
        pass
    return hasattr(device, "get_device_ids") or hasattr(device, "get_devices")


class TtDecoderLayer:
    """Native ttnn Qwen3 decoder block, column/row-parallel over the mesh."""

    def __init__(self, device, torch_module) -> None:
        self.device = device
        self.mesh = _is_mesh(device)
        self.tp = _num_devices(device) if self.mesh else 1

        attn = torch_module.self_attn
        mlp = torch_module.mlp
        cfg = getattr(torch_module, "config", None) or getattr(attn, "config", None)

        sd = {k: v.detach().float() for k, v in torch_module.state_dict().items()}
        self.head_dim = int(getattr(attn, "head_dim", 0) or getattr(cfg, "head_dim", 0) or 128)
        self.scaling = float(getattr(attn, "scaling", self.head_dim**-0.5))

        wq = sd["self_attn.q_proj.weight"]
        wk = sd["self_attn.k_proj.weight"]
        wv = sd["self_attn.v_proj.weight"]
        wo = sd["self_attn.o_proj.weight"]
        wg = sd["mlp.gate_proj.weight"]
        wu = sd["mlp.up_proj.weight"]
        wd = sd["mlp.down_proj.weight"]

        self.hidden = int(wq.shape[1])
        self.n_heads = int(wq.shape[0]) // self.head_dim
        self.n_kv_heads = int(wk.shape[0]) // self.head_dim
        self.intermediate = int(wg.shape[0])

        if self.n_heads % self.tp or self.n_kv_heads % self.tp or self.intermediate % (self.tp * TILE):
            # TP must divide BOTH head counts (else a chip holds a partial GQA
            # group) and cut the intermediate axis on tile boundaries.
            self.tp = 1
        self.n_local_heads = self.n_heads // self.tp
        self.n_local_kv_heads = self.n_kv_heads // self.tp
        self.n_rep = self.n_local_heads // self.n_local_kv_heads

        def _shard(t, dim):
            mapper = ttnn.ShardTensorToMesh(device, dim=dim) if (self.mesh and self.tp > 1) else None
            return ttnn.from_torch(
                t.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        wq_t, wk_t, wv_t = wq.t().contiguous(), wk.t().contiguous(), wv.t().contiguous()

        # --- attention: column-parallel q, packed [K_i | V_i], row-parallel o
        self.wq = _shard(wq_t, -1)
        kvw = self.n_local_kv_heads * self.head_dim
        self.wkv = _shard(
            torch.cat(
                [
                    torch.cat([wk_t[:, i * kvw : (i + 1) * kvw], wv_t[:, i * kvw : (i + 1) * kvw]], dim=-1)
                    for i in range(self.tp)
                ],
                dim=-1,
            ).contiguous(),
            -1,
        )
        self.wo = _shard(wo.t().contiguous(), 0)

        # --- mlp: column-parallel gate/up, row-parallel down
        self.w_gate = _shard(wg.t().contiguous(), -1)
        self.w_up = _shard(wu.t().contiguous(), -1)
        self.w_down = _shard(wd.t().contiguous(), 0)

        act_name = str(getattr(cfg, "hidden_act", "silu") or "silu").lower()
        self.act = _ACT.get(act_name, ttnn.silu)

        # --- replicated norm gammas, in the (1, 1, dim//32, 32) ROW_MAJOR form
        self.eps = float(getattr(cfg, "rms_norm_eps", 0.0) or 1e-6)
        self.attn_eps = float(getattr(getattr(attn, "q_norm", None), "variance_epsilon", 0.0) or self.eps)

        def _gamma(t):
            dim = int(t.numel())
            return ttnn.from_torch(
                t.reshape(1, 1, dim // TILE, TILE).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device) if self.mesh else None,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.input_norm = _gamma(sd["input_layernorm.weight"])
        self.post_attn_norm = _gamma(sd["post_attention_layernorm.weight"])
        qn, kn = sd.get("self_attn.q_norm.weight"), sd.get("self_attn.k_norm.weight")
        self.q_norm = _gamma(qn) if qn is not None else None
        self.k_norm = _gamma(kn) if kn is not None else None

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._rope_cache = {}
        self._mask_cache = {}

    # ---------------------------------------------------------------- helpers
    def _stage(self, t, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=layout,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if self.mesh else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _rope_tables(self, position_embeddings):
        key = id(position_embeddings)
        hit = self._rope_cache.get(key)
        if hit is not None:
            return hit
        cos, sin = position_embeddings

        def _b(t, n_local):
            t = t.reshape(1, 1, t.shape[-2], t.shape[-1]).expand(1, n_local, -1, -1).contiguous()
            return self._stage(t)

        tables = (
            _b(cos, self.n_local_heads),
            _b(sin, self.n_local_heads),
            _b(cos, self.n_local_kv_heads),
            _b(sin, self.n_local_kv_heads),
        )
        self._rope_cache[key] = tables
        return tables

    def _rotate_half(self, x):
        half = self.head_dim // 2
        s = list(x.shape)
        x1 = ttnn.slice(x, [0, 0, 0, 0], [s[0], s[1], s[2], half])
        x2 = ttnn.slice(x, [0, 0, 0, half], [s[0], s[1], s[2], self.head_dim])
        return ttnn.concat([ttnn.neg(x2), x1], dim=-1)

    def _apply_rope(self, x, cos, sin):
        return ttnn.add(ttnn.mul(x, cos), ttnn.mul(self._rotate_half(x), sin))

    def _score_bias(self, attention_mask, seq_len):
        """The additive bias HF folds into the score matrix.

        No mask supplied -> `sdpa_attention_forward` runs with is_causal=True,
        so build the causal bias. A supplied mask is used as given, unless it
        is constant along the score axis (which cancels inside softmax).
        """
        key = ("causal", seq_len) if attention_mask is None else id(attention_mask)
        if key in self._mask_cache:
            return self._mask_cache[key]
        if attention_mask is None:
            bias = torch.zeros(seq_len, seq_len).masked_fill(
                torch.ones(seq_len, seq_len, dtype=torch.bool).triu(1), -1e9
            )
            bias = bias.reshape(1, 1, seq_len, seq_len)
        elif torch.is_tensor(attention_mask):
            bias = torch.zeros(1, 1, seq_len, seq_len) + attention_mask.to(torch.float32)
            if torch.allclose(bias, bias[..., :1].expand_as(bias)):
                self._mask_cache[key] = None
                return None
        else:
            self._mask_cache[key] = None
            return None
        out = self._stage(bias.expand(bias.shape[0], self.n_local_heads, seq_len, seq_len).contiguous())
        self._mask_cache[key] = out
        return out

    def _repeat_kv(self, x):
        if self.n_rep == 1:
            return x
        if self.n_local_kv_heads == 1:
            return ttnn.repeat(x, ttnn.Shape([1, self.n_rep, 1, 1]))
        return ttnn.repeat_interleave(x, self.n_rep, 1)

    def _attention(self, x, position_embeddings, attention_mask, seq_len):
        xq = ttnn.linear(x, self.wq, compute_kernel_config=self.compute_kernel_config)
        xkv = ttnn.linear(x, self.wkv, compute_kernel_config=self.compute_kernel_config)

        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xq,
            xkv,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False,
        )
        if self.q_norm is not None:
            q = ttnn.rms_norm(
                q, epsilon=self.attn_eps, weight=self.q_norm, compute_kernel_config=self.compute_kernel_config
            )
        if self.k_norm is not None:
            k = ttnn.rms_norm(
                k, epsilon=self.attn_eps, weight=self.k_norm, compute_kernel_config=self.compute_kernel_config
            )
        if position_embeddings is not None:
            cos_q, sin_q, cos_k, sin_k = self._rope_tables(position_embeddings)
            q = self._apply_rope(q, cos_q, sin_q)
            k = self._apply_rope(k, cos_k, sin_k)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1), compute_kernel_config=self.compute_kernel_config)
        scores = ttnn.mul(scores, self.scaling)
        bias = self._score_bias(attention_mask, seq_len)
        if bias is not None:
            scores = ttnn.add(scores, bias)
        scores = ttnn.softmax(scores, dim=-1, compute_kernel_config=self.compute_kernel_config)

        context = ttnn.matmul(scores, v, compute_kernel_config=self.compute_kernel_config)
        context = ttnn.experimental.nlp_concat_heads(context)

        out = ttnn.linear(context, self.wo, compute_kernel_config=self.compute_kernel_config)
        return ttnn.all_reduce(out) if self.tp > 1 else out

    def _mlp(self, x):
        gate = ttnn.linear(x, self.w_gate, compute_kernel_config=self.compute_kernel_config)
        up = ttnn.linear(x, self.w_up, compute_kernel_config=self.compute_kernel_config)
        h = ttnn.mul(self.act(gate), up)
        out = ttnn.linear(h, self.w_down, compute_kernel_config=self.compute_kernel_config)
        return ttnn.all_reduce(out) if self.tp > 1 else out

    # ---------------------------------------------------------------- forward
    def __call__(self, *args, **kwargs):
        hidden_states = kwargs.pop("hidden_states", None)
        position_embeddings = kwargs.pop("position_embeddings", None)
        attention_mask = kwargs.pop("attention_mask", None)

        for a in args:
            if isinstance(a, tuple) and len(a) == 2 and position_embeddings is None:
                position_embeddings = a
            elif hidden_states is None:
                hidden_states = a
            elif attention_mask is None and a is not None:
                attention_mask = a
        if hidden_states is None:
            raise ValueError("decoder_layer stub: no hidden_states supplied")

        x = self._stage(hidden_states) if torch.is_tensor(hidden_states) else hidden_states
        in_shape = list(x.shape)
        seq_len = int(in_shape[-2])
        if len(in_shape) != 4:
            x = ttnn.reshape(x, (x.shape[0], 1, seq_len, in_shape[-1]))

        h = ttnn.rms_norm(x, epsilon=self.eps, weight=self.input_norm, compute_kernel_config=self.compute_kernel_config)
        x = ttnn.add(x, self._attention(h, position_embeddings, attention_mask, seq_len))

        h = ttnn.rms_norm(
            x, epsilon=self.eps, weight=self.post_attn_norm, compute_kernel_config=self.compute_kernel_config
        )
        x = ttnn.add(x, self._mlp(h))

        if len(in_shape) != 4:
            x = ttnn.reshape(x, tuple(int(d) for d in in_shape))
        return x

    @classmethod
    def build(cls, device, torch_module):
        if torch_module is None:
            raise RuntimeError("decoder_layer stub needs the torch reference module to source its weights")
        return cls(device, torch_module)


def build(device, torch_module=None):
    return TtDecoderLayer.build(device, torch_module)


def decoder_layer(device, torch_module=None):
    return TtDecoderLayer.build(device, torch_module)

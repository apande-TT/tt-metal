# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN, tensor-parallel port of the Qwen3 transformer stack.

Component `encoder_stack` of `/tmp/tt_hw_planner_components/flux_2_klein_9b_text_encoder`
= `model` (`Qwen3Model`): embed_tokens + 36 x Qwen3DecoderLayer + final
RMSNorm, with RoPE computed once for the whole stack.

This checkpoint is decoder-only, so the component's "stack" role is filled by
`model`; the scaffold's llama_vision_encoder paths resolve to nothing here.

Shapes: hidden=4096, head_dim=128, n_heads=32, n_kv_heads=8 (GQA group 4),
intermediate=12288, SwiGLU, 36 layers, rope_theta=1e6, no projection biases.

Tensor-parallel scheme (TP = number of mesh devices; 8 here)
------------------------------------------------------------
Per layer, two column-then-row pairs with one collective each:

* Attention. q/k/v_proj COLUMN-parallel -- their outputs feed per-head work
  (RMSNorm, RoPE, softmax), so each chip owns n_heads/TP = 4 query heads and
  n_kv_heads/TP = 1 kv head. Query head `j` attends to kv head `j // 4`, so
  the contiguous split keeps query heads 4i..4i+3 and kv head i on chip i:
  each chip holds a whole GQA group and the attention core needs no
  cross-chip traffic. o_proj is ROW-parallel over that head axis and yields
  a PARTIAL sum -> `ttnn.all_reduce`.
* MLP. gate_proj/up_proj COLUMN-parallel (12288/8 = 1536 columns per chip,
  feeding the SiLU gate elementwise); down_proj ROW-parallel over the
  intermediate axis -> a second `ttnn.all_reduce`.

So the residual stream is full-width and identical on every chip at each
layer boundary, which is what lets the layers stack without any further
communication, and what the harness reads back.

Replicated (never sharded): every RMSNorm gamma (input, post-attention,
final, and the per-head q_norm/k_norm), the RoPE cos/sin tables, and the
token-embedding table -- all lookups or elementwise ops on unsplit axes.
Only the six per-layer projection matrices are sharded.

The k and v projections are packed per chip as [K_i | V_i] so
`nlp_create_qkv_heads` can split heads tile-natively (this build's row-major
reshape kernel does not compile, so reshapes that split the last dim are
avoided entirely).

Causal masking: `Qwen3Model` builds `create_causal_mask(...)` and its layers
dispatch through `sdpa_attention_forward`, so with no caller mask the
reference attention is CAUSAL; this port applies a causal additive bias when
no mask is given and uses the caller's mask when one is.

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


class TtEncoderStack:
    """Native ttnn Qwen3 transformer stack, column/row-parallel over the mesh."""

    def __init__(self, device, torch_module) -> None:
        self.device = device
        self.mesh = _is_mesh(device)
        self.tp = _num_devices(device) if self.mesh else 1

        cfg = torch_module.config
        layers = torch_module.layers
        attn0 = layers[0].self_attn

        self.head_dim = int(getattr(attn0, "head_dim", 0) or getattr(cfg, "head_dim", 0) or 128)
        self.scaling = float(getattr(attn0, "scaling", self.head_dim**-0.5))
        self.hidden = int(cfg.hidden_size)
        self.n_heads = int(cfg.num_attention_heads)
        self.n_kv_heads = int(cfg.num_key_value_heads)
        self.intermediate = int(cfg.intermediate_size)
        self.eps = float(getattr(cfg, "rms_norm_eps", 0.0) or 1e-6)
        self.attn_eps = float(getattr(getattr(attn0, "q_norm", None), "variance_epsilon", 0.0) or self.eps)
        rope = getattr(cfg, "rope_parameters", None) or {}
        self.rope_theta = float(rope.get("rope_theta", getattr(cfg, "rope_theta", 10000.0)))
        self.act = _ACT.get(str(getattr(cfg, "hidden_act", "silu") or "silu").lower(), ttnn.silu)

        if self.n_heads % self.tp or self.n_kv_heads % self.tp or self.intermediate % (self.tp * TILE):
            # TP must divide BOTH head counts (else a chip holds a partial GQA
            # group) and cut the intermediate axis on tile boundaries.
            self.tp = 1
        self.n_local_heads = self.n_heads // self.tp
        self.n_local_kv_heads = self.n_kv_heads // self.tp
        self.n_rep = self.n_local_heads // self.n_local_kv_heads

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Build the per-layer weights one layer at a time so no full-model
        # float copy is ever materialised on the host.
        kvw = self.n_local_kv_heads * self.head_dim
        self.layers = []
        for layer in layers:
            sd = layer.state_dict()
            wk_t = sd["self_attn.k_proj.weight"].detach().to(torch.bfloat16).t().contiguous()
            wv_t = sd["self_attn.v_proj.weight"].detach().to(torch.bfloat16).t().contiguous()
            self.layers.append(
                {
                    "wq": self._shard(sd["self_attn.q_proj.weight"], -1, transpose=True),
                    "wkv": self._pack_kv(wk_t, wv_t, kvw),
                    "wo": self._shard(sd["self_attn.o_proj.weight"], 0, transpose=True),
                    "w_gate": self._shard(sd["mlp.gate_proj.weight"], -1, transpose=True),
                    "w_up": self._shard(sd["mlp.up_proj.weight"], -1, transpose=True),
                    "w_down": self._shard(sd["mlp.down_proj.weight"], 0, transpose=True),
                    "input_norm": self._gamma(sd["input_layernorm.weight"]),
                    "post_attn_norm": self._gamma(sd["post_attention_layernorm.weight"]),
                    "q_norm": self._gamma(sd["self_attn.q_norm.weight"]) if "self_attn.q_norm.weight" in sd else None,
                    "k_norm": self._gamma(sd["self_attn.k_norm.weight"]) if "self_attn.k_norm.weight" in sd else None,
                }
            )

        self.final_norm = self._gamma(torch_module.state_dict()["norm.weight"])
        # The embedding table is only needed on the input_ids path; keep the
        # host tensor and stage it lazily so the (replicated) 151936 x 4096
        # table does not cost DRAM on every chip when driven by inputs_embeds.
        self._embed_torch = getattr(torch_module, "embed_tokens", None)
        self._embed = None

        self._rope_cache = {}
        self._mask_cache = {}

    # ------------------------------------------------------------ weight prep
    def _shard(self, t, dim, transpose=False):
        t = t.detach().to(torch.bfloat16)
        if transpose:  # nn.Linear stores [out, in]; ttnn.linear wants [in, out]
            t = t.t().contiguous()
        mapper = ttnn.ShardTensorToMesh(self.device, dim=dim) if (self.mesh and self.tp > 1) else None
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _pack_kv(self, wk_t, wv_t, kvw):
        """Lay K and V out per chip as [K_i | V_i] so one dim=-1 shard hands
        each chip its own kv head pair; a plain cat([K, V]) would instead give
        chip 0 two K heads and no V."""
        packed = torch.cat(
            [
                torch.cat([wk_t[:, i * kvw : (i + 1) * kvw], wv_t[:, i * kvw : (i + 1) * kvw]], dim=-1)
                for i in range(self.tp)
            ],
            dim=-1,
        ).contiguous()
        return self._shard(packed, -1)

    def _gamma(self, t):
        dim = int(t.numel())
        return ttnn.from_torch(
            t.detach().reshape(1, 1, dim // TILE, TILE).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if self.mesh else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _stage(self, t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=layout,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if self.mesh else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # ---------------------------------------------------------------- helpers
    def _rope_tables(self, seq_len):
        """cos/sin for positions 0..S-1, matching Qwen3RotaryEmbedding's
        default rope_type (attention_scaling == 1.0), pre-broadcast to this
        chip's query and kv head counts."""
        hit = self._rope_cache.get(seq_len)
        if hit is not None:
            return hit
        half = self.head_dim // 2
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.int64).float() / self.head_dim)
        )
        freqs = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv_freq)  # (S, hd/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (S, hd)
        assert emb.shape[-1] == 2 * half

        def _b(t, n_local):
            return self._stage(
                t.reshape(1, 1, seq_len, self.head_dim).expand(1, n_local, -1, -1).to(torch.bfloat16).contiguous()
            )

        cos, sin = emb.cos(), emb.sin()
        tables = (
            _b(cos, self.n_local_heads),
            _b(sin, self.n_local_heads),
            _b(cos, self.n_local_kv_heads),
            _b(sin, self.n_local_kv_heads),
        )
        self._rope_cache[seq_len] = tables
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
        out = self._stage(
            bias.expand(bias.shape[0], self.n_local_heads, seq_len, seq_len).to(torch.bfloat16).contiguous()
        )
        self._mask_cache[key] = out
        return out

    def _repeat_kv(self, x):
        if self.n_rep == 1:
            return x
        if self.n_local_kv_heads == 1:
            return ttnn.repeat(x, ttnn.Shape([1, self.n_rep, 1, 1]))
        return ttnn.repeat_interleave(x, self.n_rep, 1)

    def _embed_table(self):
        if self._embed is None:
            if self._embed_torch is None:
                raise ValueError("encoder_stack stub: no embed_tokens to look up input_ids with")
            self._embed = self._stage(
                self._embed_torch.weight.detach().to(torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT
            )
        return self._embed

    # ------------------------------------------------------------------ block
    def _attention(self, w, x, cos_q, sin_q, cos_k, sin_k, bias):
        xq = ttnn.linear(x, w["wq"], compute_kernel_config=self.compute_kernel_config)
        xkv = ttnn.linear(x, w["wkv"], compute_kernel_config=self.compute_kernel_config)

        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xq,
            xkv,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False,
        )
        if w["q_norm"] is not None:
            q = ttnn.rms_norm(
                q, epsilon=self.attn_eps, weight=w["q_norm"], compute_kernel_config=self.compute_kernel_config
            )
        if w["k_norm"] is not None:
            k = ttnn.rms_norm(
                k, epsilon=self.attn_eps, weight=w["k_norm"], compute_kernel_config=self.compute_kernel_config
            )
        q = self._apply_rope(q, cos_q, sin_q)
        k = self._apply_rope(k, cos_k, sin_k)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1), compute_kernel_config=self.compute_kernel_config)
        scores = ttnn.mul(scores, self.scaling)
        if bias is not None:
            scores = ttnn.add(scores, bias)
        scores = ttnn.softmax(scores, dim=-1, compute_kernel_config=self.compute_kernel_config)

        context = ttnn.matmul(scores, v, compute_kernel_config=self.compute_kernel_config)
        context = ttnn.experimental.nlp_concat_heads(context)

        out = ttnn.linear(context, w["wo"], compute_kernel_config=self.compute_kernel_config)
        return ttnn.all_reduce(out) if self.tp > 1 else out

    def _mlp(self, w, x):
        gate = ttnn.linear(x, w["w_gate"], compute_kernel_config=self.compute_kernel_config)
        up = ttnn.linear(x, w["w_up"], compute_kernel_config=self.compute_kernel_config)
        h = ttnn.mul(self.act(gate), up)
        out = ttnn.linear(h, w["w_down"], compute_kernel_config=self.compute_kernel_config)
        return ttnn.all_reduce(out) if self.tp > 1 else out

    # ---------------------------------------------------------------- forward
    def __call__(self, *args, **kwargs):
        inputs_embeds = kwargs.pop("inputs_embeds", None)
        input_ids = kwargs.pop("input_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        for a in args:
            if a is None:
                continue
            if inputs_embeds is None and input_ids is None:
                if torch.is_tensor(a) and not a.is_floating_point():
                    input_ids = a
                else:
                    inputs_embeds = a
            elif attention_mask is None:
                attention_mask = a

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("encoder_stack stub: neither inputs_embeds nor input_ids supplied")
            ids = input_ids
            if torch.is_tensor(ids):
                ids = self._stage(ids.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32)
            x = ttnn.embedding(ids, self._embed_table(), layout=ttnn.TILE_LAYOUT)
        elif torch.is_tensor(inputs_embeds):
            x = self._stage(inputs_embeds.to(torch.bfloat16))
        else:
            x = inputs_embeds

        in_shape = list(x.shape)
        seq_len = int(in_shape[-2])
        if len(in_shape) != 4:
            x = ttnn.reshape(x, (x.shape[0], 1, seq_len, in_shape[-1]))

        cos_q, sin_q, cos_k, sin_k = self._rope_tables(seq_len)
        bias = self._score_bias(attention_mask, seq_len)

        for w in self.layers:
            h = ttnn.rms_norm(
                x, epsilon=self.eps, weight=w["input_norm"], compute_kernel_config=self.compute_kernel_config
            )
            x = ttnn.add(x, self._attention(w, h, cos_q, sin_q, cos_k, sin_k, bias))
            h = ttnn.rms_norm(
                x, epsilon=self.eps, weight=w["post_attn_norm"], compute_kernel_config=self.compute_kernel_config
            )
            x = ttnn.add(x, self._mlp(w, h))

        x = ttnn.rms_norm(x, epsilon=self.eps, weight=self.final_norm, compute_kernel_config=self.compute_kernel_config)
        if len(in_shape) != 4:
            x = ttnn.reshape(x, tuple(int(d) for d in in_shape))
        return x

    @classmethod
    def build(cls, device, torch_module):
        if torch_module is None:
            raise RuntimeError("encoder_stack stub needs the torch reference module to source its weights")
        return cls(device, torch_module)


def build(device, torch_module=None):
    return TtEncoderStack.build(device, torch_module)


def encoder_stack(device, torch_module=None):
    return TtEncoderStack.build(device, torch_module)

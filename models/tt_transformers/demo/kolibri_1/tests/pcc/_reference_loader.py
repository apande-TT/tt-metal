# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Pure-PyTorch reference for Aleph-Alpha/Kolibri-1 (`model_type: kolibri1`).

Kolibri 1 is not in transformers. Its only implementation is the vLLM plugin in the
`aleph-alpha-inference` package (aleph_alpha_inference/kolibri1.py, v1.0.0), which needs vLLM
and a GPU. This file ports that model to plain torch, op for op:

  * Qwen3-MoE skeleton (the plugin's config class subclasses Qwen3MoeConfig).
  * Attention: GQA 48/4 heads, head_dim 128, per-head q/k RMSNorm, scale head_dim**-0.5.
    `sliding_attention` layers apply neox RoPE (theta 10000) and see the current token plus the
    512 before it (sliding_window=513). `full_attention` layers are causal with NO positional
    encoding (RNoPE).
  * Sandwich norms: h += post_attn_norm(attn(input_layernorm(h)));
                    h += post_ffn_norm(moe(post_attention_layernorm(h))).
  * MoE on every layer: fp32 router logits, top-6 picked on `logits + expert_bias`, weighted by
    the UNBIASED sigmoid(logits), no renormalisation (norm_topk_prob=false), no routed scaling,
    plus one ungated shared expert added on top. SwiGLU experts.
  * Final RMSNorm, untied lm_head computed in fp32 (`head_dtype: float32`).

Weights are the shipped FP8 checkpoint as-is: every quantised projection keeps `weight`
(float8_e4m3fn) and `weight_scale_inv` (fp32, one per 128x128 block) exactly as the repo names them,
and dequantises when it runs (`Kolibri1FP8Linear.dequantize()`). Holding them dequantised would
take ~156 GB of host RAM, versus ~78 GB like this. Code that reads `.weight` to build a port MUST go
through `dequantize()`; casting the fp8 tensor alone drops the block scales. Everything else
(embeddings, norms, router, lm_head) is shipped bf16 and kept bf16. The checkpoint's
`model.layers.N.moe.router.expert_bias` is held as `model.layers.N.mlp.gate.e_score_correction_bias`,
which is where the plugin's own weight mapper puts it.

The full-model forward runs in fp32. Each submodule computes in the dtype of its input.
"""

import json
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

REFERENCE_LOADER_CONTRACT = 2

_FP8_BLOCK = 128
_ROUTER_BIAS_SRC = ".moe.router.expert_bias"
_ROUTER_BIAS_DST = ".mlp.gate.e_score_correction_bias"


def _config_class():
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

    class Kolibri1Config(Qwen3MoeConfig):
        model_type = "kolibri1"

    return Kolibri1Config


class Kolibri1FP8Linear(nn.Module):
    """Bias-free linear over block-quantised FP8 weights (128x128 blocks, fp32 inverse scales)."""

    def __init__(self, in_features: int, out_features: int, block: int = _FP8_BLOCK):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block = block
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        self.weight_scale_inv = nn.Parameter(
            torch.empty(-(-out_features // block), -(-in_features // block), dtype=torch.float32),
            requires_grad=False,
        )

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """weight * weight_scale_inv per 128x128 block (transformers' finegrained-FP8 dequant)."""
        sr, sc = self.weight_scale_inv.shape
        w = self.weight.to(torch.float32).view(sr, self.out_features // sr, sc, self.in_features // sc)
        w = w * self.weight_scale_inv[:, None, :, None]
        return w.view(self.out_features, self.in_features).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.dequantize(x.dtype))

    def _apply(self, fn, recurse=True):
        # A dtype cast of a parent (`model.float()`, which the PCC harness does) would otherwise turn the
        # fp8 payload into fp32: 4x the host RAM, ~300 GB for the full model. dequantize() casts per call,
        # so keeping fp8 is numerically identical. Anything that moves the tensor still applies.
        probe = fn(torch.empty(0, dtype=torch.float8_e4m3fn, device=self.weight.device))
        if probe.device == self.weight.device and probe.dtype != torch.float8_e4m3fn:
            fp8 = self.weight
            self._parameters["weight"] = None  # _apply skips None; the key keeps its place in state_dict
            try:
                return super()._apply(fn, recurse)
            finally:
                self._parameters["weight"] = fp8
        return super()._apply(fn, recurse)


class Kolibri1RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16), requires_grad=False)
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return h.to(x.dtype) * self.weight.to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Kolibri1Attention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        # The plugin: RoPE + window on sliding layers only; full layers are causal NoPE.
        self.sliding_window = config.sliding_window if self.is_sliding else None
        self.rope_theta = float(config.rope_parameters["rope_theta"])
        h = config.hidden_size
        self.q_proj = Kolibri1FP8Linear(h, self.num_heads * self.head_dim)
        self.k_proj = Kolibri1FP8Linear(h, self.num_kv_heads * self.head_dim)
        self.v_proj = Kolibri1FP8Linear(h, self.num_kv_heads * self.head_dim)
        self.o_proj = Kolibri1FP8Linear(self.num_heads * self.head_dim, h)
        self.q_norm = Kolibri1RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Kolibri1RMSNorm(self.head_dim, config.rms_norm_eps)

    def _rope(self, q, k, position_ids):
        d = self.head_dim
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
        freqs = position_ids.float()[..., None] * inv_freq  # [B, T, d/2]
        emb = torch.cat((freqs, freqs), dim=-1)[:, None]  # [B, 1, T, d]
        cos, sin = emb.cos(), emb.sin()
        rot = lambda t: (t.float() * cos + _rotate_half(t.float()) * sin).to(t.dtype)  # noqa: E731
        return rot(q), rot(k)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[list] = None,
    ) -> torch.Tensor:
        """`attention_mask` is [B, keys] (1 = real token). `past_key_value` is an optional mutable
        list this call extends with [key positions, k, v]; generate() uses it, nothing else needs to."""
        B, T, _ = hidden_states.shape
        if position_ids is None:
            position_ids = torch.arange(T, device=hidden_states.device)[None].expand(B, T)
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        if self.is_sliding:
            q, k = self._rope(q, k, position_ids)
        key_pos = position_ids
        if past_key_value is not None:
            if past_key_value:
                key_pos = torch.cat([past_key_value[0], key_pos], dim=1)
                k = torch.cat([past_key_value[1], k], dim=2)
                v = torch.cat([past_key_value[2], v], dim=2)
            past_key_value[:] = [key_pos, k, v]

        # Visibility from positions, so left padding keeps the window measured in real tokens.
        pq = position_ids[:, :, None]
        pk = key_pos[:, None, :]
        allowed = pk <= pq
        if self.sliding_window is not None:
            allowed = allowed & (pq - pk < self.sliding_window)
        if attention_mask is not None:
            allowed = allowed & attention_mask[:, None, :].bool()
        allowed = allowed[:, None]  # [B, 1, T, T]

        groups = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.scaling
        scores = scores.masked_fill(~allowed, torch.finfo(torch.float32).min)
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v.float()).to(hidden_states.dtype)
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out)


class Kolibri1MLP(nn.Module):
    """SwiGLU expert; used for both the routed experts and the shared expert."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = Kolibri1FP8Linear(hidden_size, intermediate_size)
        self.up_proj = Kolibri1FP8Linear(hidden_size, intermediate_size)
        self.down_proj = Kolibri1FP8Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Kolibri1Router(nn.Module):
    """bf16 router weight with fp32 logits, plus the selection-only bias (shipped bf16, used in fp32)."""

    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size, dtype=torch.bfloat16), requires_grad=False)
        self.e_score_correction_bias = nn.Parameter(torch.zeros(num_experts, dtype=torch.bfloat16), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.float(), self.weight.float())


class Kolibri1SparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = Kolibri1Router(config.hidden_size, config.num_experts)
        self.experts = nn.ModuleList(
            Kolibri1MLP(config.hidden_size, config.moe_intermediate_size) for _ in range(config.num_experts)
        )
        self.shared_experts = Kolibri1MLP(config.hidden_size, config.shared_expert_intermediate_size)

    def route(self, x: torch.Tensor):
        """(topk_weights fp32 [N, k], topk_ids [N, k]) for flattened tokens x [N, H]."""
        logits = self.gate(x)
        topk_ids = torch.topk(logits + self.gate.e_score_correction_bias.float(), k=self.top_k, dim=-1).indices
        topk_weights = torch.sigmoid(logits.gather(1, topk_ids))
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_weights, topk_ids

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        topk_weights, topk_ids = self.route(x)
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        for e in torch.unique(topk_ids).tolist():
            rows, slot = torch.nonzero(topk_ids == e, as_tuple=True)
            y = self.experts[e](x[rows]).float() * topk_weights[rows, slot, None]
            out.index_add_(0, rows, y)
        out = out + self.shared_experts(x).float()
        return out.to(hidden_states.dtype).reshape(shape)


class Kolibri1DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        eps = config.rms_norm_eps
        self.self_attn = Kolibri1Attention(config, layer_idx)
        self.mlp = Kolibri1SparseMoeBlock(config)
        self.input_layernorm = Kolibri1RMSNorm(config.hidden_size, eps)
        self.post_attn_norm = Kolibri1RMSNorm(config.hidden_size, eps)
        self.post_attention_layernorm = Kolibri1RMSNorm(config.hidden_size, eps)
        self.post_ffn_norm = Kolibri1RMSNorm(config.hidden_size, eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[list] = None,
    ) -> torch.Tensor:
        a = self.self_attn(self.input_layernorm(hidden_states), position_ids, attention_mask, past_key_value)
        hidden_states = hidden_states + self.post_attn_norm(a)
        m = self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states + self.post_ffn_norm(m)


class Kolibri1Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=torch.bfloat16)
        self.embed_tokens.weight.requires_grad_(False)
        self.layers = nn.ModuleList(Kolibri1DecoderLayer(config, i) for i in range(config.num_hidden_layers))
        self.norm = Kolibri1RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        compute_dtype=torch.float32,
        all_hidden=None,
    ):
        h = self.embed_tokens(input_ids).to(compute_dtype)
        if position_ids is None:
            B, T = input_ids.shape
            if attention_mask is not None:
                position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)[:, -T:]
            else:
                position_ids = torch.arange(T, device=input_ids.device)[None].expand(B, T)
        for i, layer in enumerate(self.layers):
            if all_hidden is not None:
                all_hidden.append(h)
            cache = past_key_values[i] if past_key_values is not None else None
            h = layer(h, position_ids, attention_mask, cache)
        h = self.norm(h)
        if all_hidden is not None:
            all_hidden.append(h)
        return h


class Kolibri1ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Kolibri1Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=torch.bfloat16)
        self.lm_head.weight.requires_grad_(False)

    def can_generate(self) -> bool:
        return True

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        output_hidden_states: bool = False,
        **kwargs,
    ):
        from transformers.modeling_outputs import CausalLMOutput

        hidden = [] if output_hidden_states else None
        h = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            all_hidden=hidden,
        )
        logits = F.linear(h.float(), self.lm_head.weight.float())  # head_dtype: float32
        return CausalLMOutput(logits=logits, hidden_states=tuple(hidden) if hidden is not None else None)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens: int = 20, eos_token_id=None, **kwargs):
        """Greedy decoding with a per-layer KV cache. Returns prompt + generated ids."""
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        eos = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])
        B = input_ids.shape[0]
        mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids)
        pos = (mask.long().cumsum(-1) - 1).clamp_min(0)
        cache = [[] for _ in self.model.layers]
        seq, step_ids, step_pos = input_ids, input_ids, pos
        done = torch.zeros(B, dtype=torch.bool)
        for _ in range(max_new_tokens):
            logits = self(step_ids, attention_mask=mask, position_ids=step_pos, past_key_values=cache).logits
            nxt = logits[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, self.config.pad_token_id), nxt)
            seq = torch.cat([seq, nxt[:, None]], dim=1)
            mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
            step_ids, step_pos = nxt[:, None], step_pos[:, -1:] + 1
            done |= torch.tensor([int(t) in eos for t in nxt])
            if bool(done.all()):
                break
        return seq


def _repo_dir(model_id: str) -> Path:
    if os.path.isdir(model_id):
        return Path(model_id)
    from huggingface_hub import snapshot_download

    patterns = ["*.json", "*.safetensors"]
    try:
        return Path(snapshot_download(model_id, allow_patterns=patterns))
    except Exception:  # offline: whatever is complete in the cache
        return Path(snapshot_download(model_id, allow_patterns=patterns, local_files_only=True))


def _build_config(repo: Path):
    raw = json.loads((repo / "config.json").read_text())
    raw.pop("architectures", None)
    raw.pop("model_type", None)
    return _config_class()(**raw)


def _assign(model: nn.Module, name: str, tensor: torch.Tensor, owners: dict):
    owner, attr = owners[name]
    old = getattr(owner, attr)
    if tuple(old.shape) != tuple(tensor.shape):
        raise ValueError(f"{name}: checkpoint shape {tuple(tensor.shape)} != reference {tuple(old.shape)}")
    if old.dtype != tensor.dtype:
        raise ValueError(f"{name}: checkpoint dtype {tensor.dtype} != reference {old.dtype}")
    # clone(): safetensors hands back file-backed mmap storage, which ttnn uploads stall on.
    setattr(owner, attr, nn.Parameter(tensor.clone(), requires_grad=False))


def load_reference_model(model_id: str):
    """Return an nn.Module (in eval mode) equivalent to the HF reference for this model, loaded from whatever real format the repo actually ships."""
    from safetensors import safe_open

    repo = _repo_dir(model_id)
    config = _build_config(repo)
    with torch.device("meta"):
        model = Kolibri1ForCausalLM(config)

    owners = {}
    for mod_name, mod in model.named_modules():
        for p_name, _ in mod.named_parameters(recurse=False):
            owners[f"{mod_name}.{p_name}" if mod_name else p_name] = (mod, p_name)

    weight_map = json.loads((repo / "model.safetensors.index.json").read_text())["weight_map"]
    loaded = set()
    for shard in sorted(set(weight_map.values())):
        with safe_open(str(repo / shard), framework="pt") as f:
            for key in f.keys():
                name = key.replace(_ROUTER_BIAS_SRC, _ROUTER_BIAS_DST)
                if name not in owners:
                    raise KeyError(f"checkpoint tensor {key} has no place in the Kolibri1 reference")
                _assign(model, name, f.get_tensor(key), owners)
                loaded.add(name)
    missing = sorted(set(owners) - loaded)
    if missing:
        raise KeyError(f"{len(missing)} reference parameters not in the checkpoint, e.g. {missing[:5]}")
    return model.eval()

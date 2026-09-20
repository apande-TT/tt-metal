# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The checkpoint's `acoustic_transformer` section, ported to TTNN on the graduated stubs.

WHY THIS EXISTS
---------------
`consolidated.safetensors` holds THREE repeated block stacks, not one:

    layers.<0..25>                        26  the text backbone  (tt/generation.py, tt/hidden_states.py)
    audio_tokenizer.decoder_blocks.<0..7>  8  the audio tokenizer / vocoder
    acoustic_transformer.layers.<0..2>     3  this file

A stack the structural walk cannot see is never capped, never marked, and its depth is inferred
for the whole run -- so a section the pipeline does not hold is a section every downstream
measurement gets wrong. Two of the three were missing entirely; this adds the one that can be
ported honestly.

WHAT IS GROUNDED AND WHAT IS NOT
--------------------------------
GROUNDED. `params.json -> multimodal.audio_model_args.acoustic_transformer_args` specifies this
section completely: `input_dim 3072, dim 3072, n_layers 3, head_dim 128, hidden_dim 9216,
n_heads 32, n_kv_heads 8, use_biases false, rope_theta 10000`. Its tensor names are the SAME
native-Mistral block names the text backbone uses -- `attention.{wq,wk,wv,wo}`, `attention_norm`,
`feed_forward.{w1,w2,w3}`, `ffn_norm` -- so the identical key map and the identical RoPE permute
apply, and that transform is the one `tests/pcc/_reference_loader.py` validated BIT-IDENTICALLY
against Mistral's own published HF conversion of the declared base model. The blocks are
therefore plain `MistralDecoderLayer`s at a different rope_theta, and the reference below is
built out of the real `transformers` classes, not a hand-rolled imitation.

NOT GROUNDED, AND THEREFORE NOT DRIVEN. The section also carries `input_projection` (36 -> 3072),
`time_projection` (3072 -> 3072) and `acoustic_codebook_output` (3072 -> 36). Those are the
flow-matching sampler's surface: the noisy acoustic latents and the diffusion timestep. This
checkpoint ships NO reference implementation for that sampler -- `transformers` has `voxtral` and
`voxtral_realtime`, and neither matches this section (the realtime TTS decoder conditions through
an `ada_rms_norm` whose weights do not exist here) -- so how the three conditioning terms combine
is not something the checkpoint states. Guessing it would be fiction dressed as a port, so this
file does not: it drives the part whose role the names and shapes make unambiguous,

    lm_hidden [B, S, 3072] --llm_projection--> 3 x decoder block --norm--> semantic_codebook_output
                                                                                -> [B, S, 8320]

and leaves the rest documented as a hole (see README.md and e2e_plan.json). The PCC gate compares
this TTNN chain against the torch chain built from the SAME weights in `reference_acoustic()`, so
what is measured is the fidelity of the port -- which is the deliverable -- over a composition
both sides share.
"""
from __future__ import annotations

import json
import os

import torch

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import common

# The section's own tensor prefix in consolidated.safetensors.
_PREFIX = "acoustic_transformer"

# Native-Mistral block key -> transformers block key. Identical to the text backbone's map, because
# the blocks are the same format; kept here rather than imported so this file reads standalone.
_LAYER_MAP = {
    "attention.wq.weight": "self_attn.q_proj.weight",
    "attention.wk.weight": "self_attn.k_proj.weight",
    "attention.wv.weight": "self_attn.v_proj.weight",
    "attention.wo.weight": "self_attn.o_proj.weight",
    "feed_forward.w1.weight": "mlp.gate_proj.weight",
    "feed_forward.w2.weight": "mlp.down_proj.weight",
    "feed_forward.w3.weight": "mlp.up_proj.weight",
    "attention_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
}

# Additive mask fill; finite in bfloat16 so a fully-masked row cannot become inf - inf.
_MASK_NEG = -1e9


def _permute_rope(w: torch.Tensor, n_heads: int, head_dim: int, hidden: int) -> torch.Tensor:
    """Native Mistral interleaved-RoPE layout -> transformers `rotate_half` layout."""
    return w.view(n_heads, head_dim // 2, 2, hidden).transpose(1, 2).reshape(n_heads * head_dim, hidden)


def acoustic_args(model_id: str = common.HF_MODEL_ID) -> dict:
    """`params.json`'s own description of this section. Raises if the checkpoint stops declaring it."""
    params = common.load_params(model_id)
    args = (params.get("multimodal") or {}).get("audio_model_args", {}).get("acoustic_transformer_args")
    if not args:
        raise RuntimeError("params.json declares no multimodal.audio_model_args.acoustic_transformer_args")
    return args


def acoustic_config(model_id: str = common.HF_MODEL_ID):
    """A `MistralConfig` for the acoustic blocks, every field read off `params.json`."""
    from transformers import MistralConfig

    args = acoustic_args(model_id)
    params = common.load_params(model_id)
    return MistralConfig(
        vocab_size=int(args["dim"]),  # unused: this section has no token embedding
        hidden_size=int(args["dim"]),
        intermediate_size=int(args["hidden_dim"]),
        num_hidden_layers=int(args["n_layers"]),
        num_attention_heads=int(args["n_heads"]),
        num_key_value_heads=int(args["n_kv_heads"]),
        head_dim=int(args["head_dim"]),
        rms_norm_eps=float(args.get("sigma", params["norm_eps"])),
        max_position_embeddings=int(params.get("max_seq_len", 65536)),
        sliding_window=None,
        tie_word_embeddings=False,
        use_cache=False,
        dtype=torch.float32,
        attn_implementation="eager",
        rope_parameters={"rope_type": "default", "rope_theta": float(args["rope_theta"])},
    )


class AcousticReference(torch.nn.Module):
    """The torch GOLDEN for this section, assembled from real `transformers` Mistral modules.

    `layers` is a `ModuleList` of `MistralDecoderLayer`, `norm` a `MistralRMSNorm`, and the two
    driven projections plain `nn.Linear` -- all loaded with the checkpoint's own tensors. Nothing
    here is a reimplementation: the reference and the port read the same weights through the same
    library classes, which is what makes the PCC number mean something.
    """

    def __init__(self, config, model_id: str = common.HF_MODEL_ID) -> None:
        super().__init__()
        from transformers.models.mistral.modeling_mistral import (
            MistralDecoderLayer,
            MistralRMSNorm,
            MistralRotaryEmbedding,
        )

        self.config = config
        self.layers = torch.nn.ModuleList(
            [MistralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MistralRotaryEmbedding(config=config, device="cpu")
        self.llm_projection = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.semantic_codebook_output = torch.nn.Linear(config.hidden_size, _semantic_width(model_id), bias=False)
        self.load_state_dict(_acoustic_state(config, model_id), strict=False, assign=True)
        self.eval()
        self.requires_grad_(False)

    def forward(self, lm_hidden: torch.Tensor) -> dict:
        hidden = self.llm_projection(lm_hidden)
        seq_len = int(hidden.shape[1])
        position_ids = torch.arange(seq_len, dtype=torch.long).reshape(1, seq_len)
        position_embeddings = self.rotary_emb(hidden, position_ids)
        blocked = torch.ones(seq_len, seq_len, dtype=torch.bool).triu(1)
        mask = torch.zeros(1, 1, seq_len, seq_len, dtype=hidden.dtype).masked_fill_(blocked, _MASK_NEG)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=mask, position_embeddings=position_embeddings)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
        hidden = self.norm(hidden)
        return {"hidden": hidden, "semantic_logits": self.semantic_codebook_output(hidden)}


def _reader(model_id: str):
    from safetensors import safe_open

    path = os.path.join(common.resolve_repo(model_id), "consolidated.safetensors")
    return safe_open(path, framework="pt")


def _semantic_width(model_id: str = common.HF_MODEL_ID) -> int:
    with _reader(model_id) as reader:
        return int(reader.get_slice(f"{_PREFIX}.semantic_codebook_output.weight").get_shape()[0])


def _acoustic_state(config, model_id: str = common.HF_MODEL_ID) -> dict:
    """Checkpoint tensors -> `AcousticReference` parameter names, with the RoPE permute applied."""
    hidden, head_dim = config.hidden_size, config.head_dim
    state = {}
    with _reader(model_id) as reader:
        state["norm.weight"] = reader.get_tensor(f"{_PREFIX}.norm.weight")
        state["llm_projection.weight"] = reader.get_tensor(f"{_PREFIX}.llm_projection.weight")
        state["semantic_codebook_output.weight"] = reader.get_tensor(f"{_PREFIX}.semantic_codebook_output.weight")
        for i in range(config.num_hidden_layers):
            for src, dst in _LAYER_MAP.items():
                w = reader.get_tensor(f"{_PREFIX}.layers.{i}.{src}")
                if src == "attention.wq.weight":
                    w = _permute_rope(w, config.num_attention_heads, head_dim, hidden)
                elif src == "attention.wk.weight":
                    w = _permute_rope(w, config.num_key_value_heads, head_dim, hidden)
                state[f"layers.{i}.{dst}"] = w
    return {k: v.to(torch.float32) for k, v in state.items()}


class AcousticBlock:
    """ONE block of the acoustic stack, composed from the graduated leaf stubs.

    Same residual shape as the text backbone's split blocks -- norm, attention, residual, norm,
    SwiGLU, residual -- because these are the same Mistral blocks at a different rope_theta. Held
    as ONE concrete class so the stack stays a list of same-typed elements a structural walk can
    find and size. No `__slots__`: the walk identifies a block by its `__dict__`.
    """

    def __init__(self, index: int, parts) -> None:
        self.index = int(index)
        self.kind = "acoustic_layer"
        self.parts = list(parts)  # [(graduated component name, role, counter-wrapped stub)]
        self._by_role = {role: stub for _, role, stub in self.parts}

    @property
    def component_names(self) -> tuple:
        return tuple(name for name, _, _ in self.parts)

    def part(self, role):
        return self._by_role[role]

    def __repr__(self) -> str:
        return f"AcousticBlock(index={self.index}, parts={list(self.component_names)})"

    def __call__(self, hidden_states, position_embeddings=None, attention_mask=None):
        normed = self._by_role["input_layernorm"](hidden_states)
        attn = self._by_role["attention"](
            normed, position_embeddings=position_embeddings, attention_mask=attention_mask
        )
        ttnn.deallocate(normed)
        mid = ttnn.add(hidden_states, attn)
        ttnn.deallocate(attn)

        normed = self._by_role["post_attention_layernorm"](mid)
        ffn = self._by_role["feed_forward"](normed)
        ttnn.deallocate(normed)
        out = ttnn.add(mid, ffn)
        ttnn.deallocate(ffn)
        ttnn.deallocate(mid)
        return out


class VoxtralAcousticStack:
    """The resident TTNN port of `acoustic_transformer`: 3 blocks plus the driven projections."""

    def __init__(self, device, reference, blocks, rotary, final_norm, llm_projection, semantic_head, counter):
        self.device = device
        self.reference = reference
        self.config = reference.config
        self.layers = list(blocks)  # plain list, all elements AcousticBlock
        self.n_layers = len(self.layers)
        self.rotary = rotary
        self.final_norm = final_norm
        self.llm_projection = llm_projection
        self.semantic_head = semantic_head
        self.counter = counter
        self.act_dtype = ttnn.float32
        self._prep_cache: dict = {}

    def stage_constants(self, seq_len: int):
        """`(position_ids, causal mask)` for one length, cached. The acoustic stack is causal too."""
        seq_len = int(seq_len)
        cached = self._prep_cache.get(seq_len)
        if cached is None:
            position_ids = torch.arange(seq_len, dtype=torch.long).reshape(1, seq_len)
            blocked = torch.ones(seq_len, seq_len, dtype=torch.bool).triu(1)
            mask = torch.zeros(seq_len, seq_len, dtype=torch.float32).masked_fill_(blocked, _MASK_NEG)
            cached = (
                position_ids,
                # bfloat16, not `act_dtype`: see generation.stage_constants -- the mask feeds the
                # fused flash-attention op, which takes no format wider than bf16.
                ttnn.from_torch(
                    mask.reshape(1, 1, seq_len, seq_len).contiguous(),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                ),
            )
            self._prep_cache[seq_len] = cached
        return cached

    def forward_resident(self, lm_hidden, position_embeddings, attention_mask):
        """The HOST-FREE core: a device hidden state in, the acoustic hidden state out."""
        hidden = self.llm_projection(lm_hidden)
        for block in self.layers:
            nxt = block(hidden, position_embeddings=position_embeddings, attention_mask=attention_mask)
            ttnn.deallocate(hidden)
            hidden = nxt
        out = self.final_norm(hidden)
        ttnn.deallocate(hidden)
        return out

    def forward_semantic_logits(self, lm_hidden):
        """`lm_hidden [B, S, 3072]` (a device tensor) -> `(acoustic hidden, semantic logits)`."""
        position_ids, mask = self.stage_constants(int(lm_hidden.shape[-2]))
        cos, sin = self.rotary(position_ids=position_ids)
        rope = (
            ttnn.reshape(cos, (1, 1, cos.shape[-2], cos.shape[-1])),
            ttnn.reshape(sin, (1, 1, sin.shape[-2], sin.shape[-1])),
        )
        hidden = self.forward_resident(lm_hidden, rope, mask)
        ttnn.deallocate(rope[0])
        ttnn.deallocate(rope[1])
        return hidden, self.semantic_head(hidden)

    def describe(self) -> dict:
        return {
            "head": "acoustic_transformer",
            "n_layers": self.n_layers,
            "declared_layers": int(acoustic_args()["n_layers"]),
            "rope_theta": float(acoustic_args()["rope_theta"]),
            "graduated_modules_routed": sorted({n for b in self.layers for n in b.component_names}),
        }


def build_acoustic_stack(device, counter=None, model_id: str = common.HF_MODEL_ID) -> VoxtralAcousticStack:
    """Compose the graduated leaf stubs over the checkpoint's `acoustic_transformer` weights.

    The depth is NOT capped. The section declares three blocks in total, and the structural walk
    that sizes stacks needs at least three same-typed members to see one at all, so a cap here
    would hide the stack it is meant to size -- the opposite of the point.
    """
    counter = common.InvocationCounter() if counter is None else counter
    config = acoustic_config(model_id)
    reference = AcousticReference(config, model_id=model_id)

    blocks = []
    for index, torch_layer in enumerate(reference.layers):
        parts = [
            (
                "r_m_s_norm",
                "input_layernorm",
                counter.wrap("r_m_s_norm", common.build_stub("r_m_s_norm", device, torch_layer.input_layernorm)),
            ),
            (
                "attention",
                "attention",
                counter.wrap("attention", common.build_stub("attention", device, torch_layer.self_attn)),
            ),
            (
                "r_m_s_norm",
                "post_attention_layernorm",
                counter.wrap(
                    "r_m_s_norm", common.build_stub("r_m_s_norm", device, torch_layer.post_attention_layernorm)
                ),
            ),
            ("mlp", "feed_forward", counter.wrap("mlp", common.build_stub("mlp", device, torch_layer.mlp))),
        ]
        blocks.append(AcousticBlock(index, parts))

    rotary = counter.wrap("rotary_embedding", common.build_stub("rotary_embedding", device, reference.rotary_emb))
    final_norm = counter.wrap("r_m_s_norm", common.build_stub("r_m_s_norm", device, reference.norm))
    llm_projection = counter.wrap("decoder_head", common.build_stub("decoder_head", device, reference.llm_projection))
    semantic_head = counter.wrap(
        "decoder_head", common.build_stub("decoder_head", device, reference.semantic_codebook_output)
    )

    return VoxtralAcousticStack(device, reference, blocks, rotary, final_norm, llm_projection, semantic_head, counter)


def expected_invocation_counts(n_layers: int) -> dict:
    """Gate-2's expected counts for ONE `forward_semantic_logits` at `n_layers`."""
    return {
        "r_m_s_norm": 2 * n_layers + 1,
        "attention": n_layers,
        "mlp": n_layers,
        "rotary_embedding": 1,
        "decoder_head": 2,
    }


def run_acoustic(stack: VoxtralAcousticStack, lm_hidden, **kwargs) -> dict:
    """Call 3: text-backbone hidden state -> acoustic hidden state + semantic-codebook logits.

    `lm_hidden` may be a device tensor (the normal case -- it comes straight out of the text
    stack's forward and never touches the host) or a torch tensor, which is staged once here.
    """
    keep_device_tensor = kwargs.pop("keep_device_tensor", False)
    if kwargs:
        raise TypeError(f"unexpected kwargs: {sorted(kwargs)}")

    staged = None
    if not isinstance(lm_hidden, ttnn.Tensor):
        staged = ttnn.from_torch(
            lm_hidden.to(torch.float32).contiguous(),
            dtype=stack.act_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=stack.device,
        )
        lm_hidden = staged

    hidden, semantic_logits = stack.forward_semantic_logits(lm_hidden)
    out = {
        "acoustic_hidden": ttnn.to_torch(hidden).to(torch.float32),
        "semantic_logits": ttnn.to_torch(semantic_logits).to(torch.float32),
        "tt_acoustic_hidden": hidden,
        "n_layers": stack.n_layers,
    }
    ttnn.deallocate(semantic_logits)
    if staged is not None:
        ttnn.deallocate(staged)
    if not keep_device_tensor:
        ttnn.deallocate(hidden)
        out["tt_acoustic_hidden"] = None
    return out


# --------------------------------------------------------------------------------------
# The golden. Nothing below this banner runs on the TT path.
# --------------------------------------------------------------------------------------


def hf_reference_acoustic(stack: VoxtralAcousticStack, lm_hidden: torch.Tensor) -> dict:
    """The torch chain over the same weights, in float32. Used only for PCC."""
    with torch.no_grad():
        out = stack.reference(lm_hidden.to(torch.float32))
    return {k: v.to(torch.float32) for k, v in out.items()}


def dump_section_report(model_id: str = common.HF_MODEL_ID) -> str:
    """One line per declared section, so 'what is ported' is readable without reading the code."""
    args = acoustic_args(model_id)
    return json.dumps(
        {
            "acoustic_transformer_args": args,
            "ported": ["layers", "norm", "llm_projection", "semantic_codebook_output"],
            "not_driven": ["input_projection", "time_projection", "acoustic_codebook_output"],
            "reason": "no reference exists for this checkpoint's flow-matching sampler",
        },
        indent=2,
    )

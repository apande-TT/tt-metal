# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Call 2 -- `hidden_states`: text -> `last_hidden_state`, on the graduated whole-stack port.

This head routes exactly ONE graduated module, `model` (Source B's
`_stubs/model.py::TtModel`), the TTNN port of
`transformers.models.mistral.modeling_mistral.MistralModel`. Its declared
output IS `last_hidden_state`, and that is literally the tensor the bring-up
tool captured for it (`_captured/model/output.pt`, and the 64-token
`golden_cache_s0.pt` at `[1, 64, 3072]`).

The chain is exactly what `e2e_plan.json -> task_heads[1].chain` declares::

    ids[B, S]                                            (tekken tokenizer, 32 real prompts)
    device_ids, (cos, sin), mask = stub.prepare_inputs(...)   # host staging, OUTSIDE the forward
    h = stub(device_ids, position_embeddings=(cos, sin), attention_mask=mask)
    -> [B, S, 3072]

BATCH: B=32 goes through as one leading batch dim on ONE program. There is no
python loop over samples anywhere on the device path -- the 32 samples ride the
same 26-layer forward.

STRICT TT-ONLY: nothing in `run_hidden_states()` or in the stub's `__call__`
touches HF or torch compute. HF appears here only in `build_*` (weight
extraction / config reads) and inside `hf_reference_hidden_states()`, which is
the GOLDEN helper and is never on the hot path.
"""
from __future__ import annotations

import torch

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import common


class _DepthCappedBackbone:
    """A read-only view of `MistralModel` with `.layers` truncated to the first N.

    `layers` caps the DEPTH only. The embedding table, the final norm and the rotary tables are
    passed through untouched, so a capped build still exercises every DISTINCT op the full model
    runs -- just fewer decoder blocks. This is a separate object rather than an in-place edit, so
    the caller's `hf_model` is never mutated and stays usable as the golden.
    """

    def __init__(self, torch_module, depth: int) -> None:
        self.config = torch_module.config
        self.embed_tokens = torch_module.embed_tokens
        self.layers = torch.nn.ModuleList(list(torch_module.layers)[:depth])
        self.norm = torch_module.norm
        self.rotary_emb = torch_module.rotary_emb


class VoxtralHiddenStatesStack:
    """The resident Call-2 stack: the graduated `model` stub plus what a structural walk needs.

    `.layers` is the stub's OWN list of repeated blocks (plain python list, all
    `TtDecoderLayer`), so a walk can find the stack, size it, and attribute work to it.
    `.hf` keeps the HF reference reachable -- it is ground truth for the section structure.
    """

    def __init__(self, device, hf_model, stub, counter, requested_layers=None) -> None:
        self.device = device
        self.hf = hf_model
        self.stub = stub
        self.counter = counter
        self.requested_layers = requested_layers
        self.n_layers = len(common.unwrap(stub).layers)
        self.available_layers = len(hf_model.model.layers)
        self.hidden_size = int(hf_model.config.hidden_size)

    @property
    def graduated_stub(self):
        """The bare `TtModel` behind the Gate-2 invocation counter."""
        return common.unwrap(self.stub)

    @property
    def layers(self) -> list:
        """The stub's OWN repeated blocks -- a VIEW, deliberately not a second attribute.

        Binding this list to an instance attribute as well gave the structural walk two attribute
        paths (`hidden_states.layers` and `hidden_states.stub.layers`) to the one stack, and it
        counted the section TWICE. One section, one stack; a property lives on the class, so only
        the stub's own path is discoverable.
        """
        return self.graduated_stub.layers

    def prepare_inputs(self, input_ids):
        """Host staging (ids, rotary tables, causal mask) -- runs OUTSIDE the forward."""
        return self.stub.prepare_inputs(self.device, input_ids=input_ids)

    def describe(self) -> dict:
        return {
            "head": "hidden_states",
            "graduated_modules_routed": ["model"],
            "n_layers": self.n_layers,
            "available_layers": self.available_layers,
            "hidden_size": self.hidden_size,
        }


def build_hidden_states_stack(device, hf_model, layers=None, counter=None) -> VoxtralHiddenStatesStack:
    """Build Call 2's stack: ONE graduated `model` stub, depth-capped to `layers`.

    `layers=None` builds every layer (26); a positive `layers` builds the first N decoder blocks
    and keeps everything else intact. `layers<=0` is rejected -- a zero-layer model is not a model.
    """
    counter = counter if counter is not None else common.InvocationCounter()
    backbone = hf_model.model
    available = len(backbone.layers)

    if layers is not None:
        depth = int(layers)
        if depth <= 0:
            raise ValueError(f"layers={depth} would build a zero-layer stack; pass None for every layer")
        depth = min(depth, available)
        if depth < available:
            backbone = _DepthCappedBackbone(backbone, depth)

    stub = common.build_stub("model", device, backbone)
    return VoxtralHiddenStatesStack(device, hf_model, counter.wrap("model", stub), counter, requested_layers=layers)


def run_hidden_states(stack: VoxtralHiddenStatesStack, input_ids, **kwargs) -> dict:
    """The TT forward: `input_ids [B, S]` -> `last_hidden_state [B, S, 3072]`.

    ONE program for all B samples. Everything between `prepare_inputs` and the read-back is pure
    ttnn -- no HF call, no torch compute, no per-sample loop.
    """
    keep_device_tensor = kwargs.pop("keep_device_tensor", True)
    if kwargs:
        raise TypeError(f"unexpected kwargs: {sorted(kwargs)}")

    if not torch.is_tensor(input_ids):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    batch, seq_len = int(input_ids.shape[0]), int(input_ids.shape[1])

    prepared = stack.prepare_inputs(input_ids)
    device_ids = prepared["input_ids"]
    cos, sin = prepared["position_embeddings"]
    mask = prepared["attention_mask"]

    tt_out = stack.stub(device_ids, position_embeddings=(cos, sin), attention_mask=mask)

    torch_out = ttnn.to_torch(tt_out).to(torch.float32).reshape(batch, seq_len, stack.hidden_size)

    for staged in (device_ids, cos, sin, mask):
        ttnn.deallocate(staged)
    if not keep_device_tensor:
        ttnn.deallocate(tt_out)
        tt_out = None

    return {
        "last_hidden_state": torch_out,
        "tt_last_hidden_state": tt_out,
        "batch": batch,
        "seq_len": seq_len,
        "n_layers": stack.n_layers,
        "l2_norm": torch_out.reshape(batch, -1).norm(dim=-1),
    }


def hf_reference_hidden_states(hf_model, input_ids, layers=None) -> torch.Tensor:
    """The GOLDEN: `MistralModel.forward().last_hidden_state`, in float32.

    This is the ONLY place in Call 2 that calls HF. `hf_model` is a `MistralForCausalLM`, so
    `.model` is the `MistralModel` whose declared output is the tensor Call 2 reproduces.
    `layers` mirrors the TT depth cap by temporarily truncating the reference stack; the original
    `ModuleList` is always restored, including on failure.
    """
    if not torch.is_tensor(input_ids):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    backbone = hf_model.model
    original = backbone.layers
    if layers is not None and int(layers) < len(original):
        backbone.layers = torch.nn.ModuleList(list(original)[: int(layers)])
    try:
        with torch.no_grad():
            out = backbone(input_ids=input_ids).last_hidden_state
    finally:
        backbone.layers = original
    return out.to(torch.float32)

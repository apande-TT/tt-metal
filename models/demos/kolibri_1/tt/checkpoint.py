# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kolibri-1 weights from the shipped FP8 checkpoint, one decoder layer at a time.

The modules are the reference's own classes (tests/pcc/_reference_loader.py, the torch port of the
aleph-alpha-inference vLLM plugin), so the graduated stubs build from exactly what they were PCC'd
against. Streaming keeps the host footprint at one layer (~1.6 GB fp8) instead of the whole ~78 GB model.
"""
from __future__ import annotations

import json
from collections import defaultdict

import torch
from safetensors import safe_open

from models.demos.kolibri_1.tests.pcc import _reference_loader as rl
from models.demos.kolibri_1.tt.inputs import MODEL_ID


def layer_indices(config, layers=None) -> list:
    """Indices of the decoder layers a build of depth `layers` holds (None = every layer).

    A capped build keeps every distinct layer kind: the model interleaves sliding-window RoPE layers
    with full-attention NoPE layers (4:1), so with layers >= 2 the last pick is swapped for the first
    layer of any kind the prefix would miss."""
    n = int(config.num_hidden_layers)
    if layers is None or int(layers) >= n:
        return list(range(n))
    if int(layers) < 1:
        raise ValueError(f"layers must be >= 1 (None = all {n}), got {layers}")
    idx = list(range(int(layers)))
    types = list(config.layer_types)
    for kind in dict.fromkeys(types):
        if all(types[i] != kind for i in idx) and len(idx) >= 2:
            missing = types.index(kind)
            for j in range(len(idx) - 1, 0, -1):
                if sum(types[i] == types[idx[j]] for i in idx) > 1:
                    idx[j] = missing
                    break
    return sorted(idx)


class Checkpoint:
    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self.repo = rl._repo_dir(model_id)
        self.config = rl._build_config(self.repo)
        self._weight_map = json.loads((self.repo / "model.safetensors.index.json").read_text())["weight_map"]

    def _fill(self, module: torch.nn.Module, prefix: str) -> torch.nn.Module:
        owners = {}
        for mod_name, mod in module.named_modules():
            for p_name, _ in mod.named_parameters(recurse=False):
                owners[f"{mod_name}.{p_name}" if mod_name else p_name] = (mod, p_name)
        by_shard = defaultdict(list)
        for name in owners:
            key = (prefix + name).replace(rl._ROUTER_BIAS_DST, rl._ROUTER_BIAS_SRC)
            by_shard[self._weight_map[key]].append((name, key))
        for shard, pairs in sorted(by_shard.items()):
            with safe_open(str(self.repo / shard), framework="pt") as f:
                for name, key in pairs:
                    rl._assign(module, name, f.get_tensor(key), owners)
        return module.eval()

    def decoder_layer(self, i: int) -> torch.nn.Module:
        with torch.device("meta"):
            layer = rl.Kolibri1DecoderLayer(self.config, i)
        return self._fill(layer, f"model.layers.{i}.")

    def shell(self) -> torch.nn.Module:
        """The full-depth reference model on the meta device (no weights): ground truth for the model's
        section structure (one 50-deep decoder stack)."""
        with torch.device("meta"):
            return rl.Kolibri1ForCausalLM(self.config).eval()

    def module(self, name: str, factory) -> torch.nn.Module:
        """A non-layer submodule (embed_tokens, norm, lm_head) with its checkpoint weights."""
        with torch.device("meta"):
            mod = factory()
        return self._fill(mod, f"{name}.")

    def embed_tokens(self):
        c = self.config
        return self.module(
            "model.embed_tokens", lambda: torch.nn.Embedding(c.vocab_size, c.hidden_size, dtype=torch.bfloat16)
        )

    def final_norm(self):
        c = self.config
        return self.module("model.norm", lambda: rl.Kolibri1RMSNorm(c.hidden_size, c.rms_norm_eps))

    def lm_head(self):
        c = self.config
        return self.module(
            "lm_head", lambda: torch.nn.Linear(c.hidden_size, c.vocab_size, bias=False, dtype=torch.bfloat16)
        )

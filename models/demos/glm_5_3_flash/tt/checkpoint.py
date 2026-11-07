# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side reader of GLM-5.3-Flash weights, one block at a time, from the FP8 checkpoint.

The whole model in bf16 (~640 GB) exceeds this host's RAM, so neither the TT build nor the HF
golden may call ``from_pretrained`` on it. This module reads ONE block (a text decoder layer, the
vision tower, the embedding / head) straight from the safetensors shards, applies exactly the key
renames HF registers for ``glm5_next`` (``transformers/conversion_mapping.py``) and dequantises the
FP8 weights with exactly HF's ``Fp8Dequantize._dequantize_one`` math ((q.float() * scale) -> bf16
per 128x128 block), then materialises the REAL HF module class for that block. Equality with HF's
own loader.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from functools import lru_cache

import torch

HF_MODEL_ID = "zai-org/GLM-5.3-Flash"


@lru_cache(maxsize=2)
def snapshot_dir(model_id: str = HF_MODEL_ID) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id, local_files_only=True, allow_patterns=["*.json", "*.jinja"])


@lru_cache(maxsize=2)
def load_config(model_id: str = HF_MODEL_ID):
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_id)


def _dequant(q: torch.Tensor, s: torch.Tensor, out_dtype=torch.bfloat16) -> torch.Tensor:
    """HF Fp8Dequantize._dequantize_one (fp32 math, per-block scale), cast to the parameter dtype."""
    qf = q.to(torch.float32)
    rows, cols = qf.shape[-2:]
    sr, sc = s.shape[-2:]
    bm, bn = rows // sr, cols // sc
    shp = qf.shape
    qq = qf.reshape(-1, sr, bm, sc, bn)
    ss = s.to(torch.float32).reshape(-1, sr, sc).unsqueeze(-1).unsqueeze(2)
    return (qq * ss).to(out_dtype).reshape(shp)


# HF's WeightRenaming rules for glm5_next (source -> target), applied to the per-block relative key.
_RENAMES = [
    (r"self_attn\.f_a_proj\.", "self_attn.forget_gate.f_a_proj."),
    (r"self_attn\.f_b_proj\.", "self_attn.forget_gate.f_b_proj."),
    (r"self_attn\.dt_bias", "self_attn.forget_gate.dt_bias"),
    (r"self_attn\.A_log", "self_attn.forget_gate.A_log"),
    (r"hc_attn_fn", "attn_hc.fn"),
    (r"hc_attn_base", "attn_hc.base"),
    (r"hc_attn_scale", "attn_hc.scale"),
    (r"hc_ffn_fn", "ffn_hc.fn"),
    (r"hc_ffn_base", "ffn_hc.base"),
    (r"hc_ffn_scale", "ffn_hc.scale"),
]


class CheckpointReader:
    """Random access to the checkpoint's tensors by block, without loading the model."""

    def __init__(self, model_id: str = HF_MODEL_ID):
        self.model_id = model_id
        from huggingface_hub import snapshot_download

        self.dir = snapshot_download(model_id, local_files_only=True)
        with open(os.path.join(self.dir, "model.safetensors.index.json")) as f:
            self.weight_map = json.load(f)["weight_map"]
        self.config = load_config(model_id)
        self.tcfg = self.config.text_config
        self.vcfg = self.config.vision_config

    # ------------------------------------------------------------------ raw access
    def _read(self, keys):
        from safetensors import safe_open

        by_file = defaultdict(list)
        for k in keys:
            by_file[self.weight_map[k]].append(k)
        out = {}
        for fn, ks in by_file.items():
            with safe_open(os.path.join(self.dir, fn), framework="pt") as f:
                for k in ks:
                    out[k] = f.get_tensor(k)
        return out

    def _block(self, prefix: str):
        """{relative key: tensor} for every checkpoint key under ``prefix`` (fp8 weights dequantised)."""
        keys = [k for k in self.weight_map if k.startswith(prefix)]
        raw = self._read(keys)
        out = {}
        for k, t in raw.items():
            if k.endswith("weight_scale_inv"):
                continue
            rel = k[len(prefix) :]
            if t.dtype == torch.float8_e4m3fn:
                t = _dequant(t, raw[k[: -len("weight")] + "weight_scale_inv"])
            out[rel] = t
        return out

    @staticmethod
    def _assign(module, sd):
        """Load ``sd`` into a meta-initialised HF module, casting to each parameter's declared dtype."""
        want = dict(module.named_parameters())
        want.update(dict(module.named_buffers()))
        fixed = {}
        for k, t in sd.items():
            if k not in want:
                raise KeyError(f"unexpected checkpoint key {k} for {type(module).__name__}")
            # Tensors the checkpoint stores in fp32 (mHC base/scale, A_log, dt_bias, router bias) are
            # params HF keeps in fp32 (_keep_in_fp32); everything else takes the declared dtype.
            fixed[k] = t if t.dtype == torch.float32 else t.to(want[k].dtype)
        missing = [k for k, p in want.items() if k not in fixed and p.device.type == "meta"]
        if missing:
            raise KeyError(f"{type(module).__name__}: checkpoint is missing {missing[:8]}")
        module.load_state_dict(fixed, strict=False, assign=True)
        return module.eval()

    @staticmethod
    def _meta_build(fn):
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device("meta"):
                return fn()
        finally:
            torch.set_default_dtype(prev)

    # ------------------------------------------------------------------ blocks
    def text_layer_state(self, i: int):
        sd = self._block(f"model.language_model.layers.{i}.")
        out = {}
        experts = defaultdict(dict)
        convs = {}
        for k, t in sd.items():
            for src, dst in _RENAMES:
                k = re.sub(src, dst, k)
            m = re.match(r"mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$", k)
            if m:
                experts[m.group(2)][int(m.group(1))] = t
                continue
            m = re.match(r"self_attn\.([qkv])_conv1d\.weight$", k)
            if m:
                convs[m.group(1)] = t
                continue
            out[k] = t
        if experts:
            E = len(experts["gate_proj"])
            gate = torch.stack([experts["gate_proj"][e] for e in range(E)], 0)
            up = torch.stack([experts["up_proj"][e] for e in range(E)], 0)
            # MergeModulelist(0) + Concatenate(1): [E, I, D] x2 -> [E, 2I, D] (gate rows, then up rows)
            out["mlp.experts.gate_up_proj"] = torch.stack([gate, up], dim=1).flatten(1, 2)
            out["mlp.experts.down_proj"] = torch.stack([experts["down_proj"][e] for e in range(E)], 0)
        if convs:
            out["self_attn.conv1d.weight"] = torch.stack([convs["q"], convs["k"], convs["v"]], 0).flatten(0, 1)  # q|k|v
        return out

    def text_layer(self, i: int):
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextDecoderLayer

        layer = self._meta_build(lambda: Glm5NextTextDecoderLayer(self.tcfg, i))
        return self._assign(layer, self.text_layer_state(i))

    def vision_model(self, depth: int | None = None):
        import copy

        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel

        vcfg = copy.deepcopy(self.vcfg)
        if depth is not None:
            vcfg.depth = int(depth)
        vm = self._meta_build(lambda: Glm5NextVisionModel(vcfg))
        sd = self._block("model.visual.")
        sd = {k: v for k, v in sd.items() if not (k.startswith("blocks.") and int(k.split(".")[1]) >= vcfg.depth)}
        # Non-persistent buffers (rotary inv_freq) were built on meta: recompute them on CPU.
        with torch.device("cpu"):
            fresh = type(vm.rotary_pos_emb)(vcfg)
        vm.rotary_pos_emb = fresh
        return self._assign(vm, sd)

    def tensor(self, key: str, dtype=torch.bfloat16):
        t = self._read([key])[key]
        return t.to(dtype)

    def embed_weight(self):
        return self.tensor("model.language_model.embed_tokens.weight")

    def lm_head_weight(self):
        return self.tensor("lm_head.weight")

    def final_norm(self):
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextRMSNorm

        n = Glm5NextTextRMSNorm(self.tcfg.hidden_size, eps=self.tcfg.rms_norm_eps)
        n.weight.data = self.tensor("model.language_model.norm.weight")
        return n.eval()

    def hc_head(self):
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextHyperHead

        return Glm5NextTextHyperHead().eval()

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The HF reference model (Source A), held in host memory at the checkpoint's own FP8 width.

The checkpoint ships its 40 text-decoder layers as FP8 (e4m3) linears with a per-tensor weight_scale_inv. On a
host without a GPU, transformers loads it by DEQUANTIZING every one of them to bf16 (Fp8Dequantize:
bf16 = (fp8.float() * scale).to(bf16)) -- ~48 GB of host RAM. Here those linears keep their FP8 codes (~22 GB)
and produce the very same bf16 weight on access: each of the 256 FP8 codes is dequantized once with
transformers' own arithmetic into a lookup table, so every element equals what the dequantizing load would
have stored. load_reference_model() checks that bit-for-bit against transformers' Fp8Dequantize on one layer.
Everything else (embeddings, norms, lm_head, vision tower, projector) is the checkpoint's bf16 as transformers
loads it. Both the TT weight build (stubs read `<proj>.weight`) and the CPU golden see identical values.
"""
from __future__ import annotations

import contextlib
import functools
import time

import torch

_FP8 = torch.float8_e4m3fn


class Fp8ResidentLinear(torch.nn.Module):
    """A bias-free nn.Linear whose weight is stored as FP8 codes + a per-tensor scale.

    `weight` is a bf16 tensor rebuilt on every access (never cached), equal element-for-element to
    transformers' dequantized weight. The codes and the table are plain attributes, not parameters or buffers,
    so Module.float() / .to(dtype) on the whole model cannot widen them."""

    ROWS_PER_CHUNK = 256  # rows looked up per pass: keeps the int32 index scratch cache-resident

    def __init__(self, fp8_linear):
        super().__init__()
        scale = fp8_linear.weight_scale_inv.detach().to(torch.float32)
        assert scale.numel() == 1, f"per-tensor FP8 scale expected, got shape {tuple(scale.shape)}"
        self.in_features, self.out_features = fp8_linear.in_features, fp8_linear.out_features
        self.bias = None
        self.__dict__["codes"] = fp8_linear.weight.detach().view(torch.uint8)
        self.__dict__["table"] = dequant_table(scale).reshape(256, 1)

    @property
    def weight(self):
        codes = self.codes
        out = torch.empty(codes.shape, dtype=torch.bfloat16)
        idx = torch.empty((min(self.ROWS_PER_CHUNK, codes.shape[0]), codes.shape[1]), dtype=torch.int32)
        for r in range(0, codes.shape[0], self.ROWS_PER_CHUNK):
            n = min(self.ROWS_PER_CHUNK, codes.shape[0] - r)
            idx[:n].copy_(codes[r : r + n])
            out[r : r + n] = torch.nn.functional.embedding(idx[:n], self.table).view(n, -1)
        return out

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, fp8 codes + per-tensor scale"


def dequant_table(scale):
    """bf16 value of each FP8 code under `scale`, computed exactly as transformers' Fp8Dequantize does."""
    codes = torch.arange(256, dtype=torch.uint8).view(_FP8)
    return (codes.to(torch.float32) * scale.to(torch.float32).reshape(-1)[0]).to(torch.bfloat16)


@contextlib.contextmanager
def _keep_fp8_on_cpu():
    """transformers forces dequantize=True when no GPU is present; keep the FP8 tensors as loaded instead."""
    from transformers.quantizers import quantizer_finegrained_fp8 as q

    orig = q.FineGrainedFP8HfQuantizer.validate_environment
    q.FineGrainedFP8HfQuantizer.validate_environment = lambda self, *a, **k: None
    try:
        yield
    finally:
        q.FineGrainedFP8HfQuantizer.validate_environment = orig


def _checkpoint_tensor(model_id, key):
    """One raw tensor from the hub checkpoint's safetensors shards."""
    import json
    from pathlib import Path

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    index = json.loads(Path(hf_hub_download(model_id, "model.safetensors.index.json")).read_text())
    shard = hf_hub_download(model_id, index["weight_map"][key])
    with safe_open(shard, framework="pt") as f:
        return f.get_tensor(key)


def _assert_matches_transformers(model_id, module, prefix):
    """The resident weight of `module` equals transformers' Fp8Dequantize of the raw checkpoint tensors."""
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize

    q, s = _checkpoint_tensor(model_id, prefix + ".weight"), _checkpoint_tensor(model_id, prefix + ".weight_scale_inv")
    ref = Fp8Dequantize(None)._dequantize_one(q, s, output_dtype=torch.bfloat16)
    got = module.weight
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), f"{prefix}: FP8 dequant differs from transformers"


@functools.lru_cache(maxsize=1)
def load_reference_model(model_id):
    """Mistral3ForConditionalGeneration with its FP8 text linears resident as FP8 (see module docstring)."""
    import transformers
    from transformers.integrations.finegrained_fp8 import FP8Linear

    t0 = time.time()
    with _keep_fp8_on_cpu():
        model = transformers.AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
    swapped = 0
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, FP8Linear):
                setattr(mod, child_name, Fp8ResidentLinear(child))
                swapped += 1
    assert swapped, "no FP8 linears found: the checkpoint layout changed"
    # transformers leaves every loaded tensor as a view into a private mmap of the safetensors shard. The bf16
    # tensors (embeddings, norms, lm_head, vision, projector: ~3.6 GB) are handed to ttnn.from_torch as they are,
    # and tt-kmd's host-pin of a private file-backed mapping can stall the upload, so they get their own memory.
    # The FP8 codes stay file-backed: only the table lookup reads them, and the page cache is shared between
    # processes and reclaimable.
    for t in list(model.parameters()) + list(model.buffers()):
        t.data = t.data.clone()
    # The model now computes in bf16 throughout, exactly like transformers' dequantizing load.
    model.hf_quantizer = None
    with contextlib.suppress(AttributeError):
        model.is_quantized = False
    model = model.eval()
    _assert_matches_transformers(
        model_id, model.model.language_model.layers[0].mlp.down_proj, "language_model.model.layers.0.mlp.down_proj"
    )
    print(f"[reference] {model_id}: {swapped} FP8 linears resident as FP8 + scale, loaded in {time.time() - t0:.0f}s")
    return model

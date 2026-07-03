# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the ACE-Step v1.5 end-to-end TTNN pipeline.

This module is imported by BOTH the demo entrypoints and the e2e tests. It
provides: HF reference-model loading (same cascade the per-component PCC tests
use), deterministic e2e input construction (exactly the shapes captured in
Source B), the host-side glue that AceStepConditionGenerationModel.generate_audio
runs *around* the graduated modules (tokenize pre-processing, context-latent
assembly, noise, the ODE euler update), and mesh-safe tensor round-trips.
"""
from __future__ import annotations

import math

import torch

import ttnn

HF_MODEL_ID = "ACE-Step/acestep-v15-base"

# The exact e2e gate configuration from e2e_plan.json::fixed_gate_config.
GATE_CONFIG = {
    "batch": 1,
    "seq_len_latent": 50,
    "text_seq": 50,
    "lyric_seq": 58,
    "refer_frames": 100,
    "text_hidden_dim": 1024,
    "audio_acoustic_hidden_dim": 64,
    "pool_window_size": 5,
    "infer_steps": 4,
    "seed": 1234,
    "diffusion_guidance_scale": 1.0,
    "audio_cover_strength": 1.0,
    "infer_method": "ode",
    "is_covers": 1,
}


def load_hf_model(dtype: str = "bfloat16"):
    """Load the ACE-Step reference model via the planner loader cascade.

    Returns a torch model in float32 on CPU (eval). float32 gives an accurate
    golden reference and still builds bf16 device weights fine when passed to a
    stub's build(device, torch_module)."""
    from scripts.tt_hw_planner.agentic.probe import load_hf_model_cascade

    model, loader_or_err = load_hf_model_cascade(HF_MODEL_ID, torch_dtype=dtype, verbose=False)
    if model is None:
        raise RuntimeError(f"Could not load {HF_MODEL_ID}; last error: {loader_or_err}")
    model.eval()
    model.config._attn_implementation = "eager"
    model = model.float()
    return model


def resolve(module, dotted: str):
    cur = module
    for tok in dotted.replace("[", ".").replace("]", "").split("."):
        if tok == "":
            continue
        cur = cur[int(tok)] if tok.isdigit() else getattr(cur, tok)
    return cur


def _captured_dir():
    import os

    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_captured")


def build_inputs(seed: int = None, dtype=torch.float32, use_captured: bool = True):
    """Deterministic e2e inputs.

    Per the task ("input exactly as collected from Sources A+B"), by default the
    conditioning tensors are the REAL captured Source-B tensors: text/lyric/refer
    embeddings from _captured/ace_step_condition_encoder (the upstream
    processor/text-encoder/feature-extractor outputs), src_latents reconstructed
    by un-patchifying the _captured/ace_step_audio_tokenizer input, and
    chunk_masks read from the _captured/ace_step_di_t_model context_latents. This
    feeds every stage ~its per-component captured input, so the chained TTNN
    pipeline is exercised on the same distribution each stub graduated against.
    Falls back to seeded randn if the captures are absent. Fed IDENTICALLY to the
    HF golden chain and the TT pipeline."""
    cfg = GATE_CONFIG
    if seed is None:
        seed = cfg["seed"]
    B, L, D = cfg["batch"], cfg["seq_len_latent"], cfg["audio_acoustic_hidden_dim"]

    if use_captured:
        try:
            import os

            from einops import rearrange

            cap = _captured_dir()
            ce = torch.load(os.path.join(cap, "ace_step_condition_encoder", "kwargs.pt"), weights_only=False)
            tok_in = torch.load(os.path.join(cap, "ace_step_audio_tokenizer", "args.pt"), weights_only=False)[0]
            dit = torch.load(os.path.join(cap, "ace_step_di_t_model", "kwargs.pt"), weights_only=False)

            def f(x):
                return x.to(dtype) if isinstance(x, torch.Tensor) and x.is_floating_point() else x

            src_latents = rearrange(tok_in, "n t p d -> n (t p) d").to(dtype)
            chunk_masks = dit["context_latents"][..., D:].to(dtype)
            inputs = {
                "text_hidden_states": f(ce["text_hidden_states"]),
                "text_attention_mask": f(ce["text_attention_mask"]),
                "lyric_hidden_states": f(ce["lyric_hidden_states"]),
                "lyric_attention_mask": f(ce["lyric_attention_mask"]),
                "refer_audio_acoustic_hidden_states_packed": f(ce["refer_audio_acoustic_hidden_states_packed"]),
                "refer_audio_order_mask": ce["refer_audio_order_mask"].to(torch.int64),
                "src_latents": src_latents,
                "chunk_masks": chunk_masks,
                "silence_latent": torch.zeros(B, L, D, dtype=dtype),
                "attention_mask": torch.ones(B, L, dtype=dtype),
                "is_covers": torch.tensor([cfg["is_covers"]] * B, dtype=torch.int64),
            }
            return inputs
        except Exception as e:
            print(f"[common] captured inputs unavailable ({e}); falling back to seeded randn", flush=True)

    g = torch.Generator().manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=g, dtype=dtype)

    return {
        "text_hidden_states": rnd(B, cfg["text_seq"], cfg["text_hidden_dim"]),
        "text_attention_mask": torch.ones(B, cfg["text_seq"], dtype=dtype),
        "lyric_hidden_states": rnd(B, cfg["lyric_seq"], cfg["text_hidden_dim"]),
        "lyric_attention_mask": torch.ones(B, cfg["lyric_seq"], dtype=dtype),
        "refer_audio_acoustic_hidden_states_packed": rnd(B, cfg["refer_frames"], D),
        "refer_audio_order_mask": torch.zeros(B, dtype=torch.int64),
        "src_latents": rnd(B, L, D),
        "chunk_masks": torch.ones(B, L, D, dtype=dtype),
        "silence_latent": rnd(B, L, D),
        "attention_mask": torch.ones(B, L, dtype=dtype),
        "is_covers": torch.tensor([cfg["is_covers"]] * B, dtype=torch.int64),
    }


def tokenize_preprocess(x, silence_latent, attention_mask, pool_window_size):
    """Host-torch replica of AceStepConditionGenerationModel.tokenize's steps
    BEFORE self.tokenizer(x): pad to a multiple of pool_window_size, rearrange
    into [N, T//P, P, D] patches, and max-pool the attention mask. Returns
    (x_patched, pooled_attention_mask). The graduated audio_tokenizer stub
    consumes x_patched directly."""
    import torch.nn.functional as F
    from einops import rearrange

    if x.shape[1] % pool_window_size != 0:
        pad_len = pool_window_size - (x.shape[1] % pool_window_size)
        x = torch.cat([x, silence_latent[:1, :pad_len].repeat(x.shape[0], 1, 1)], dim=1)
        attention_mask = F.pad(attention_mask, (0, pad_len), mode="constant", value=0)
    x = rearrange(x, "n (t_patch p) d -> n t_patch p d", p=pool_window_size)
    seq_len = x.shape[1]
    chunk = math.ceil(attention_mask.shape[1] / seq_len)
    attention_mask = attention_mask.to(x.dtype)
    attention_mask = F.max_pool1d(attention_mask.unsqueeze(1), kernel_size=chunk, stride=chunk, ceil_mode=True).squeeze(
        1
    )
    return x, attention_mask


def assemble_context_latents(lm_hints_25hz, src_latents, chunk_masks, is_covers):
    """Host-torch replica of prepare_condition's tail: crop hints to src length,
    select via is_covers, concat chunk_masks -> context_latents [B, L, 128]."""
    lm_hints_25hz = lm_hints_25hz[:, : src_latents.shape[1], :]
    src_latents = torch.where(
        is_covers.unsqueeze(-1).unsqueeze(-1) > 0, lm_hints_25hz.to(src_latents.dtype), src_latents
    )
    context_latents = torch.cat([src_latents, chunk_masks.to(src_latents.dtype)], dim=-1)
    return context_latents


def prepare_noise(context_latents, seed):
    """Replica of AceStepConditionGenerationModel.prepare_noise for a single
    seed. Depends only on shape+seed, so HF and TT get identical noise."""
    shape = (context_latents.shape[0], context_latents.shape[1], context_latents.shape[-1] // 2)
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def ode_timesteps(infer_steps, shift=1.0):
    t = torch.linspace(1.0, 0.0, infer_steps + 1, dtype=torch.float32)
    if shift != 1.0:
        t = shift * t / (1 + (shift - 1) * t)
    return t


def is_mesh_device(device):
    try:
        if isinstance(device, ttnn.MeshDevice):
            return True
    except AttributeError:
        pass
    return hasattr(device, "get_device_ids") or hasattr(device, "get_devices")


def from_torch(tensor, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    t = tensor.to(torch.bfloat16) if dtype == ttnn.bfloat16 else tensor
    if is_mesh_device(device):
        try:
            return ttnn.from_torch(
                t, dtype=dtype, layout=layout, device=device, mesh_mapper=ttnn.ReplicateTensorToMesh(device)
            )
        except (AttributeError, TypeError):
            pass
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


def to_torch(ttnn_tensor, device):
    if isinstance(ttnn_tensor, torch.Tensor):
        return ttnn_tensor
    try:
        if hasattr(ttnn, "synchronize_device"):
            ttnn.synchronize_device(device)
    except Exception:
        pass
    if is_mesh_device(device):
        for mk in (
            lambda: ttnn.concat_mesh_to_tensor_composer(device, 0),
            lambda: ttnn.ConcatMeshToTensor(device, dim=0),
        ):
            try:
                composer = mk()
            except (AttributeError, TypeError):
                continue
            try:
                t = ttnn.to_torch(ttnn_tensor, mesh_composer=composer)
                if t is None:
                    continue
                if t.ndim >= 1 and t.shape[0] > 1:
                    n = 1
                    try:
                        ids = device.get_device_ids() if hasattr(device, "get_device_ids") else []
                        n = len(ids) or 1
                    except Exception:
                        pass
                    if n > 1 and t.shape[0] % n == 0:
                        t = t[: t.shape[0] // n]
                return t
            except Exception:
                continue
    return ttnn.to_torch(ttnn_tensor)


def pcc(golden: torch.Tensor, actual: torch.Tensor):
    """Pearson correlation of two tensors, flattened; matches comp_pcc's metric."""
    from models.common.utility_functions import comp_pcc

    ok, value = comp_pcc(golden.to(torch.float32), actual.to(torch.float32), 0.99)
    return ok, value

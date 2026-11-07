# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF golden for GLM-5.3-Flash image-text-to-text (REFERENCE ONLY -- never called by the TT pipeline).

``Glm5NextForConditionalGeneration.from_pretrained`` cannot run here: in bf16 the model is ~640 GB and the
host has 566 GB. The golden therefore runs HF's OWN module classes (Glm5NextVisionModel,
Glm5NextTextDecoderLayer x45, Glm5NextTextHyperHead, Glm5NextTextRMSNorm, the lm_head weight) with the
checkpoint's weights, in bf16 with the implementations from_pretrained selects (sdpa attention, grouped_mm
experts), reading one decoder layer at a time from the checkpoint -- the same arithmetic as ``model(...)``, ordered layer-major.

``reference_logits`` is a full-sequence (prefill-style) forward over prompt + candidate continuation; its
argmax at position t is exactly what greedy ``generate()`` emits after that prefix. ``teacher_forced``
returns those per-prefix predictions: a continuation equals generate()'s greedy output iff it agrees with
them at every position (by induction over the prefix). ``teacher_forced_references`` runs the same forward at
fp32 (the exact math: the e2e gate's reference) and at bf16 (HF's own precision: the noise floor a correct
bf16 implementation shows on this model), cached per prompt + token trajectory.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import torch

from models.demos.glm_5_3_flash.tt.checkpoint import HF_MODEL_ID, CheckpointReader

# token rows (batch rows x sequence) one text-layer call of the reference processes at a time; host-RAM bound only
REF_ROW_TOKENS = 16384


def _log(msg):
    print(f"[hf-golden] {msg}", flush=True)


def _set_impl(cfg):
    for c in (cfg, getattr(cfg, "text_config", None), getattr(cfg, "vision_config", None)):
        if c is not None:
            c._attn_implementation = "sdpa"
            c._experts_implementation = "grouped_mm"


@torch.no_grad()
def reference_logits(
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    first_pos: int,
    ckpt: Optional[CheckpointReader] = None,
    n_text_layers: Optional[int] = None,
    n_vision_layers: Optional[int] = None,
    keep_hidden: bool = False,
    dtype: torch.dtype = torch.bfloat16,
):
    """input_ids [B, L] (prompt + continuation), pixel_values for ONE image (each row carries the same image),
    image_grid_thw [1,3]. Returns (logits [B, L - first_pos, V] bf16 for positions first_pos..L-1,
    image_embeds [B*R, D], hidden [B, L, D] or None) -- all as Glm5NextForConditionalGeneration.forward."""
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextHyperHead

    cs = ckpt or CheckpointReader(HF_MODEL_ID)
    cfg = cs.config
    _set_impl(cfg)
    tcfg = cfg.text_config
    B, L = input_ids.shape
    n_text = int(n_text_layers or tcfg.num_hidden_layers)
    t0 = time.time()
    # --- vision tower (B images; Glm5NextModel.get_image_features: pixel_values cast to the tower dtype)
    vm = cs.vision_model(depth=n_vision_layers).to(dtype)
    _set_impl(vm.config)
    grid = image_grid_thw.repeat(B, 1)
    img = vm(pixel_values.repeat(B, 1).to(dtype), grid_thw=grid).pooler_output
    del vm
    _log(f"vision done {tuple(img.shape)} in {time.time() - t0:.0f}s")
    # --- embeddings + masked_scatter of the image rows (Glm5NextModel.forward)
    emb = torch.nn.functional.embedding(input_ids, cs.embed_weight().to(dtype))
    mask = (input_ids == int(cfg.image_token_id)).unsqueeze(-1).expand_as(emb)
    emb = emb.masked_scatter(mask, img.to(emb.dtype))
    # --- text model (Glm5NextTextModel.forward), one checkpoint layer at a time
    H = emb.unsqueeze(2).expand(-1, -1, int(tcfg.hc_mult), -1).contiguous()
    am = torch.ones(B, L, dtype=torch.bool)
    pos = torch.arange(L)[None]
    topk = None
    rows = max(1, min(B, REF_ROW_TOKENS // L))
    for i in range(n_text):
        t1 = time.time()
        layer = cs.text_layer(i)
        if dtype != torch.bfloat16:
            layer = layer.to(dtype)
        # batch rows are independent: run the layer over row chunks so the per-layer temporaries (attention /
        # indexer scores grow with L^2 per row) stay bounded on long prompts at large B
        hs, ts = [], []
        for r0 in range(0, B, rows):
            r = slice(r0, r0 + rows)
            h_, t_ = layer(
                H[r],
                attention_mask=am[r],
                position_ids=pos,
                past_key_values=None,
                position_embeddings=None,
                prev_topk_indices=None if topk is None else topk[r],
            )
            hs.append(h_)
            ts.append(t_)
        H = hs[0] if len(hs) == 1 else torch.cat(hs, dim=0)
        topk = None if ts[0] is None else (ts[0] if len(ts) == 1 else torch.cat(ts, dim=0))
        del layer, hs, ts
        _log(f"text layer {i + 1}/{n_text} ({time.time() - t1:.1f}s)")
    h = cs.final_norm().to(dtype)(Glm5NextTextHyperHead()(H))
    W = cs.lm_head_weight().to(dtype)
    logits = torch.nn.functional.linear(h[:, first_pos:], W)  # bf16, as the HF lm_head
    _log(f"reference forward over {B}x{L} tokens in {time.time() - t0:.0f}s")
    return logits, img, (h if keep_hidden else None)


@torch.no_grad()
def reference_image_embeds(pixel_values, image_grid_thw, B, ckpt=None, n_vision_layers=None, dtype=torch.float32):
    """HF Glm5NextVisionModel pooler_output for B copies of the image, at ``dtype`` (fp32 = the exact math)."""
    cs = ckpt or CheckpointReader(HF_MODEL_ID)
    vm = cs.vision_model(depth=n_vision_layers).to(dtype)
    _set_impl(vm.config)
    return vm(pixel_values.repeat(B, 1).to(dtype), grid_thw=image_grid_thw.repeat(B, 1)).pooler_output.float()


def teacher_forced(
    prompt_ids: torch.Tensor,
    continuation,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    ckpt=None,
    n_text_layers=None,
    n_vision_layers=None,
    keep_hidden=False,
):
    """One full-sequence HF forward over prompt + continuation (per row, padded with the row's last token
    beyond its own length). Returns (pred [B, n+1] = HF's greedy next token after each prefix of length
    0..n, logits [B, n+1, V] bf16, image_embeds, final normed hidden [B, L, D] or None). pred[b, t] is what greedy generate() emits after the
    first t continuation tokens, so a continuation c is generate()'s output iff c[t] == pred[b, t] for
    every t and c ends at the first eos (or the length cap)."""
    rows = [list(map(int, r)) for r in continuation]
    n = max(len(r) for r in rows)
    pad = [r + [r[-1] if r else 0] * (n - len(r)) for r in rows]
    seq = torch.cat([prompt_ids, torch.tensor(pad, dtype=prompt_ids.dtype)], dim=1) if n else prompt_ids
    L0 = prompt_ids.shape[1]
    logits, img, hidden = reference_logits(
        seq, pixel_values, image_grid_thw, L0 - 1, ckpt, n_text_layers, n_vision_layers, keep_hidden
    )
    return logits.float().argmax(-1), logits, img, hidden


def golden_cache_path(tag: str) -> str:
    # outside the source tree: one cached reference pair is ~19 GB
    d = os.environ.get("GLM53_GOLDEN_DIR", os.path.expanduser("~/.cache/tt_glm_5_3_flash_golden"))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{tag}.pt")


def trajectory_key(prompt_ids: torch.Tensor, continuation, n_text_layers, n_vision_layers) -> str:
    """The references below are a deterministic function of exactly these inputs (the image is the captured
    example's); a cached entry is reused only for an identical prompt + token trajectory + depth."""
    import hashlib
    import json

    h = hashlib.sha256()
    h.update(json.dumps([prompt_ids.tolist(), [list(map(int, r)) for r in continuation]]).encode())
    h.update(f"t{n_text_layers}_v{n_vision_layers}".encode())
    return h.hexdigest()[:20]


def teacher_forced_references(
    prompt_ids: torch.Tensor,
    continuation,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    ckpt=None,
    n_text_layers=None,
    n_vision_layers=None,
):
    """HF's own modules over prompt + continuation at TWO precisions:
      * ``f32``  -- the exact math (bf16 checkpoint weights, every activation fp32): THE reference;
      * ``bf16`` -- HF's own production precision, the noise floor a correct bf16 implementation shows
                   against the exact math on this model.
    Each: dict(logits [B, n+1, V] (row t = the prediction for continuation token t), hid [B, L0, D] final
    normed hidden over the prompt, img [B*R, D] vision tower output). Cached per trajectory_key."""
    L0 = prompt_ids.shape[1]
    key = trajectory_key(prompt_ids, continuation, n_text_layers, n_vision_layers)
    path = golden_cache_path(f"ref_{key}")
    if os.path.exists(path):
        _log(f"references from cache {path}")
        return torch.load(path, weights_only=False)
    rows = [list(map(int, r)) for r in continuation]
    n = max(len(r) for r in rows)
    pad = [r + [r[-1]] * (n - len(r)) for r in rows]
    seq = torch.cat([prompt_ids, torch.tensor(pad, dtype=prompt_ids.dtype)], dim=1)
    out = {}
    for name, dt in (("f32", torch.float32), ("bf16", torch.bfloat16)):
        logits, img, hid = reference_logits(
            seq, pixel_values, image_grid_thw, L0 - 1, ckpt, n_text_layers, n_vision_layers, True, dt
        )
        out[name] = dict(
            logits=logits if dt == torch.float32 else logits.to(torch.bfloat16),
            hid=hid[:, :L0].float().clone(),
            img=img.float(),
        )
        del logits, hid
    # keep only the latest trajectory's entry (a stale one is never reused)
    d = os.path.dirname(path)
    for f in os.listdir(d):
        if f.startswith("ref_") and f.endswith(".pt"):
            os.remove(os.path.join(d, f))
    torch.save(out, path)
    _log(f"references cached at {path}")
    return out

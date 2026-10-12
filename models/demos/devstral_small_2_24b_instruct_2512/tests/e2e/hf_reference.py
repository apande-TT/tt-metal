# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF golden for the Devstral e2e tests (Source A: the transformers reference on CPU, bf16).

The vision tower + projector of the reference run in fp32 (fp32 copies of the same weights; the 24B text model
stays bf16 and receives the image features cast to bf16, as HF does). The Pixtral encoder grows massive
activations around layer 13 and whether a patch row crosses that onset hinges on low bits a bf16 CPU run drops:
on the gate's image the bf16 reference itself sits at row PCC 0.73 against fp32 on one patch. The reference must
be the model's output, not one rounding's accident, so the stage that is unstable in bf16 is computed in fp32.

Sampling rule (shared with tt/pipeline.py): the checkpoint samples at temperature T = 0.15. Both sides use the
Gumbel-max form -- token = argmax over the top-K logits of logits / T + g -- with the same per-row seeded,
bf16-rounded Gumbel table g, so a matching model draws the same token. On the HF side the rule runs as a
LogitsProcessor inside model.generate() (do_sample=False then picks exactly the drawn token).

The prompt is identical in every row, so it is prefilled once at batch 1 and the cache is replicated to B rows
before model.generate() continues; generate() then runs the real batched decode with its own EOS / pad rule.
Goldens are cached on disk (keyed by everything that determines them) so reruns of the gate do not pay the CPU
reference again.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import os
import time
from pathlib import Path

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from models.demos.devstral_small_2_24b_instruct_2512.tt.pipeline import MODEL_ID, gumbel_table

# Bumped whenever the reference's own computation changes, so a stale cached golden is never reused.
REFERENCE_VERSION = "vision-fp32"
CACHE_DIR = Path(os.environ.get("DEVSTRAL_GOLDEN_CACHE", Path.home() / ".cache" / "devstral_e2e_golden"))


class GumbelTopKSampler(LogitsProcessor):
    """logits -> one-hot of argmax_topK(logits / T + g[step, row, id % buckets]); the TT on-device sampler's rule."""

    def __init__(self, noise, temperature, top_k, prompt_len):
        self.noise, self.temperature, self.top_k, self.prompt_len = noise, float(temperature), int(top_k), prompt_len

    def choose(self, scores, step):
        vals, idx = scores.float().topk(self.top_k, dim=-1)
        rows = torch.arange(idx.shape[0])[:, None]
        noised = vals / self.temperature + self.noise[step][rows, idx % self.noise.shape[-1]]
        order = noised.sort(dim=-1, descending=True)
        pick = idx.gather(-1, order.indices[:, :1])
        margin = (order.values[:, 0] - order.values[:, 1]) * self.temperature  # in logit units
        return pick[:, 0], margin

    def __call__(self, input_ids, scores):
        step = input_ids.shape[1] - self.prompt_len
        if step % 16 == 0:
            print(f"[hf-golden] generate step {step}", flush=True)
        pick, _ = self.choose(scores, step)
        out = torch.full_like(scores, float("-inf"))
        out.scatter_(-1, pick[:, None], 0.0)
        return out


def _key(req, extra):
    h = hashlib.sha256()
    h.update(MODEL_ID.encode())
    h.update(REFERENCE_VERSION.encode())
    h.update(req["tokens"][0, : req["real_len"]].numpy().tobytes())
    h.update(repr((req["batch"], req["seeds"], req["max_new_tokens"], extra)).encode())
    if "pixel_values" in req:
        h.update(req["pixel_values"][0].to(torch.float32).numpy().tobytes())
    return h.hexdigest()[:20]


def _cached(name, req, extra, build):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}_{_key(req, extra)}.pt"
    if path.is_file():
        print(f"[hf-golden] cache hit {path}", flush=True)
        return torch.load(path, weights_only=False)
    out = build()
    buf = io.BytesIO()
    torch.save(out, buf)
    path.write_bytes(buf.getvalue())
    print(f"[hf-golden] cached {path}", flush=True)
    return out


def _prompt_inputs(req):
    hf_inputs = {
        k: v for k, v in req["hf_inputs"].items() if k in ("input_ids", "attention_mask", "pixel_values", "image_sizes")
    }
    if "pixel_values" in hf_inputs:
        hf_inputs["pixel_values"] = hf_inputs["pixel_values"].to(torch.bfloat16)
    return hf_inputs


@contextlib.contextmanager
def _fp32_vision(model):
    """Run the reference's vision tower + projector from fp32 copies of their weights for the duration."""
    m = model.model
    saved = m.vision_tower, m.multi_modal_projector
    m.vision_tower = copy.deepcopy(saved[0]).float()
    m.multi_modal_projector = copy.deepcopy(saved[1]).float()
    try:
        yield
    finally:
        m.vision_tower, m.multi_modal_projector = saved


def _prefill_once(model, req):
    """Batch-1 prefill of the prompt: logits of the last row, post-norm hidden of every prompt row, image
    features (when there is an image) and the KV cache."""
    caught = {}
    with torch.no_grad(), _fp32_vision(model):
        hooks = [model.model.language_model.norm.register_forward_hook(lambda m, i, o: caught.__setitem__("hidden", o))]
        hooks.append(
            model.model.multi_modal_projector.register_forward_hook(lambda m, i, o: caught.__setitem__("image", o))
        )
        try:
            out = model(**_prompt_inputs(req), use_cache=True, logits_to_keep=1)
        finally:
            for h in hooks:
                h.remove()
    return out, caught


def _hf_reference_generate(model, req, top_k, temperature):
    """The INDEPENDENT golden: model.generate() for all B rows under the shared sampling + stop rule."""

    def build():
        torch.set_num_threads(os.cpu_count() or 16)
        t0 = time.time()
        b, L = req["batch"], req["real_len"]
        out, caught = _prefill_once(model, req)
        print(f"[hf-golden] prefill 1x{L} in {time.time() - t0:.1f}s", flush=True)
        cache = out.past_key_values
        cache.crop(L - 1)
        cache.batch_repeat_interleave(b)
        noise = gumbel_table(req["seeds"], req["max_new_tokens"] + 1)
        sampler = GumbelTopKSampler(noise, temperature, top_k, L)
        prompt = req["tokens"][:, :L]
        gen = model.generation_config
        with torch.no_grad():
            res = model.generate(
                input_ids=prompt,
                attention_mask=torch.ones_like(prompt),
                past_key_values=cache,
                logits_processor=LogitsProcessorList([sampler]),
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                max_new_tokens=req["max_new_tokens"],
                eos_token_id=gen.eos_token_id,
                pad_token_id=gen.pad_token_id,
                output_logits=True,
                return_dict_in_generate=True,
            )
        tokens = res.sequences[:, L:]
        logits = torch.stack(res.logits, dim=0).to(torch.bfloat16)  # [T, B, V] raw (pre-processor) logits
        print(f"[hf-golden] generate: {tokens.shape[1]} steps for {b} rows in {time.time() - t0:.1f}s", flush=True)
        return {
            "tokens": tokens,
            "logits": logits,
            "prefill_hidden": caught["hidden"][0].float(),
            "image_features": caught.get("image").float() if caught.get("image") is not None else None,
            "prompt_logits": out.logits[0, -1].float(),
        }

    return _cached("generate", req, (top_k, temperature), build)


def _hf_reference_teacher_forced(model, req, tt_tokens, top_k, temperature, chunk=64):
    """HF teacher-forced onto the TT trajectory: logits at every step given the TT-fed tokens, and HF's own
    choice under the shared sampler (with its noised top-2 margin)."""
    b, L = req["batch"], req["real_len"]
    t_total = tt_tokens.shape[1]
    torch.set_num_threads(os.cpu_count() or 16)
    t0 = time.time()
    out, _ = _prefill_once(model, req)
    cache = out.past_key_values
    cache.batch_repeat_interleave(b)
    step_logits = [out.logits[:, -1].float().expand(b, -1)]
    with torch.no_grad():
        for s in range(0, t_total - 1, chunk):
            feed = tt_tokens[:, s : min(s + chunk, t_total - 1)]
            o = model(input_ids=feed, past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            step_logits += [o.logits[:, j].float() for j in range(feed.shape[1])]
            print(f"[hf-golden] teacher-forced {s + feed.shape[1]}/{t_total - 1} ({time.time() - t0:.1f}s)", flush=True)
    noise = gumbel_table(req["seeds"], req["max_new_tokens"] + 1)
    sampler = GumbelTopKSampler(noise, temperature, top_k, 0)
    picks, margins = zip(*(sampler.choose(lg, t) for t, lg in enumerate(step_logits)))
    return {
        "logits": torch.stack(step_logits, dim=0).to(torch.bfloat16),
        "choice": torch.stack(picks, dim=1),
        "margin": torch.stack(margins, dim=1),
    }

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reference (golden) side of the Kolibri-1 e2e test. Host torch only; nothing here runs on the device.

The reference model is the torch port in tests/pcc/_reference_loader.py (fp32 forward, FP8 weights
dequantised per call). Two goldens are built from it:

  * independent: the reference generates on its OWN, free-running, with the same sampling rule and the
    same per-(sample, position) uniforms as the TT pipeline. Nothing in it depends on the TT run, so it
    is computed once and cached on disk (keyed by everything that determines it).
  * teacher-forced: one causal forward over the TT pipeline's own sequences (prompt + generated
    tokens), streamed one decoder layer at a time, giving the reference logits for exactly the prefix
    every TT step saw.
  * precision baseline: the reference run at bf16 activations (its config dtype) against its fp32 run on
    the independent golden's trajectories -- the bar the TT pipeline's discrete agreement is held to.

The sampling rule is generation_config's (top_k=128 -> top_p=0.97 -> temperature 1.0), drawn by inverse
CDF with one uniform per (sample, position); the CDF runs in TOKEN-ID order, so a small change in the
logits moves every CDF boundary by at most twice the total-variation distance between the two
distributions (sorted-order CDFs jump when two near-equal tokens swap places).

Usage (precompute the independent golden; the e2e test also builds it on a cache miss):
    python -m models.demos.kolibri_1.tests.e2e.golden
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

import torch

from models.demos.kolibri_1.tests.pcc import _reference_loader as rl
from models.demos.kolibri_1.tt import inputs as kin
from models.demos.kolibri_1.tt.checkpoint import Checkpoint

GOLDEN_SEEDS = 32  # the independent golden covers seeds 0..31; a smaller batch uses its first rows
CACHE_DIR = Path(os.environ.get("KOLIBRI_E2E_CACHE", Path.home() / ".cache" / "tt_kolibri_1_e2e"))


# ----------------------------------------------------------------------------------------------- sampler
def host_sample(logits: torch.Tensor, u: torch.Tensor, settings: kin.SamplingSettings):
    """The pipeline's sampling rule on host, in float64. logits [N, V], u [N].

    Returns (token [N], q [N, V] normalised kept-set probabilities, cdf [N, V] inclusive in token-id order)."""
    x = logits.double() / settings.temperature
    vals = torch.topk(x, settings.top_k, dim=-1).values
    p = torch.softmax(vals, dim=-1)
    excl = torch.cumsum(p, dim=-1) - p
    n_keep = (excl < settings.top_p).sum(-1).clamp_min(1)
    cut = vals.gather(1, (n_keep - 1)[:, None])
    q = torch.where(x >= cut, torch.exp(x - vals[:, :1]), torch.zeros_like(x))
    cdf = torch.cumsum(q, dim=-1)
    z = cdf[:, -1:]
    tok = (cdf <= u.double()[:, None] * z).sum(-1)
    return tok, q / z, cdf / z


def interval_margin(cdf: torch.Tensor, tok: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Distance from u to the nearer edge of the chosen token's CDF interval [cdf[tok-1], cdf[tok])."""
    hi = cdf.gather(1, tok[:, None])[:, 0]
    lo = torch.where(tok > 0, cdf.gather(1, (tok - 1).clamp_min(0)[:, None])[:, 0], torch.zeros_like(hi))
    u = u.double()
    return torch.minimum(u - lo, hi - u)


# ---------------------------------------------------------------------------------------- cache identity
def _source_digest(uniforms: torch.Tensor) -> str:
    """Everything else that determines the golden: the reference model's code, the sampling rule, the
    generation loop, and the uniforms themselves (not this file as a whole, so editing the comparison
    code does not throw away an hours-long golden)."""
    h = hashlib.sha256(Path(rl.__file__).read_bytes())
    for fn in (host_sample, _hf_reference_text_generation):
        h.update(inspect.getsource(fn).encode())
    h.update(uniforms.numpy().tobytes())
    return h.hexdigest()[:16]


def golden_key(prompt_ids, settings, capacity: int, uniforms: torch.Tensor) -> str:
    blob = json.dumps(
        {
            "prompt": list(prompt_ids),
            "seeds": GOLDEN_SEEDS,
            "settings": [settings.top_k, settings.top_p, settings.temperature, list(settings.stop_ids)],
            "capacity": int(capacity),
            "src": _source_digest(uniforms),
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def golden_path(prompt_ids, settings, capacity: int, uniforms: torch.Tensor) -> Path:
    return CACHE_DIR / f"independent_{golden_key(prompt_ids, settings, capacity, uniforms)}.pt"


# ---------------------------------------------------------------------------- independent (free-running)
def _hf_reference_text_generation(ref, prompt_ids, uniforms, settings, capacity: int, log=print) -> dict:
    """Free-running reference generation over GOLDEN_SEEDS samples (same prompt, seed b on row b).

    The token drawn from the logits at position p lands at position p + 1; a sample stops on its first
    stop token; nothing is produced past position capacity - 1 (the TT KV capacity, the same cap)."""
    B, T = uniforms.shape[1], len(prompt_ids)
    stop = set(settings.stop_ids)
    ids = torch.tensor([list(prompt_ids)] * B)
    cache = [[] for _ in ref.model.layers]
    t0 = time.time()
    with torch.no_grad():
        out = ref(ids, past_key_values=cache, output_hidden_states=True)
    prefill_hidden = out.hidden_states[-1].float().clone()
    prefill_logits = out.logits[:, -1].float().clone()
    tok, _, _ = host_sample(prefill_logits, uniforms[T - 1], settings)
    seqs = [[int(t)] for t in tok]
    active = [b for b in range(B) if int(tok[b]) not in stop]
    log(f"[golden] prefill {time.time() - t0:.0f}s, {len(active)} rows active", flush=True)
    cache = [[c[0][active], c[1][active], c[2][active]] for c in cache]
    pos = T
    while active and pos + 1 <= capacity - 1:
        t1 = time.time()
        rows = torch.tensor(active)
        step_ids = torch.tensor([[seqs[b][-1]] for b in active])
        step_pos = torch.full((len(active), 1), pos, dtype=torch.long)
        with torch.no_grad():
            logits = ref(step_ids, position_ids=step_pos, past_key_values=cache).logits[:, -1].float()
        tok, _, _ = host_sample(logits, uniforms[pos, rows], settings)
        keep = []
        for j, b in enumerate(active):
            seqs[b].append(int(tok[j]))
            if int(tok[j]) not in stop:
                keep.append(j)
        cache = [[c[0][keep], c[1][keep], c[2][keep]] for c in cache]
        active = [active[j] for j in keep]
        pos += 1
        log(f"[golden] position {pos}: {len(active)} rows active ({time.time() - t1:.1f}s)", flush=True)
    return {
        "tokens": seqs,
        "ended_on_stop": [s[-1] in stop for s in seqs],
        "prefill_hidden": prefill_hidden,
        "prefill_logits": prefill_logits,
        "prompt_ids": list(prompt_ids),
        "capacity": int(capacity),
    }


def independent_golden(capacity: int, log=print) -> dict:
    """The cached independent golden; built (with the whole reference model in host RAM) on a miss."""
    tok = kin.load_tokenizer()
    prompt = kin.encode_prompt(tok)
    settings = kin.sampling_settings()
    uniforms = kin.sampling_uniforms(range(GOLDEN_SEEDS), kin.uniforms_length())
    path = golden_path(prompt, settings, capacity, uniforms)
    if path.is_file():
        g = torch.load(path, weights_only=False)
        g["_path"] = str(path)
        return g
    log(f"[golden] cache miss -> generating {path.name} (fp32 reference on CPU, slow)", flush=True)
    torch.set_num_threads(int(os.environ.get("KOLIBRI_REF_THREADS", "8")))
    ref = rl.load_reference_model(kin.MODEL_ID)
    g = _hf_reference_text_generation(ref, prompt, uniforms, settings, capacity, log=log)
    del ref
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(g, tmp)
    tmp.replace(path)
    g["_path"] = str(path)
    return g


# ------------------------------------------------------------------------------------ teacher-forced
def _hf_reference_teacher_forced(sequences, layer_ids=None, compute_dtype=torch.float32, log=print) -> dict:
    """Reference final hidden states for every position of each sequence (one causal pass, streamed one
    decoder layer at a time). Rows are right-padded; causal attention keeps padding out of real positions.
    compute_dtype is the reference model's own activation-dtype option (Kolibri1Model.forward).

    Returns {"hidden": [B, L, H] fp32 after the final norm, "lm_head": [V, H] fp32 weight}."""
    ck = Checkpoint()
    B, L = len(sequences), max(len(s) for s in sequences)
    pad = int(ck.config.pad_token_id)
    ids = torch.tensor([list(s) + [pad] * (L - len(s)) for s in sequences])
    pos = torch.arange(L)[None].expand(B, L)
    with torch.no_grad():
        h = ck.embed_tokens()(ids).to(compute_dtype)
        t0 = time.time()
        ids_ = list(range(int(ck.config.num_hidden_layers))) if layer_ids is None else list(layer_ids)
        for n, i in enumerate(ids_):
            h = ck.decoder_layer(i)(h, pos, None, None)
            if n % 10 == 9 or n == len(ids_) - 1:
                log(f"[golden] teacher-forced layer {n + 1}/{len(ids_)} ({time.time() - t0:.0f}s)", flush=True)
        h = ck.final_norm()(h).float()
    return {"hidden": h, "lm_head": ck.lm_head().weight.float()}


def step_agreement(lt, lr, u, settings) -> dict:
    """Discrete agreement of one batch of steps: candidate logits lt vs reference logits lr [N, V], same
    uniforms u [N]. decisive = the reference's sampling margin exceeds 2 x TV, so a candidate that samples
    by the same rule from its own logits MUST pick the reference's token there."""
    cand, q_c, cdf_c = host_sample(lt, u, settings)
    ref, q_r, cdf_r = host_sample(lr, u, settings)
    tv = 0.5 * (q_c - q_r).abs().sum(-1)
    margin = interval_margin(cdf_r, ref, u)
    return {
        "cand": cand,
        "cand_margin": interval_margin(cdf_c, cand, u),
        "ref": ref,
        "tv": tv,
        "margin": margin,
        "decisive": margin > 2 * tv + 1e-5,
    }


def precision_baseline(gold: dict, log=print) -> dict:
    """The bar for discrete agreement, read from the reference itself: on the independent golden's own
    trajectories, how often the reference run at bf16 activations (the checkpoint's config dtype) samples
    exactly what its fp32 run samples, and on what share of steps that is guaranteed (decisive). Cached
    beside the golden it is computed from."""
    settings = kin.sampling_settings()
    path = Path(str(gold["_path"]).replace("independent_", "bf16_baseline_")) if "_path" in gold else None
    if path is not None and path.is_file():
        return torch.load(path, weights_only=False)
    prompt = gold["prompt_ids"]
    T = len(prompt)
    seqs = [list(prompt) + list(t) for t in gold["tokens"]]
    f32 = _hf_reference_teacher_forced(seqs, log=log)
    b16 = _hf_reference_teacher_forced(seqs, compute_dtype=torch.bfloat16, log=log)
    W = f32["lm_head"]
    uni = kin.sampling_uniforms(range(len(seqs)), kin.uniforms_length())
    agree = decisive = total = 0
    tvs = []
    for t in range(max(len(g) for g in gold["tokens"])):
        rows = [b for b, g in enumerate(gold["tokens"]) if t < len(g)]
        r = torch.tensor(rows)
        st = step_agreement(
            b16["hidden"][r, T - 1 + t] @ W.t(), f32["hidden"][r, T - 1 + t] @ W.t(), uni[T - 1 + t, r], settings
        )
        agree += int((st["cand"] == st["ref"]).sum())
        decisive += int(st["decisive"].sum())
        total += len(rows)
        tvs.append(st["tv"])
    out = {
        "agree_frac": agree / total,
        "decisive_frac": decisive / total,
        "tv_mean": float(torch.cat(tvs).mean()),
        "steps": total,
    }
    log(f"[golden] bf16-activation reference vs fp32: {out}", flush=True)
    if path is not None:
        torch.save(out, path)
    return out


if __name__ == "__main__":
    from models.demos.kolibri_1.tt.pipeline import KV_CAPACITY

    g = independent_golden(int(sys.argv[1]) if len(sys.argv) > 1 else KV_CAPACITY)
    precision_baseline(g)
    tok = kin.load_tokenizer()
    lens = [len(s) for s in g["tokens"]]
    print(
        f"[golden] done: lengths min {min(lens)} max {max(lens)}; ended on stop {sum(g['ended_on_stop'])}/{len(lens)}"
    )
    for b in range(min(4, len(lens))):
        print(f"[golden] sample {b}: {tok.decode(g['tokens'][b])!r}")

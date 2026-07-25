# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF-referenced end-to-end PCC gate for the self-contained Llama-3.1-8B-Instruct demo.

WHY THIS EXISTS
    The previous gate (``test_pcc.py``) asserts top-1 TOKEN ACCURACY, which cannot bound PCC.
    Quantization error is proportional to magnitude, so it preserves the argmax ordering while
    corrupting the values. Measured on realistic logits: a bf4_b-style weight lever sits at
    PCC 0.513 with 100% top-1 match -- it clears an 0.86 accuracy floor untouched. Since optimize
    walks knob:dtype bf16 -> bf8_b -> bf4_b, that is not hypothetical.

WHAT IT MEASURES
    Raw decode LOGITS from the resident TT generator vs the same logits from HuggingFace, on a
    fixed greedy prompt. Logits are captured BEFORE argmax -- capturing after is exactly what makes
    a token-accuracy gate blind.

REFERENCE CACHING
    This gate runs after EVERY edit, so HF must not run every time: the reference logits are
    computed once and cached next to this file. Delete the cache to regenerate.

    Emits ``PCC: <float>`` on stdout, which is the contract the optimize loop parses.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

os.environ.setdefault("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
HF_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# The ONE correctness floor for this gate (the optimize loop lifts this number out of the source).
LLAMA31_8B_PCC_MIN = 0.95

# Fixed, deterministic prompt: the only thing allowed to vary between runs is the model math.
PROMPT = "The capital of France is"
N_DECODE_STEPS = 4

# Cached OUTSIDE the repo: optimize runs each iteration from a fresh temp worktree, so a cache
# living next to this file would be absent every time and force a full HF CPU run per iteration.
_REF_CACHE = Path(
    os.environ.get("LLAMA_PCC_REF_CACHE") or (Path.home() / ".cache" / "tt_pcc_ref" / "llama31_8b_instruct_logits.pt")
)


def _reference_logits(tokenizer_ids: torch.Tensor) -> torch.Tensor:
    """Reference decode logits from HF, cached on disk after the first computation."""
    if _REF_CACHE.is_file():
        blob = torch.load(_REF_CACHE, map_location="cpu")
        if blob.get("prompt") == PROMPT and blob.get("steps") == N_DECODE_STEPS:
            return blob["logits"]

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(HF_MODEL_ID, torch_dtype=torch.float32, device_map="cpu")
    model.eval()
    ids = tokenizer_ids.clone()
    steps = []
    with torch.no_grad():
        for _ in range(N_DECODE_STEPS):
            out = model(ids).logits[:, -1, :]  # logits for the NEXT token
            steps.append(out.float().cpu())
            ids = torch.cat([ids, out.argmax(-1, keepdim=True)], dim=1)  # greedy teacher-forcing
    logits = torch.stack(steps, dim=1)  # [batch, steps, vocab]
    _REF_CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prompt": PROMPT, "steps": N_DECODE_STEPS, "logits": logits}, _REF_CACHE)
    return logits


def _as_logits(out):
    """decode_forward/prefill_forward_text may hand back a bare tensor or a tuple; take the tensor."""
    if isinstance(out, (tuple, list)):
        for item in out:
            if isinstance(item, torch.Tensor):
                return item
        raise TypeError("no tensor in generator output: %r" % (type(out),))
    return out


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation over the flattened tensors.

    A zero denominator means one side is CONSTANT after centering. The reference is never constant,
    so that indicates a degenerate device output -- report 0.0, never 1.0.
    """
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom == 0:
        return 0.0
    return float((a @ b).item() / denom)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_e2e_pcc_hf(mesh_device, reset_seeds):
    """TT decode logits vs HF decode logits on a fixed prompt; gate on PCC."""
    from transformers import AutoTokenizer

    from models.demos.llama3_1_8b_p150.tt.pipeline import build_pipeline

    tok = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    ids = tok(PROMPT, return_tensors="pt").input_ids

    ref = _reference_logits(ids)
    # TEACHER FORCING: feed the REFERENCE's tokens into the TT decode, not TT's own predictions.
    # Letting each side continue from its own argmax means one divergence makes every later step
    # compare different contexts, so the PCC measures drift in the prompt rather than in the math.
    ref_tokens = ref.argmax(-1)  # [batch, steps]

    generator = build_pipeline(mesh_device, max_seq_len=1024, batch_size=1)
    prompt = ids
    decoding_pos = [int(prompt.shape[1])]

    # sampling_params=None keeps RAW LOGITS: with greedy params the generator samples on device and
    # hands back a token, which is precisely the information a correctness gate must not lose.
    prefill_out = generator.prefill_forward_text(
        prompt,
        page_table=generator.page_table,
        kv_cache=generator.tt_kv_cache,
        prompt_lens=decoding_pos,
        sampling_params=None,
        warmup_prefill=True,
        enable_trace=True,
    )
    prefill_logits = _as_logits(prefill_out)
    got = [prefill_logits.float().cpu().reshape(1, -1)]
    cur_tok = ref_tokens[:, 0].reshape(1, 1)  # teacher-forced, not TT's own argmax
    cur_pos = torch.tensor(decoding_pos)

    for i in range(N_DECODE_STEPS - 1):
        out = generator.decode_forward(
            cur_tok,
            cur_pos,
            enable_trace=True,
            page_table=generator.page_table,
            kv_cache=generator.tt_kv_cache,
            reset_batch=(i == 0),
            sampling_params=None,
            prompt_tokens=prompt,
            output_tokens=cur_tok,
        )
        step_logits = _as_logits(out)
        got.append(step_logits.float().cpu().reshape(1, -1))
        cur_tok = ref_tokens[:, i + 1].reshape(1, 1)  # teacher-forced
        cur_pos = cur_pos + 1

    tt_logits = torch.stack(got, dim=1)
    n = min(tt_logits.shape[1], ref.shape[1])
    v = min(tt_logits.shape[-1], ref.shape[-1])
    pcc = _pcc(tt_logits[:, :n, :v], ref[:, :n, :v])

    print(f"PCC: {pcc:.6f}")  # the contract the optimize loop parses
    assert pcc >= LLAMA31_8B_PCC_MIN, f"e2e logits PCC {pcc:.6f} < floor {LLAMA31_8B_PCC_MIN}"

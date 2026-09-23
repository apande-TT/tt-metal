# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Section gate for `tt/text_stack.py` -- the composed, KV-cached 26-layer Voxtral text decoder.

Driven at B=32 with the 32 DISTINCT prompts `common.build_batch_inputs()` builds, and every
number is printed on every run, pass or fail:

  * all NINE stubs this section owns are INVOKED inside the real forward path
    (`common.InvocationCounter`), with their output feeding the next layer;
  * PCC(TT prefill hidden, `hf_model.model(input_ids).last_hidden_state`) >= 0.99 PER SAMPLE;
  * PCC(TT `decode_step` chain over 4 steps, the reference's OWN cached decode) >= 0.99 per
    sample and per step;
  * PCC(TT cached decode, TT full-RECOMPUTE prefill of the same grown sequence) -- the proof that
    the KV cache is what the recompute would have produced, not merely self-consistent;
  * the 32 outputs are pairwise DISTINCT: 32 identical rows would make every PCC above
    meaningless, and a stub hardcoding a leading 1 drops samples 1..31 silently rather than
    failing.

    cd /home/ttuser/tt-metal && flock /tmp/tt_dev0.lock ./python_env/bin/python -m pytest \\
        models/demos/voxtral_4b_tts_2603/tests/section/test_text_stack.py -svv
"""
from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import common
from models.demos.voxtral_4b_tts_2603.tt import text_stack as ts

# The repo pytest.ini's 300 s default is a generic unit-test guard: this module builds a 26-layer
# 4 B decoder on device with a 3.4 B fp32 HF golden beside it on CPU. A bound is still enforced,
# so a genuine hang fails here rather than hanging forever.
pytestmark = pytest.mark.timeout(1800)

DEVICE_ID = int(os.environ.get("VOXTRAL_SECTION_DEVICE_ID", "0"))
PCC_TARGET = 0.99
CACHE_PCC_TARGET = 0.9999
BATCH = common.DEFAULT_BATCH
SEQ_LEN = common.DEFAULT_SEQ_LEN
DECODE_STEPS = 4


@pytest.fixture(scope="module")
def hf_model():
    common.use_all_cpu_threads()
    return common.load_reference_model()


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(
        device_id=DEVICE_ID,
        l1_small_size=24576,
        trace_region_size=200 * 1024 * 1024,
        num_command_queues=1,
    )
    try:
        yield dev
    finally:
        ttnn.close_device(dev)


def _inputs():
    """`(prefill_ids [B, S], step_ids [B, DECODE_STEPS])` -- all rows genuine, pairwise distinct.

    The continuation tokens are each row's OWN tokens 1..4 rather than tokens 32.. : the shortest
    of the 32 prompts tokenizes to 35, so 32 + 4 real positions do not exist for every row, and
    padding one would stop it being a genuine sample. Token 0 is the shared BOS, which is why the
    slice starts at 1 -- the same ids are fed to BOTH sides, so the comparison is unaffected either
    way, but per-row-distinct step tokens keep the distinctness check meaningful.
    """
    ids, texts = common.build_batch_inputs(BATCH, seq_len=SEQ_LEN)
    return ids[:, :SEQ_LEN].contiguous(), ids[:, 1 : 1 + DECODE_STEPS].contiguous(), texts


def _reference(hf, prefill_ids, step_ids):
    """`(prefill_hidden [B, S, H], decode_hidden [steps][B, H])` from the HF reference's own cache."""

    def compute():
        from transformers.cache_utils import DynamicCache

        with torch.no_grad():
            cache = DynamicCache()
            out = hf.model(input_ids=prefill_ids, past_key_values=cache, use_cache=True)
            prefill_hidden = out.last_hidden_state.detach().to(torch.float32).clone()
            past = out.past_key_values
            steps = []
            for t in range(int(step_ids.shape[1])):
                out = hf.model(
                    input_ids=step_ids[:, t : t + 1],
                    past_key_values=past,
                    use_cache=True,
                )
                steps.append(out.last_hidden_state[:, -1, :].detach().to(torch.float32).clone())
                past = out.past_key_values
        return prefill_hidden, steps

    key = common.golden_key(
        what="text_stack_section",
        prefill_ids=prefill_ids,
        step_ids=step_ids,
        batch=int(prefill_ids.shape[0]),
    )
    return common.cached_golden(key, compute)


def _per_sample_pcc(tt, ref):
    return [common.pcc(tt[i], ref[i]) for i in range(int(tt.shape[0]))]


def _report(label, values, target):
    lo = min(values)
    print(f"{label} min PCC={lo:.6f} (target {target}) over {len(values)} samples", flush=True)
    print(f"{label} per-sample PCC={[round(v, 6) for v in values]}", flush=True)
    return lo


def _ids_tt(device, ids):
    return ttnn.from_torch(ids.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def _pairwise_distinct(rows):
    """True when no two rows are bit-identical. `rows` is `[B, H]` on the host."""
    batch = int(rows.shape[0])
    for i in range(batch):
        for j in range(i + 1, batch):
            if torch.equal(rows[i], rows[j]):
                return False, (i, j)
    return True, None


def test_text_stack_batch32(device, hf_model):
    prefill_ids, step_ids, texts = _inputs()
    assert len({tuple(r.tolist()) for r in prefill_ids}) == BATCH, "prefill rows not distinct"
    print(f"batch={BATCH} seq_len={SEQ_LEN} decode_steps={DECODE_STEPS}", flush=True)

    counter = common.InvocationCounter()
    stack = ts.build_text_stack(device, hf_model, counter=counter)
    print(f"built {stack!r}", flush=True)
    assert stack.n_layers == len(hf_model.model.layers)
    assert isinstance(stack.blocks, list)
    assert len({type(b) for b in stack.blocks}) == 1, "blocks are not same-typed"
    assert not hasattr(ts.TextBlock, "__slots__")
    assert {b.kind for b in stack.blocks} == set(ts.BLOCK_KINDS), "a block kind never got routed"

    ref_prefill, ref_steps = _reference(hf_model, prefill_ids, step_ids)
    print(f"reference prefill hidden {tuple(ref_prefill.shape)}", flush=True)

    # ---- prefill ---------------------------------------------------------------------
    ids_tt = _ids_tt(device, prefill_ids)
    hidden, last = stack.prefill(ids_tt)
    assert list(hidden.shape) == [BATCH, 1, SEQ_LEN, stack.hidden_size]
    assert list(last.shape) == [BATCH, stack.hidden_size]
    tt_hidden = ttnn.to_torch(hidden).to(torch.float32).reshape(BATCH, SEQ_LEN, stack.hidden_size)
    tt_last = ttnn.to_torch(last).to(torch.float32)
    ttnn.deallocate(hidden)
    ttnn.deallocate(last)

    prefill_pccs = _per_sample_pcc(tt_hidden, ref_prefill)
    prefill_min = _report("prefill hidden", prefill_pccs, PCC_TARGET)
    last_pccs = _per_sample_pcc(tt_last, ref_prefill[:, -1, :])
    last_min = _report("prefill last_hidden", last_pccs, PCC_TARGET)

    ok, pair = _pairwise_distinct(tt_last)
    print(f"prefill last_hidden pairwise distinct={ok}" + (f" (rows {pair} identical)" if pair else ""), flush=True)

    # ---- cached decode ---------------------------------------------------------------
    tt_steps = []
    for t in range(DECODE_STEPS):
        step_tt = _ids_tt(device, step_ids[:, t : t + 1])
        embeds = stack.embed(step_tt)
        assert list(embeds.shape) == [BATCH, 1, 1, stack.hidden_size]
        out = stack.decode_step(embeds, SEQ_LEN + t)
        assert list(out.shape) == [BATCH, stack.hidden_size]
        tt_steps.append(ttnn.to_torch(out).to(torch.float32))
        ttnn.deallocate(out)

    decode_mins = []
    for t in range(DECODE_STEPS):
        pccs = _per_sample_pcc(tt_steps[t], ref_steps[t])
        decode_mins.append(_report(f"decode step {t} (pos {SEQ_LEN + t})", pccs, PCC_TARGET))
    decode_min = min(decode_mins)

    distinct_flags = []
    for t in range(DECODE_STEPS):
        ok_t, pair_t = _pairwise_distinct(tt_steps[t])
        distinct_flags.append(ok_t)
        print(
            f"decode step {t} pairwise distinct={ok_t}" + (f" (rows {pair_t} identical)" if pair_t else ""),
            flush=True,
        )

    # ---- the cache against a full RECOMPUTE of the same grown sequence ----------------
    # The decode chain is only right if it equals what recomputing the whole grown prefix would
    # have produced. Same stack, same weights, cache OFF: prefill the 32 + DECODE_STEPS sequence
    # and read the last row.
    grown = torch.cat([prefill_ids, step_ids], dim=1)
    pad = (-grown.shape[1]) % ttnn.TILE_SIZE
    if pad:
        grown = torch.cat([grown, grown[:, :pad]], dim=1)
    stack.reset_cache()
    grown_tt = _ids_tt(device, grown)
    recompute_hidden, _ = stack.prefill_embeds(stack.embed(grown_tt))
    rec = ttnn.to_torch(recompute_hidden).to(torch.float32).reshape(BATCH, int(grown.shape[1]), stack.hidden_size)
    ttnn.deallocate(recompute_hidden)
    cache_pccs = _per_sample_pcc(tt_steps[DECODE_STEPS - 1], rec[:, SEQ_LEN + DECODE_STEPS - 1, :])
    cache_min = _report("cached decode vs full recompute", cache_pccs, CACHE_PCC_TARGET)

    # ---- gate 2: every owned stub invoked --------------------------------------------
    counts = dict(counter.counts)
    print(f"invocation counts={counts}", flush=True)
    missing = [n for n in ts.OWNED_STUBS if counts.get(n, 0) < 1]
    print(f"owned stubs never invoked={missing}", flush=True)

    print(f"text_stack prefill PCC={prefill_min}", flush=True)
    print(f"text_stack decode PCC={decode_min}", flush=True)
    print(f"text_stack cache-vs-recompute PCC={cache_min}", flush=True)

    assert not missing, f"stubs never invoked: {missing}"
    assert all(distinct_flags), "decode rows are not pairwise distinct"
    assert ok, "prefill rows are not pairwise distinct"
    assert prefill_min >= PCC_TARGET, f"prefill hidden PCC {prefill_min} < {PCC_TARGET}"
    assert last_min >= PCC_TARGET, f"prefill last_hidden PCC {last_min} < {PCC_TARGET}"
    assert decode_min >= PCC_TARGET, f"decode PCC {decode_min} < {PCC_TARGET}"
    assert cache_min >= CACHE_PCC_TARGET, f"cache-vs-recompute PCC {cache_min} < {CACHE_PCC_TARGET}"

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Call 2 demo -- text -> `last_hidden_state` on the graduated TTNN `model` stub.

Run it::

    PYTHONPATH=$PWD ./python_env/bin/python -m models.demos.voxtral_4b_tts_2603.demo.demo_hidden_states
    ... --batch 8 --seq-len 32 --layers 4 --device-id 1

It builds through `tt.pipeline.build_pipeline(...)` and calls
`pipeline.run_hidden_states(...)` -- the SAME entry point the e2e test drives,
so a green test guarantees a working demo.

For every sample it prints the prompt, the output shape, the post-final-norm L2
next to HF's, and a few leading values of the last token's hidden state.
"""
from __future__ import annotations

import argparse

import torch

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import common
from models.demos.voxtral_4b_tts_2603.tt import hidden_states as hs
from models.demos.voxtral_4b_tts_2603.tt import pipeline as pipeline_mod


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Voxtral-4B-TTS-2603 Call 2: hidden-state extraction on TTNN")
    p.add_argument("--batch", type=int, default=common.DEFAULT_BATCH, help="independent prompts driven together")
    p.add_argument("--seq-len", type=int, default=common.DEFAULT_SEQ_LEN, help="real tokens per prompt (no padding)")
    p.add_argument(
        "--layers",
        type=int,
        default=None,
        help="cap the decoder depth built (default: every layer). Below full depth the HF golden "
        "is truncated to match, so the PCC stays meaningful.",
    )
    p.add_argument("--device-id", type=int, default=1, help="Tenstorrent device to open")
    p.add_argument("--show", type=int, default=4, help="leading hidden values to print per sample")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    input_ids, texts = common.build_batch_inputs(batch=args.batch, seq_len=args.seq_len)
    hf_model = common.load_reference_model()

    device = ttnn.open_device(device_id=args.device_id, l1_small_size=24576)
    try:
        pipeline = pipeline_mod.build_pipeline(
            device,
            model=hf_model,
            layers=args.layers,
            heads=("hidden_states",),
            batch=args.batch,
        )
        stack = pipeline.hidden_states

        print(
            f"[voxtral_4b_tts_2603] head=hidden_states batch={pipeline.batch} seq_len={args.seq_len} "
            f"layers={stack.n_layers}/{stack.available_layers} device_id={args.device_id}"
        )

        result = pipeline.run_hidden_states(input_ids)
        tt_hidden = result["last_hidden_state"]
        hf_hidden = hs.hf_reference_hidden_states(hf_model, input_ids, layers=stack.n_layers)

        print(
            f"[voxtral_4b_tts_2603] last_hidden_state shape={tuple(tt_hidden.shape)} "
            f"invocations={pipeline.counter.counts}"
        )

        pccs = []
        for b in range(tt_hidden.shape[0]):
            sample_pcc = common.pcc(tt_hidden[b], hf_hidden[b])
            pccs.append(sample_pcc)
            tt_l2 = float(tt_hidden[b].norm())
            hf_l2 = float(hf_hidden[b].norm())
            tt_tok_l2 = float(tt_hidden[b].norm(dim=-1).mean())
            hf_tok_l2 = float(hf_hidden[b].norm(dim=-1).mean())
            head = ", ".join(f"{v:+.5f}" for v in tt_hidden[b, -1, : args.show].tolist())
            print(f"\nsample {b:2d} | prompt: {texts[b][:88]!r}")
            print(f"           shape={tuple(tt_hidden[b].shape)}  PCC={sample_pcc:.6f}")
            print(f"           post-norm L2  TT={tt_l2:10.4f}   HF={hf_l2:10.4f}")
            print(f"           per-token L2  TT={tt_tok_l2:10.4f}   HF={hf_tok_l2:10.4f}")
            print(f"           last-token hidden[:{args.show}] = [{head}]")

        # Distinctness is measured over the WHOLE sample. Every prompt starts with the same BOS
        # token, so position 0's hidden state is identical by construction -- a prefix comparison
        # would report 1 distinct output on a perfectly healthy batch.
        flat = tt_hidden.reshape(tt_hidden.shape[0], -1).to(torch.float64)
        gaps = [
            float((flat[i] - flat[j]).abs().max()) for i in range(flat.shape[0]) for j in range(i + 1, flat.shape[0])
        ]
        print(
            f"\n[voxtral_4b_tts_2603] batch driven={pipeline.batch}  "
            f"min PCC={min(pccs):.6f}  max PCC={max(pccs):.6f}  "
            f"pairwise min max|diff| over {tt_hidden.shape[0]} outputs={min(gaps) if gaps else float('nan'):.6f} "
            f"(0 would mean samples were dropped)"
        )

        if result["tt_last_hidden_state"] is not None:
            ttnn.deallocate(result["tt_last_hidden_state"])
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Section gate for `tt/vocode_stage.py` -- the codec decoder, codes -> waveform, at B=32.

The inputs are REAL codes, produced by running the reference chain the pipeline runs:

    32 distinct prompts -> hf.model(input_ids).last_hidden_state[:, -1]
                        -> hf.acoustic_transformer(h, cfg_alpha), one call per frame
                        -> codes [32, 37, T] in the SHIFTED space the acoustic head emits

and the golden is the reference codec on the same codes, unshifted: `hf.audio_tokenizer(codes - 2)`.

What is asserted, per sample rather than in aggregate: 32 identical waveforms would pass a
whole-batch PCC and prove nothing, so every row is compared to ITS OWN golden and the 32 outputs
are checked pairwise distinct. Both halves of the batch split are covered -- rows [0:16] run the
explicit part-stub chain and rows [16:32] the whole-section body -- and the invocation counter
proves all 14 stubs ran.

The device is opened HERE (never inside `tt/`), by the standard `device` fixture.
"""
from __future__ import annotations

import math
import os

import pytest
import torch

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import common, vocode_stage

pytestmark = pytest.mark.timeout(1800)

# Every stub this section owns. All 14 must be invoked by one `decode` + one
# `audio_token_embedding`; nothing is called for the counter's sake.
OWNED_STUBS = (
    "voxtral_t_t_s_audio_tokenizer",
    "codec_transformer",
    "codec_transformer_block",
    "codec_attention",
    "causal_conv1d",
    "causal_conv_transpose1d",
    "parametrized_conv1d",
    "parametrized_conv_transpose1d",
    "weight_norm",
    "parametrization_list",
    "mistral_audio_codebook",
    "semantic_codebook",
    "acoustic_codebook",
    "multi_vocab_embeddings",
)

BATCH = common.DEFAULT_BATCH
SEQ_LEN = common.DEFAULT_SEQ_LEN
CFG_ALPHA = 3.0
# The acoustic sampler draws noise, so the frame sequence is only reproducible against a pinned
# seed per frame -- which is also what makes the cached golden meaningful.
NOISE_SEED = 1234
PCC_TARGET = 0.99


def _reference(hf, input_ids, n_frames):
    """The reference chain: prompts -> hidden -> per-frame codes -> waveform + audio embedding."""
    batch = int(input_ids.shape[0])
    cfg_alpha = torch.full((batch,), CFG_ALPHA)
    with torch.no_grad():
        hidden = hf.model(input_ids=input_ids).last_hidden_state[:, -1]
        frames = []
        for step in range(n_frames):
            torch.manual_seed(NOISE_SEED + step)
            frames.append(hf.acoustic_transformer(hidden, cfg_alpha))
        codes = torch.stack(frames, dim=-1)

        offset = common.n_audio_special_tokens(hf)
        waveform = hf.audio_tokenizer(codes - offset)
        embeds = hf.audio_tokenizer.audio_token_embedding(codes).sum(dim=1)
    return {"codes": codes, "waveform": waveform, "embeds": embeds, "offset": offset}


def _per_sample_pcc(label, tt, ref):
    """PCC for every row, PRINTED whether it passes or fails, minimum returned."""
    values = [common.pcc(tt[i], ref[i]) for i in range(tt.shape[0])]
    for i, value in enumerate(values):
        print(f"[vocode] {label} sample {i:2d} PCC={value:.6f}")
    worst = min(values)
    print(f"[vocode] {label}: min PCC={worst:.6f} over {len(values)} samples (target {PCC_TARGET})")
    return worst, values


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_vocode_stage(device_params, device):
    print(f"[vocode] torch threads={common.use_all_cpu_threads()}")
    hf = common.load_reference_model()

    input_ids, texts = common.build_batch_inputs(BATCH, SEQ_LEN)
    # A SECTION test of the codec on a fixed-length code block, not the e2e horizon (which is the
    # model's stop rule, asserted in tests/e2e): one second of audio at the model's own frame rate.
    n_frames = int(-(-float(hf.audio_tokenizer.frame_rate) // 1))
    provenance = f"ceil(frame_rate {hf.audio_tokenizer.frame_rate}) -- one second, section-level"
    print(f"[vocode] frames={n_frames} ({provenance})")

    key = common.golden_key(
        section="vocode",
        input_ids=input_ids,
        frames=n_frames,
        cfg_alpha=CFG_ALPHA,
        noise_seed=NOISE_SEED,
    )
    ref = common.cached_golden(key, lambda: _reference(hf, input_ids, n_frames))
    codes, offset = ref["codes"], ref["offset"]
    assert codes.shape == (BATCH, 37, n_frames), codes.shape
    assert (
        int((codes - offset).min()) >= 0
    ), f"a row emitted a special token: min unshifted code {int((codes - offset).min())}"
    assert (
        len({tuple(row.flatten().tolist()) for row in codes}) == BATCH
    ), "the 32 code sequences are not pairwise distinct -- the gate would be meaningless"

    counter = common.InvocationCounter()
    stage = vocode_stage.build_vocode_stage(device, hf, counter=counter)
    print(
        f"[vocode] blocks={len(stage.blocks)} n_layers={stage.n_layers} "
        f"kinds={sorted({b.kind for b in stage.blocks})} windows={stage.group_windows}"
    )
    print(
        f"[vocode] frame_rate={stage.frame_rate} sampling_rate={stage.sampling_rate} "
        f"samples_per_frame={stage.samples_per_frame} max_frames={stage.max_frames} "
        f"({stage.max_frames_provenance})"
    )

    # The blocks list is one flat list of same-typed members, `n_layers` per group, in order.
    assert all(isinstance(b, vocode_stage.CodecBlock) for b in stage.blocks)
    assert len(stage.blocks) == 4 * stage.n_layers
    assert [b.group for b in stage.blocks] == sorted(b.group for b in stage.blocks)
    assert stage.group_windows == (2, 4, 8, 16)
    assert stage.samples_per_frame == 1920
    assert stage.sampling_rate == 24000
    assert stage.frame_rate == 12.5

    codes_tt = ttnn.from_torch(
        codes.to(torch.int32).contiguous(),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    waveform = ttnn.to_torch(stage.decode(codes_tt)).to(torch.float32)
    embeds = ttnn.to_torch(stage.audio_token_embedding(codes_tt)).to(torch.float32)

    # -- every stub ran --------------------------------------------------------------------
    print(f"[vocode] invocations={counter.counts}")
    missing = [name for name in OWNED_STUBS if counter.counts.get(name, 0) == 0]
    assert not missing, f"stubs never invoked: {missing}"

    # -- shapes ---------------------------------------------------------------------------
    expected_samples = n_frames * stage.samples_per_frame
    assert list(waveform.shape) == [
        BATCH,
        1,
        expected_samples,
    ], f"waveform {list(waveform.shape)} != [{BATCH}, 1, {expected_samples}]"
    assert list(ref["waveform"].shape) == [BATCH, 1, expected_samples], ref["waveform"].shape
    assert list(embeds.shape) == [BATCH, 1, n_frames, 3072], list(embeds.shape)

    # -- PCC, per sample, printed pass or fail ---------------------------------------------
    wf_worst, _ = _per_sample_pcc("waveform", waveform.reshape(BATCH, -1), ref["waveform"].reshape(BATCH, -1))
    emb_worst, _ = _per_sample_pcc(
        "audio_token_embedding",
        embeds.reshape(BATCH, n_frames, 3072),
        ref["embeds"].reshape(BATCH, n_frames, 3072),
    )

    # -- the 32 samples are genuinely 32 samples -------------------------------------------
    flat = waveform.reshape(BATCH, -1)
    for i in range(BATCH):
        assert float(flat[i].std()) > 1e-4, f"sample {i} is constant (std {float(flat[i].std()):.3g})"
        peak = float(flat[i].abs().max())
        assert peak <= 4.0, f"sample {i} peaks at {peak:.3f}, outside the audio range"
    for i in range(BATCH):
        for j in range(i + 1, BATCH):
            assert not torch.allclose(flat[i], flat[j], atol=1e-4), f"samples {i} and {j} are identical"
    print(
        f"[vocode] 32 waveforms pairwise distinct, peak={float(flat.abs().max()):.4f}, "
        f"golden peak={float(ref['waveform'].abs().max()):.4f}"
    )

    # -- the frame ceiling is the stage's, measured rather than claimed ---------------------
    probe_frames = int(os.environ.get("VOXTRAL_VOCODE_PROBE_FRAMES", "64"))
    probe_frames = min(probe_frames, stage.max_frames)
    if probe_frames > n_frames:
        long_codes = codes.repeat(1, 1, math.ceil(probe_frames / n_frames))[:, :, :probe_frames]
        long_tt = ttnn.from_torch(
            long_codes.to(torch.int32).contiguous(),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        long_wf = stage.decode(long_tt)
        assert list(long_wf.shape) == [BATCH, 1, probe_frames * stage.samples_per_frame]
        print(
            f"[vocode] frame-axis probe: B={BATCH} T={probe_frames} frames "
            f"({probe_frames * 8} attention rows) decoded, waveform {list(long_wf.shape)}"
        )

    assert wf_worst >= PCC_TARGET, f"waveform min PCC {wf_worst:.6f} < {PCC_TARGET}"
    assert emb_worst >= PCC_TARGET, f"embedding min PCC {emb_worst:.6f} < {PCC_TARGET}"

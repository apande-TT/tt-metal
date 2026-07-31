# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Shared chained TTNN pipeline for coqui/XTTS-v2 text-to-speech synthesis.

This is THE single forward pass over the graduated TTNN stubs. BOTH the demo
(``demo/demo_tts.py``) and the e2e test (``tests/e2e/test_e2e_tts.py``) import
``build_pipeline`` from here and call the same object, so a green test
guarantees a working demo.

Pipeline (XTTS real synthesis, text + speaker-reference audio -> 24 kHz wave):

    encode (host, input-encoding)  tokenize text; DVAE cond-mel; 16 kHz ref wav
    speaker      (TT)  res_net_speaker_encoder(ref_wav)      -> g   [1,512,1]
    conditioning (TT)  conditioning_encoder -> perceiver_resampler -> cond_latents [1,32,1024]
    prefill+decode (TT, AR)  gpt_gpt_inference greedy over the shared 30-block
                             transformer (logits head) -> audio codes
    latent       (TT)  g_p_t(emb) return_latent (same transformer, latent head) -> latents [B,M,1024]
    vocode       (TT)  hifi_decoder(latents, g)              -> waveform [1,1,W]

Every graduated stub's computation is invoked through its canonical stub; the
overlap groups are documented in ../e2e_plan.json. Mesh: TP=8 x DP=1 (1x8).
The GPT + perceiver stages shard (ShardTensorToMesh + all_gather) so this is a
genuine TP=8 placement, not pure replication.

DECODE BATCH = 4 (``DECODE_BATCH``)
-----------------------------------
The pipeline runs ``DECODE_BATCH`` INDEPENDENT streams -- 4 different sentences
spoken by 4 different reference speakers -- through ONE decode program. A single
stream would leave 31 of the 32 rows of a matmul tile empty, so filling 4 of them
raises aggregate throughput ~4x at unchanged per-stream latency.

Where the batch axis lives:
  * the GPT prefix / prefill / latent head carry it as the LEADING dim
    (``[B, T, 1024]``): B*T matmul rows in one program, one sharded weight set.
  * the AR decode step carries it as B ROWS of one tile (``[1, B, 1024]``) -- the
    layout ``nlp_create_qkv_heads_decode`` calls "users" -- and the KV cache holds
    ``[B, heads, C, head_dim]``, i.e. B independent sequences, one slot per stream,
    written by ONE ``paged_fused_update_cache`` with B update indices and read by
    ONE decode-SDPA with B positions. No python loop over streams.
  * the collectives carry the batch rows along; batch is a SEPARATE axis from the
    TP-sharded weight axis and is never sharded.
  * the conv/pool-shaped stages -- speaker encoder, conditioning encoder and the
    vocoder -- are graduated as batch-1 bodies (their reshapes/slices fix the
    leading dim), so those run per stream over the B streams. They sit OUTSIDE the
    AR decode loop (encode side / one final vocode), which is why batching them
    buys nothing: the decode step is the op that repeats hundreds of times.
"""
from __future__ import annotations

import math
import os

import torch

import ttnn
from models.common.utility_functions import comp_pcc

from models.demos.xtts_v2._stubs import (
    conditioning_encoder as _m_cond,
    g_p_t as _m_gpt_latent,
    gpt_gpt_inference as _m_gpt_logits,
    hifi_decoder as _m_hifi,
    learned_position_embeddings as _m_pos,
    perceiver_resampler as _m_perc,
    res_net_speaker_encoder as _m_spk,
)

HF_MODEL_ID = "coqui/XTTS-v2"

# XTTS gpt is an autoregressive ForCausalLM-style core and the model emits
# speech, so the perf/2CQ stages are prefill -> decode -> vocode.
PIPELINE_STAGES = ["prefill", "decode", "vocode"]

# DECODE BATCH: how many independent utterances the decode program serves at once.
DECODE_BATCH = 4

# The B distinct streams: a different shipped speaker reference AND a different sentence
# per stream, so each stream must produce its own audio codes and its own waveform (a
# pipeline that merely shape-supports B would emit 4 identical outputs).
#
# The pairs are chosen so the REFERENCE ITSELF produces non-degenerate speech. Greedy XTTS
# collapses into a repeated audio code for some speaker/text pairs (measured: fr_sample +
# "every stream ..." emits code 323 thirty times in a row), and a sustained tone is useless
# for a sample-aligned waveform metric: its PCC measures PHASE, which the vocoder's own
# ~1e-3 numerics move around (feeding the reference's OWN latents and speaker embedding to
# the ttnn vocoder scored 0.499 on such a stream, and perturbing the latents by 1e-3 raised
# it to 0.987 -- non-monotonic, i.e. noise). Each pair below yields 11-12 distinct codes in
# 12 steps with no repeat, i.e. real speech, so the metric measures the pipeline.
SPEAKER_SAMPLES = ("en_sample.wav", "de_sample.wav", "tr_sample.wav", "pt_sample.wav")
STREAM_TEXTS = (
    "it took me quite a long time to develop a voice",
    "every stream in this batch speaks its own sentence now",
    "she sells sea shells by the sea shore in summer",
    "the quick brown fox jumps over the lazy dog today",
)
# Reference-audio window per stream. The shipped samples are 2.96 s and up, so one
# common window length that all of them can supply keeps ref_wav / cond_mel a single
# stacked tensor.
STREAM_SECONDS = 2.9

# The six canonical graduated stubs the real forward invokes. Each covers a
# hierarchy group of finer/coarser graduated modules that reimplement the same
# proven body inline (see e2e_plan.json graduated_module_coverage).
CANONICAL_STUBS = [
    "res_net_speaker_encoder",
    "conditioning_encoder",
    "perceiver_resampler",
    "gpt_gpt_inference",
    "g_p_t",
    "hifi_decoder",
]
COVERAGE_MAP = {
    "res_net_speaker_encoder": ["res_net_speaker_encoder", "s_e_basic_block", "s_e_layer",
                                "adaptive_avg_pool2d", "instance_norm1d", "mel_spectrogram",
                                "mel_scale", "pre_emphasis"],
    "conditioning_encoder": ["conditioning_encoder", "gpt_conditioning_encoder", "attention_block",
                             "group_norm32", "q_k_v_attention_legacy"],
    "perceiver_resampler": ["perceiver_resampler", "gpt_conditioning_perceiver", "attention",
                            "attend", "g_e_g_l_u"],
    "gpt_gpt_inference": ["gpt_gpt_inference", "g_p_t2_inference_model"],
    "g_p_t": ["g_p_t", "gpt_gpt", "g_p_t2_model", "g_p_t2_block", "g_p_t2_attention",
              "g_p_t2_m_l_p", "conv1_d", "learned_position_embeddings", "dropout1d"],
    "hifi_decoder": ["hifi_decoder", "hifigan_generator", "res_block1", "parametrized_conv1d",
                     "parametrized_conv_transpose1d", "parametrization_list", "weight_norm"],
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _is_mesh(device):
    try:
        return isinstance(device, ttnn.MeshDevice)
    except AttributeError:
        return hasattr(device, "get_device_ids")


def _mesh_n(device):
    try:
        return len(device.get_device_ids())
    except Exception:
        return 1


def load_reference_model(model_id: str = HF_MODEL_ID):
    """Load the native Coqui XTTS reference module (eval, grads off)."""
    import importlib.util as ilu

    here = os.path.dirname(os.path.abspath(__file__))
    rl_path = os.path.normpath(os.path.join(here, "..", "tests", "pcc", "_reference_loader.py"))
    spec = ilu.spec_from_file_location("_xtts_reference_loader", rl_path)
    rl = ilu.module_from_spec(spec)
    spec.loader.exec_module(rl)
    m = rl.load_reference_model(model_id)
    m.eval()
    return m


def _load_captured_args(name):
    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.normpath(os.path.join(here, "..", "_captured", name))
    a = torch.load(os.path.join(base, "args.pt"), map_location="cpu", weights_only=False)
    return list(a) if isinstance(a, (list, tuple)) else [a]


# --------------------------------------------------------------------------- #
# HOST input encoding (the processor/feature-extraction side of the task)
# --------------------------------------------------------------------------- #
def _sample_wav_paths():
    """The speaker-reference wavs shipped in the coqui/XTTS-v2 repo."""
    import glob

    pats = os.path.expanduser(
        "~/.cache/huggingface/hub/models--coqui--XTTS-v2/snapshots/*/samples/*.wav"
    )
    return sorted(glob.glob(pats))


def _read_wav_mono(path):
    """Read a wav as mono float32 ``[1, T]`` + its sample rate (stdlib/scipy only:
    no torchcodec/ffmpeg, which is unavailable on headless boxes)."""
    import numpy as np
    from scipy.io import wavfile

    sr, data = wavfile.read(path)
    data = np.asarray(data)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    else:
        data = data.astype(np.float32)
        if float(np.abs(data).max() or 0.0) > 1.5:
            data = data / float(np.abs(data).max())
    if data.ndim > 1:
        data = data.mean(axis=1)
    return torch.from_numpy(data.copy()).unsqueeze(0), int(sr)


def _fit_len(w, n):
    """Crop/pad a ``[1, T]`` waveform to exactly ``n`` samples (tile the signal when
    the reference clip is shorter, so the window is real audio rather than silence)."""
    t = int(w.shape[-1])
    if t >= n:
        return w[:, :n]
    reps = int(math.ceil(n / max(1, t)))
    return w.repeat(1, reps)[:, :n]


def load_reference_audio_16k(max_samples: int = 32000):
    """The shipped English speaker sample as a mono 16 kHz waveform ``[1,1,T]``.

    Single-stream convenience (the perf harness and ``--batch 1`` runs use it);
    ``build_streams`` is the batched equivalent.
    """
    import torchaudio

    hits = [p for p in _sample_wav_paths() if p.endswith("en_sample.wav")] or _sample_wav_paths()
    if not hits:
        # deterministic synthetic fallback
        g = torch.Generator().manual_seed(0)
        return torch.randn(1, 1, max_samples, generator=g) * 0.1
    wav, sr = _read_wav_mono(hits[0])
    wav16 = torchaudio.functional.resample(wav, sr, 16000)[:, :max_samples]
    return wav16.unsqueeze(0)


def _cond_mel_from_wav(model, wav22):
    """The DVAE conditioning mel XTTS itself feeds ``gpt.get_style_emb`` -- computed
    with the model's OWN front end and its shipped mel statistics (host-side input
    encoding, exactly like a HF feature extractor)."""
    from TTS.tts.models.xtts import wav_to_mel_cloning

    return wav_to_mel_cloning(
        wav22, mel_norms=model.mel_stats.cpu(), n_fft=2048, hop_length=256,
        win_length=1024, power=2, normalized=False, sample_rate=22050,
        f_min=0, f_max=8000, n_mels=80,
    )


def build_streams(model=None, texts=None, speakers=None, batch: int = DECODE_BATCH,
                  language: str = "en", seconds: float = STREAM_SECONDS):
    """Encode B DISTINCT real streams -> the pipeline's 4-tuple input contract.

    Returns ``(cond_mel[B,80,Tc], ref_wav16[B,1,T], text_ids[B,L], audio_codes[B,M])``.

    Stream i is a DIFFERENT sentence read with a DIFFERENT shipped speaker
    reference, so no two streams can share an output. Both audio views come from
    the same clip: 16 kHz for the ResNet speaker encoder and 22.05 kHz for the DVAE
    conditioning mel, which is how XTTS consumes a reference clip.

    Text ids are the real XTTS tokenizer's output, TRUNCATED to the common length
    across the B streams: one batched program needs one prefix length, and cropping
    real token sequences avoids inventing a padding convention the reference would
    not have used.
    """
    import torchaudio

    if model is None:
        model = load_reference_model()
    B = int(batch)
    names = list(speakers) if speakers else list(SPEAKER_SAMPLES)
    all_paths = _sample_wav_paths()
    picks = []
    for i in range(B):
        want = names[i % len(names)]
        hit = [p for p in all_paths if p.endswith(want)]
        picks.append(hit[0] if hit else (all_paths[i % len(all_paths)] if all_paths else None))

    # ---- text ----
    tl = list(texts) if texts else list(STREAM_TEXTS)
    tok = []
    for i in range(B):
        try:
            enc = list(model.tokenizer.encode(tl[i % len(tl)].strip().lower(), lang=language))
        except Exception:  # noqa: BLE001 - tokenizer unavailable: fall back to captured ids
            enc = [int(v) for v in _load_captured_args("g_p_t")[0].reshape(-1)]
        tok.append(enc if enc else [0])
    L = max(1, min(len(e) for e in tok))
    text_ids = torch.zeros(B, L, dtype=torch.long)
    for i, e in enumerate(tok):
        text_ids[i, :] = torch.as_tensor(e[:L], dtype=torch.long)

    # ---- audio (two rates from the same window) ----
    n16, n22 = int(seconds * 16000), int(seconds * 22050)
    ref = torch.zeros(B, 1, n16, dtype=torch.float32)
    mels = None
    for i, p in enumerate(picks):
        if p is None:
            g = torch.Generator().manual_seed(i)
            w, sr = torch.randn(1, n22, generator=g) * 0.1, 22050
        else:
            w, sr = _read_wav_mono(p)
        ref[i, 0, :] = _fit_len(torchaudio.functional.resample(w, sr, 16000), n16)[0]
        m = _cond_mel_from_wav(model, _fit_len(torchaudio.functional.resample(w, sr, 22050), n22))
        if mels is None:
            mels = torch.zeros(B, int(m.shape[-2]), int(m.shape[-1]), dtype=torch.float32)
        mels[i, :, :] = m.reshape(int(m.shape[-2]), int(m.shape[-1])).float()

    # Audio codes are an input only for the DETERMINISTIC/trace paths (the real chain
    # generates its own on device). The captured golden codes, one row per stream.
    cap = _load_captured_args("g_p_t")[2].long()
    codes = torch.zeros(B, int(cap.shape[-1]), dtype=torch.long)
    codes[:, :] = cap.reshape(1, -1)
    return mels, ref, text_ids, codes


def _default_selftest_inputs(model=None, batch: int = DECODE_BATCH):
    """The standard zero-knowledge input tuple every ``<stage>_trace_inputs`` returns."""
    return build_streams(model=model, batch=batch)


# --------------------------------------------------------------------------- #
# pipeline object
# --------------------------------------------------------------------------- #
class XttsPipeline:
    """Resident chained TTNN pipeline object (built once, run many times)."""

    def __init__(self, device, model=None, batch: int = DECODE_BATCH, vocode_len=None):
        self.device = device
        self.model = model if model is not None else load_reference_model()
        self.gpt = self.model.gpt
        self.hd = self.model.hifigan_decoder
        self.n = _mesh_n(device)
        self.batch = int(batch)
        self.vocode_len = int(vocode_len) if vocode_len else int(self.VOCODE_LEN)
        self._invoked = set()
        self._streams = None

        gpt = self.gpt
        self.start_audio = int(gpt.start_audio_token)
        self.stop_audio = int(gpt.stop_audio_token)
        self.num_audio = int(gpt.num_audio_tokens)
        self.code_stride = int(gpt.code_stride_len)
        self.max_audio_tokens = int(getattr(gpt, "max_mel_tokens", 605))

        # ---- build stubs once (weights uploaded to the mesh) ----
        self.f_spk = _m_spk.build(device, self.hd.speaker_encoder)
        self.f_cond = _m_cond.build(device, gpt.conditioning_encoder)
        self.f_perc = _m_perc.build(device, gpt.conditioning_perceiver)
        self.f_gpt_logits = _m_gpt_logits.build(device, gpt.gpt_inference)
        self.f_tpos = _m_pos.build(device, gpt.text_pos_embedding)
        self.f_mpos = _m_pos.build(device, gpt.mel_pos_embedding)
        # Both heads that bake a LENGTH are built per length and cached: the vocoder
        # pins its interpolation/dilation matrices to the latent count it synthesizes,
        # and g_p_t bakes the mel-latent slice length.
        self._hifi_cache = {}
        self._gpt_latent_cache = {}
        self.f_hifi = self._build_vocoder(self.vocode_len)

        # embedding tables staged on the mesh (replicated).
        self.txt_w = self._rep(gpt.text_embedding.weight.detach(), layout=ttnn.ROW_MAJOR_LAYOUT)
        self.mel_w = self._rep(gpt.mel_embedding.weight.detach(), layout=ttnn.ROW_MAJOR_LAYOUT)
        self.dim = int(gpt.mel_embedding.weight.shape[1])
        # Raw mel positional table [num_pos, dim] staged on the mesh so the AR
        # decode can look up a single position row ON DEVICE (no host arange).
        self.mel_pos_w = self._rep(gpt.mel_pos_embedding.emb.weight.detach())

    # ---- tensor movement ----
    def _rep(self, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        tt = t.to(torch.bfloat16) if dtype == ttnn.bfloat16 else t.to(torch.float32)
        if _is_mesh(self.device):
            return ttnn.from_torch(tt, dtype=dtype, layout=layout, device=self.device,
                                   mesh_mapper=ttnn.ReplicateTensorToMesh(self.device))
        return ttnn.from_torch(tt, dtype=dtype, layout=layout, device=self.device)

    def _to_torch(self, t, shape=None):
        try:
            ttnn.synchronize_device(self.device)
        except Exception:
            pass
        if _is_mesh(self.device):
            comp = ttnn.ConcatMeshToTensor(self.device, dim=0)
            out = ttnn.to_torch(t, mesh_composer=comp)
            if out.shape[0] % self.n == 0 and out.shape[0] > 1:
                out = out[: out.shape[0] // self.n]
        else:
            out = ttnn.to_torch(t)
        out = out.float()
        if shape is not None:
            out = out.reshape(shape)
        return out

    # ---- token embedding (on device) ----
    def _embed(self, token_index, weight_tt, fpos):
        """Embed a ``[B, L]`` id block: table gather + the learned positional rows.

        Positions are shared by every stream (all B prefixes have the same length),
        so the ``[1, L, dim]`` positional block broadcasts over the batch.
        """
        idx_tt = ttnn.from_torch(
            token_index.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if _is_mesh(self.device) else None,
        )
        e = ttnn.embedding(idx_tt, weight_tt)
        e = ttnn.to_layout(e, ttnn.TILE_LAYOUT)
        e = ttnn.typecast(e, ttnn.bfloat16)
        pos = fpos(token_index)  # [L, dim] tile
        pos = ttnn.reshape(ttnn.typecast(pos, ttnn.bfloat16), [1, int(token_index.shape[1]), -1])
        return ttnn.add(e, pos)

    def _pos_row(self, mel_pos, B):
        """The mel positional row for ``mel_pos``, as ``[1, B, dim]``.

        One on-device slice of the staged table, then replicated to the B decode rows
        (every stream is at the same mel position, so it is the same row B times). No
        host arange, no per-call upload -- both would make the decode step untraceable.
        """
        prow = ttnn.slice(self.mel_pos_w, [int(mel_pos), 0], [int(mel_pos) + 1, self.dim])
        prow = ttnn.reshape(ttnn.typecast(prow, ttnn.bfloat16), [1, 1, self.dim])
        if int(B) == 1:
            return prow
        return ttnn.concat([prow] * int(B), dim=1)

    def _embed_next(self, next_id_tt, mel_pos, B=None):
        """Embed the B streams' next tokens whose ids already live ON DEVICE (the
        ``ttnn.argmax`` result) at absolute mel position ``mel_pos``.

        No host readback and no token re-upload: the ids feed ``ttnn.embedding``
        directly and the positional row is an on-device slice. Returns ``[1, B, dim]``
        -- B rows of one tile, the layout the batched decode step consumes.
        """
        b = int(B) if B is not None else int(next_id_tt.volume())
        idx = ttnn.reshape(ttnn.typecast(ttnn.to_layout(next_id_tt, ttnn.ROW_MAJOR_LAYOUT),
                                         ttnn.uint32), [1, b])
        e = ttnn.embedding(idx, self.mel_w)                # [1,B,dim]
        e = ttnn.typecast(ttnn.to_layout(e, ttnn.TILE_LAYOUT), ttnn.bfloat16)
        return ttnn.add(e, self._pos_row(mel_pos, b))

    def _build_vocoder(self, latent_len):
        """BUILD (or fetch) the vocoder baked to ``latent_len``. Setup path only."""
        key = int(latent_len)
        if key not in self._hifi_cache:
            self.hd._tt_latent_len = key
            self._hifi_cache[key] = _m_hifi.build(self.device, self.hd)
        return self._hifi_cache[key]

    def _prebuilt_vocoder(self, latent_len):
        """The resident vocoder for ``latent_len`` — LOOKUP ONLY, never a build.

        The vocoder bakes its interpolation/dilation matrices AND its weight-norm-
        materialized conv weights to the latent count, so *building* one runs a few
        hundred host tensor ops. From inside the forward that is per-call weight
        streaming, and it is what the host-op observer catches. Every length the
        pipeline can feed is built once in ``__init__``; a miss here is a wiring bug
        (a caller fed the vocoder a latent count it was not built for), so say so
        instead of silently rebuilding on the compute path.
        """
        f = self._hifi_cache.get(int(latent_len))
        if f is None:
            raise RuntimeError(
                f"no resident vocoder for {latent_len} latents (built: "
                f"{sorted(self._hifi_cache)}). The vocoder is LENGTH-BAKED: feed it "
                f"exactly vocode_len={self.vocode_len} latents (see _codes_for_vocode), "
                f"or construct with build_pipeline(..., vocode_len={latent_len})."
            )
        return f

    def _build_latent_head(self, mel_len, sub=5):
        key = (int(mel_len), int(sub))
        if key not in self._gpt_latent_cache:
            self.gpt._tt_mel_len = int(mel_len)
            self.gpt._tt_sub = int(sub)
            self._gpt_latent_cache[key] = _m_gpt_latent.build(self.device, self.gpt)
        return self._gpt_latent_cache[key]

    def streams(self, **kw):
        """The pipeline's own B distinct encoded streams (cached)."""
        if self._streams is None:
            self._streams = build_streams(model=self.model, batch=self.batch, **kw)
        return self._streams

    # ===================================================================== #
    # HOST input-encoding glue (mirrors GPT.forward / compute_embeddings).
    # This is token-id manipulation only (feature-extraction-like), NOT model
    # math; embeddings/transformer/etc. all run on device.
    # ===================================================================== #
    def _pad_text_ids(self, text_inputs):
        gpt = self.gpt
        t = text_inputs[:, : gpt.max_text_tokens] if hasattr(gpt, "max_text_tokens") else text_inputs
        t = torch.nn.functional.pad(t, (0, 1), value=gpt.stop_text_token)
        t = torch.nn.functional.pad(t, (1, 0), value=gpt.start_text_token)
        return t

    def _pad_mel_ids(self, audio_codes, wav_lengths):
        """Reproduce GPT.forward mel-id glue for the return_latent path (per stream)."""
        gpt = self.gpt
        code_lengths = torch.ceil(wav_lengths / self.code_stride).long() + 3
        max_mel_len = int(code_lengths.max())
        codes = audio_codes[:, :max_mel_len]
        if codes.shape[-1] < max_mel_len:
            codes = torch.nn.functional.pad(codes, (0, max_mel_len - codes.shape[-1]), value=0)
        codes = torch.nn.functional.pad(codes, (0, 1), value=self.stop_audio)
        # set_mel_padding: positions >= (code_lengths-3) become stop_audio
        for b in range(codes.shape[0]):
            actual = int(code_lengths[b] - 3)
            codes[b, actual:] = self.stop_audio
        codes = torch.nn.functional.pad(codes, (1, 0), value=self.start_audio)
        return codes

    def _codes_for_vocode(self, audio_codes):
        """Slice/pad a code block to EXACTLY ``vocode_len`` columns (host id glue).

        The latent head emits one latent per code, and the vocoder is baked to a fixed
        latent count, so this is the single place that pins the code count every path
        (real forward, trace setup, on-device self-test) hands downstream. Without it a
        path carrying the shorter captured code row would ask ``vocode`` for a vocoder
        that was never built.
        """
        vl = int(self.vocode_len)
        codes = audio_codes[:, :vl].long()
        if int(codes.shape[-1]) < vl:
            codes = torch.nn.functional.pad(codes, (0, vl - int(codes.shape[-1])), value=0)
        return codes

    # ===================================================================== #
    # stages
    # ===================================================================== #
    def encode_conditioning(self, cond_mel_tt):
        """cond_mel (TT) [1,80,Tc] -> cond_latents (TT) [1,32,1024]. On-device."""
        self._invoked.add("conditioning_encoder")
        self._invoked.add("perceiver_resampler")
        ce = self.f_cond(cond_mel_tt)                 # [1,1024,Tc]
        perc_in = ttnn.transpose(ce, 1, 2)            # [1,Tc,1024]  (on device)
        return self.f_perc(perc_in)                   # TT [1,32,1024]

    def encode_conditioning_batch(self, cond_mel):
        """B streams' cond mels -> ONE batched cond_latents [B,32,1024] (TT).

        The conditioning encoder + perceiver graduated with batch-1 bodies, so each
        stream runs its own call and the results are stacked ON DEVICE (32 latent rows
        per stream == exactly one tile row, so the concat is free).
        """
        B = int(cond_mel.shape[0])
        parts = [self.encode_conditioning(self._rep(cond_mel[b:b + 1].float())) for b in range(B)]
        return parts[0] if B == 1 else ttnn.concat(parts, dim=0)

    def encode_speaker(self, ref_wav_tt):
        """ref_wav (TT) [1,1,T] -> g (TT) [1,512,1], unit-norm. On-device."""
        self._invoked.add("res_net_speaker_encoder")
        g = self.f_spk(ref_wav_tt)                    # [1,512] raw fc
        # XTTS get_speaker_embedding uses l2_norm=True (unit-norm g). The stub
        # returns the raw fc output (~67x larger); PCC on g is scale-invariant so
        # it hides this, but the vocoder adds g LINEARLY, so we must L2-normalise
        # here (on device) to match the reference — else the wave is garbage.
        g = ttnn.reshape(g, [1, 512])
        inv = ttnn.rsqrt(ttnn.sum(ttnn.multiply(g, g), dim=-1, keepdim=True))
        g = ttnn.multiply(g, inv)                     # unit-norm [1,512]
        return ttnn.reshape(g, [1, 512, 1])           # TT [1,512,1]

    def encode_speaker_batch(self, ref_wav):
        """B streams' reference clips -> a list of B speaker embeddings (TT).

        The speaker encoder is a graduated batch-1 conv2d/pool body and it runs ONCE
        per synthesis (not per decode step), so per-stream calls are the right shape
        here; ``g`` stays per stream because the vocoder consumes it per stream.
        """
        B = int(ref_wav.shape[0])
        return [self.encode_speaker(self._rep(ref_wav[b:b + 1].float(), dtype=ttnn.float32))
                for b in range(B)]

    def _assemble_emb(self, cl_tt, text_ids, mel_ids):
        text_emb = self._embed(text_ids, self.txt_w, self.f_tpos)
        mel_emb = self._embed(mel_ids, self.mel_w, self.f_mpos)
        return ttnn.concat([cl_tt, text_emb, mel_emb], dim=1)

    def _kv_capable(self):
        """True when the logits head exposes the KV-cache decode contract."""
        return callable(getattr(self.f_gpt_logits, "decode_one", None))

    def _kv_pos(self, p, B=1):
        """The decode position as a DEVICE tensor holding ONE position PER STREAM.

        It has to live on device or a captured trace bakes in a stale constant (the
        cache would be read/written at one fixed slot for every replay), and it has to
        be length B because the fused cache write and decode-SDPA index per stream.
        """
        pos_vec = torch.zeros(int(B), dtype=torch.int32) + int(p)
        return ttnn.from_torch(
            pos_vec, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if _is_mesh(self.device) else None,
        )

    def _tile_ceil(self, n):
        return int(((int(n) + 31) // 32) * 32)

    def _prefix_ids(self, text_inputs):
        """HOST id glue for the decode prefix: the padded text ids + the start-audio row.

        Split out so the on-device prefix assembly can be handed ALREADY-ENCODED ids --
        the host-op self-test needs the id padding to happen outside its observed region.
        """
        text_ids = self._pad_text_ids(text_inputs)
        first = torch.zeros(int(text_inputs.shape[0]), 1, dtype=torch.long) + self.start_audio
        return text_ids, first

    def _prefix_emb(self, cl_tt, text_inputs=None, capacity=None, ids=None):
        """cond | text | start_audio for all B streams -> ``[B, C, dim]`` (TT).

        ``capacity`` pins the sequence axis to a fixed C by appending zero rows; causal
        attention makes that tail inert, and decode-SDPA masks everything past the
        current position, so the pinned shape changes no result.
        """
        text_ids, first = ids if ids is not None else self._prefix_ids(text_inputs)
        B = int(text_ids.shape[0])
        text_emb = self._embed(text_ids, self.txt_w, self.f_tpos)
        emb = ttnn.concat([cl_tt, text_emb, self._embed(first, self.mel_w, self.f_mpos)], dim=1)
        real = int(emb.shape[1])
        if capacity and int(capacity) > real:
            tail = ttnn.multiply(ttnn.slice(emb, [0, 0, 0],
                                            [B, int(capacity) - real, self.dim]), 0.0)
            emb = ttnn.concat([emb, tail], dim=1)
        return emb, real

    def decode_codes(self, cl_tt, text_inputs, horizon):
        """Greedy AR decode of B streams through ONE program, fully on device.

        Prefill runs the B prefixes once and keeps every layer's K/V -- ``[B, heads, C,
        head_dim]``, one cache slot per stream. Each AR step then feeds ONE
        ``[1, B, dim]`` row block (the B streams' current tokens) through the 30 blocks
        against those caches: one fused cache write with B update indices, one
        decode-SDPA with B positions, one set of projections carrying B rows. The
        next-token ids never leave the device inside the loop -- ``ttnn.argmax`` feeds
        ``ttnn.embedding`` directly -- so nothing here is host-serialized per step.

        The ids are copied to host ONCE after the loop, to report the sequences and
        truncate each stream at its own stop token. Returns ``(codes[B, hz], None)``.
        """
        self._invoked.add("gpt_gpt_inference")
        B = int(text_inputs.shape[0])
        # Horizon is model-grounded: capped by the config's max audio-token context.
        hz = max(1, min(int(horizon), self.max_audio_tokens))
        next_ids = []
        step_logits = []
        if self._kv_capable():
            emb, real = self._prefix_emb(cl_tt, text_inputs)
            C = self._tile_ceil(real + hz)
            if C > real:
                tail = ttnn.multiply(ttnn.slice(emb, [0, 0, 0], [B, C - real, self.dim]), 0.0)
                emb = ttnn.concat([emb, tail], dim=1)      # causal => padding is inert
            logits = self.f_gpt_logits.prefill_cache(emb)   # seeds B independent slots
            pos = real - 1
            for step in range(hz):
                # Token 0 comes from the PREFILL logits at each stream's last real
                # position; every later token comes from the cached decode step. The
                # first token must NOT be re-derived by replaying decode_one at
                # `real - 1`: that position is inside the prefix the cache was seeded
                # from, and decode-SDPA there does not reproduce the prefill result
                # (measured: it returns a junk argmax), so the AR chain would start
                # from a wrong token. decode_one is only ever called at positions PAST
                # the prefix, which is exactly where an AR step belongs.
                last = ttnn.slice(logits, [0, pos, 0], [B, pos + 1, self.num_audio]) \
                    if step == 0 else logits
                nid = ttnn.argmax(last, dim=-1)            # [B,1] / [1,B] ON DEVICE
                next_ids.append(nid)
                step_logits.append(last)
                if step + 1 < hz:
                    pos += 1
                    row = self._embed_next(nid, step + 1, B=B)   # [1,B,dim]
                    logits = self.f_gpt_logits.decode_one(row, self._kv_pos(pos, B))
        else:
            emb, real = self._prefix_emb(cl_tt, text_inputs)
            for step in range(hz):
                logits = self.f_gpt_logits(emb)            # [B, T, num_audio]
                t = int(emb.shape[1])
                last = ttnn.slice(logits, [0, t - 1, 0], [B, t, self.num_audio])
                nid = ttnn.argmax(last, dim=-1)            # [B,1] ON DEVICE
                next_ids.append(nid)
                step_logits.append(last)
                emb = ttnn.concat([emb, self._embed_next(nid, step + 1, B=B)], dim=1)
        # single post-loop host copy of the generated ids + per-step logits
        # (reporting / stop truncation / the test's tie certification only -- nothing inside
        # the loop reads them back).
        codes = torch.zeros(B, hz, dtype=torch.long)
        for s, nid in enumerate(next_ids):
            row_h = self._to_torch(nid).reshape(-1)
            for b in range(B):
                codes[b, s] = int(row_h[b])
        lg_host = [self._to_torch(t).reshape(B, -1)[:, : self.num_audio] for t in step_logits]
        return codes, lg_host

    def stop_truncate(self, codes, length):
        """Per-stream stop-token truncation, then pin to ``length`` codes.

        The stop rule is the model's own ``stop_audio_token``; it is applied to the
        copied ids AFTER the device loop so no step needs a host readback. Streams that
        never stop simply keep their codes (the loop horizon is capped by the config's
        max audio-token context).
        """
        B = int(codes.shape[0])
        out = torch.zeros(B, int(length), dtype=torch.long)
        kept = []
        for b in range(B):
            seq = []
            for v in codes[b].reshape(-1):
                iv = int(v)
                if iv == self.stop_audio:
                    break
                seq.append(iv)
            if not seq:
                seq = [0]
            kept.append(len(seq))
            n = min(len(seq), int(length))
            out[b, :n] = torch.as_tensor(seq[:n], dtype=torch.long)
        return out, kept

    def latents_from_codes(self, cl_tt, text_inputs, audio_codes):
        """audio_codes [B,M] -> GPT latents (TT) [B,M,1024] via the g_p_t latent head.

        One batched program: the B streams' cond|text|mel embeddings are stacked on the
        leading axis and the 30 blocks run B*T rows at once.
        """
        self._invoked.add("g_p_t")
        B = int(audio_codes.shape[0])
        wav_lengths = torch.zeros(B) + float(audio_codes.shape[-1] * self.code_stride)
        text_ids = self._pad_text_ids(text_inputs)
        mel_ids = self._pad_mel_ids(audio_codes, wav_lengths)
        emb = self._assemble_emb(cl_tt, text_ids, mel_ids)
        f_latent = self._build_latent_head(mel_len=int(mel_ids.shape[1]), sub=5)
        return f_latent(emb)  # TT [B,M,1024]

    def vocode(self, latents_tt, g):
        """latents (TT) [B,M,1024] + per-stream g -> a list of B waveforms (TT) [1,1,W].

        The vocoder graduated as a batch-1 body (its conv/interpolation shapes fix the
        leading dim), and it runs ONCE per utterance rather than per decode step, so the
        B streams are vocoded as B calls into the same resident stub -- each with its
        OWN latents slice and its OWN speaker embedding, so the outputs really differ.
        """
        self._invoked.add("hifi_decoder")
        if not isinstance(latents_tt, ttnn.Tensor):
            latents_tt = self._rep(latents_tt, dtype=ttnn.float32)
        gs = g if isinstance(g, (list, tuple)) else [g]
        B = int(latents_tt.shape[0])
        T, D = int(latents_tt.shape[-2]), int(latents_tt.shape[-1])
        f_hifi = self._prebuilt_vocoder(T)   # resident: no weight build on the forward
        out = []
        for b in range(B):
            lat_b = latents_tt if B == 1 else ttnn.slice(latents_tt, [b, 0, 0], [b + 1, T, D])
            out.append(f_hifi(lat_b, g=gs[b % len(gs)]))
        return out

    # ===================================================================== #
    # top-level chained forward
    # ===================================================================== #
    # AUDIO-CODE LENGTH of one synthesis, and therefore the decode horizon's default:
    # it is exactly the number of codes the final waveform consumes. The vocoder pins its
    # interpolation matrices to it (``_tt_latent_len``), so it is a build-time constant per
    # length, not a dynamic axis.
    #
    # 32 codes = 32 * code_stride(1024) samples at 22.05 kHz -> 1.49 s of speech (35584
    # samples at 24 kHz). The captured PCC case is 6, and 6 codes is a POOR gate signal:
    # greedy XTTS repeats a code early, so a 6-code waveform is ~0.28 s of near-periodic
    # audio whose sample-aligned PCC collapses on a sub-sample phase shift (measured: a
    # latents PCC of 0.9991 gave a waveform PCC of 0.71 on such a stream, while the same
    # vocoder fed the reference latents scored 0.998). A 1.5 s waveform has real structure,
    # so the metric measures the pipeline instead of the periodicity.
    VOCODE_LEN = 32

    def run_tts(self, cond_mel, ref_wav_16k, text_inputs, audio_codes=None,
                horizon=None, generate=True, vocode_len=None, **_ignore):
        """Full chained forward over B streams. Returns the waveforms + intermediates.

        Inputs carry a LEADING batch dim (``cond_mel[B,80,Tc]``, ``ref_wav[B,1,T]``,
        ``text_inputs[B,L]``); B=1 is the degenerate single-stream case.

        - generate=True: run the AR decode stage to produce each stream's audio codes
          (these feed the latent stage). If ``audio_codes`` is also given, the decode
          still runs and its codes are reported, but the latent stage uses
          ``audio_codes`` for a deterministic comparison.
        """
        B = int(text_inputs.shape[0])
        cl_tt = self.encode_conditioning_batch(cond_mel)
        g_list = self.encode_speaker_batch(ref_wav_16k)
        g_host = [self._to_torch(g, (1, 512)) for g in g_list]

        # The vocoder is baked to ONE latent count at construction, so the synthesized
        # length is a build-time knob, not a per-call one: honour an explicit request
        # only when it matches, rather than rebuilding weights mid-forward.
        vl = int(self.vocode_len)
        if vocode_len is not None and int(vocode_len) != vl:
            raise ValueError(
                f"vocode_len={int(vocode_len)} but this pipeline's vocoder is built for "
                f"{vl} latents; pass vocode_len to build_pipeline() instead."
            )
        codes_gen = None
        step_logits = None
        kept = None
        if generate:
            hz = int(horizon) if horizon is not None else vl
            codes_gen, step_logits = self.decode_codes(cl_tt, text_inputs, hz)

        if audio_codes is not None:
            codes = self._codes_for_vocode(audio_codes)
        else:
            codes, kept = self.stop_truncate(codes_gen, vl)

        latents_tt = self.latents_from_codes(cl_tt, text_inputs, codes)
        wav_tt = self.vocode(latents_tt, g_list)
        waves = [self._to_torch(w) for w in wav_tt]
        wav_host = torch.zeros(B, 1, int(waves[0].reshape(-1).shape[0]))
        for b in range(B):
            wav_host[b, 0, :] = waves[b].reshape(-1)
        return {
            "batch": B,
            "waveforms_tt": wav_tt,
            "waveform_tt": wav_tt[0],
            "waveforms": waves,
            "waveform": wav_host,
            "cond_latents_tt": cl_tt,
            "g_tt": g_list,
            "g_host": g_host,
            "latents_tt": latents_tt,
            "codes_gen": codes_gen,
            "codes_used": codes,
            "codes_kept": kept,
            "step_logits": step_logits,
            "invoked": set(self._invoked),
        }

    # ===================================================================== #
    # COMMAND 3 — trace + 2CQ contract (host-free per-stage capture)
    #
    # Stages (from the config: AR ForCausalLM core emitting speech):
    #   prefill  — gpt_gpt_inference over the cond|text|start prefix at fixed C
    #   decode   — one batched AR step of gpt_gpt_inference at fixed C
    #   vocode   — hifi_decoder at the fixed latent length (6)
    # Each stage pins its VARIABLE (sequence) dim to a fixed capacity C and runs
    # a host-op-free forward reading ONLY pre-uploaded persistent buffers, at the
    # full DECODE_BATCH so the traced shapes are the shapes the pipeline runs.
    # ===================================================================== #
    def _prep_seed(self, cond_mel, ref_wav_16k, text_inputs, audio_codes):
        """Host input-encoding shared by the trace stages (done OUTSIDE traces)."""
        cl_tt = self.encode_conditioning_batch(cond_mel)
        text_ids = self._pad_text_ids(text_inputs)
        B = int(audio_codes.shape[0])
        wav_lengths = torch.zeros(B) + float(int(audio_codes.shape[-1]) * self.code_stride)
        mel_ids = self._pad_mel_ids(audio_codes, wav_lengths)
        return cl_tt, text_ids, mel_ids

    # ---- the ZERO-ARG standard seam the perf engine calls per stage ----
    def prefill_trace_inputs(self):
        return self.streams()

    def decode_trace_inputs(self):
        return self.streams()

    def vocode_trace_inputs(self):
        return self.streams()

    # ---- prefill ----
    def prefill_trace_setup(self, inputs):
        cond_mel, ref_wav, text_inputs, audio_codes = inputs
        cl_tt = self.encode_conditioning_batch(cond_mel)
        emb, real = self._prefix_emb(cl_tt, text_inputs)
        self._prefill_B = int(text_inputs.shape[0])
        self._prefill_real_len = real
        self._prefill_C = self._tile_ceil(real)
        if self._prefill_C > real:
            emb, _ = self._prefix_emb(cl_tt, text_inputs, capacity=self._prefill_C)
        self._prefill_emb = emb                            # persistent buffer [B,C,1024]
        return self._prefill_emb

    def prefill_trace_step(self):
        # pure-ttnn forward over the persistent padded emb (causal => [:real_len] exact)
        return self.f_gpt_logits(self._prefill_emb)

    def prefill_write_inputs(self, new_inputs=None):
        # one-shot stage: re-stage the prefix buffer (2CQ CQ1 upload point)
        if new_inputs is not None:
            self.prefill_trace_setup(new_inputs)
        return self._prefill_emb

    # ---- decode (AR) ----
    def decode_prefill(self, inputs):
        """Seed the resident decode state at fixed capacity C: the B prefixes' self-attn
        KV caches (one slot per stream) plus the resident decode row and position."""
        cond_mel, ref_wav, text_inputs, audio_codes = inputs
        B = int(text_inputs.shape[0])
        cl_tt = self.encode_conditioning_batch(cond_mel)
        emb, real = self._prefix_emb(cl_tt, text_inputs)
        self._decode_B = B
        self._decode_C = self._tile_ceil(real + self.vocode_len)
        emb, real = self._prefix_emb(cl_tt, text_inputs, capacity=self._decode_C)
        self._decode_emb = emb
        self._decode_pos = real - 1
        # MEL position is its OWN axis: the start_audio token that ends the prefix is
        # mel position 0, so the next generated code is mel position 1 — NOT the
        # absolute row, which counts the 32 cond + text rows ahead of it.
        self._decode_mel_pos = 0
        self._decode_next = None
        self._kv = self._kv_capable()
        if self._kv:
            # Fill the caches from the prefix ONCE here (outside the traced step) and stage
            # the resident [1,B,dim] row + the on-device B positions the step reads. Without
            # this the traced step is a repeat_prefill -- a full C-row forward per token.
            logits = self.f_gpt_logits.prefill_cache(self._decode_emb)
            # Seed the resident row with the FIRST GENERATED token (from the prefill logits,
            # device-side) at mel position 1, so the traced step is a genuine steady-state AR
            # step at a position PAST the prefix -- the same call the decode loop replays.
            last = ttnn.slice(logits, [0, self._decode_pos, 0],
                              [B, self._decode_pos + 1, self.num_audio])
            self._decode_pos += 1
            self._decode_mel_pos = 1
            self._decode_row = self._embed_next(ttnn.argmax(last, dim=-1), 1, B=B)
            self._decode_pos_tt = self._kv_pos(self._decode_pos, B)
        return self._decode_emb

    def decode_trace_setup(self, inputs):
        return self.decode_prefill(inputs)

    def decode_trace_step(self):
        if getattr(self, "_kv", False):
            # seq_len=1 per stream against cached K/V, NOT a re-prefill of the prefix.
            return self.f_gpt_logits.decode_one(self._decode_row, self._decode_pos_tt)
        return self.f_gpt_logits(self._decode_emb)

    def decode_step(self):
        """ONE fixed-shape, host-op-free AR step for ALL B streams.

        Runs the cached decode at the fixed capacity C, takes the on-device argmax and
        returns the B next-token ids ``[1, B]`` still ON DEVICE (no readback, no host
        token loop). ``decode_write_inputs`` stages them back on the next command queue.
        """
        logits = self.decode_trace_step()
        if int(logits.shape[-2]) != int(getattr(self, "_decode_B", 1)):
            b = int(getattr(self, "_decode_B", 1))
            logits = ttnn.slice(logits, [0, self._decode_pos, 0],
                                [b, self._decode_pos + 1, self.num_audio])
        nid = ttnn.argmax(logits, dim=-1)                         # [1,B] ON DEVICE
        self._decode_next = nid
        return nid

    def decode_write_inputs(self, next_id=None):
        """Stage the B next tokens into the resident decode row ON DEVICE (CQ1).

        Consumes the on-device ids from ``decode_step`` (never a host int), embeds them
        via ``_embed_next`` (``ttnn.embedding`` of the argmax result) and advances the
        resident position tensor. No host re-upload, no O(capacity) recompute.
        """
        nid = next_id if next_id is not None else self._decode_next
        if nid is None:
            return self._decode_row if getattr(self, "_kv", False) else self._decode_emb
        B = int(getattr(self, "_decode_B", 1))
        self._decode_mel_pos = int(getattr(self, "_decode_mel_pos", 0)) + 1
        self._decode_pos += 1
        tok_emb = self._embed_next(nid, self._decode_mel_pos, B=B)
        if getattr(self, "_kv", False):
            self._decode_row = tok_emb
            self._decode_pos_tt = self._kv_pos(self._decode_pos, B)
            return self._decode_row
        emb = ttnn.concat([self._decode_emb, tok_emb], dim=1)
        # keep capacity fixed C (drop one trailing pad slot)
        self._decode_emb = ttnn.slice(emb, [0, 0, 0], [B, self._decode_C, 1024])
        return self._decode_emb

    # ---- vocode ----
    def vocode_trace_setup(self, inputs):
        cond_mel, ref_wav, text_inputs, audio_codes = inputs
        # EXACTLY vocode_len codes, so the traced stage runs the resident vocoder at the
        # length the pipeline synthesizes at (a short code row would ask for a second,
        # differently-baked vocoder).
        codes = self._codes_for_vocode(audio_codes)
        cl_tt, text_ids, mel_ids = self._prep_seed(cond_mel, ref_wav, text_inputs, codes)
        emb = self._assemble_emb(cl_tt, text_ids, mel_ids)
        f_lat = self._build_latent_head(mel_len=int(mel_ids.shape[1]), sub=5)
        self._vocode_latents = f_lat(emb)                              # [B,6,1024] persistent
        self._vocode_g = self.encode_speaker_batch(ref_wav)            # B persistent embeddings
        self._vocode_B = int(cond_mel.shape[0])
        return self._vocode_latents

    def vocode_trace_step(self):
        outs = self.vocode(self._vocode_latents, self._vocode_g)
        return outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=0)

    def vocode_write_inputs(self, new_latents=None):
        if new_latents is not None:
            self._vocode_latents = new_latents
        return self._vocode_latents

    def trace_capture_selftest(self, device=None, inputs=None):
        """Capture ONE step per PIPELINE_STAGE host-free, execute, verify PCC.

        Captures each stage's trace in isolation (release before the next) and
        checks the traced output matches the eager reference. Returns True iff
        every stage captured host-free AND matched. Prints any fallback.
        """
        device = device or self.device
        if inputs is None:
            inputs = self.streams()
        ok = True
        for stage in PIPELINE_STAGES:
            setup = getattr(self, f"{stage}_trace_setup")
            step = getattr(self, f"{stage}_trace_step")
            cur = inputs
            for attempt in (0, 1):
                try:
                    setup(cur)
                    ref = self._to_torch(step())                          # eager reference
                    tid = ttnn.begin_trace_capture(device, cq_id=0)
                    out = step()
                    ttnn.end_trace_capture(device, tid, cq_id=0)
                    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
                    got = self._to_torch(out)
                    ttnn.release_trace(device, tid)
                    pcc = comp_pcc(ref.reshape(-1), got.reshape(-1), 0.99)[1]
                    good = pcc >= 0.99
                    ok = ok and good
                    print(f"[trace] stage={stage} B={int(cur[0].shape[0])} captured host-free, "
                          f"PCC={pcc:.5f} {'OK' if good else 'LOW'}")
                    break
                except Exception as e:  # pragma: no cover
                    if attempt == 0 and int(cur[0].shape[0]) > 1:
                        # The trace region is sized from the LARGEST stage; if a capture
                        # overflows it, shrink the pinned capacity (here: the stream count,
                        # which multiplies every stage's rows) and SAY SO — never silently.
                        cur = tuple(t[:1] for t in cur)
                        print(f"[trace] stage={stage} capture failed at B={self.batch} "
                              f"({type(e).__name__}: {e}); FALLBACK to B=1 capture")
                        continue
                    ok = False
                    print(f"[trace] stage={stage} FALLBACK to single-CQ / capture failed: "
                          f"{type(e).__name__}: {e}")
                    break
        return ok

    def host_op_selftest(self, inputs=None):
        """Authoritative fully-on-device check: run the model math under
        observe_host_ops with input-ENCODING + weight build done OUTSIDE the
        observed region. A truly on-device forward fires ZERO host aten ops."""
        from scripts.tt_hw_planner.host_op_observer import observe_host_ops, verdict

        if inputs is None:
            inputs = self.streams()
        cond_mel, ref_wav, text_inputs, audio_codes = inputs
        B = int(text_inputs.shape[0])
        codes = self._codes_for_vocode(audio_codes)   # the length the vocoder is built for
        # ---- OUTSIDE observed: encoding (tokens/mel/pad), uploads, weight build ----
        text_ids = self._pad_text_ids(text_inputs)
        prefix_ids = self._prefix_ids(text_inputs)
        wav_lengths = torch.zeros(B) + float(int(codes.shape[-1]) * self.code_stride)
        mel_ids = self._pad_mel_ids(codes, wav_lengths)
        cond_tt = [self._rep(cond_mel[b:b + 1].float()) for b in range(B)]
        wav_tt = [self._rep(ref_wav[b:b + 1].float(), dtype=ttnn.float32) for b in range(B)]
        f_lat = self._build_latent_head(mel_len=int(mel_ids.shape[1]), sub=5)  # weight build
        # The prefix length is pure shape arithmetic (perceiver latents + padded text +
        # the mel start row), so the decode position tensor can be staged out here --
        # a torch.zeros inside the observed region would itself be a host op.
        n_cond = int(self.gpt.conditioning_perceiver.latents.shape[0])
        real = n_cond + int(prefix_ids[0].shape[1]) + 1
        C = self._tile_ceil(real + 1)
        pos_tt = self._kv_pos(real, B)                   # first GENERATED token's position
        # ---- INSIDE observed: pure model math (encoded inputs -> waveform), B streams ----
        with observe_host_ops() as ops:
            cl = ttnn.concat([self.encode_conditioning(c) for c in cond_tt], dim=0) \
                if B > 1 else self.encode_conditioning(cond_tt[0])
            g = [self.encode_speaker(w) for w in wav_tt]
            # prefill -> first token -> ONE cached AR step (the real decode chain)
            emb_d, _ = self._prefix_emb(cl, capacity=C, ids=prefix_ids)
            pref = self.f_gpt_logits.prefill_cache(emb_d)
            first = ttnn.argmax(ttnn.slice(pref, [0, real - 1, 0],
                                          [B, real, self.num_audio]), dim=-1)
            row = self._embed_next(first, 1, B=B)
            step_logits = self.f_gpt_logits.decode_one(row, pos_tt)
            nxt = self._embed_next(ttnn.argmax(step_logits, dim=-1), 2, B=B)
            # latent head + vocoder
            emb = self._assemble_emb(cl, text_ids, mel_ids)
            lat = f_lat(emb)
            wav = self.vocode(lat, g)
            try:
                ttnn.synchronize_device(self.device)
            except Exception:
                pass
            del wav, nxt
        v = verdict(ops)
        print(f"[host_op] B={B} on_device={v['on_device']} n_host_ops={v['n_host_ops']} "
              f"{v['reason']}")
        return v

    # ===================================================================== #
    # reference (HF) helpers — SETUP/REFERENCE only, NOT the TT forward path
    # ===================================================================== #
    def hf_reference(self, cond_mel, ref_wav_16k, text_inputs, audio_codes):
        """Deterministic golden for ONE stream: HF conditioning + speaker + latent + vocode."""
        gpt, hd = self.gpt, self.hd
        with torch.no_grad():
            ce = gpt.conditioning_encoder(cond_mel.float())
            cl = gpt.conditioning_perceiver(ce.transpose(1, 2))  # [1,32,1024]
            g = hd.speaker_encoder(ref_wav_16k.float(), l2_norm=True).unsqueeze(-1)  # [1,512,1]
            text_lengths = torch.tensor([text_inputs.shape[-1]])
            wav_lengths = torch.tensor([audio_codes.shape[-1] * self.code_stride])
            latent = gpt(text_inputs, text_lengths, audio_codes, wav_lengths,
                         cond_latents=cl, return_latent=True)
            wav = hd(latent, g=g)
        return {"cond_latents": cl, "g": g, "latents": latent, "waveform": wav}

    def hf_greedy_codes(self, cond_mel, text_inputs, horizon):
        """HF greedy (do_sample=False) audio codes for ONE stream over the SAME
        model-grounded horizon the TT decode uses."""
        gpt = self.gpt
        with torch.no_grad():
            ce = gpt.conditioning_encoder(cond_mel.float())
            cl = gpt.conditioning_perceiver(ce.transpose(1, 2))
            gen = gpt.generate(
                cond_latents=cl, text_inputs=text_inputs,
                do_sample=False, num_beams=1, temperature=1.0, top_k=None, top_p=None,
                repetition_penalty=1.0, length_penalty=1.0, num_return_sequences=1,
                output_attentions=False, max_new_tokens=int(horizon),
            )
        return gen

    def hf_step_logits(self, cond_mel, text_inputs, code_prefix):
        """Reference next-code logits GIVEN a code prefix (teacher forcing), for ONE stream.

        This is generate()'s own per-step distribution: cond | text | start_audio | codes...
        through the reference transformer + lm_head. Used by the e2e test to certify a greedy
        divergence as a numerical near-tie rather than a wiring error.
        """
        gpt = self.gpt
        with torch.no_grad():
            ce = gpt.conditioning_encoder(cond_mel.float())
            cl = gpt.conditioning_perceiver(ce.transpose(1, 2))
            tids = self._pad_text_ids(text_inputs)
            temb = gpt.text_embedding(tids) + gpt.text_pos_embedding(tids)
            n = int(code_prefix.reshape(-1).shape[0]) if code_prefix is not None else 0
            mids = torch.zeros(1, n + 1, dtype=torch.long)
            mids[0, 0] = self.start_audio            # the mel stream opens on start_audio
            if n:
                mids[0, 1:] = code_prefix.reshape(-1).long()
            memb = gpt.mel_embedding(mids) + gpt.mel_pos_embedding(mids)
            # cond | text | mel, written into one preallocated block: the row offsets ARE
            # the prefix layout, so spell them out rather than relying on a concat order.
            rows = [int(p.shape[1]) for p in (cl, temb, memb)]
            emb = cl.new_zeros(1, sum(rows), int(cl.shape[2]))
            off = 0
            for part, r in zip((cl, temb, memb), rows):
                emb[:, off:off + r] = part
                off += r
            hid = gpt.gpt_inference.transformer(inputs_embeds=emb).last_hidden_state
            lg = gpt.gpt_inference.lm_head(hid)
        return lg.reshape(1, -1, lg.shape[-1])[0, -1, : self.num_audio].float()

    def hf_reference_streams(self, inputs, horizon=None, codes=None):
        """Per-stream goldens for the whole batch.

        Each stream gets its OWN ``generate()`` code sequence (``gold["codes"]``, the
        behavioural reference for the decode) and its OWN deterministic latent+vocode
        golden. ``codes`` optionally supplies the per-stream code sequence the golden
        waveform is built over -- pass the TT-generated codes to compare the two chains
        over the SAME tokens when greedy hits a numerical tie. Nothing here feeds the TT
        pipeline; it is reference-side only.
        """
        cond_mel, ref_wav, text_inputs, _codes = inputs
        B = int(text_inputs.shape[0])
        hz = int(horizon) if horizon is not None else self.vocode_len
        out = []
        for b in range(B):
            cm = cond_mel[b:b + 1]
            rw = ref_wav[b:b + 1]
            ti = text_inputs[b:b + 1]
            gen_codes = self.hf_greedy_codes(cm, ti, hz).reshape(1, -1).long()
            use = gen_codes if codes is None else codes[b:b + 1].reshape(1, -1).long()
            gold = self.hf_reference(cm, rw, ti, use[:, : self.vocode_len])
            gold["codes"] = gen_codes
            gold["codes_used"] = use[:, : self.vocode_len]
            out.append(gold)
        return out


# --------------------------------------------------------------------------- #
# module-level factory the perf/2CQ harness + demo + test all call
# --------------------------------------------------------------------------- #
def build_pipeline(device, model=None, batch: int = DECODE_BATCH, vocode_len=None, **kwargs):
    """Construct and RETURN the resident XttsPipeline object (does not run it)."""
    return XttsPipeline(device, model=model, batch=batch, vocode_len=vocode_len)


# --------------------------------------------------------------------------- #
# module-level self-test hooks — the observer/probe calls these by NAME on the
# module (getattr(pipeline, "host_op_selftest")). The probe passes NO device, so
# these stand one up via the OUT-OF-PACKAGE opener (_selftest_device). The tt/
# package itself never opens a device, keeping the fixture the sole opener of the
# num_command_queues=2 trace+2CQ device.
# --------------------------------------------------------------------------- #
def host_op_selftest(inputs=None):
    """Module-level authoritative fully-on-device check (stands up its own device)."""
    from models.demos.xtts_v2._selftest_device import close_selftest_device, open_selftest_device

    dev, is_mesh = open_selftest_device(trace=False)
    try:
        return build_pipeline(dev).host_op_selftest(inputs)
    finally:
        close_selftest_device(dev, is_mesh)


def trace_capture_selftest(device=None, inputs=None):
    """Module-level trace+2CQ capture check. Uses ``device`` if given, else stands
    up a trace-enabled mesh via the out-of-package opener. Returns True iff every
    stage captured host-free and matched its eager reference."""
    if device is not None:
        return build_pipeline(device).trace_capture_selftest(device, inputs)
    from models.demos.xtts_v2._selftest_device import close_selftest_device, open_selftest_device

    dev, is_mesh = open_selftest_device(trace=True)
    try:
        return build_pipeline(dev).trace_capture_selftest(dev, inputs)
    finally:
        close_selftest_device(dev, is_mesh)

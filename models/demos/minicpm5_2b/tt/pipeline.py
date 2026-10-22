# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The ONE chained TTNN pipeline for openbmb/MiniCPM5-2B text generation (LlamaForCausalLM).

Both demo/demo_text_generation.py and tests/e2e/ call this file; nothing else holds the wiring.

Forward (every op on device, every graduated stub inside it):
  ids -> token_embed -> encoder_stack[ layer i: decoder_layer (even i) | layer (odd i), each holding
         r_m_s_norm -> attention (resident KV) -> r_m_s_norm -> mlp (even i) | m_l_p (odd i) ]
      -> r_m_s_norm (final) -> decoder_head -> on-device sampler -> next token
  rotary_embedding turns the device position ids into the cos/sin every attention reads.

Sampling is generation_config's chain (top_k -> top_p -> temperature 1.0), drawn by Gumbel-max with
per-sample seeded noise (tt/inputs.py) so a seed gives the same sample on TT and in the HF golden.
"""
from __future__ import annotations

import collections
import contextlib
import math
import os
import struct
import sys
import time

import torch

import ttnn

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 4)))
if _REPO not in sys.path:  # tt/ is also imported as a top-level package by the perf tooling
    sys.path.insert(0, _REPO)

from models.demos.minicpm5_2b._stubs import attention as attention_stub  # noqa: E402
from models.demos.minicpm5_2b._stubs import decoder_head as decoder_head_stub  # noqa: E402
from models.demos.minicpm5_2b._stubs import decoder_layer as decoder_layer_stub  # noqa: E402
from models.demos.minicpm5_2b._stubs import encoder_stack as encoder_stack_stub  # noqa: E402
from models.demos.minicpm5_2b._stubs import layer as layer_stub  # noqa: E402
from models.demos.minicpm5_2b._stubs import m_l_p as m_l_p_stub  # noqa: E402
from models.demos.minicpm5_2b._stubs import mlp as mlp_stub  # noqa: E402
from models.demos.minicpm5_2b._stubs import r_m_s_norm as r_m_s_norm_stub  # noqa: E402
from models.demos.minicpm5_2b._stubs import rotary_embedding as rotary_embedding_stub  # noqa: E402
from models.demos.minicpm5_2b._stubs import token_embed as token_embed_stub  # noqa: E402
from models.demos.minicpm5_2b.tt import inputs as tt_inputs  # noqa: E402

PIPELINE_STAGES = ["prefill", "decode"]

# Every graduated module of the bring-up (bringup_status.json + _stubs/*.py.last_good_native).
GRADUATED_STUBS = (
    "token_embed",
    "rotary_embedding",
    "encoder_stack",
    "decoder_layer",
    "layer",
    "attention",
    "mlp",
    "m_l_p",
    "r_m_s_norm",
    "decoder_head",
)

_TILE = 32
_NEG = -1e9
# Decode: the fused attention picks its key-mask tiles from a fixed 32-tile table (mask_table) instead of the
# [B, 1, 32, C] mask built every step (three ops).
_MASK_TABLE = True
# Decode: cos / signed sin rows gathered per step from tables of every cache position (one 4-core data-movement
# program) instead of the position * inv_freq, cos, sin, typecasts, L1 copy and * rot_sign ops.
_ROPE_TABLE = True
_L1_CONSTS_KERNEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_stubs", "kernels", "l1_consts"
)
_POS_STEP_KERNEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_stubs", "kernels", "pos_step"
)
_NOISE_SEG_W = 1280  # 40 tiles: under gather's 60-tile single-row threshold; 130560 / 1280 = 102 segments
# Candidate noise read straight from the segmented buffer by one data-movement kernel (else: per-segment gather).
_USE_NOISE_KERNEL = True
# Argmax over the candidates + token-id lookup written straight into tok_buf by one kernel (else: untilize,
# argmax, tilize, gather, typecasts, reshape, copy).
_USE_PICK_KERNEL = True
# The LM head writes its logits as [1, S, B, V / S] (the same tile pages as [1, B, V]) so the sampler's top-k runs in
# two stages: ttnn.topk's top-_SEG_K of each segment (S * B rows over the grid: its large-indices route splits by
# rows, so one [B, V] row set kept 32 cores busy) and a k-way merge of the S sorted lists (kernels/sampler).
_LOGIT_SEGMENTS = 17
_SEG_K = 64  # >= top_k (50); half the per-segment sort and merge of 128
_SAMPLER_KERNELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_stubs", "kernels", "sampler")


def _round_up(n, m=_TILE):
    return ((int(n) + m - 1) // m) * m


class CountedStub:
    """A graduated stub plus its Gate-2 invocation counter; calls go straight through to the stub."""

    def __init__(self, name, stub, counts):
        self.name, self.stub, self.counts = name, stub, counts

    def __call__(self, *args, **kwargs):
        self.counts[self.name] += 1
        return self.stub(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self.__dict__["stub"], item)


def sampling_params(hf_model):
    """(top_k, top_p, temperature) generate() applies for this checkpoint's generation_config.

    top_k is unset in generation_config.json, so generate() fills transformers' global default.
    """
    gc = hf_model.generation_config
    defaults = gc._get_default_generation_params()
    top_k = gc.top_k if gc.top_k is not None else defaults["top_k"]
    top_p = gc.top_p if gc.top_p is not None else defaults["top_p"]
    temperature = gc.temperature if gc.temperature is not None else defaults["temperature"]
    return int(top_k), float(top_p), float(temperature)


def eos_token_ids(hf_model):
    eos = hf_model.generation_config.eos_token_id
    return [int(e) for e in (eos if isinstance(eos, (list, tuple)) else [eos])]


class MiniCPM5Pipeline:
    """Resident TT text-generation pipeline: weights, KV caches and every per-step buffer on device."""

    def __init__(self, device, hf_model, layers=None, batch=None, prompt_len=None, max_new_tokens=None):
        cfg = hf_model.config
        self.device = device
        self.hf_model = hf_model  # HF reference: ground truth for section structure and goldens
        self.config = cfg
        n_total = cfg.num_hidden_layers
        self.n_layers = n_total if layers is None else max(1, min(int(layers), n_total))
        self.batch = int(batch) if batch is not None else tt_inputs.batch_size()
        self.prompt_len = int(prompt_len) if prompt_len is not None else int(tt_inputs.example_input_ids().shape[-1])
        self.max_new_tokens = int(max_new_tokens) if max_new_tokens is not None else tt_inputs.EXAMPLE_MAX_NEW_TOKENS
        self.vocab = cfg.vocab_size
        self.hidden = cfg.hidden_size
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        # Fixed capacities: prefill length padded to a tile, KV slots for prompt + every new token.
        self.prefill_len = _round_up(self.prompt_len)
        self.capacity = _round_up(self.prompt_len + self.max_new_tokens)
        assert self.capacity <= cfg.max_position_embeddings, "KV capacity exceeds max_position_embeddings"
        self.top_k, self.top_p, self.temperature = sampling_params(hf_model)
        self.eos_ids = eos_token_ids(hf_model)
        self.pad_id = int(hf_model.generation_config.pad_token_id)

        self.invocations = collections.Counter()
        hf = hf_model.model

        def counted(name, stub):
            return CountedStub(name, stub, self.invocations)

        self.token_embed = counted("token_embed", token_embed_stub.build(device, hf.embed_tokens))
        self.rotary_embedding = counted("rotary_embedding", rotary_embedding_stub.build(device, hf.rotary_emb))
        blocks = []
        for i in range(self.n_layers):
            hl = hf.layers[i]
            even = i % 2 == 0
            parts = {
                "input_layernorm": counted("r_m_s_norm", r_m_s_norm_stub.build(device, hl.input_layernorm)),
                "self_attn": counted("attention", attention_stub.build(device, hl.self_attn)),
                "post_attention_layernorm": counted(
                    "r_m_s_norm", r_m_s_norm_stub.build(device, hl.post_attention_layernorm)
                ),
                "mlp": counted("mlp", mlp_stub.build(device, hl.mlp))
                if even
                else counted("m_l_p", m_l_p_stub.build(device, hl.mlp)),
            }
            if even:
                blocks.append(counted("decoder_layer", decoder_layer_stub.build(device, hl, parts=parts)))
            else:
                blocks.append(counted("layer", layer_stub.build(device, hl, parts=parts)))
        self.encoder_stack = counted("encoder_stack", encoder_stack_stub.build(device, hf.layers, blocks=blocks))
        self.final_norm = counted("r_m_s_norm", r_m_s_norm_stub.build(device, hf.norm))
        self.decoder_head = counted("decoder_head", decoder_head_stub.build(device, hf_model.lm_head))
        bind = getattr(self.decoder_head, "bind_prenorm", None)
        self._head_prenorm = bool(bind and bind(self.final_norm))

        self.kv_caches = [
            attention_stub.KVCache(device, self.batch, self.n_kv_heads, self.capacity, self.head_dim)
            for _ in range(self.n_layers)
        ]
        self._alloc_buffers()

    # ------------------------------------------------------------------ persistent device buffers
    def _tt(self, t, dtype, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=self.device)

    def _alloc_buffers(self):
        B, Sp, C = self.batch, self.prefill_len, self.capacity
        causal = torch.full((Sp, Sp), _NEG).triu(diagonal=1).reshape(1, 1, Sp, Sp)  # HF causal mask, additive
        self.prefill_mask = self._tt(causal, ttnn.bfloat16)
        self.prefill_pos = self._tt(torch.arange(Sp, dtype=torch.float32).repeat(B, 1), ttnn.float32)
        self.key_index_row = self._tt(torch.arange(C, dtype=torch.float32).reshape(1, 1, 1, C), ttnn.float32)
        # The key index broadcast over a tile's 32 rows: the decode key mask is built at its [B, 1, 32, C] size.
        self.key_index_full = self._tt(torch.arange(C, dtype=torch.float32).repeat(1, 1, 32, 1), ttnn.float32)
        # The decode key mask's only distinct tiles, made once by the per-step mask's own ops: tile r is a key tile
        # holding the position at column r (columns c > r masked); tile 31 the all-unmasked tile (0 * NEG = -0).
        # The fused attention reads tile pos % 32 for the position tile and tile 31 before it (none after it).
        col = self._tt(torch.arange(32, dtype=torch.float32).repeat(1, 1, 32, 1), ttnn.float32)
        at = self._tt(torch.arange(32, dtype=torch.float32).reshape(32, 1, 1, 1), ttnn.float32)
        self.mask_table = ttnn.typecast(ttnn.multiply(ttnn.gt(col, at), _NEG), ttnn.bfloat16)  # [32, 1, 32, 32]
        # ... followed by one copy per attention core (R = B * n_kv row blocks) of the tiles every core reads -- the
        # unmasked tile, the score scale, the max scaler -- so no single L1 page is read by all R cores at once; the
        # fp32 sum scaler likewise as R tiles. One per-step L1 copy of each brings them in.
        att = self.encoder_stack.layers[0].self_attn
        R = B * self.n_kv_heads
        reps = ttnn.Shape([R, 1, 1, 1])
        unmasked = ttnn.slice(self.mask_table, (31, 0, 0, 0), (32, 1, 32, 32))
        self.mask_table = ttnn.concat(
            [self.mask_table] + [ttnn.repeat(t, reps) for t in (unmasked, att.scale_tile, att.max_scaler)], dim=0
        )  # [32 + 3R, 1, 32, 32]
        self.sum_scalers = ttnn.repeat(att.sum_scaler, reps)  # [R, 1, 32, 32] fp32
        self.mask_replicas = R
        kv = self.n_kv_heads
        self.key_index_kv = self._tt(torch.arange(C, dtype=torch.float32).repeat(1, kv, 1, 1), ttnn.float32)
        # rotate_half's sign over head_dim (-1 on the first half), folded into sin for the decode rope kernel.
        d = self.head_dim
        sign = torch.cat([-torch.ones(d // 2), torch.ones(d // 2)]).reshape(1, 1, 1, d)
        self.rot_sign = self._tt(sign, ttnn.bfloat16)
        if _ROPE_TABLE and attention_stub._USE_QKV_ROPE and B == 32:
            # Decode cos / signed sin of every cache position, made once by the per-step rows ops themselves.
            self.rotary_embedding.build_rows_table(C, self.rot_sign)
        k = self.top_k
        self.cumsum_upper = self._tt(torch.triu(torch.ones(k, k)), ttnn.float32)  # p @ U = inclusive cumsum
        self.ids_buf = self._tt(torch.full((B, Sp), self.pad_id, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        # [1, B]: one row-major page, the sampler kernel's output and the decode embedding's input as is.
        self.tok_buf = self._tt(torch.zeros(1, B, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.pos_buf = self._tt(torch.zeros(B, 1), ttnn.float32)
        # The same positions as int32 [1, B]: the in-place KV-cache write slot of each row.
        self.pos_i32 = self._tt(torch.zeros(1, B, dtype=torch.int32), ttnn.int32)
        # ... and as the row-major [B] page the fused decode attention reads (cur_pos), stepped by the same kernel.
        self.pos_rm = self._tt(torch.zeros(B, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        # Sampling noise lives as [1, S, B, V/S]: S vocab segments of _NOISE_SEG_W columns, so the per-step
        # gather of the top-k candidates' noise reads one short segment row per core instead of one core
        # streaming the whole vocab row.
        S, W = self.vocab // _NOISE_SEG_W, _NOISE_SEG_W
        self.noise_buf = self._tt(torch.zeros(1, S, B, W), ttnn.bfloat16)
        self.noise_seg_base = self._tt(torch.arange(S, dtype=torch.float32).mul_(W).reshape(1, S, 1, 1), ttnn.float32)
        self.noise_steps = []
        # The noise the sampler reads: noise_buf (traced steps copy their noise into it), or the step's own
        # resident noise tensor in eager decode_step (no 8 MB copy per token).
        self.noise_src = self.noise_buf
        self.real_len = self.prompt_len

    def load_request(self, input_ids, noise):
        """Upload one request: prompt ids [B, S] and the per-step sampling noise [T, B, V] (bf16).

        Host->device input encoding only; the model math runs from these resident buffers.
        """
        B, S = input_ids.shape
        assert B == self.batch, f"request batch {B} != pipeline batch {self.batch}"
        assert S <= self.prefill_len, f"prompt {S} longer than the prefill capacity {self.prefill_len}"
        ids = torch.full((B, self.prefill_len), self.pad_id, dtype=torch.int32)
        ids[:, :S] = input_ids.to(torch.int32)
        ttnn.copy(self._tt(ids, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT), self.ids_buf)
        ttnn.copy(self._tt(torch.full((B, 1), float(S)), ttnn.float32), self.pos_buf)
        ttnn.copy(self._tt(torch.full((1, B), S, dtype=torch.int32), ttnn.int32), self.pos_i32)
        ttnn.copy(self._tt(torch.full((B,), S, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT), self.pos_rm)
        self.real_len = S
        for t in self.noise_steps:
            ttnn.deallocate(t)
        S, W = self.vocab // _NOISE_SEG_W, _NOISE_SEG_W
        self.noise_steps = [
            self._tt(noise[t].reshape(B, S, W).transpose(0, 1).reshape(1, S, B, W).contiguous(), ttnn.bfloat16)
            for t in range(noise.shape[0])
        ]
        ttnn.copy(self.noise_steps[0], self.noise_buf)
        self.noise_src = self.noise_buf

    # ------------------------------------------------------------------ on-device sampler
    def _sample(self, logits):
        """generate()'s sampler on device: top_k -> top_p filter, Gumbel-max with the resident noise.

        logits [B, 1, V] bf16 -> token ids [1, 1, B] uint32. Mirrors HF's TopK/TopP warpers: keep the
        top_k logits (ties kept), then drop the low tail whose ascending cumulative mass <= 1 - top_p.
        """
        B, V, k = self.batch, self.vocab, self.top_k
        if len(logits.shape) == 4 and logits.shape[1] > 1:
            # Segmented logits [1, S, B, V / S] (temperature 1.0 only): the two-stage top-k.
            top_vals, top_idx = self._topk_segmented(logits, k)  # fp32 values, uint32 ids
        elif self.temperature != 1.0:
            logits = ttnn.reshape(logits, (1, 1, B, V))
            scores = ttnn.multiply(ttnn.typecast(logits, ttnn.float32), 1.0 / self.temperature)
            top_vals, top_idx = ttnn.topk(scores, k=k, dim=-1)  # [1, 1, B, k], descending
        else:
            logits = ttnn.reshape(logits, (1, 1, B, V))
            # bf16 -> fp32 is exact, so take the top k on the bf16 logits: that input routes to the
            # multi-core topk (an fp32 input falls to the single-core factory at this width).
            top_vals, top_idx = ttnn.topk(logits, k=k, dim=-1)
            top_vals = ttnn.typecast(top_vals, ttnn.float32)
        probs = ttnn.softmax(top_vals, dim=-1)
        incl = ttnn.matmul(probs, self.cumsum_upper)  # mass of ranks <= j
        tail_mass = ttnn.add(ttnn.subtract(probs, incl), 1.0)  # mass of ranks >= j (HF's ascending cumsum)
        if _USE_PICK_KERNEL and B == 32 and top_vals.dtype == ttnn.float32 and top_idx.dtype == ttnn.uint32:
            # keep / threshold / allowed / argmax are compares and selects only: the pick kernel does them on the
            # fp32 bits; the one rounding add (top_vals + noise) stays the same ttnn op as below.
            scored = ttnn.add(top_vals, self._candidate_noise(top_idx))
            self._pick_kernel(tail_mass, top_vals, scored, top_idx)  # writes self.tok_buf
            return None
        keep = ttnn.gt(tail_mass, 1.0 - self.top_p)
        kept_vals = ttnn.add(top_vals, ttnn.multiply(ttnn.subtract(keep, 1.0), _NEG))  # dropped -> +1e9
        threshold = ttnn.min(kept_vals, dim=-1, keepdim=True)  # smallest kept logit, [1, 1, B, 1]
        # Every allowed token is one of the top k, so the Gumbel-max runs over the k candidates only:
        # gather each candidate's noise, argmax over k, then map the winning rank back to its token id.
        allowed = ttnn.ge(top_vals, threshold)
        noise = self._candidate_noise(top_idx)
        perturbed = ttnn.add(ttnn.add(top_vals, noise), ttnn.multiply(ttnn.subtract(allowed, 1.0), -_NEG))
        rank = ttnn.argmax(ttnn.untilize(perturbed), dim=-1, keepdim=True)  # [1, 1, B, 1] uint32
        rank = ttnn.to_layout(rank, ttnn.TILE_LAYOUT)
        tok = ttnn.gather(top_idx, -1, ttnn.typecast(rank, top_idx.dtype) if rank.dtype != top_idx.dtype else rank)
        return ttnn.to_layout(ttnn.typecast(tok, ttnn.uint32) if tok.dtype != ttnn.uint32 else tok, ttnn.ROW_MAJOR_LAYOUT)

    def _topk_segmented(self, logits, k):
        """top-k over segmented logits [1, S, B, W]: each segment's top-_SEG_K by ttnn.topk (k <= _SEG_K, so every
        global top-k token survives in its segment), then the k-way merge of the S sorted lists as one generic_op
        (kernels/sampler/topk_merge.cpp, one core per batch row). Returns fp32 values and uint32 global ids,
        [1, 1, B, k]."""
        S, B, W = logits.shape[1], logits.shape[2], logits.shape[3]
        vals, ids = ttnn.topk(logits, k=_SEG_K, dim=-1)  # [1, S, B, _SEG_K], descending, in-segment ids
        l1 = ttnn.L1_MEMORY_CONFIG
        top_vals = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, B, k]), ttnn.float32, ttnn.TILE_LAYOUT, self.device, l1)
        top_idx = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, B, k]), ttnn.uint32, ttnn.TILE_LAYOUT, self.device, l1)
        k_tiles = top_vals.padded_shape[-1] // 32
        grid = self.device.compute_with_storage_grid_size()
        full_rows, rem = divmod(B, grid.x)
        ranges = []
        if full_rows:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, full_rows - 1)))
        if rem:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
        cores = ttnn.CoreRangeSet(ranges)
        rt = ttnn.RuntimeArgs()
        for row in range(B):
            rt[row % grid.x][row // grid.x] = [row]
        ks_tiles = vals.padded_shape[-1] // 32
        tile = 2 * 32 * 32
        scratch = 2 * S * ks_tiles * 2 * 64 + 2 * k_tiles * 2 * 64
        cb = ttnn.CBDescriptor(
            total_size=-(-scratch // tile) * tile,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.bfloat16, tile)],
        )
        reader = ttnn.KernelDescriptor(
            kernel_source=os.path.join(_SAMPLER_KERNELS, "topk_merge.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[k, k_tiles, S, W, ks_tiles, 0]
            + list(ttnn.TensorAccessorArgs(vals).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(ids).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(top_vals).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(top_idx).get_compile_time_args()),
            runtime_args=rt,
            common_runtime_args=[
                vals.buffer_address(),
                ids.buffer_address(),
                top_vals.buffer_address(),
                top_idx.buffer_address(),
            ],
            config=ttnn.ReaderConfigDescriptor(),
        )
        program = ttnn.ProgramDescriptor(kernels=[reader], semaphores=[], cbs=[cb])
        ttnn.generic_op([vals, ids, top_vals, top_idx], program)
        return top_vals, top_idx

    def logits_to_torch(self, logits):
        """Host [B, V] float logits from either layout ([1, B, V] or segmented [1, S, B, V / S])."""
        t = ttnn.to_torch(logits)
        if t.dim() == 4 and t.shape[1] > 1:
            t = t.permute(0, 2, 1, 3)
        return t.reshape(self.batch, self.vocab).float()

    def _candidate_noise(self, top_idx):
        """Noise of each top-k candidate, [1, 1, B, k] fp32, gathered from the segmented noise buffer.

        Candidate token t sits in segment t // W at column t % W: every segment gathers at its local
        column (clamped in range), and only the segment that holds t contributes to the sum over segments.
        """
        if _USE_NOISE_KERNEL and top_idx.dtype == ttnn.uint32 and self.batch == 32:
            return self._noise_kernel(top_idx)
        W = _NOISE_SEG_W
        local = ttnn.subtract(ttnn.typecast(top_idx, ttnn.float32), self.noise_seg_base)  # [1, S, B, k]
        inside = ttnn.multiply(ttnn.ge(local, 0.0), ttnn.lt(local, float(W)))
        local = ttnn.typecast(ttnn.clamp(local, 0.0, float(W - 1)), ttnn.uint32)
        # The gathered noise follows the candidates (L1 in decode) rather than the resident DRAM noise buffer.
        seg_noise = ttnn.gather(self.noise_src, -1, local, memory_config=local.memory_config())
        seg_noise = ttnn.typecast(seg_noise, ttnn.float32)
        return ttnn.sum(ttnn.multiply(seg_noise, inside), dim=1, keepdim=True)

    def _noise_kernel(self, top_idx):
        """_candidate_noise as one generic_op (kernels/sampler/noise_gather.cpp): each candidate's bf16 noise read
        from its segment and widened to fp32 -- the value the masked segment sum produces."""
        B, k = self.batch, top_idx.shape[-1]
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, B, k]), ttnn.float32, ttnn.TILE_LAYOUT, self.device, ttnn.L1_MEMORY_CONFIG
        )
        k_tiles = top_idx.padded_shape[-1] // 32
        grid = self.device.compute_with_storage_grid_size()
        full_rows, rem = divmod(B, grid.x)
        ranges = []
        if full_rows:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, full_rows - 1)))
        if rem:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
        cores = ttnn.CoreRangeSet(ranges)
        rt = ttnn.RuntimeArgs()
        for b in range(B):
            rt[b % grid.x][b // grid.x] = [b, 1]
        tile = 2 * 32 * 32
        scratch = k_tiles * (2 * 64 + 32 * 64 + 2 * 64)
        cb = ttnn.CBDescriptor(
            total_size=-(-scratch // tile) * tile,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.bfloat16, tile)],
        )
        reader = ttnn.KernelDescriptor(
            kernel_source=os.path.join(_SAMPLER_KERNELS, "noise_gather.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[k, k_tiles, _NOISE_SEG_W, 0]
            + list(ttnn.TensorAccessorArgs(top_idx).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(self.noise_src).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
            runtime_args=rt,
            common_runtime_args=[top_idx.buffer_address(), self.noise_src.buffer_address(), out.buffer_address()],
            config=ttnn.ReaderConfigDescriptor(),
        )
        program = ttnn.ProgramDescriptor(kernels=[reader], semaphores=[], cbs=[cb])
        return ttnn.generic_op([top_idx, self.noise_src, out], program)

    def _pick_kernel(self, tail_mass, top_vals, scored, top_idx):
        """The top-p filter (keep, smallest kept logit, allowed) and the argmax of scored = top_vals + noise over the
        allowed candidates (ttnn.argmax's comparison, first max wins), then that rank's token id, written into
        tok_buf by one generic_op (kernels/sampler/pick_token_mc.cpp): one core per batch row, the B ids gathered
        into core (0, 0)'s L1 behind a semaphore and written from there."""
        B = self.batch
        k_tiles = scored.padded_shape[-1] // 32
        keep_above_bits = struct.unpack("<I", struct.pack("<f", 1.0 - self.top_p))[0]
        grid = self.device.compute_with_storage_grid_size()
        full_rows, rem = divmod(B, grid.x)
        ranges = []
        if full_rows:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, full_rows - 1)))
        if rem:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
        cores = ttnn.CoreRangeSet(ranges)
        rt = ttnn.RuntimeArgs()
        for b in range(B):
            rt[b % grid.x][b // grid.x] = [b]
        leader = self.device.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
        tile = 2 * 32 * 32
        scratch = 4 * k_tiles * 32 * 4 + 16 * B + 4 * B
        cb = ttnn.CBDescriptor(
            total_size=-(-scratch // tile) * tile,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.bfloat16, tile)],
        )
        reader = ttnn.KernelDescriptor(
            kernel_source=os.path.join(_SAMPLER_KERNELS, "pick_token_mc.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[scored.shape[-1], k_tiles, B, 0, 0, keep_above_bits]
            + list(ttnn.TensorAccessorArgs(tail_mass).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(top_vals).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(scored).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(top_idx).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(self.tok_buf).get_compile_time_args()),
            runtime_args=rt,
            common_runtime_args=[
                tail_mass.buffer_address(),
                top_vals.buffer_address(),
                scored.buffer_address(),
                top_idx.buffer_address(),
                self.tok_buf.buffer_address(),
                leader.x,
                leader.y,
            ],
            config=ttnn.ReaderConfigDescriptor(),
        )
        sem = ttnn.SemaphoreDescriptor(id=0, core_ranges=cores, initial_value=0)
        program = ttnn.ProgramDescriptor(kernels=[reader], semaphores=[sem], cbs=[cb])
        ttnn.generic_op([tail_mass, top_vals, scored, top_idx, self.tok_buf], program)

    def _emit(self, logits):
        tok = self._sample(logits)
        if tok is not None:  # else the sampler kernel already wrote tok_buf
            ttnn.copy(ttnn.reshape(tok, (1, self.batch)), self.tok_buf)

    # ------------------------------------------------------------------ the chained forward
    def _prefill_forward(self):
        """Prompt pass over the resident ids: seeds every layer's KV cache, emits the first token."""
        B, H = self.batch, self.hidden
        x = self.token_embed(self.ids_buf)
        cos_sin = self.rotary_embedding(x, position_ids=self.prefill_pos)
        # cos / signed sin [B, 1, S, D] once per prefill for every layer's rope kernel (rows are positions).
        cos, sin = (ttnn.reshape(t, (B, 1, t.shape[-2], self.head_dim)) for t in cos_sin)
        ctx = attention_stub.AttnContext("prefill", self.prefill_mask, rope=(cos, ttnn.multiply(sin, self.rot_sign)))
        h = self.encoder_stack(x, position_embeddings=cos_sin, kv_caches=self.kv_caches, attn_ctx=ctx)
        self.prefill_hidden = self.final_norm(h)  # [B, Sp, H]: the state prefill hands every decode step
        last = ttnn.slice(self.prefill_hidden, (0, self.real_len - 1, 0), (B, self.real_len, H))
        last = ttnn.reshape(last, (1, B, H))  # B real rows in one tile row for the vocab matmul
        logits = self.decoder_head(last, segments=self._logit_segments())
        self._emit(logits)
        return logits

    def _decode_forward(self):
        """One token per row from the resident token/position: reads + writes the KV cache, emits the next."""
        B = self.batch
        # [1, B, H]: the B single-token rows share one tile row instead of padding each to a tile
        # (the ids are looked up as one [1, B] row, so the embedding already lands in that layout).
        x = self.token_embed(self.tok_buf)
        rows = attention_stub._USE_QKV_ROPE and B == 32
        # rows: cos / sin as one tile row (row b = batch b) for the rope on the QKV layout, the only rope decode runs.
        table = rows and _ROPE_TABLE
        cos_sin = self.rotary_embedding(
            x,
            position_ids=self.pos_buf,
            rows=rows,
            table_pos=self.pos_i32 if table else None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        cur_pos = self.pos_rm  # [B] int32, one contiguous page (pos_i32 row-major, kept in step by _pos_step)
        # [B, 1, 1, 1] positions: only the non-kernel KV write and the non-table masks read them.
        needs_pos = not attention_stub._USE_KV_WRITE_KERNEL or not (rows and _MASK_TABLE)
        pos = ttnn.reshape(self.pos_buf, (B, 1, 1, 1)) if needs_pos else None
        # The one-hot cache-slot selector is read only by the non-kernel KV write.
        write = None
        if not attention_stub._USE_KV_WRITE_KERNEL:
            write = ttnn.typecast(ttnn.eq(self.key_index_kv, pos), ttnn.bfloat16)  # [B, kv, 1, C]
        if rows:
            # The fused attention reads only the row-broadcast mask: the same gt / * NEG / typecast per element at
            # [B, 1, 32, C] directly, instead of [B, 1, 1, C] untilized, repeated and re-tilized.
            mask = None
        else:
            mask = ttnn.typecast(ttnn.multiply(ttnn.gt(self.key_index_row, pos), _NEG), ttnn.bfloat16)  # [B, 1, 1, C]
        l1 = ttnn.L1_MEMORY_CONFIG
        sums_l1 = None
        if rows and _MASK_TABLE:
            # Both per-step L1 constant copies (mask table + sum scalers) as one program.
            mask_full, sums_l1 = self._attn_consts_l1()
        elif rows:
            mask_full = ttnn.typecast(
                ttnn.multiply(ttnn.gt(self.key_index_full, pos), _NEG), ttnn.bfloat16, memory_config=l1
            )
        else:
            mask_full = ttnn.repeat(mask, ttnn.Shape([1, 1, 32, 1]), memory_config=l1)
        if rows:
            if table:
                cos_rows, sin_rows = cos_sin  # gathered straight into L1, the sign already folded into sin
            else:
                cos_rows = ttnn.to_memory_config(cos_sin[0], l1)
                sin_rows = ttnn.multiply(cos_sin[1], self.rot_sign, memory_config=l1)  # rotate_half's sign folded in
            ctx = attention_stub.AttnContext(
                "decode", mask, write, cur_pos, mask_full=mask_full, rope_rows=(cos_rows, sin_rows)
            )
            ctx.mask_table = self.mask_replicas if _MASK_TABLE else 0
            # The attention constants copied to L1 for this step (persistent L1 copies clash with prefill's CBs).
            att = self.encoder_stack.layers[0].self_attn
            if _MASK_TABLE:
                ctx.consts = (mask_full, mask_full, mask_full, sums_l1)
            else:
                ctx.consts = tuple(
                    ttnn.to_memory_config(t, l1)
                    for t in (att.scale_tile, att.zero_tile, att.max_scaler, att.sum_scaler)
                )
            position_embeddings = None
        else:
            # cos / signed sin broadcast over a tile's rows once per step, shared by every layer's rope kernel.
            cos, sin = (ttnn.reshape(t, (B, 1, 1, self.head_dim)) for t in cos_sin)
            cos_full = ttnn.repeat(cos, ttnn.Shape([1, 1, 32, 1]), memory_config=l1)
            sin_signed = ttnn.multiply(ttnn.repeat(sin, ttnn.Shape([1, 1, 32, 1])), self.rot_sign, memory_config=l1)
            ctx = attention_stub.AttnContext(
                "decode", mask, write, cur_pos, rope=(cos_full, sin_signed), mask_full=mask_full
            )
            position_embeddings = cos_sin
        h = self.encoder_stack(x, position_embeddings=position_embeddings, kv_caches=self.kv_caches, attn_ctx=ctx)
        # Decode logits (8 MB) stay in L1 for the sampler's topk.
        l1 = ttnn.L1_MEMORY_CONFIG
        if self._head_prenorm:
            # The LM head applies the final norm itself (as the prologue of its program).
            logits = self.decoder_head(h, memory_config=l1, segments=self._logit_segments(), prenorm=self.final_norm)
        else:
            logits = self.decoder_head(
                self.final_norm(h, memory_config=l1), memory_config=l1, segments=self._logit_segments()
            )
        self._emit(logits)
        self._pos_step()
        return logits

    def _attn_consts_l1(self, n_cores=32):
        """L1 copies of the mask table and the sum-scaler replicas (read on the fused attention's critical path)
        by one data-movement generic_op (kernels/l1_consts) instead of two ttnn copies."""
        srcs = (self.mask_table, self.sum_scalers)
        dsts = [
            ttnn.allocate_tensor_on_device(t.shape, t.dtype, ttnn.TILE_LAYOUT, self.device, ttnn.L1_MEMORY_CONFIG)
            for t in srcs
        ]
        pages = [math.prod(list(t.padded_shape)) // (32 * 32) for t in srcs]
        sizes = [32 * 32 * (4 if t.dtype == ttnn.float32 else 2) for t in srcs]
        grid = self.device.compute_with_storage_grid_size()
        per = [-(-n // n_cores) for n in pages]
        rows, rem = divmod(n_cores, grid.x)
        ranges = []
        if rows:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, rows - 1)))
        if rem:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, rows), ttnn.CoreCoord(rem - 1, rows)))
        cores = ttnn.CoreRangeSet(ranges)
        rt = ttnn.RuntimeArgs()
        for i in range(n_cores):
            args = []
            for n, q in zip(pages, per):
                start = min(i * q, n)
                args += [start, min(q, n - start)]
            rt[i % grid.x][i // grid.x] = args
        accessor_ct = []
        for src, dst in zip(srcs, dsts):
            accessor_ct += list(ttnn.TensorAccessorArgs(src).get_compile_time_args())
            accessor_ct += list(ttnn.TensorAccessorArgs(dst).get_compile_time_args())
        buf = per[0] * sizes[0] + per[1] * sizes[1]
        reader = ttnn.KernelDescriptor(
            kernel_source=os.path.join(_L1_CONSTS_KERNEL_DIR, "copy_pages.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=[0, sizes[0], sizes[1]] + accessor_ct,
            runtime_args=rt,
            common_runtime_args=[srcs[0].buffer_address(), dsts[0].buffer_address()]
            + [srcs[1].buffer_address(), dsts[1].buffer_address()],
            config=ttnn.ReaderConfigDescriptor(),
        )
        cbs = [
            ttnn.CBDescriptor(
                total_size=buf, core_ranges=cores, format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.bfloat16, buf)]
            )
        ]
        program = ttnn.ProgramDescriptor(kernels=[reader], semaphores=[], cbs=cbs)
        ttnn.generic_op(list(srcs) + dsts, program)
        return dsts[0], dsts[1]

    def _pos_step(self):
        """pos_buf, pos_i32 and pos_rm += 1 in place as one 1-core data-movement generic_op (kernels/pos_step)."""
        core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        tile = 4 * 32 * 32
        reader = ttnn.KernelDescriptor(
            kernel_source=os.path.join(_POS_STEP_KERNEL_DIR, "pos_step.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core,
            compile_time_args=[0, 1, tile]
            + list(ttnn.TensorAccessorArgs(self.pos_buf).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(self.pos_i32).get_compile_time_args())
            + [2, 4 * self.batch]
            + list(ttnn.TensorAccessorArgs(self.pos_rm).get_compile_time_args()),
            runtime_args=ttnn.RuntimeArgs(),
            common_runtime_args=[
                self.pos_buf.buffer_address(),
                self.pos_i32.buffer_address(),
                self.pos_rm.buffer_address(),
            ],
            config=ttnn.ReaderConfigDescriptor(),
        )
        cbs = [
            ttnn.CBDescriptor(
                total_size=tile,
                core_ranges=core,
                format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.float32, tile)],
            ),
            ttnn.CBDescriptor(
                total_size=tile,
                core_ranges=core,
                format_descriptors=[ttnn.CBFormatDescriptor(1, ttnn.int32, tile)],
            ),
            ttnn.CBDescriptor(
                total_size=4 * self.batch,
                core_ranges=core,
                format_descriptors=[ttnn.CBFormatDescriptor(2, ttnn.int32, 4 * self.batch)],
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=[reader], semaphores=[], cbs=cbs)
        ttnn.generic_op([self.pos_buf, self.pos_i32, self.pos_rm], program)

    def _logit_segments(self):
        ok = self.vocab % (_LOGIT_SEGMENTS * _TILE) == 0 and self.temperature == 1.0 and self.top_k <= _SEG_K
        return _LOGIT_SEGMENTS if ok and self.batch == 32 else None

    def prefill_step(self):
        return self._prefill_forward()

    def decode_step(self, step):
        """Decode new token `step` (1-based after the prefill's token 0) with that step's noise (read in place)."""
        self.noise_src = self.noise_steps[step]
        try:
            return self._decode_forward()
        finally:
            self.noise_src = self.noise_buf

    # AR decode contract: decode_prefill seeds the resident self-attn KV; decode_step reads it.
    def decode_prefill(self, input_ids, noise):
        self.load_request(input_ids, noise)
        return self.prefill_step()

    def read_tokens(self):
        return ttnn.to_torch(self.tok_buf).reshape(self.batch).to(torch.int64)

    def run_text_generation(self, input_ids, noise, max_new_tokens=None, collect_logits=False, log=print):
        """Generate until every row hit an eos id or max_new_tokens (generate()'s stop rule).

        Returns dict(tokens [B, T] (pad after a row's eos), lengths [B], stop_reason, steps, logits
        (list of [B, V] float per step, when collect_logits), prefill_hidden [B, S, H]).
        """
        max_new_tokens = int(max_new_tokens or self.max_new_tokens)
        safety_cap = self.config.max_position_embeddings - input_ids.shape[-1]
        assert input_ids.shape[-1] + max_new_tokens <= self.capacity, "KV capacity below prompt + max_new_tokens"
        B = self.batch
        out_tokens, step_logits = [], []
        finished = torch.zeros(B, dtype=torch.bool)
        lengths = torch.full((B,), max_new_tokens, dtype=torch.int64)
        stop_reason = "max_new_tokens"
        prefill_logits = self.decode_prefill(input_ids, noise)
        prefill_hidden = ttnn.to_torch(self.prefill_hidden)[:, : self.real_len].float()
        t0 = time.time()
        for step in range(max_new_tokens):
            logits = self.decode_step(step) if step > 0 else prefill_logits
            ttnn.synchronize_device(self.device)
            tok = self.read_tokens()
            if collect_logits:
                step_logits.append(self.logits_to_torch(logits))
            tok = torch.where(finished, torch.full_like(tok, self.pad_id), tok)
            out_tokens.append(tok)
            newly = (~finished) & torch.isin(tok, torch.tensor(self.eos_ids))
            lengths[newly] = step + 1
            finished |= newly
            log(
                f"[minicpm5_2b] token {step + 1}/{max_new_tokens} done "
                f"({int(finished.sum())}/{B} rows finished, {time.time() - t0:.1f}s)"
            )
            if bool(finished.all()):
                stop_reason = "eos"
                break
            assert step + 1 < safety_cap, "decode reached the max_position_embeddings safety cap"
        return {
            "tokens": torch.stack(out_tokens, dim=1),
            "lengths": lengths,
            "stop_reason": stop_reason,
            "steps": len(out_tokens),
            "logits": step_logits,
            "prefill_hidden": prefill_hidden,
        }

    # ------------------------------------------------------------------ trace contract
    def _stored_request(self):
        return tt_inputs.stored_request(self.batch)

    def prefill_trace_inputs(self):
        ids, seeds = self._stored_request()
        return {"input_ids": ids, "noise": tt_inputs.gumbel_noise(seeds, 1, self.vocab)}

    def prefill_trace_setup(self, inputs):
        self.load_request(inputs["input_ids"], inputs["noise"])

    def prefill_trace_step(self):
        return self._prefill_forward()

    def prefill_trace_items(self):
        return self.batch * self.prefill_len

    def decode_trace_inputs(self):
        ids, seeds = self._stored_request()
        return {"input_ids": ids, "noise": tt_inputs.gumbel_noise(seeds, 2, self.vocab)}

    def decode_trace_setup(self, inputs):
        self.load_request(inputs["input_ids"], inputs["noise"])
        self._prefill_forward()  # seeds the resident KV + first token outside the trace
        ttnn.copy(self.noise_steps[1], self.noise_buf)
        self.noise_src = self.noise_buf

    def decode_trace_step(self):
        return self._decode_forward()

    def decode_trace_items(self):
        return self.batch

    def decode_trace_repeats(self):
        return self.max_new_tokens - 1  # the prefill emits token 1; decode emits the rest

    def trace_region_bytes(self):
        """Trace buffer for the largest stage (prefill at Sp x layers), with headroom."""
        per_layer = 2 * 1024 * 1024
        return int(64 * 1024 * 1024 + per_layer * self.n_layers)

    def trace_capture_selftest(self, pcc_min=0.99):
        return trace_capture_selftest(self.device, pipeline=self, pcc_min=pcc_min)

    def host_op_selftest(self):
        return host_op_selftest(self.device, pipeline=self)

    # ------------------------------------------------------------------ HF reference helpers (setup only)
    def hf_reference_logits(self, sequences):
        """HF logits of `sequences` [B, L] with this build's depth (all layers unless capped)."""
        hf = self.hf_model
        full = hf.model.layers
        n_cfg = hf.config.num_hidden_layers
        try:
            if self.n_layers != n_cfg:
                hf.model.layers = torch.nn.ModuleList(list(full)[: self.n_layers])
                hf.config.num_hidden_layers = self.n_layers
            with torch.no_grad():
                return hf(input_ids=sequences, use_cache=False).logits.float()
        finally:
            hf.model.layers = full
            hf.config.num_hidden_layers = n_cfg


def load_hf_model():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(tt_inputs.MODEL_ID, torch_dtype=torch.float32)
    return model.eval()


def build_pipeline(device, model=None, layers=None, prefill_layers=None, decode_layers=None, **kwargs):
    """Construct and return the resident pipeline object (does NOT run it).

    `layers` caps the depth of the model's one repeated stack (model.layers); None = all 42. Prefill and
    decode run the SAME stack, so prefill_layers / decode_layers are accepted as aliases of it (the
    first one given wins). Demo kwargs (prompt, text, ...) are accepted and ignored; `batch`,
    `prompt_len` and `max_new_tokens` size the resident buffers.
    """
    depth = next((d for d in (prefill_layers, decode_layers, layers) if d is not None), None)
    if depth is None and os.environ.get("TT_PERF_LAYERS"):
        depth = int(os.environ["TT_PERF_LAYERS"])
    hf_model = model if model is not None else load_hf_model()
    return MiniCPM5Pipeline(
        device,
        hf_model,
        layers=depth,
        batch=kwargs.get("batch"),
        prompt_len=kwargs.get("prompt_len"),
        max_new_tokens=kwargs.get("max_new_tokens"),
    )


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def _stage_reference(pipe, stage, inputs, traced_tokens_before):
    """HF logits the stage's step should produce (computed from the stage's own inputs)."""
    ids = inputs["input_ids"]
    if stage == "prefill":
        return pipe.hf_reference_logits(ids[:, :])[:, -1, :]
    seq = torch.empty(ids.shape[0], ids.shape[1] + 1, dtype=ids.dtype)
    seq[:, :-1], seq[:, -1] = ids, traced_tokens_before.reshape(-1).to(ids.dtype)  # prompt + fed token
    return pipe.hf_reference_logits(seq)[:, -1, :]


@contextlib.contextmanager
def _device_or_default(device):
    """Yield `device`; a bare call (the gate's probes) borrows the demo's device open instead."""
    if device is not None:
        yield device
        return
    from models.demos.minicpm5_2b.demo.device import opened_device

    with opened_device() as dev:
        yield dev


def trace_capture_selftest(device=None, pipeline=None, pcc_min=0.99, layers=None):
    """Capture ONE step per stage in a trace, replay it, release it, and compare to the HF reference.

    True only if every stage captured (host-free) and its replayed logits match HF (PCC, every row).
    """
    if pipeline is None:
        with _device_or_default(device) as dev:
            return _trace_capture_selftest(dev, build_pipeline(dev, layers=layers), pcc_min)
    return _trace_capture_selftest(device, pipeline, pcc_min)


def _trace_capture_selftest(device, pipe, pcc_min):
    ok = True
    for stage in PIPELINE_STAGES:
        setup = getattr(pipe, f"{stage}_trace_setup")
        step = getattr(pipe, f"{stage}_trace_step")
        inputs = getattr(pipe, f"{stage}_trace_inputs")()
        setup(inputs)
        step()  # compile outside the trace
        setup(inputs)
        fed = pipe.read_tokens() if stage == "decode" else None
        try:
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            out = step()
            ttnn.end_trace_capture(device, tid, cq_id=0)
        except Exception as exc:  # noqa: BLE001
            print(f"TRACE_CAPTURE_FAILED[{stage}]: {exc}; capacity C={pipe.capacity} cannot be traced")
            return False
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        got = pipe.logits_to_torch(out)
        ttnn.release_trace(device, tid)
        ref = _stage_reference(pipe, stage, inputs, fed)
        worst = min(_pcc(got[b], ref[b]) for b in range(pipe.batch))
        print(f"trace selftest stage={stage} captured+replayed, min-row PCC vs HF={worst:.5f}")
        ok = ok and worst >= pcc_min
    return ok


def host_op_selftest(device=None, pipeline=None, layers=None):
    """The authoritative on-device check: no host aten op may fire inside the model math.

    Tokenisation, noise generation, upload and weight build happen OUTSIDE the observed region;
    prefill + decode steps (embedding through sampling) run INSIDE it.
    """
    if pipeline is None:
        with _device_or_default(device) as dev:
            return _host_op_selftest(build_pipeline(dev, layers=layers))
    return _host_op_selftest(pipeline)


def _host_op_selftest(pipe):
    from scripts.tt_hw_planner import host_op_observer

    ids, seeds = tt_inputs.batch_inputs(pipe.batch)
    pipe.load_request(ids, tt_inputs.gumbel_noise(seeds, 3, pipe.vocab))
    ttnn.synchronize_device(pipe.device)
    with host_op_observer.observe_host_ops() as ops:
        pipe.prefill_step()
        pipe.decode_step(1)
        pipe.decode_step(2)
        ttnn.synchronize_device(pipe.device)
    return host_op_observer.verdict(list(ops))

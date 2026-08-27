# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN stub for LlamaModel (language_model of Voxtral-Mini-3B-2507).

Embedding → 30 × (RMSNorm → GQA Attention + RoPE → Add → RMSNorm → SwiGLU MLP → Add) → RMSNorm.

Build contract
--------------
``build(device, torch_module, layer_range=None, skip_embedding=False,
        rope_capacity=1024, apply_final_norm=True)``

  * ``layer_range=(lo, hi)``   -- upload ONLY ``torch_module.layers[lo:hi]``
    (saves several GB of device DRAM when the model is sharded across runs).
    ``None`` uploads all layers, exactly as before.
  * ``skip_embedding=True``    -- do not upload the 131072x3072 embedding table
    (~0.8 GB); only legal if the caller always supplies ``inputs_embeds``.
  * ``rope_capacity``          -- rows in the internal fallback RoPE table
    (was hard-coded 256).  Longer table, same values: slicing the first S rows
    is bit-identical to the old table for S <= 256.
  * ``apply_final_norm``       -- default for the trailing rms_norm.

Call contract
-------------
``__call__(x=None, *, inputs_embeds=None, rope=None, kv_slots=None,
           layer_range=None, mode="prefill", apply_final_norm=None, **legacy)``

  * ``inputs_embeds``  -- ttnn tensor used directly as ``h`` (skips the internal
    ``ttnn.embedding``).  Needed by Voxtral, which scatters audio embeddings
    into the text embeddings on the host/encoder side.
  * ``x`` (token ids) with ``inputs_embeds=None`` -- today's path, unchanged.
  * ``rope=(cos_tt, sin_tt)`` -- overrides the internal table for every layer.
  * ``kv_slots``  -- list of ``KVSlot``, one per layer ACTUALLY RUN (indexed
    relative to ``layer_range``).
  * ``layer_range=(lo, hi)``  -- run only ``self.layer_weights[lo:hi]``.
  * ``mode="prefill"|"decode"``.
  * ``**legacy`` -- ``position_ids`` / ``position_embeddings`` / etc. from the
    generated PCC harness, handled exactly as before.
"""
from __future__ import annotations

import torch

import ttnn

_HIFI4_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)

_DEFAULT_ROPE_CAPACITY = 1024


def _as_bf16(t):
    """Narrow a weight to bfloat16 ON THE HOST, before it is ever uploaded.

    ``from_torch(t, dtype=ttnn.bfloat16, ...)`` does NOT convert a float32 `t` on the host: it
    uploads the fp32 bytes, converts the LAYOUT on device in fp32, and only then emits a device
    Typecast to bf16.  Every call site here hands over a `.float()` tensor, so the layout
    conversion was moving 4 bytes per element to produce a 2-byte tensor and paying for a whole
    extra device op to do it.  Handing over bf16 makes the requested dtype the dtype that arrives:
    the conversion moves half the bytes and the typecast has nothing left to do.

    The VALUES are the same either way -- the fp32 -> bf16 rounding happens regardless, on the host
    here instead of on the device one op later.
    """
    return t.bfloat16() if hasattr(t, "bfloat16") else t


def _to_device(t, device):
    t = _as_bf16(t)
    try:
        if isinstance(device, ttnn.MeshDevice):
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
    except (AttributeError, TypeError):
        pass
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def _to_device_rm(t, device):
    t = _as_bf16(t)
    try:
        if isinstance(device, ttnn.MeshDevice):
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
    except (AttributeError, TypeError):
        pass
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


# ---------------------------------------------------------------------------
# Shared prefill/decode plumbing.
#
# Duplicated verbatim in llama_attention.py / llama_decoder_layer.py /
# llama_model.py so every stub stays importable on its own (the bring-up
# harness imports them one at a time and `_stubs` is a namespace package with
# no __init__.py).  Keep these signatures identical in all three files.
#
# ttnn layouts relied upon (docstrings + tests/ttnn unit tests):
#   ttnn.fill_cache(cache, x, batch_idx)
#       cache [B, nkv, C, hd] TILE ; x [1, nkv, S, hd] TILE (interleaved ok)
#       -> tests/tt_eager/python_api_testing/unit_testing/misc/
#          test_update_cache.py::TestUpdateCache::test_fill_cache
#   ttnn.update_cache(cache, x, update_index)
#       cache [B, nkv, C, hd] ; x [1, nkv, B(padded to 32), hd]
#       i.e. the BATCH axis is dim -2 of the input.
#       -> same file, ::test_update_cache  (xt = x[B,nkv,1,hd].permute(2,1,0,3))
#   ttnn.experimental.paged_update_cache(cache, x, update_idxs_tensor=idx)
#       cache [B, nkv, C, hd] ; x [1, B, nkv, hd] HEIGHT_SHARDED on B L1 cores
#       -> tests/ttnn/nightly/unit_tests/operations/transformers/
#          test_paged_update_cache.py::run_test_update_cache_decode
#       Opt-in only (KVSlot.paged=True) because it needs an explicit shard spec;
#       the interleaved ttnn.update_cache path is the default.
#   ttnn.transformer.scaled_dot_product_attention_decode(q, k, v,
#           cur_pos_tensor=<[B] int32 ROW_MAJOR device tensor>, scale=...)
#       q [1, B, nh, hd] ; k/v [B, nkv, S, hd] ; out [1, B, padded_nh, hd]
#       -> docstring + tests/ttnn/unit_tests/operations/sdpa/sdpa_test_utils.py
#          ::run_test_sdpa_decode_single_iter (interleaved DRAM q/k/v is fine)
# ---------------------------------------------------------------------------


class KVSlot:
    """Resident per-layer KV cache.

    ``k`` / ``v`` are ttnn tensors ``[B, n_kv_heads, C, head_dim]``
    (TILE_LAYOUT, ``C`` = cache capacity in tokens).  ``cur_pos_tt`` is a
    resident ttnn ROW_MAJOR int32 tensor of shape ``[B]`` holding each stream's
    current write index; ``cur_pos`` is the python int mirror (all streams
    share it in this pipeline).

    ``allocate`` / ``set_pos`` / ``advance`` are HOST-side bookkeeping used by
    the pipeline BETWEEN steps -- they are never called from any ``__call__``.
    """

    def __init__(self, k, v, cur_pos_tt=None, cur_pos=0, device=None, paged=False):
        self.k = k
        self.v = v
        self.cur_pos_tt = cur_pos_tt
        self.cur_pos = int(cur_pos)
        self.device = device
        self.paged = bool(paged)

    @staticmethod
    def allocate(device, batch, n_kv_heads, capacity, head_dim, dtype=ttnn.bfloat16):
        shape = (int(batch), int(n_kv_heads), int(capacity), int(head_dim))
        k = ttnn.zeros(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        v = ttnn.zeros(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        slot = KVSlot(k, v, device=device)
        slot.set_pos(0)
        return slot

    def set_pos(self, pos, device=None):
        self.cur_pos = int(pos)
        dev = device if device is not None else self.device
        batch = int(self.k.shape[0])
        idx = torch.full((batch,), int(pos), dtype=torch.int32)
        self.cur_pos_tt = ttnn.from_torch(idx, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        return self.cur_pos_tt

    def advance(self, step=1):
        return self.set_pos(self.cur_pos + int(step))


def _rank4(t):
    """Left-pad a ttnn tensor's shape with 1s until it is rank 4, so cos/sin
    broadcast against [B, nh, S, hd] (prefill) and [1, B, nh, hd] (decode)."""
    shp = list(t.shape)
    if len(shp) >= 4:
        return t
    return ttnn.reshape(t, tuple([1] * (4 - len(shp)) + shp))


def _apply_rotary_tt(x, cos, sin):
    """x*cos + rotate_half(x)*sin, rotate_half(x) == concat([-x2, x1], -1).

    Mirrors LlamaModel._apply_rotary; pure ttnn, slices only the last dim so
    it is rank-agnostic."""
    shp = list(x.shape)
    last = len(shp) - 1
    half = shp[last] // 2
    starts_lo = [0] * len(shp)
    ends_lo = list(shp)
    ends_lo[last] = half
    starts_hi = [0] * len(shp)
    starts_hi[last] = half
    x1 = ttnn.slice(x, tuple(starts_lo), tuple(ends_lo))
    x2 = ttnn.slice(x, tuple(starts_hi), tuple(shp))
    rotated = ttnn.concat([ttnn.neg(x2), x1], dim=last)
    return ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(rotated, sin))


def _fill_kv_prefill(kv, k, v):
    """Write a full prefill K/V into the resident cache at sequence offset 0.
    k/v are [B, n_kv, S, head_dim]; ttnn.fill_cache wants [1, n_kv, S, hd]."""
    shp = list(k.shape)
    batch = int(shp[0])
    for b in range(batch):
        if batch == 1:
            kb, vb = k, v
        else:
            kb = ttnn.slice(k, (b, 0, 0, 0), (b + 1, shp[1], shp[2], shp[3]))
            vb = ttnn.slice(v, (b, 0, 0, 0), (b + 1, shp[1], shp[2], shp[3]))
        ttnn.fill_cache(kv.k, kb, b)
        ttnn.fill_cache(kv.v, vb, b)


def _height_shard_decode_input(x, device, batch):
    """[1, B, nkv, hd] TILE -> HEIGHT_SHARDED L1 on B cores (one [32, hd] tile
    row per user), which is what paged_update_cache requires."""
    grid = device.compute_with_storage_grid_size()
    shard_grid = ttnn.num_cores_to_corerangeset(int(batch), grid, True)
    padded = x.padded_shape
    shard_spec = ttnn.ShardSpec(shard_grid, [int(padded[-2]), int(padded[-1])], ttnn.ShardOrientation.ROW_MAJOR)
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    return ttnn.to_memory_config(x, mem)


def _write_kv_decode(kv, k, v, device=None):
    """k/v are [1, B, n_kv, head_dim] (the decode projection layout)."""
    if getattr(kv, "paged", False):
        dev = device if device is not None else kv.device
        batch = int(k.shape[1])
        ttnn.experimental.paged_update_cache(
            kv.k,
            _height_shard_decode_input(k, dev, batch),
            update_idxs_tensor=kv.cur_pos_tt,
            share_cache=False,
        )
        ttnn.experimental.paged_update_cache(
            kv.v,
            _height_shard_decode_input(v, dev, batch),
            update_idxs_tensor=kv.cur_pos_tt,
            share_cache=False,
        )
        return
    # ttnn.update_cache wants the BATCH axis at dim -2: [1, n_kv, B, head_dim].
    ttnn.update_cache(kv.k, ttnn.transpose(k, 1, 2), kv.cur_pos)
    ttnn.update_cache(kv.v, ttnn.transpose(v, 1, 2), kv.cur_pos)


def _decode_pos_kwargs(kv, batch):
    if kv.cur_pos_tt is not None:
        return {"cur_pos_tensor": kv.cur_pos_tt}
    return {"cur_pos": [int(kv.cur_pos)] * int(batch)}


def _decode_batch(shape):
    """[B, 1, hidden] or [1, B, hidden] -> B."""
    shp = list(shape)
    if len(shp) < 3:
        return 1
    return int(shp[1]) if int(shp[0]) == 1 else int(shp[0])


def allocate_kv_slot(device, batch, n_kv_heads, capacity, head_dim, dtype=ttnn.bfloat16):
    """Convenience alias for ``KVSlot.allocate`` (build/allocation time only)."""
    return KVSlot.allocate(device, batch, n_kv_heads, capacity, head_dim, dtype=dtype)


# ---------------------------------------------------------------------------


class LlamaModel:
    def __init__(
        self,
        device,
        torch_module,
        layer_range=None,
        skip_embedding=False,
        rope_capacity=_DEFAULT_ROPE_CAPACITY,
        apply_final_norm=True,
    ):
        self.device = device
        cfg = torch_module.layers[0].self_attn.config
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.hidden_size = cfg.hidden_size
        self.scaling = self.head_dim**-0.5
        self.num_layers = len(torch_module.layers)
        self.apply_final_norm = bool(apply_final_norm)

        # Which of the 30 HF layers get uploaded to device DRAM.
        if layer_range is None:
            lo, hi = 0, self.num_layers
        else:
            lo = max(0, min(int(layer_range[0]), self.num_layers))
            hi = max(lo, min(int(layer_range[1]), self.num_layers))
        self.build_layer_range = (lo, hi)
        self.num_resident_layers = hi - lo

        self.skip_embedding = bool(skip_embedding)
        if self.skip_embedding:
            self.embed_weight = None
        else:
            self.embed_weight = _to_device_rm(torch_module.embed_tokens.weight.float(), device)

        # Fallback RoPE table.  Positions are arange(0, capacity), so slicing
        # the first S rows yields exactly the values the old 256-row table did.
        max_seq = int(((int(rope_capacity) + 31) // 32) * 32)
        self.rope_capacity = max_seq
        with torch.no_grad():
            pos_ids = torch.arange(max_seq).unsqueeze(0)
            cos_t, sin_t = torch_module.rotary_emb(torch.zeros(1, max_seq, self.hidden_size), pos_ids)
            self.cos_tt = _to_device(cos_t.unsqueeze(1).float(), device)
            self.sin_tt = _to_device(sin_t.unsqueeze(1).float(), device)

        self.layer_weights = []
        for i in range(lo, hi):
            layer = torch_module.layers[i]
            lw = {
                "in_ln_w": _to_device(layer.input_layernorm.weight.unsqueeze(0).unsqueeze(0).float(), device),
                "in_ln_eps": layer.input_layernorm.variance_epsilon,
                "q_w": _to_device(layer.self_attn.q_proj.weight.T.contiguous().float(), device),
                "k_w": _to_device(layer.self_attn.k_proj.weight.T.contiguous().float(), device),
                "v_w": _to_device(layer.self_attn.v_proj.weight.T.contiguous().float(), device),
                "o_w": _to_device(layer.self_attn.o_proj.weight.T.contiguous().float(), device),
                "post_ln_w": _to_device(
                    layer.post_attention_layernorm.weight.unsqueeze(0).unsqueeze(0).float(), device
                ),
                "post_ln_eps": layer.post_attention_layernorm.variance_epsilon,
                "gate_w": _to_device(layer.mlp.gate_proj.weight.T.contiguous().float(), device),
                "up_w": _to_device(layer.mlp.up_proj.weight.T.contiguous().float(), device),
                "down_w": _to_device(layer.mlp.down_proj.weight.T.contiguous().float(), device),
            }
            self.layer_weights.append(lw)

        self.norm_w = _to_device(torch_module.norm.weight.unsqueeze(0).unsqueeze(0).float(), device)
        self.norm_eps = torch_module.norm.variance_epsilon

    def _apply_rotary(self, x, cos_s, sin_s):
        half = self.head_dim // 2
        B, H, S, D = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x1 = ttnn.slice(x, (0, 0, 0, 0), (B, H, S, half))
        x2 = ttnn.slice(x, (0, 0, 0, half), (B, H, S, D))
        neg_x2 = ttnn.neg(x2)
        rotated = ttnn.concat([neg_x2, x1], dim=3)
        return ttnn.add(ttnn.multiply(x, cos_s), ttnn.multiply(rotated, sin_s))

    def _attn_decode(self, h, lw, cos_s, sin_s, kv, B):
        """h is [1, B, hidden]; returns [1, B, hidden]."""
        q = ttnn.linear(h, lw["q_w"], compute_kernel_config=_HIFI4_CFG)
        k = ttnn.linear(h, lw["k_w"], compute_kernel_config=_HIFI4_CFG)
        v = ttnn.linear(h, lw["v_w"], compute_kernel_config=_HIFI4_CFG)

        q = ttnn.reshape(q, (1, B, self.num_heads, self.head_dim))
        k = ttnn.reshape(k, (1, B, self.num_kv_heads, self.head_dim))
        v = ttnn.reshape(v, (1, B, self.num_kv_heads, self.head_dim))

        if cos_s is not None:
            q = _apply_rotary_tt(q, cos_s, sin_s)
            k = _apply_rotary_tt(k, cos_s, sin_s)

        _write_kv_decode(kv, k, v, self.device)

        attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
            q, kv.k, kv.v, scale=self.scaling, **_decode_pos_kwargs(kv, B), compute_kernel_config=_HIFI4_CFG
        )
        # [1, B, padded_nh, hd] -> [1, B, nh*hd]; nh == 32 here so no padding.
        attn_out = ttnn.reshape(attn_out, (1, B, self.num_heads * self.head_dim))
        return ttnn.linear(attn_out, lw["o_w"], compute_kernel_config=_HIFI4_CFG)

    def __call__(
        self,
        x=None,
        *,
        inputs_embeds=None,
        rope=None,
        kv_slots=None,
        layer_range=None,
        mode="prefill",
        apply_final_norm=None,
        **legacy,
    ):
        decode = mode == "decode"
        apply_norm = self.apply_final_norm if apply_final_norm is None else bool(apply_final_norm)

        if inputs_embeds is None:
            if self.embed_weight is None:
                raise ValueError("LlamaModel was built with skip_embedding=True; pass inputs_embeds=<ttnn tensor>")
            # ---- legacy / token-id path (numerically unchanged) ----
            B = x.shape[0]
            S = x.shape[-1]
            h = ttnn.embedding(x, self.embed_weight)
            h = ttnn.to_layout(h, ttnn.TILE_LAYOUT)
            orig_shape = None
        else:
            h = inputs_embeds
            orig_shape = list(h.shape)
            if decode:
                B = _decode_batch(orig_shape)
                S = 1
                H = int(orig_shape[-1])
                # [B, 1, H] and [1, B, H] share row-major element order.
                if orig_shape != [1, B, H]:
                    h = ttnn.reshape(h, (1, B, H))
                else:
                    orig_shape = None
            else:
                B = int(orig_shape[0])
                S = int(orig_shape[1])
                orig_shape = None

        use_rope = "position_embeddings" not in legacy
        if rope is not None:
            cos_s = _rank4(rope[0])
            sin_s = _rank4(rope[1])
        elif decode:
            raise ValueError(
                "LlamaModel decode mode needs rope=(cos_tt, sin_tt) for the current "
                "position; the internal table is only sliced from row 0."
            )
        elif use_rope:
            cos_s = ttnn.slice(self.cos_tt, (0, 0, 0, 0), (1, 1, S, self.head_dim))
            sin_s = ttnn.slice(self.sin_tt, (0, 0, 0, 0), (1, 1, S, self.head_dim))
        else:
            cos_s = None
            sin_s = None

        if layer_range is None:
            layers = self.layer_weights
        else:
            layers = self.layer_weights[int(layer_range[0]) : int(layer_range[1])]

        for li, lw in enumerate(layers):
            kv = kv_slots[li] if kv_slots is not None else None

            residual = h
            h = ttnn.rms_norm(h, weight=lw["in_ln_w"], epsilon=lw["in_ln_eps"], compute_kernel_config=_HIFI4_CFG)

            if decode:
                if kv is None:
                    raise ValueError("LlamaModel decode mode requires one KVSlot per layer run")
                attn_out = self._attn_decode(h, lw, cos_s, sin_s, kv, B)
            else:
                q = ttnn.linear(h, lw["q_w"], compute_kernel_config=_HIFI4_CFG)
                k = ttnn.linear(h, lw["k_w"], compute_kernel_config=_HIFI4_CFG)
                v = ttnn.linear(h, lw["v_w"], compute_kernel_config=_HIFI4_CFG)

                q = ttnn.reshape(q, (B, S, self.num_heads, self.head_dim))
                q = ttnn.transpose(q, 1, 2)
                k = ttnn.reshape(k, (B, S, self.num_kv_heads, self.head_dim))
                k = ttnn.transpose(k, 1, 2)
                v = ttnn.reshape(v, (B, S, self.num_kv_heads, self.head_dim))
                v = ttnn.transpose(v, 1, 2)

                if cos_s is not None:
                    q = self._apply_rotary(q, cos_s, sin_s)
                    k = self._apply_rotary(k, cos_s, sin_s)

                if kv is not None:
                    _fill_kv_prefill(kv, k, v)

                attn_out = ttnn.transformer.scaled_dot_product_attention(
                    q, k, v, is_causal=True, scale=self.scaling, compute_kernel_config=_HIFI4_CFG
                )
                attn_out = ttnn.transformer.concatenate_heads(attn_out)
                attn_out = ttnn.linear(attn_out, lw["o_w"], compute_kernel_config=_HIFI4_CFG)

            h = ttnn.add(residual, attn_out)

            residual = h
            h = ttnn.rms_norm(h, weight=lw["post_ln_w"], epsilon=lw["post_ln_eps"], compute_kernel_config=_HIFI4_CFG)

            gate = ttnn.linear(h, lw["gate_w"], compute_kernel_config=_HIFI4_CFG)
            gate = ttnn.silu(gate)
            up = ttnn.linear(h, lw["up_w"], compute_kernel_config=_HIFI4_CFG)
            h = ttnn.multiply(gate, up)
            h = ttnn.linear(h, lw["down_w"], compute_kernel_config=_HIFI4_CFG)

            h = ttnn.add(residual, h)

        if apply_norm:
            h = ttnn.rms_norm(h, weight=self.norm_w, epsilon=self.norm_eps, compute_kernel_config=_HIFI4_CFG)
        if orig_shape is not None:
            h = ttnn.reshape(h, tuple(orig_shape))
        return h


def build(
    device,
    torch_module,
    layer_range=None,
    skip_embedding=False,
    rope_capacity=_DEFAULT_ROPE_CAPACITY,
    apply_final_norm=True,
):
    return LlamaModel(
        device,
        torch_module,
        layer_range=layer_range,
        skip_embedding=skip_embedding,
        rope_capacity=rope_capacity,
        apply_final_norm=apply_final_norm,
    )

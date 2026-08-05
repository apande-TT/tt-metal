# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN stub for LlamaDecoderLayer (language_model.layers[i]).

RMSNorm + GQA Attention + Add + RMSNorm + SwiGLU MLP + Add.

Call contract
-------------
``__call__(x, *, rope=None, kv=None, mode="prefill", **legacy)``

  * ``rope=(cos_tt, sin_tt)``  -- ttnn tensors, rotate_half convention.
  * ``kv=<KVSlot>``           -- resident cache, written in place.
  * ``mode="prefill"|"decode"``
  * ``**legacy``              -- everything the generated PCC harness passes
    (``position_ids`` / ``position_embeddings`` / ``attention_mask`` /
    ``past_key_values`` / ``use_cache``) is accepted and IGNORED, exactly as
    before.  With ``rope=None`` and ``kv=None`` this file is numerically
    identical to the graduated version.

The norm / MLP halves are untouched.
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


def _to_device(t, device):
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


# ---------------------------------------------------------------------------


class TtLlamaDecoderLayer:
    def __init__(self, device, torch_module):
        self.device = device
        cfg = torch_module.self_attn.config
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.scaling = self.head_dim**-0.5

        attn = torch_module.self_attn
        self.q_weight = _to_device(attn.q_proj.weight.T.contiguous().float(), device)
        self.k_weight = _to_device(attn.k_proj.weight.T.contiguous().float(), device)
        self.v_weight = _to_device(attn.v_proj.weight.T.contiguous().float(), device)
        self.o_weight = _to_device(attn.o_proj.weight.T.contiguous().float(), device)

        self.in_ln_w = _to_device(torch_module.input_layernorm.weight.unsqueeze(0).unsqueeze(0).float(), device)
        self.in_ln_eps = torch_module.input_layernorm.variance_epsilon

        self.post_ln_w = _to_device(
            torch_module.post_attention_layernorm.weight.unsqueeze(0).unsqueeze(0).float(), device
        )
        self.post_ln_eps = torch_module.post_attention_layernorm.variance_epsilon

        mlp = torch_module.mlp
        self.gate_weight = _to_device(mlp.gate_proj.weight.T.contiguous().float(), device)
        self.up_weight = _to_device(mlp.up_proj.weight.T.contiguous().float(), device)
        self.down_weight = _to_device(mlp.down_proj.weight.T.contiguous().float(), device)

    def __call__(self, x, *, rope=None, kv=None, mode="prefill", **legacy):
        decode = mode == "decode"

        orig_shape = list(x.shape)
        if decode:
            if kv is None:
                raise ValueError("TtLlamaDecoderLayer decode mode requires kv=<KVSlot>")
            B = _decode_batch(orig_shape)
            H = int(orig_shape[-1])
            # [B, 1, H] and [1, B, H] have the same row-major element order.
            if orig_shape != [1, B, H]:
                x = ttnn.reshape(x, (1, B, H))
            S = 1
        else:
            B = x.shape[0]
            S = x.shape[1] if len(x.shape) == 3 else x.shape[-2]

        residual = x
        x = ttnn.rms_norm(x, weight=self.in_ln_w, epsilon=self.in_ln_eps, compute_kernel_config=_HIFI4_CFG)

        q = ttnn.linear(x, self.q_weight, compute_kernel_config=_HIFI4_CFG)
        k = ttnn.linear(x, self.k_weight, compute_kernel_config=_HIFI4_CFG)
        v = ttnn.linear(x, self.v_weight, compute_kernel_config=_HIFI4_CFG)

        if decode:
            q = ttnn.reshape(q, (1, B, self.num_heads, self.head_dim))
            k = ttnn.reshape(k, (1, B, self.num_kv_heads, self.head_dim))
            v = ttnn.reshape(v, (1, B, self.num_kv_heads, self.head_dim))

            if rope is not None:
                cos_s = _rank4(rope[0])
                sin_s = _rank4(rope[1])
                q = _apply_rotary_tt(q, cos_s, sin_s)
                k = _apply_rotary_tt(k, cos_s, sin_s)

            _write_kv_decode(kv, k, v, self.device)

            attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
                q, kv.k, kv.v, scale=self.scaling, **_decode_pos_kwargs(kv, B), compute_kernel_config=_HIFI4_CFG
            )
            # [1, B, padded_nh, hd] -> [1, B, nh*hd]; nh == 32 here so no padding.
            attn_out = ttnn.reshape(attn_out, (1, B, self.num_heads * self.head_dim))
            attn_out = ttnn.linear(attn_out, self.o_weight, compute_kernel_config=_HIFI4_CFG)
        else:
            q = ttnn.reshape(q, (B, S, self.num_heads, self.head_dim))
            q = ttnn.transpose(q, 1, 2)
            k = ttnn.reshape(k, (B, S, self.num_kv_heads, self.head_dim))
            k = ttnn.transpose(k, 1, 2)
            v = ttnn.reshape(v, (B, S, self.num_kv_heads, self.head_dim))
            v = ttnn.transpose(v, 1, 2)

            if rope is not None:
                cos_s = _rank4(rope[0])
                sin_s = _rank4(rope[1])
                q = _apply_rotary_tt(q, cos_s, sin_s)
                k = _apply_rotary_tt(k, cos_s, sin_s)

            if kv is not None:
                _fill_kv_prefill(kv, k, v)

            attn_out = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=self.scaling, compute_kernel_config=_HIFI4_CFG
            )
            attn_out = ttnn.transformer.concatenate_heads(attn_out)
            attn_out = ttnn.linear(attn_out, self.o_weight, compute_kernel_config=_HIFI4_CFG)

        x = ttnn.add(residual, attn_out)

        residual = x
        x = ttnn.rms_norm(x, weight=self.post_ln_w, epsilon=self.post_ln_eps, compute_kernel_config=_HIFI4_CFG)

        gate = ttnn.linear(x, self.gate_weight, compute_kernel_config=_HIFI4_CFG)
        gate = ttnn.silu(gate)
        up = ttnn.linear(x, self.up_weight, compute_kernel_config=_HIFI4_CFG)
        x = ttnn.multiply(gate, up)
        x = ttnn.linear(x, self.down_weight, compute_kernel_config=_HIFI4_CFG)

        x = ttnn.add(residual, x)
        if decode and orig_shape != [1, B, H]:
            x = ttnn.reshape(x, tuple(orig_shape))
        return x


def build(device, torch_module=None):
    return TtLlamaDecoderLayer(device, torch_module)

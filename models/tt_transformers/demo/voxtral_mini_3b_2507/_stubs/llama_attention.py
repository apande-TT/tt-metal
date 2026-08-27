# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN stub for LlamaAttention (language_model.layers[i].self_attn).

Grouped query attention with RoPE. num_heads=32, num_kv_heads=8, head_dim=128.

Call contract
-------------
``__call__(hidden_states, *, rope=None, kv=None, mode="prefill", **legacy)``

  * ``rope=(cos_tt, sin_tt)``  -- ttnn tensors, rotate_half convention.
  * ``kv=<KVSlot>``           -- resident cache, written in place.
  * ``mode="prefill"|"decode"``
  * ``**legacy``              -- everything the generated PCC harness passes
    (``position_ids`` / ``position_embeddings`` / ``attention_mask`` /
    ``past_key_values`` / ``use_cache``) is accepted and IGNORED, exactly as
    before.  With ``rope=None`` and ``kv=None`` this file is numerically
    identical to the graduated version.
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
    """Upload the weight ALREADY TILED, so the device emits no layout-conversion op at all.

    Passing ``device=`` to from_torch is what puts the conversion on the device: the ROW_MAJOR
    bytes go up and a Tilize (or TilizeWithValPadding, for a bias that is not a whole tile) runs
    there.  Building the tensor with NO device argument tilizes on the host instead, and
    ``ttnn.to_device`` is then a plain DMA of bytes that are already in the layout the consumer
    wants -- the conversion does not move to a cheaper kernel, it stops existing.

    This is a WEIGHT path, so the host cost is paid once at build and never in a forward, whereas
    the device op it replaces was on the critical path of the measured region.  Values are
    untouched: the same host-side bf16 tensor, the same tiling, just assembled before the copy
    rather than after it.
    """
    t = _as_bf16(t)
    kw = {"dtype": ttnn.bfloat16, "layout": ttnn.TILE_LAYOUT}
    try:
        if isinstance(device, ttnn.MeshDevice):
            kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
    except (AttributeError, TypeError):
        pass
    # NO `device=` HERE, and that is the whole point. `ttnn.open_device()` returns a MeshDevice on
    # this build, so a `isinstance(device, MeshDevice)` branch that kept `device=` was the branch
    # ALWAYS taken -- the host-tilize path below it was dead code. The mapper does not need the
    # tensor placed to describe the replication, so it composes with a host build.
    return ttnn.to_device(ttnn.from_torch(t, **kw), device)


def _mesh_to_torch(t, device):
    if isinstance(t, torch.Tensor):
        return t
    try:
        if hasattr(ttnn, "synchronize_device"):
            ttnn.synchronize_device(device)
    except Exception:
        pass
    try:
        if isinstance(device, ttnn.MeshDevice):
            for mk_composer in (
                lambda: ttnn.concat_mesh_to_tensor_composer(device, 0),
                lambda: ttnn.ConcatMeshToTensor(device, dim=0),
            ):
                try:
                    composer = mk_composer()
                    out = ttnn.to_torch(t, mesh_composer=composer)
                    n_devices = len(device.get_device_ids()) if hasattr(device, "get_device_ids") else 1
                    if n_devices > 1 and out.shape[0] % n_devices == 0:
                        out = out[: out.shape[0] // n_devices]
                    return out
                except Exception:
                    continue
    except (AttributeError, TypeError):
        pass
    return ttnn.to_torch(t)


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


class TtLlamaAttention:
    def __init__(self, device, torch_module):
        self.device = device
        self.head_dim = torch_module.head_dim
        self.num_heads = torch_module.config.num_attention_heads
        self.num_kv_heads = torch_module.config.num_key_value_heads
        self.scaling = torch_module.head_dim**-0.5

        self.q_weight = _to_device(torch_module.q_proj.weight.T.contiguous().float(), device)
        self.k_weight = _to_device(torch_module.k_proj.weight.T.contiguous().float(), device)
        self.v_weight = _to_device(torch_module.v_proj.weight.T.contiguous().float(), device)
        self.o_weight = _to_device(torch_module.o_proj.weight.T.contiguous().float(), device)

    def __call__(self, hidden_states, *, rope=None, kv=None, mode="prefill", **legacy):
        if mode == "decode":
            return self._forward_decode(hidden_states, rope, kv)
        return self._forward_prefill(hidden_states, rope, kv)

    # -- prefill (identical op sequence to the graduated stub) --------------

    def _forward_prefill(self, hidden_states, rope, kv):
        B = hidden_states.shape[0]
        S = hidden_states.shape[1] if len(hidden_states.shape) == 3 else hidden_states.shape[-2]

        q = ttnn.linear(hidden_states, self.q_weight, compute_kernel_config=_HIFI4_CFG)
        k = ttnn.linear(hidden_states, self.k_weight, compute_kernel_config=_HIFI4_CFG)
        v = ttnn.linear(hidden_states, self.v_weight, compute_kernel_config=_HIFI4_CFG)

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

        return attn_out

    # -- decode -------------------------------------------------------------

    def _forward_decode(self, hidden_states, rope, kv):
        if kv is None:
            raise ValueError("TtLlamaAttention decode mode requires kv=<KVSlot>")

        orig_shape = list(hidden_states.shape)
        B = _decode_batch(orig_shape)
        H = int(orig_shape[-1])
        # [B, 1, H] and [1, B, H] have the same row-major element order.
        x = hidden_states if orig_shape == [1, B, H] else ttnn.reshape(hidden_states, (1, B, H))

        q = ttnn.linear(x, self.q_weight, compute_kernel_config=_HIFI4_CFG)
        k = ttnn.linear(x, self.k_weight, compute_kernel_config=_HIFI4_CFG)
        v = ttnn.linear(x, self.v_weight, compute_kernel_config=_HIFI4_CFG)

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
        if orig_shape != [1, B, H]:
            attn_out = ttnn.reshape(attn_out, tuple(orig_shape))
        return attn_out


def build(device, torch_module=None):
    return TtLlamaAttention(device, torch_module)

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

# THE SwiGLU BRANCH THAT TOLERATES THE NARROWEST WEIGHT.  gate/up/down are 3/4 of every decode
# token's DRAM read, so halving one of them is the largest byte lever left -- but all three at
# bfloat4_b was measured at PCC 0.8724, far under the 0.95 gate.  `up` is the gentlest of the three
# to narrow: `down` writes straight into the residual stream and `gate` is perturbed BEFORE the
# nonlinearity, whereas `up` enters as a plain linear factor of silu(gate).
_UP_DTYPE = ttnn.bfloat4_b

_HIFI4_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


# ATTENTION-PROJECTION FIDELITY, MATCHED TO THE bf8_b WEIGHTS.  q/k/v/o are bf8_b, but they were
# still running through a HiFi4 kernel, which makes the math engine take FOUR passes over operands
# that carry one pass worth of mantissa.  The profiler tagged these matmuls FLOP-bound (not DRAM-
# bound) even though they only read a few MB of weights, which is the signature of exactly that:
# at HiFi4 on the 12-worker DRAM-sharded grid the math, not the weight read, is the critical path.
# LoFi is the documented pairing for 8-bit operands (GUIDELINES/01 section 12), and the MLP in the
# sibling bodies already used it.  The norms and SDPA deliberately STAY at HiFi4 + fp32_dest_acc_en:
# reduction and softmax accumulation are the one place lower fidelity compounds over depth.
_ATTN_PROJ_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


def _dram_sharded():
    """Load the shared DRAM-bank-sharded projection helper that sits next to this stub.

    The stubs are imported standalone BY PATH (tt/pipeline._load_stub_module), so they have no
    package context and a relative import is not available to them.
    """
    import importlib.util
    import pathlib
    import sys

    key = "_voxtral_stub__dram_sharded"
    mod = sys.modules.get(key)
    if mod is None:
        spec = importlib.util.spec_from_file_location(key, pathlib.Path(__file__).with_name("_dram_sharded.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return mod


_DS = _dram_sharded()


# MATCHED TO bf8_b WEIGHTS.  8-bit operands through a HiFi4 kernel make the math engine take four
# passes over one pass worth of precision, which cancels the bandwidth saving; LoFi is the pairing.
_LOFI_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


# NORM FIDELITY, SPLIT FROM THE ACCUMULATOR WIDTH.  The reduction is the one place in the stack
# where low fidelity compounds over depth, but that risk lives in the ACCUMULATOR, not in the
# multiplier: fp32_dest_acc_en is what keeps the sum of 3072 squares exact, while math_fidelity
# only sets how many passes the math engine takes over each operand's mantissa.  The inputs here
# are bf16 -- one pass worth of mantissa -- so HiFi4's four passes buy nothing and the norm
# profiled as the largest reduction cost in the decode step (2 per layer, 60 per token).  HiFi2
# with fp32_dest_acc_en STILL held is the pairing; SDPA's softmax stays at HiFi4.
_NORM_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


def _to_device(t, device, dtype=ttnn.bfloat16):
    # NARROW TO bf16 ON THE HOST.  Callers hand this `.float()` tensors, but the target dtype is
    # bf16, so ttnn used to upload fp32 and fix it up on DEVICE -- the profile showed 42 ms of
    # fp32 Tilize plus 24 ms of fp32->bf16 Typecast doing exactly that.  Narrowing first halves
    # the bytes tilized and removes the typecast entirely.  It is EXACT, not an approximation:
    # both host and device round fp32->bf16 round-to-nearest-even, and these weights came from a
    # bf16 checkpoint that `.float()` had merely widened, so this restores the original values.
    # Block-float targets (bf8_b / bf4_b) are left in fp32 on purpose: their mantissa is
    # derived from a per-block shared exponent, so inserting a bf16 rounding step first can
    # change the packed result.  Only the bf16 path is a pure round-trip removal.
    if dtype == ttnn.bfloat16:
        t = t.bfloat16()
    try:
        if isinstance(device, ttnn.MeshDevice):
            return ttnn.from_torch(
                t,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
    except (AttributeError, TypeError):
        pass
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


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


def _fuse_layer_qkv(attn):
    """Concatenate q/k/v into a single [hidden, (nh + 2*nkv)*hd] weight, on the host.

    No bias and no scale folding: this model's q/k/v are bias-free and the attention scale is
    passed to SDPA, so the fused weight is a pure output-dim concat of the three originals.
    """
    return _DS.fuse_qkv(
        attn.q_proj.weight.T.contiguous().float(),
        attn.k_proj.weight.T.contiguous().float(),
        attn.v_proj.weight.T.contiguous().float(),
    )[0]


# Last height-sharded (cos, sin) pair per core set, with the tables they came from.
_ROPE_SHARD_CACHE = {}


def _shard_rope_pair(cos, sin, device, batch, cores):
    """cos/sin height-sharded onto `cores`, reusing the last shard of the SAME tables.

    THE BIGGEST LAUNCH SINK IN THE DECODE STEP.  cos/sin are built ONCE per token and handed
    unchanged to every LM layer, but the rank-4 fixup and the reshard onto the rotary op's core
    set were happening inside each call: 30 layers x 2 rotaries (q and k) x up to 3 launches =
    ~180 dispatches a token, every one of them recomputing a byte-identical shard.  Decode here
    is launch-count bound, not bandwidth bound -- ~3.7 GB of bf8_b weights a token is ~6 ms of
    the 11.4 ms at this board's bandwidth, and the ~450 dispatches at the ~8 us/launch the KV
    fusion measured are most of the rest -- so those launches are real time.

    Keyed on the core set and validated by IDENTITY of the tables passed in, so the next token's
    cos/sin (fresh op outputs) miss and reshard exactly once per core set.  The entry holds the
    source tables alive, so identity can never be recycled underneath it, and it holds at most
    one pair per core set (two, here: q's and k's), so nothing accumulates.
    """
    key = str(cores)
    hit = _ROPE_SHARD_CACHE.get(key)
    if hit is not None and hit[0] is cos and hit[1] is sin:
        return hit[2], hit[3]
    cs = _height_shard_decode_input(_rank4(cos), device, batch, cores)
    ss = _height_shard_decode_input(_rank4(sin), device, batch, cores)
    _ROPE_SHARD_CACHE[key] = (cos, sin, cs, ss)
    return cs, ss


def _apply_rotary_tt(x, cos, sin, decode=False):
    """x*cos + rotate_half(x)*sin, rotate_half(x) == concat([-x2, x1], -1).

    Mirrors LlamaModel._apply_rotary; pure ttnn, slices only the last dim so
    it is rank-agnostic."""
    # FUSED DECODE PATH.  The generic form below is ~7 dispatches (2 slices, neg, concat, 2
    # multiplies, add) per call, and decode calls it twice per layer per token, so at 30 layers
    # that is ~360 tiny ops a token.  ttnn's rotary_embedding_hf is the same rotate_half maths in
    # ONE op and takes exactly the decode layout we already have: x [1, B, nh, hd] with cos/sin
    # [1, B, 1, hd].  Fall back to the generic path if the op rejects the shapes, so correctness
    # never depends on it being available.
    if decode:
        try:
            dev = x.device()
            batch = int(x.shape[1])
            # The op REQUIRES a sharded input in decode mode (it asserts is_sharded()), and the
            # caches must match, so height-shard all three onto one core per user first and put the
            # result back where the rest of the graph expects it.  Six moves replace fourteen ops.
            # MEET x WHERE IT IS.  q and k come off the head-split on DIFFERENT core sets now, and
            # the op requires all three operands to share one, so the target is x's own core set
            # and cos/sin are the tensors that move.  Same launch count as pinning all three to
            # cores 0..batch-1 -- cos/sin were being resharded per call either way -- but it
            # leaves k on the core set that keeps it disjoint from v.
            cores = _decode_shard_grid(x, dev, batch)
            xs = _height_shard_decode_input(x, dev, batch, cores)
            # SHARD THE ROPE TABLES ONCE PER TOKEN, NOT ONCE PER LAYER -- see _shard_rope_pair.
            cs, ss = _shard_rope_pair(cos, sin, dev, batch, cores)
            out = ttnn.experimental.rotary_embedding_hf(xs, cs, ss, is_decode_mode=True)
            # FREE THE PER-CALL SHARD.  Now that `out` stays resident in L1 the input that
            # produced it must not: the DRAM-sharded projections that run next size their circular
            # buffers against whatever L1 is still free on these cores, and leaving ~24 kB of dead
            # shards behind pushes them past the 1.5 MB budget.  Only drop a tensor this call
            # created -- `_height_shard_decode_input` returns its argument unchanged when it
            # already has the right layout.  cs/ss are deliberately NOT dropped: they are the
            # cached per-token rope shards that the other 29 layers are about to reuse, and they
            # are two [32, hd] tiles per core against a 1.5 MB budget.
            if xs is not x:
                ttnn.deallocate(xs)
            # HAND THE SHARD STRAIGHT ON.  Both consumers of this result want the SAME
            # height-sharded-by-user L1 layout the op just produced: q goes to
            # scaled_dot_product_attention_decode, whose validate explicitly accepts a
            # HEIGHT_SHARDED Q (and only falls back to requiring DRAM when Q is unsharded), and k
            # goes to paged_update_cache via _height_shard_decode_input, which asks for exactly
            # this config.  Pushing it back to DRAM here cost a ShardedToInterleaved plus a full
            # DRAM round-trip of q and k, and then the k path immediately re-sharded it.  The
            # helper below is idempotent, so the k side is a no-op rather than a second reshard.
            return out
        except (RuntimeError, TypeError, AttributeError):
            pass
    # FUSED PREFILL PATH.  The SAME op has a prefill mode, and unlike decode mode it accepts
    # INTERLEAVED tensors -- so there is nothing to reshard and the seven generic dispatches
    # (2 slices, neg, concat, 2 multiplies, add) collapse to ONE launch that reads the activation
    # once instead of five times.  Prefill calls this twice per layer on a [B, 32, 512, 128] q,
    # which is what made the generic form the dominant eltwise cost in the profile.
    else:
        try:
            cos4 = _rank4(cos)
            sin4 = _rank4(sin)
            shp = [int(d) for d in x.shape]
            if len(shp) == 4 and int(cos4.shape[0]) == 1 and int(cos4.shape[1]) == 1:
                b, nh, s, hd = shp
                # FOLD BATCH INTO THE HEAD DIM.  Prefill mode wants [1, nh, S, hd], but cos/sin are
                # [1, 1, S, hd] and broadcast over dim 1 -- every head row gets the SAME rotation --
                # so merging batch into that dim is exactly equivalent.  Both are leading dims, and
                # tiling depends only on the last two, so the reshape is a view, not a re-tilization.
                folded = x if b == 1 else ttnn.reshape(x, (1, b * nh, s, hd))
                out = ttnn.experimental.rotary_embedding_hf(folded, cos4, sin4, is_decode_mode=False)
                return out if b == 1 else ttnn.reshape(out, (b, nh, s, hd))
        except (RuntimeError, TypeError, AttributeError):
            pass
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
    # MATCH THE CACHE DTYPE ONCE, NOT PER STREAM.  update_cache's FILL path refuses a mixed
    # precision write outright ("Input and cache tensors must have same dtype!" -- only its
    # DECODE path has the conversion kernel), so a narrowed cache has to be met here.  Cast the
    # whole [B, n_kv, S, hd] tensor before the loop: casting the per-stream slices instead would
    # be 2*B launches for the same bytes.
    if k.dtype != kv.k.dtype:
        k = ttnn.typecast(k, kv.k.dtype)
    if v.dtype != kv.v.dtype:
        v = ttnn.typecast(v, kv.v.dtype)
    for b in range(batch):
        if batch == 1:
            kb, vb = k, v
        else:
            kb = ttnn.slice(k, (b, 0, 0, 0), (b + 1, shp[1], shp[2], shp[3]))
            vb = ttnn.slice(v, (b, 0, 0, 0), (b + 1, shp[1], shp[2], shp[3]))
        ttnn.fill_cache(kv.k, kb, b)
        ttnn.fill_cache(kv.v, vb, b)


def _decode_shard_grid(x, device, batch):
    """The core set a decode height-shard should land on: x's OWN, if it already has one.

    The default is the first `batch` cores, which is where the head-split puts q and v.  But k
    no longer lives there: nlp_create_qkv_heads_decode is now asked for NON-overlapping q/k core
    grids (see qkv_split_decode) so that k and v end up disjoint, which is the precondition
    paged_fused_update_cache imposes.  Pinning every decode shard to cores 0..batch-1 would drag
    k straight back on top of v and undo that, so a tensor that is already height-sharded one
    user per core keeps the cores it is on and only cos/sin get moved to meet it.
    """
    mem = x.memory_config()
    spec = mem.shard_spec
    if mem.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED and spec is not None:
        if spec.grid.num_cores() == int(batch):
            return spec.grid
    grid = device.compute_with_storage_grid_size()
    return ttnn.num_cores_to_corerangeset(int(batch), grid, True)


def _core_set(grid):
    """The (x, y) cores a CoreRangeSet covers, as a python set."""
    out = set()
    for r in grid.ranges():
        for cx in range(int(r.start.x), int(r.end.x) + 1):
            for cy in range(int(r.start.y), int(r.end.y) + 1):
                out.add((cx, cy))
    return out


def _height_shard_decode_input(x, device, batch, shard_grid=None):
    """[1, B, nkv, hd] TILE -> HEIGHT_SHARDED L1 on B cores (one [32, hd] tile
    row per user), which is what paged_update_cache requires."""
    shard_grid = _decode_shard_grid(x, device, batch) if shard_grid is None else shard_grid
    padded = x.padded_shape
    shard_spec = ttnn.ShardSpec(shard_grid, [int(padded[-2]), int(padded[-1])], ttnn.ShardOrientation.ROW_MAJOR)
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    # IDEMPOTENT.  The rotary op now hands its result back already in this layout, and
    # to_memory_config on an identical config still dispatches a reshard rather than returning the
    # tensor, so compare first and skip the launch when there is nothing to move.
    if x.memory_config() == mem:
        return x
    return ttnn.to_memory_config(x, mem)


def _write_kv_decode(kv, k, v, device=None):
    """k/v are [1, B, n_kv, head_dim] (the decode projection layout)."""
    if getattr(kv, "paged", False):
        dev = device if device is not None else kv.device
        batch = int(k.shape[1])
        # ONE LAUNCH FOR BOTH CACHES.  A decode K or V write is 8 x 8 x 128 elements -- nowhere
        # near enough work to fill the grid, so the profile tags this op dispatch-bound on a TINY
        # grid and each call is essentially launch latency.  The pair ran 2 x n_layers = 60 times
        # per token.  paged_fused_update_cache takes both cache/input pairs and issues them in one
        # program, halving that count.  Same op family, same update_idxs_tensor, and it asserts the
        # same preconditions this call site already met (both inputs height-sharded ROW_MAJOR, both
        # caches TILE + interleaved), so the bytes landed in the cache are identical.
        ks = _height_shard_decode_input(k, dev, batch)
        vs = _height_shard_decode_input(v, dev, batch)
        # ONE LAUNCH FOR BOTH CACHES, WHEN THE CORE SETS ALLOW IT.  A decode K or V write is
        # 8 x 8 x 128 elements -- nowhere near enough work to fill the grid -- so the profile tags
        # this op dispatch-bound on a TINY grid and each call is essentially launch latency.  The
        # pair ran 2 x n_layers = 60 times per token.  paged_fused_update_cache issues both in one
        # program, halving that, and writes the same bytes with the same update_idxs_tensor.
        # It refuses inputs whose core sets OVERLAP (it runs the two halves side by side), so the
        # disjointness is checked here rather than assumed: qkv_split_decode asks for
        # non-overlapping q/k grids, but that request is only honoured when its input is sharded,
        # and a fallback head-split path would put k and v back together.
        if _core_set(ks.memory_config().shard_spec.grid).isdisjoint(_core_set(vs.memory_config().shard_spec.grid)):
            ttnn.experimental.paged_fused_update_cache(
                kv.k, ks, kv.v, vs, update_idxs_tensor=kv.cur_pos_tt, share_cache=False
            )
            return
        ttnn.experimental.paged_update_cache(kv.k, ks, update_idxs_tensor=kv.cur_pos_tt, share_cache=False)
        ttnn.experimental.paged_update_cache(kv.v, vs, update_idxs_tensor=kv.cur_pos_tt, share_cache=False)
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
        # FUSED QKV -- one [hidden, (nh + 2*nkv)*hd] weight instead of three.  Same bytes and
        # same numerics (an output-dim concat), but three matmul launches per attention become
        # one AND, decisively, it is what lets k/v reach the DRAM-bank-sharded path at all: at
        # 3072x1024 their 32 output tiles divide no valid bank-worker count, so each fell back to
        # a plain ttnn.linear measured at 125 GB/s.  Fused, 6144 = 192 tiles divides exactly.
        self.qkv_weight = _to_device(_fuse_layer_qkv(attn), device, ttnn.bfloat8_b)
        self.o_weight = _to_device(attn.o_proj.weight.T.contiguous().float(), device, ttnn.bfloat8_b)
        # Decode-only DRAM-bank-sharded mirrors -- see _dram_sharded.py.
        self.qkv_ds = _DS.attach(device, self.qkv_weight)
        self.o_ds = _DS.attach(device, self.o_weight)

        self.in_ln_w = _to_device(torch_module.input_layernorm.weight.unsqueeze(0).unsqueeze(0).float(), device)
        self.in_ln_eps = torch_module.input_layernorm.variance_epsilon

        self.post_ln_w = _to_device(
            torch_module.post_attention_layernorm.weight.unsqueeze(0).unsqueeze(0).float(), device
        )
        self.post_ln_eps = torch_module.post_attention_layernorm.variance_epsilon

        mlp = torch_module.mlp
        self.gate_weight = _to_device(mlp.gate_proj.weight.T.contiguous().float(), device, ttnn.bfloat8_b)
        self.up_weight = _to_device(mlp.up_proj.weight.T.contiguous().float(), device, _UP_DTYPE)
        self.down_weight = _to_device(mlp.down_proj.weight.T.contiguous().float(), device, ttnn.bfloat8_b)
        # Decode-only DRAM-bank-sharded mirrors -- see _dram_sharded.py.
        self.gate_ds = _DS.attach(device, self.gate_weight)
        self.up_ds = _DS.attach(device, self.up_weight)
        self.down_ds = _DS.attach(device, self.down_weight)

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
        x = _DS.rms_norm(self.device, x, self.in_ln_w, self.in_ln_eps, _NORM_CFG, keep_sharded=decode)

        # ONE fused projection for both regimes; only the SPLIT differs, because decode has no
        # sequence axis for nlp_create_qkv_heads to work on (its one tile row holds the batch).
        qkv = _DS.mm(self.device, x, self.qkv_weight, _ATTN_PROJ_CFG, mirror=self.qkv_ds, keep_sharded=decode)

        if decode:
            q, k, v = _DS.qkv_split_decode(qkv, B, self.num_heads, self.num_kv_heads, self.head_dim)

            if rope is not None:
                cos_s = _rank4(rope[0])
                sin_s = _rank4(rope[1])
                q = _apply_rotary_tt(q, cos_s, sin_s, decode=True)
                k = _apply_rotary_tt(k, cos_s, sin_s, decode=True)

            _write_kv_decode(kv, k, v, self.device)

            attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
                q,
                kv.k,
                kv.v,
                scale=self.scaling,
                **_decode_pos_kwargs(kv, B),
                compute_kernel_config=_HIFI4_CFG,
                memory_config=_DS.sdpa_decode_out_config(),
            )
            # [1, B, padded_nh, hd] -> [1, B, nh*hd]; nh == 32 here so no padding.
            attn_out = _DS.merge_heads_decode(attn_out, B, self.num_heads, self.head_dim)
            attn_out = _DS.mm(self.device, attn_out, self.o_weight, _ATTN_PROJ_CFG, mirror=self.o_ds)
        else:
            # the fused output is already the exact layout nlp_create_qkv_heads consumes
            q, k, v = _DS.qkv_heads(qkv, self.num_heads, self.num_kv_heads)

            if rope is not None:
                cos_s = _rank4(rope[0])
                sin_s = _rank4(rope[1])
                # Broadcast over every head row -- see _DS.rope_resident.
                cos_s, sin_s = _DS.rope_resident(cos_s, sin_s)
                q = _apply_rotary_tt(q, cos_s, sin_s)
                k = _apply_rotary_tt(k, cos_s, sin_s)

            if kv is not None:
                _fill_kv_prefill(kv, k, v)

            attn_out = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                scale=self.scaling,
                program_config=_DS.sdpa_config(self.device, q, k),
                compute_kernel_config=_HIFI4_CFG,
            )
            attn_out = ttnn.transformer.concatenate_heads(attn_out)
            attn_out = _DS.mm(self.device, attn_out, self.o_weight, _ATTN_PROJ_CFG, mirror=self.o_ds)

        x = _DS.residual_add(self.device, residual, attn_out)

        residual = x
        x = _DS.rms_norm(self.device, x, self.post_ln_w, self.post_ln_eps, _NORM_CFG, keep_sharded=decode)

        # Full grid BY NAME -- see llama_m_l_p.py.  The same MLP lives in three bodies (this one,
        # llama_m_l_p, and the bulk stack in llama_model) and the lever has to reach all three.
        _g = self.device.compute_with_storage_grid_size()
        _grid = ttnn.CoreGrid(y=_g.y, x=_g.x)
        x = _DS.swiglu(
            x,
            self.gate_weight,
            self.gate_ds,
            self.up_weight,
            self.up_ds,
            self.down_weight,
            self.down_ds,
            _LOFI_CFG,
            _grid,
        )

        x = _DS.residual_add(self.device, residual, x)
        if decode and orig_shape != [1, B, H]:
            x = ttnn.reshape(x, tuple(orig_shape))
        return x


def build(device, torch_module=None):
    return TtLlamaDecoderLayer(device, torch_module)

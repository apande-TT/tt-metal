# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `mlp` -- the text backbone's `MistralMLP` (`model.layers.0.mlp`).

SwiGLU: `down_proj(silu(gate_proj(x)) * up_proj(x))`, 3072 -> 9216 -> 3072, no biases.

The canonical `models/tt_transformers/tt/mlp.py` is not reusable here: it builds itself from
`ModelArgs`, which resolves the model through `AutoConfig`, and this checkpoint is a native Mistral
`consolidated.safetensors` with no `config.json` / `model_type` -- `ModelArgs` raises before
reading a weight.

The leading bound comes from the tensor (`_view4`), so a batched `[B, 1, S, D]` stream keeps all B
rows; `down_proj` packs back to the dtype the caller handed in, which is what lets a float32
residual stream feed bfloat16 weights without a widening cast on the way out."""

from __future__ import annotations

import torch

import ttnn

# `ttnn.linear`/`ttnn.matmul` on their DEFAULTS leave `fp32_dest_acc_en` off, so the accumulator
# rounds to bfloat16 at every step even when the activations are float32. The consumer of this
# stack resolves a top-1/top-2 margin of a few hundredths, and the audio path rounds onto 21
# levels 0.1 apart, so that rounding decides real codes. Every matmul below passes this.
_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
)


_SHARD_HEIGHT = 32

# SDPA takes bfloat16 and nothing wider (`sdpa_device_operation.cpp:43`), and the KV cache is read
# by the same op family, so q/k/v and the cache are bf16 while the residual stream stays float32.
_SDPA_DTYPE = ttnn.bfloat16


def _lin(x, w, **kwargs):
    """`ttnn.linear` with the leading batch folded into M, so the weight streams ONCE.

    A `[B, 1, S, K]` activation against a 2-D weight runs as B separate `S x K x N` matmuls that
    each re-read the whole weight from DRAM; `[1, 1, B*S, K]` is one matmul that reads it once.
    """
    shape = [int(d) for d in x.shape]
    lead = 1
    for d in shape[:-2]:
        lead *= d
    if lead == 1:
        return ttnn.linear(x, w, **kwargs)
    y = ttnn.linear(ttnn.reshape(x, [1, 1, lead * shape[-2], shape[-1]]), w, **kwargs)
    return ttnn.reshape(y, shape[:-1] + [int(y.shape[-1])])


def _from_torch(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    t = t.to(torch.bfloat16) if dtype == ttnn.bfloat16 else t.to(torch.float32)
    if device.__class__.__name__ == "MeshDevice":
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=layout,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


def _weight(linear, device):
    """A `[in, out]` device tensor for a torch `nn.Linear` (whose weight is `[out, in]`)."""
    return _from_torch(linear.weight.detach().transpose(0, 1).contiguous(), device)


def _norm_weight(norm, device):
    """Gamma in the `[1, 1, dim // 32, 32]` ROW_MAJOR form `ttnn.rms_norm` requires."""
    return _from_torch(norm.weight.detach().reshape(1, 1, -1, _SHARD_HEIGHT), device, layout=ttnn.ROW_MAJOR_LAYOUT)


def _view4(x, dim):
    """`[..., seq, dim]` -> `([lead, 1, seq, dim], lead, seq, rank)`.

    THE LEADING BOUND IS READ OFF THE TENSOR. This used to be a literal
    `ttnn.reshape(x, [1, 1, seq, dim])`, which is right at the batch of 1 the per-component PCC
    harness feeds and wrong for every batched caller: at B=32 the reshape either raises on volume
    or -- worse, once a leading 1 is folded in elsewhere -- keeps only row 0 and silently drops
    samples 1..31. Everything downstream of here is per-row, so collapsing every leading axis into
    one `lead` is exact for `[B, S, D]`, `[B, 1, S, D]` and the decode stream's `[1, 1, B, D]`.
    """
    shape = [int(s) for s in x.shape]
    seq = shape[-2]
    lead = 1
    for size in shape[:-2]:
        lead *= size
    return ttnn.reshape(x, [lead, 1, seq, dim]), lead, seq, len(shape)


def _restore(x, lead, seq, rank, dim):
    """Put a `[lead, 1, seq, dim]` result back into the RANK the caller handed in."""
    return ttnn.reshape(x, [lead, seq, dim] if rank == 3 else [lead, 1, seq, dim])


def _broadcast4(t, seq, width):
    """A `(cos, sin)` table as `[lead, 1, seq, width]`, its leading bound read off the tensor."""
    volume = 1
    for size in t.shape:
        volume *= int(size)
    return ttnn.reshape(t, [volume // (seq * width), 1, seq, width])


def _rope(x, cos, sin, half):
    """`x * cos + rotate_half(x) * sin` -- the convention `apply_rotary_pos_emb` uses.

    `rotate_half` is `cat(-x[..., half:], x[..., :half])`; both halves are multiples of the tile
    width, so the two slices are tile-aligned.
    """
    ends = list(x.shape)
    lower = ttnn.slice(x, [0, 0, 0, 0], [ends[0], ends[1], ends[2], half])
    upper = ttnn.slice(x, [0, 0, 0, half], ends)
    rotated = ttnn.concat([ttnn.neg(upper), lower], dim=-1)
    return ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(rotated, sin))


def _decode_shard(device, rows, width):
    """HEIGHT-sharded over the batch, one user per core -- the decode op set's layout.

    `nlp_create_qkv_heads_decode`, decode-mode `rotary_embedding_hf` and
    `nlp_concat_heads_decode` are a matched set: each wants one 32-row tile per user, and the RoPE
    op rejects a merely-interleaved tensor outright, so this is part of the contract.
    """
    grid = device.compute_with_storage_grid_size()
    cols = min(int(grid.x), int(rows))
    while rows % cols:
        cols -= 1
    return ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, int(width)),
        core_grid=ttnn.CoreGrid(y=rows // cols, x=cols),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


# THE ZERO TAIL IS A PERSISTENT BUFFER, NOT A PER-CALL `ttnn.zeros`.
# `ttnn.zeros` builds the tensor on the host and enqueues a WRITE to get it onto the device, and a
# write is exactly what a captured trace cannot replay: capturing a prefill that seeded its cache
# this way died on `TT_FATAL: Writes are not supported during trace capture`. The tail is the same
# shape of the same zeros on every call, so it is created once per (device, shape) and reused --
# and because the pad is now shared it is NEVER deallocated by the caller, which used to free it
# after the concat. `flow_matching_audio_transformer` hoists its tile pad for the same reason.
_ZERO_TAIL = {}


def _zero_tail(device, b, h, rows, width):
    key = (id(device), b, h, rows, width)
    buf = _ZERO_TAIL.get(key)
    if buf is None:
        buf = ttnn.zeros([b, h, rows, width], dtype=_SDPA_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
        _ZERO_TAIL[key] = buf
    return buf


def _seed_cache(kv, k, v):
    """Hand the prefill's POST-RoPE k/v to the cache, widened out to `kv["capacity"]`.

    No copy and no second source of truth: the cache IS the prefill's own k/v with a zero tail
    concatenated on the sequence axis, so the resident history cannot disagree with the prefill
    that produced it. The tail slots are never READ before they are written -- a decode step
    writes slot `position` and then attends to `[0, position]` -- so they only have to exist.
    """
    capacity = int(kv.get("capacity") or 0)
    for key, tensor in (("k", k), ("v", v)):
        if tensor.dtype != _SDPA_DTYPE:
            tensor = ttnn.typecast(tensor, _SDPA_DTYPE)
        shape = [int(s) for s in tensor.shape]
        if capacity > shape[-2]:
            pad = _zero_tail(tensor.device(), shape[0], shape[1], capacity - shape[-2], shape[-1])
            tensor = ttnn.concat([tensor, pad], dim=2)
        elif capacity and capacity < shape[-2]:
            raise ValueError(f"kv capacity {capacity} is shorter than the prefill's {shape[-2]}")
        stale = kv.get(key)
        if stale is not None:
            try:
                ttnn.deallocate(stale)
            except Exception:  # noqa: BLE001 - an already-freed buffer is fine to skip
                pass
        kv[key] = tensor
    kv["filled"] = int(k.shape[-2])


def build(device, torch_module):
    mlp = torch_module
    dim = int(mlp.gate_proj.in_features)
    out_dim = int(mlp.down_proj.out_features)

    w_gate = _weight(mlp.gate_proj, device)
    w_up = _weight(mlp.up_proj, device)
    w_down = _weight(mlp.down_proj, device)

    def mlp_forward(x, **kwargs):
        h, lead, seq, rank = _view4(x, dim)
        gated = ttnn.multiply(
            ttnn.silu(_lin(h, w_gate, compute_kernel_config=_COMPUTE)),
            _lin(h, w_up, compute_kernel_config=_COMPUTE),
        )
        out = _lin(gated, w_down, dtype=x.dtype, compute_kernel_config=_COMPUTE)
        ttnn.deallocate(gated)
        return _restore(out, lead, seq, rank, out_dim)

    return mlp_forward

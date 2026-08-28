# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The VAE stage of the FLUX.2-klein-9B TT pipeline.

``AutoencoderKLFlux2`` is the pipeline's latent codec.  Its two halves are read
from the 0.40 diffusers source, not guessed:

    encode(x) = DiagonalGaussian( quant_conv( encoder(x) ) )
    decode(z) = decoder( post_quant_conv(z) )

so the two quant convs sit OUTSIDE both halves, and this file keeps them there:
``encode*`` returns the raw ``encoder`` output (pre-quant-conv, still 64 channels,
mean|logvar interleaved only after ``quant_conv``) and ``decode*`` takes latents
that ``post_quant_conv`` has already been applied to.  The full chains are

    moments = encode*(px);  mode = moments_to_mode(quant_conv(moments))
    image   = decode*(post_quant_conv(mode))

which is HF's order op for op.  Folding the quant convs into the halves would
work too, but then a caller who also has the standalone helpers would apply them
twice, so the split is the explicit one.

That is all ``AutoencoderKLFlux2.encode`` / ``._decode`` do -- **no** patchify
and **no** ``bn`` normalise happen inside the VAE.  ``Flux2KleinPipeline`` owns
both: ``prepare_image_latents`` calls ``_patchify_latents`` then
``(x - bn.running_mean) / sqrt(bn.running_var + batch_norm_eps)`` on the way in,
and the denoise loop's tail un-normalises and calls ``_unpatchify_latents``
before ``vae.decode``.  So this stage's contract is exactly HF's: raw pixels in
at ``encode``, raw (unpatchified, un-normalised) latents in at ``decode``.

Every route is built out of the 15 graduated bring-up stubs under
``models/tt_dit/pipelines/flux_2_klein_9b_vae/_stubs``, used verbatim, each bound
through the invocation ledger at its own position.  The only maths this file adds
is the handful of layers that are NOT any component's submodule -- the encoder's
``conv_in`` / ``conv_norm_out`` / ``conv_out`` tail, the decoder's ``conv_in`` /
``conv_norm_out`` / ``conv_out`` tail, ``quant_conv`` and ``post_quant_conv`` --
and those are built from the in-tree tt_dit primitives (``VaeConv2d``, ``_norm``)
that ``vae_blocks`` itself composes, so they inherit the layouts the graduated
stubs were verified with.  There is no torch compute anywhere in the forward
path; torch is touched only to stage weights and to build the truncated
weight-sharing module views the ``layers`` knob needs.

Tensor parallelism (channel-parallel, mesh 1x8) is entirely the stubs' business.
The one fact this file has to respect is TILE_WIDTH: a channel shard narrower
than 32 cannot be sliced in TILE layout, so at TP=8 the 128-channel stages are
not shardable and neither are the 3/32/64-channel native pieces.  Those run
REPLICATED.  The encoder's 512-channel ``conv_norm_out`` *is* shardable and is
run fractured, mirroring ``vae_blocks.Flux2VaeEncoderBody`` exactly -- because
``ttnn.group_norm``'s DRAM grid check (``groupnorm.cpp::validate_dram_grid``)
requires ``Ht = ceil(N*H*W/32)`` to be a nonzero multiple of
``num_virtual_rows``, and a fractured 512-channel norm has a period of 32
tile-rows while a replicated one has 64.

Resolution constraint (inherited, not chosen)
---------------------------------------------
That same grid check is why this stage only accepts image sides that are a
multiple of 256.  A 224x224 image gives a 28x28 latent, whose ``Ht = 25`` no core
grid divides, and the 512-channel mid block would fail before any arithmetic
happened; the graduated stubs were themselves PCC'd at 256x256 / 32x32 for this
reason (see ``tests/pcc/test_decoder.py``'s ``_NCHW_INPUT`` note).  See
``check_resolution``.

All routes take and return **ttnn tensors in NCHW** -- the contract the graduated
``NchwAdapter`` stubs speak.  ``encode*`` and ``decode*`` also accept a host
torch tensor and stage it themselves.

The leading batch axis
----------------------
Every route carries a real leading batch of ``B`` INDEPENDENT samples: ``B``
different images at ``encode*``, ``B`` different latents at ``decode*``.  No shape
in the forward path is written as a literal ``1`` -- ``moments_to_mode``'s slice
bound, every ``NchwAdapter`` permute and every ``ttnn.conv2d``'s ``batch_size``
are read off the tensor.

What a route does NOT do is put an arbitrary ``B`` through one program.  The
convolutions bound that, and well below 32: ``ttnn.conv2d`` sizes its L1_SMALL halo
from ``N * H * W`` while the slicing that would pay for it is keyed on
``(H, W, Cin, Cout)`` with no batch in it.  So a wide batch runs in CHUNKS of the
width one program does carry, concatenated on device -- exact, because every op in
these routes is per-sample.  See ``_CHUNK_START`` for the measured widths and
``_chunked`` for the driver.  Three more things do not follow on their own:

* ``ttnn.conv2d`` and ``ttnn.group_norm`` both cache work that is a function of
  ``B``, so a port built at one batch cannot be re-used at another.  Concretely
  ``Conv2d._prepared_weight`` comes out of ``ttnn.prepare_conv_weights(...,
  batch_size=b)`` on the FIRST forward and is then reused verbatim.  Every cache
  in this file is therefore keyed by ``(what, batch)``: a second batch builds its
  own ports rather than silently running the first batch's schedule.
* ``ttnn.group_norm``'s chunking heuristic cannot see the batch axis at all (see
  ``_group_norm_out_blocks``), so at ``B > 1`` it under-chunks and overflows L1.
  This file re-derives the chunk count WITH the batch axis and passes it to the
  norms IT owns -- the two ``conv_norm_out`` tails.  That is a scheduling knob:
  the arithmetic, the core grid and the ``B == 1`` behaviour are untouched (at
  ``B == 1`` it returns the same ``-1`` sentinel the norms already used).

  The norms a graduated stub owns cannot be reached the same way: tt_dit calls
  them as ``self.norm1.forward(h)`` and takes the ``num_out_blocks=-1`` default,
  and this file has no argument to add to that call.  ``unchunkable_group_norms``
  names those norms.  Chunking is what makes it moot in practice -- a chunk runs at
  a width the stock heuristic already fits -- and ``GroupNorm.forward`` upstream now
  climbs to a count that fits rather than throwing.
* A port is not batch-agnostic, so the chunk width is part of what is cached.  Ports
  are built at the CHUNK width, which is also why the batched routes reuse the very
  conv schedules the B=1 routes are green at instead of compiling a second set.
"""

from __future__ import annotations

import torch
from torch import nn

import ttnn
from models.demos.flux_2_klein_9b.tt.stubs import Ledger, load_stub_module
from models.tt_dit.layers.module import Module
from models.tt_dit.layers.normalization import GroupNorm
from models.tt_dit.pipelines.flux_2_klein_9b_vae.vae_blocks import (
    VAE_NORM,
    VaeConv2d,
    _norm,
    fracture_channels,
    gather_channels,
    is_shardable,
    make_ctx,
    replicated_ctx,
    to_nchw,
    to_nhwc,
    to_row_major,
    to_tile,
)

STAGE = "vae"

#: ``ttnn.group_norm``'s DRAM grid period, in activation positions, for the
#: widest / narrowest norm in this checkpoint.  The encoder's 512-channel norms
#: run fractured (64 local channels -> 32 tile-row period -> 1024 positions);
#: the decoder's 128-channel tail runs replicated (64 tile-row period -> 2048
#: positions).  A square image of side S has a latent of side S/8, so the binding
#: constraints are (S/8)^2 % 1024 == 0 and S^2 % 2048 == 0, i.e. S % 256 == 0.
_LATENT_POSITION_PERIOD = 1024
_IMAGE_POSITION_PERIOD = 2048
VAE_SPATIAL_DOWNSCALE = 8

#: Every DRAM-grid period this checkpoint's norms actually run at, in tile-rows.
#:
#: ``validate_dram_grid`` computes ``num_virtual_rows = (grid_x / num_virtual_cols)
#: * grid_y`` from tt_dit's pinned ``CoreGrid(8, 8)`` and the norm's PER-DEVICE
#: channel count, and then requires all three of
#:
#:     Ht >= nvr,  Ht % nvr == 0,  (nvr < N or nvr % N == 0)
#:
#: where ``Ht = N*H*W/32``.  Only four ``nvr`` values occur here:
#:
#:     nvr= 8  256- and 512-channel REPLICATED norms   (nvc = 8)
#:     nvr=16  128-channel REPLICATED norms            (nvc = 4)
#:     nvr=32  512-channel FRACTURED norms, 64 local   (nvc = 2)
#:     nvr=64  256-channel FRACTURED norms, 32 local   (nvc = 1)
#:
#: The first two conditions are what ``_LATENT_POSITION_PERIOD`` /
#: ``_IMAGE_POSITION_PERIOD`` encode, and a leading batch only multiplies ``Ht``,
#: so it cannot break them.  The THIRD is batch-only and is why
#: ``check_resolution`` takes ``batch``: it makes B=3 illegal at any resolution.
_GROUP_NORM_VIRTUAL_ROWS = (8, 16, 32, 64)


def legal_batch(batch: int) -> bool:
    """Whether ``ttnn.group_norm``'s uniform-multicast rule admits this batch.

    ``num_virtual_rows`` must either be smaller than the batch or divide it
    exactly, for every period in ``_GROUP_NORM_VIRTUAL_ROWS``.  That admits
    1, 2, 4, 8, 16, 32 and 64 -- and any batch above 64, where every period is
    already smaller than the batch -- and rejects everything else.
    """
    n = int(batch)
    if n < 1:
        return False
    return all(nvr < n or nvr % n == 0 for nvr in _GROUP_NORM_VIRTUAL_ROWS)


def check_resolution(height: int, width: int, batch: int = 1) -> None:
    """Raise unless ``ttnn.group_norm`` can legally run every norm at this size.

    Fails loudly and early rather than letting the mid block trip
    ``validate_dram_grid`` several minutes into a build.  ``batch`` is the leading
    axis the stage will carry; it multiplies ``Ht`` (so it cannot break the two
    divisibility rules the resolution has to satisfy) but it is itself checked
    against the uniform-multicast rule -- see ``_GROUP_NORM_VIRTUAL_ROWS``.
    """
    h, w = int(height), int(width)
    if h % VAE_SPATIAL_DOWNSCALE or w % VAE_SPATIAL_DOWNSCALE:
        msg = f"VAE stage needs H and W divisible by {VAE_SPATIAL_DOWNSCALE}; got {h}x{w}"
        raise ValueError(msg)
    latent_positions = (h // VAE_SPATIAL_DOWNSCALE) * (w // VAE_SPATIAL_DOWNSCALE)
    if latent_positions % _LATENT_POSITION_PERIOD or (h * w) % _IMAGE_POSITION_PERIOD:
        msg = (
            f"VAE stage cannot run at {h}x{w}: ttnn.group_norm's DRAM grid check needs the "
            f"latent to hold a multiple of {_LATENT_POSITION_PERIOD} positions (got "
            f"{latent_positions}) and the image a multiple of {_IMAGE_POSITION_PERIOD} (got "
            f"{h * w}). For square images that means a side which is a multiple of 256."
        )
        raise ValueError(msg)
    check_batch(batch)


def check_batch(batch: int) -> None:
    """Raise unless ``batch`` is a leading axis every norm's core grid can serve."""
    if not legal_batch(batch):
        msg = (
            f"VAE stage cannot run a leading batch of {int(batch)}: ttnn.group_norm's DRAM grid "
            f"check needs each norm's num_virtual_rows {list(_GROUP_NORM_VIRTUAL_ROWS)} to be "
            f"smaller than the batch or to divide it exactly, so the batch must be 1, 2, 4, 8, "
            f"16, 32, 64 or larger than 64."
        )
        raise ValueError(msg)


def _leading(x) -> int:
    """The leading (batch) extent of a host torch or staged ttnn tensor."""
    return int(x.shape[0])


# ------------------------------------------------------------- batch chunking
#
# A route's convolutions are what bound its leading batch, and the bound is well
# below 32.  ``ttnn.conv2d`` sizes its L1_SMALL halo from ``N * H * W``, but the
# slicing that would pay for it -- tt_dit's ``Conv2d.slice_params`` -- is keyed on
# ``(H, W, Cin, Cout)`` with NO batch in it, so a B=32 call asks a B=1 schedule for
# 32x the halo and the allocator refuses.  Measured on a 1x8 mesh at this stage's
# pinned resolution with the bring-up's ``l1_small_size`` of 24576 B, one FRESH
# device per point (so none of these is a leftover from an earlier route):
#
#   encode 256x256   B=4  runs.  B=32 dies in ``encoder.down_blocks``' conv
#                    (128, 128, 128, 256) asking 3200 B per bank of a region
#                    already holding 23776 B per bank.
#   decode 32x32     B=1  runs.  B=4 dies in the decoder tail conv
#                    (256, 256, 256, 128) asking 416 B per bank with 32 B free.
#
# So a batch wider than that runs in CHUNKS of it, concatenated on device.  That is
# EXACT rather than an approximation: every op in these routes is per-sample -- the
# convolutions slide over H and W only, ``ttnn.group_norm`` normalises within a
# sample, and the spatial attentions attend within one image -- so chunking changes
# the loop order and nothing else.  It also means the batched routes reuse the very
# conv schedules the B=1 routes are green at, instead of compiling a second set.
#
# These are starting points, not assumptions: ``_chunk_width`` narrows from here
# whenever a chunk's conv program exceeds L1, and remembers the width that ran.  What
# is being sized is the ACTIVATION chunk; the weights are resident for the whole run
# either way, staged once when the route is built.
_CHUNK_START = {"encode": 4, "decode": 1}

#: Allocator failures that mean "this chunk is too wide", as opposed to a defect.
#: All three are raised while the program is being BUILT, so nothing has executed
#: and narrowing the chunk and re-running is safe.
_TOO_WIDE = ("L1_SMALL buffer", "circular buffer", "beyond max L1 size")


def _is_too_wide(exc: BaseException) -> bool:
    text = str(exc)
    return "Out of Memory" in text or any(m in text for m in _TOO_WIDE)


def _slice_leading(x: ttnn.Tensor, start: int, end: int) -> ttnn.Tensor:
    """Rows ``[start, end)`` of the leading axis.

    ``ttnn.slice``, so this is a DEVICE op -- the trace steps go through it and have
    to stay host-op free.  Only dim 0 is narrowed and dim 0 is not a tiled axis, so
    this carries no tile-alignment constraint.
    """
    begins = [0] * len(x.shape)
    ends = [int(d) for d in x.shape]
    begins[0], ends[0] = int(start), int(end)
    return ttnn.slice(x, begins, ends)


# --------------------------------------------------------- group-norm chunking


def _group_norm_out_blocks(norm: GroupNorm, x: ttnn.Tensor) -> int:
    """``ttnn.group_norm``'s own chunk count, re-derived WITH the batch axis.

    ``GroupNorm.forward`` reshapes to ``[N, 1, H*W, C]`` and asks for
    ``num_out_blocks=-1``; the op then picks the count from
    ``shape[1] * shape[2] * shape[3]`` (``groupnorm_program_utils.cpp::
    groupnorm_heuristic_num_out_blocks``) -- which is ``H*W*C``, with **no N in
    it**.  Each core's block therefore grows linearly with the batch while the
    chunking stays put, and the statically-allocated CBs blow past L1:

        128 channels, 256x256, replicated (the decoder's tail):
            B=1 -> fits;  B=2 -> 1967296 B;  B=4 -> 3802304 B;  B=8 -> 7472320 B
        against "max L1 size of 1499136 B".  So the stock chunk count does not
        survive even B=2 at this resolution.

    This is the same formula with ``N`` restored, so it reduces to exactly the
    op's own answer at ``N == 1`` (and is skipped entirely there, returning the
    ``-1`` sentinel, so a B=1 run is bit-for-bit the run it always was).  It
    changes the loop structure of the op, never its arithmetic.
    """
    n, h, w = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    if n <= 1:
        return -1
    num_virtual_cols = int(norm.num_virtual_cols)
    if num_virtual_cols <= 0:
        return -1
    grid = norm.core_grid
    rows_per_y = int(grid.x) // num_virtual_cols
    virtual_cores = num_virtual_cols * rows_per_y * int(grid.y)
    if virtual_cores <= 0:
        return -1
    # HEURISTIC_BLOCK_SIZE_BASE == 256 * 256, MAX_HEURISTIC_NUM_OUT_BLOCKS == 256.
    # Chunking finer than this was measured and is NOT a lever on anything else:
    # at 4x the decoder's B=32 L1-buffer/circular-buffer clash reported the same
    # region end (1412288) to the byte, so those buffers are not the norm's.
    target = (n * h * w * int(norm.num_padded_channels)) // (65536 * virtual_cores)
    blocks = 1
    while blocks < target and blocks < 256:
        blocks *= 2
    return blocks


def unchunkable_group_norms(root, batch: int) -> list[str]:
    """The ``GroupNorm``s under ``root`` that this file cannot hand a chunk count to.

    A norm this file BUILT is called from here, so ``_group_norm_out_blocks`` can be
    passed to it as a plain argument.  A norm a graduated stub built is called by
    tt_dit (``VaeResnetBlock.forward`` -> ``self.norm1.forward(h)``), which takes the
    ``num_out_blocks=-1`` default, and there is no argument this file can add to that
    call.  Rebinding the instance's ``forward`` would be one, but assignment to a
    ``.forward`` attribute anywhere under ``tt/`` is a pipeline gate
    (``test_gates.py::test_pipeline_has_no_hf_orchestration``), and rightly so.

    So this returns the diagnosis rather than a workaround: the names of the norms
    that will overflow L1 at ``batch``, which is what the stage reports when a
    batched route fails.  The fix is one line, upstream, in
    ``models/tt_dit/layers/normalization.py``:

        def forward(self, x, num_out_blocks=-1, ...):
            batch_size, height, width, channels = x.shape
            x = x.reshape([batch_size, 1, width * height, channels])

    -- the reshape hides the batch from ``ttnn.group_norm``'s chunking heuristic,
    which reads ``shape[1] * shape[2] * shape[3]``.  Multiplying that volume by
    ``batch_size`` (or passing this file's ``_group_norm_out_blocks``) is the whole
    change, and it is a no-op at ``batch == 1``.
    """
    if int(batch) <= 1:
        return []
    return [f"{name}({int(norm.num_local_channels)}ch/device)" for name, norm in _named_group_norms(root)]


def _named_group_norms(root):
    """Every ``GroupNorm`` reachable from a built port, stub wrapper or bare module."""
    seen: set[int] = set()
    stack = [("", getattr(root, "inner", root))]
    while stack:
        prefix, node = stack.pop()
        if not isinstance(node, Module) or id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, GroupNorm):
            yield prefix or "<root>", node
        for name, child in node.named_children():
            stack.append((f"{prefix}.{name}" if prefix else name, child))


# --------------------------------------------------------------------------- torch views


def _shallow_module_copy(module: nn.Module) -> nn.Module:
    """A new ``nn.Module`` that SHARES ``module``'s weights but owns its child map.

    Weight staging only: this is how the ``layers`` knob produces a shorter model
    without copying a single parameter and without mutating ``hf_vae``.  Every
    dict/set in ``__dict__`` (``_parameters`` / ``_buffers`` / ``_modules`` /
    ``_non_persistent_buffers_set`` / the hook maps) is duplicated one level deep,
    so assigning a new ``ModuleList`` on the copy cannot reach the original.
    """
    clone = module.__class__.__new__(module.__class__)
    clone.__dict__ = dict(module.__dict__)
    for key, value in list(clone.__dict__.items()):
        if isinstance(value, dict):
            clone.__dict__[key] = type(value)(value)
        elif isinstance(value, set):
            clone.__dict__[key] = set(value)
    return clone


def _clamped(layers: int | None, available: int) -> int:
    """``layers`` resnets, clamped into ``[1, available]``.

    A down/up block with zero resnets is not a model, so 1 is the floor.
    """
    if layers is None:
        return available
    return max(1, min(int(layers), available))


def _block_view(block: nn.Module, layers: int | None) -> nn.Module:
    """``block`` with its ``resnets`` truncated to ``layers`` (weights shared)."""
    keep = _clamped(layers, len(block.resnets))
    if keep == len(block.resnets):
        return block
    view = _shallow_module_copy(block)
    view.resnets = nn.ModuleList(list(block.resnets)[:keep])
    return view


def _stack_view(parent: nn.Module, attr: str, layers: int | None) -> nn.Module:
    """``parent`` with every block of ``parent.<attr>`` resnet-truncated.

    All blocks are truncated to the SAME count, because the composite stubs
    derive one ``layers_per_block`` from ``blocks[0]`` and apply it to all of
    them.  The 4-stage channel ladder itself is never truncated: each stage has
    different channel counts and halves (encoder) or doubles (decoder) the
    resolution, so dropping one would leave the encoder's output incompatible
    with ``quant_conv`` -- i.e. not a runnable stage.
    """
    blocks = getattr(parent, attr)
    keep = _clamped(layers, len(blocks[0].resnets))
    if keep == len(blocks[0].resnets):
        return parent
    view = _shallow_module_copy(parent)
    setattr(view, attr, nn.ModuleList([_block_view(b, keep) for b in blocks]))
    return view


# --------------------------------------------------------------------------- block ports


class VaeBlockPort:
    """One stage of a decomposed VAE half: a fixed sequence of ledger-bound stub
    ports, chained in NCHW.

    ``down_blocks`` / ``up_blocks`` are lists of these, so the pipeline sees one
    uniform, same-typed handle per stage whether that stage is a single graduated
    block or a hand-decomposed resnet/attention/sampler chain.
    """

    kind = "block"

    def __init__(self, index: int, name: str, steps) -> None:
        self.index = int(index)
        self.name = name
        self.steps = list(steps)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        for step in self.steps:
            x = step(x)
        return x

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(index={self.index}, name={self.name!r}, steps={len(self.steps)})"


class DownBlockPort(VaeBlockPort):
    kind = "down"


class UpBlockPort(VaeBlockPort):
    kind = "up"


def _fresh_ladder(ladder: list) -> list:
    """Empty clones of a ladder's stages, for a SECOND batch of the same route.

    A built port is batch-specific, so the ladder ``__init__`` laid out belongs to
    whichever batch filled it first; another batch needs its own stage objects or
    it would overwrite the steps the first batch's cached route still runs.
    """
    return [type(stage)(stage.index, stage.name, []) for stage in ladder]


# --------------------------------------------------------------------------- the stage


class Flux2VaeStage:
    """The graduated FLUX.2-klein-9B VAE, composed into real encode/decode routes.

    Args:
        device: the ``MeshShape(1, TP)`` mesh device.
        hf_vae: a loaded ``AutoencoderKLFlux2`` (fp32; ``force_upcast`` is true).
            Used at build time to source weights, and never called.
        ledger: the shared invocation ledger.  One is created if omitted, so
            every port is always bound and ``self.ledger`` is always queryable.
        layers: ``None`` for full depth; ``k`` caps the resnets per down/up block
            at ``max(1, min(k, available))``.  ``conv_in``, both mid blocks, both
            tails and the quant convs stay intact, and the 4-stage channel ladder
            is never shortened (see ``_stack_view``).  A truncated stage is not
            PCC-comparable against the full HF model -- use ``hf_view`` to get a
            weight-sharing torch module of the same depth for that.

    Every route is lazy: the ports for a route are built (and bound) the first
    time that route runs, so a caller that only needs ``decode`` never pays for
    the encoder's weights.
    """

    def __init__(self, device, hf_vae, *, ledger: Ledger | None = None, layers: int | None = None) -> None:
        self.device = device
        self.hf = hf_vae
        self.ledger = ledger if ledger is not None else Ledger()
        self.layers = layers

        self._ctx = None
        self._rep_ctx = None

        # (stub name, submodule path, batch) -> built port.  The same HF submodule
        # reached by two routes at the same batch is built once and bound at both
        # positions.  `batch` is part of the key because a built port is not
        # batch-agnostic: the first forward bakes `ttnn.prepare_conv_weights(...,
        # batch_size=b)` into every conv it owns (see the module docstring), so a
        # port built at B=1 would silently run B=32 on B=1's schedule.
        self._built: dict[tuple[str, str, int], object] = {}
        self._natives: dict[tuple[str, int], object] = {}
        self._routes: dict[tuple[str, int], object] = {}

        #: route family -> the chunk width this machine actually ran, once a wider one
        #: has been shown not to fit.  Starts empty: `_CHUNK_START` is the first guess.
        self._chunk_learned: dict[str, int] = {}

        # The two channel ladders, laid out EAGERLY and filled in place by the
        # decomposed routes.  These are this checkpoint's two VAE sections, and a
        # structure walk over a freshly-built pipeline has to see both of them; if
        # the lists only came into existence when a route first ran, the walk would
        # find no VAE stack at all and its depth would be inferred for the run.
        # The ladder is never shortened -- `layers` caps the resnets INSIDE a stage
        # (see `_stack_view`), not the number of stages.
        self.down_blocks = [
            DownBlockPort(i, f"encoder.down_blocks.{i}", []) for i in range(len(hf_vae.encoder.down_blocks))
        ]
        self.up_blocks = [UpBlockPort(i, f"decoder.up_blocks.{i}", []) for i in range(len(hf_vae.decoder.up_blocks))]

    # ------------------------------------------------------------------ contexts

    @property
    def staged(self) -> bool:
        """True once any route has put weights on the device.  The pipeline reads this
        to decide whether releasing this stage frees anything."""
        return bool(self._built or self._natives or self._routes)

    @property
    def ctx(self):
        """The channel-tensor-parallel context, built on first use.

        `make_ctx` allocates a `CCLManager` on the device, so it is deferred: nothing
        in this stage's construction may touch the device (see `__init__`).
        """
        if self._ctx is None:
            self._ctx = make_ctx(self.device)
        return self._ctx

    @property
    def rep_ctx(self):
        """`ctx` with tensor parallelism off -- for the sub-tile-channel stages."""
        if self._rep_ctx is None:
            self._rep_ctx = replicated_ctx(self.ctx)
        return self._rep_ctx

    # ------------------------------------------------------------------ plumbing

    def _port(self, stub: str, submodule_path: str, position: str, torch_module, batch: int):
        """Build (or reuse) ``stub`` over ``torch_module`` at ``batch`` and bind it."""
        check_batch(batch)
        key = (stub, submodule_path, int(batch))
        port = self._built.get(key)
        if port is None:
            module = load_stub_module(STAGE, stub)
            port = module.build(self.device, torch_module)
            self._built[key] = port
        return self.ledger.bind(STAGE, stub, position, port)

    # -------------------------------------------------------------- batch chunking

    def chunk_width(self, family: str, total: int) -> int:
        """How many samples of ``total`` one program of this route family carries.

        The family's measured start (``_CHUNK_START``), narrowed to a DIVISOR of
        ``total`` so every chunk has the same shape -- a narrower last chunk would be
        a second conv schedule and a second set of prepared weights -- and further
        narrowed by whatever width ``_chunked`` has already measured as too wide for L1.
        """
        width = min(int(self._chunk_learned.get(family, _CHUNK_START[family])), int(total))
        while width > 1 and int(total) % width:
            width //= 2
        return max(1, width)

    def _narrower(self, width: int, total: int) -> int | None:
        """The next chunk width down that still divides ``total``, or None at 1."""
        candidate = int(width) // 2
        while candidate > 1 and int(total) % candidate:
            candidate //= 2
        return candidate if 1 <= candidate < int(width) else None

    def _map_over_leading(self, body, x: ttnn.Tensor, width: int) -> ttnn.Tensor:
        """``body`` over the leading axis in chunks of ``width``, joined on device."""
        total = _leading(x)
        if width >= total:
            return body(x)
        outs = [body(_slice_leading(x, start, min(start + width, total))) for start in range(0, total, width)]
        return ttnn.concat(outs, dim=0)

    def _chunked(self, family: str, x, body):
        """Stage ``x``, then run ``body`` over its leading axis in chunks.

        A chunk whose conv program exceeds L1 narrows the width and re-runs the batch,
        so the width is measured on this machine rather than assumed -- and the ports
        the rejected width built are dropped, since nothing will call them again.  This
        is a one-off calibration of the ACTIVATION chunk, remembered in
        ``_chunk_learned``; the route's weights are staged once and stay resident.
        """
        staged = self._stage(x)
        total = _leading(staged)
        width = self.chunk_width(family, total)
        while True:
            try:
                return self._map_over_leading(body, staged, width)
            except (RuntimeError, ValueError) as exc:
                narrower = self._narrower(width, total)
                if narrower is None or not _is_too_wide(exc):
                    raise
                print(
                    f"vae {family}: a chunk of {width} of the {total}-sample batch did not fit "
                    f"({type(exc).__name__}: {str(exc).strip().splitlines()[0][:160]}); "
                    f"re-running the batch in chunks of {narrower}",
                    flush=True,
                )
                self._drop_built_at(width)
                self._chunk_learned[family] = narrower
                width = narrower

    def _drop_built_at(self, batch: int) -> None:
        """Forget every port, route and native conv built at ``batch``.

        A width that was rejected will never be called again, and its prepared conv
        weights are pure residency -- which matters here more than usual, since the
        reason the width was rejected is that the device was out of room.  The ledger
        still holds its own wrapper for a bound port, so this frees the caches this
        stage owns, not every last reference.
        """
        for key in [k for k in self._built if k[2] == int(batch)]:
            self._built.pop(key, None)
        for key in [k for k in self._routes if k[1] == int(batch)]:
            self._routes.pop(key, None)
        for key in [k for k in self._natives if k[1] == int(batch)]:
            self._natives.pop(key, None)

    def _consume(self, x) -> None:
        """Record that a ledger-bound port's output was eaten by a NATIVE op.

        The ledger only sees a tensor as consumed when it reaches another bound
        port or is marked as head output.  The last stub of a decomposed route
        feeds this file's native tail, so without this the route would look like
        a coverage sweep instead of a forward path.
        """
        self.ledger.mark_final(x)

    def mark_final(self, x) -> None:
        """Pass-through so the pipeline can close the ledger on this stage's output."""
        self.ledger.mark_final(x)

    def _stage(self, x, *, layout=ttnn.TILE_LAYOUT):
        """Host torch (or an already-staged ttnn tensor) -> replicated ttnn NCHW."""
        if isinstance(x, ttnn.Tensor):
            return x
        host = x.detach().to(torch.bfloat16)
        try:
            return ttnn.from_torch(
                host,
                dtype=ttnn.bfloat16,
                layout=layout,
                device=self.device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            )
        except (AttributeError, TypeError):
            return ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=layout, device=self.device)

    # ------------------------------------------------------- native (non-stub) layers

    def _native_conv(self, name: str, torch_conv: nn.Conv2d, batch: int, *, ctx=None):
        """A replicated ``VaeConv2d`` over a diffusers ``nn.Conv2d``, at ``batch``.

        3 / 32 / 64 / 128 channels: none of these shard at TP=8 (a 16-wide or
        narrower channel shard is sub-tile and TILE layout cannot slice it), so
        every native conv here is replicated and ``tensor_parallel=False``.

        Keyed by batch for the same reason the stub ports are (see ``__init__``):
        the conv's prepared weight is built once, for the batch of its first
        forward.
        """
        check_batch(batch)
        key = (name, int(batch))
        conv = self._natives.get(key)
        if conv is not None:
            return conv
        kernel = tuple(torch_conv.kernel_size)
        padding = tuple(torch_conv.padding)
        assert kernel[0] == kernel[1], f"{name}: non-square kernel {kernel}"
        assert padding[0] == padding[1], f"{name}: asymmetric padding {padding}"
        conv = VaeConv2d(
            int(torch_conv.in_channels),
            int(torch_conv.out_channels),
            kernel_size=kernel[0],
            padding=padding[0],
            tensor_parallel=False,
            ctx=ctx if ctx is not None else self.rep_ctx,
        )
        state = {"weight": torch_conv.weight.detach().to(torch.float32)}
        if torch_conv.bias is not None:
            state["bias"] = torch_conv.bias.detach().to(torch.float32)
        conv.load_torch_state_dict(state)
        self._natives[key] = conv
        return conv

    def _native_norm(self, name: str, torch_gn: nn.GroupNorm, batch: int, *, sharded: bool):
        """``GroupNorm(32, eps=1e-6) -> SiLU`` as one fused tt_dit norm, at ``batch``."""
        check_batch(batch)
        key = (name, int(batch))
        norm = self._natives.get(key)
        if norm is not None:
            return norm
        assert int(torch_gn.num_groups) == VAE_NORM.num_groups, f"{name}: {torch_gn.num_groups} groups"
        norm = _norm(
            VAE_NORM,
            num_channels=int(torch_gn.num_channels),
            ctx=self.ctx if sharded else self.rep_ctx,
            activation_fn="silu",
        )
        norm.load_torch_state_dict(
            {
                "weight": torch_gn.weight.detach().to(torch.float32),
                "bias": torch_gn.bias.detach().to(torch.float32),
            }
        )
        self._natives[key] = norm
        return norm

    def _enc_conv_in(self, x: ttnn.Tensor) -> ttnn.Tensor:
        conv = self._native_conv("encoder.conv_in", self.hf.encoder.conv_in, _leading(x))
        return to_nchw(conv.forward(to_nhwc(x)))

    def _enc_tail(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """``conv_norm_out -> SiLU -> conv_out``, the encoder's 512 -> 64 head.

        Mirrors ``Flux2VaeEncoderBody.forward``'s tail: the norm runs channel-
        fractured (its 512 channels give a legal 64-wide shard, and a fractured
        512-channel GroupNorm has half the DRAM-grid period of a replicated one),
        then one ``all_gather`` before the replicated ``conv_out``.
        """
        encoder = self.hf.encoder
        batch = _leading(x)
        sharded = is_shardable(self.ctx, int(encoder.conv_norm_out.num_channels))
        norm = self._native_norm("encoder.conv_norm_out", encoder.conv_norm_out, batch, sharded=sharded)
        conv = self._native_conv("encoder.conv_out", encoder.conv_out, batch)

        h = to_nhwc(x)
        if sharded:
            h = fracture_channels(self.ctx, h)
        h = norm.forward(h, num_out_blocks=_group_norm_out_blocks(norm, h))
        if sharded:
            h = gather_channels(self.ctx, h)
        return to_nchw(conv.forward(to_tile(h)))

    def _dec_conv_in(self, x: ttnn.Tensor) -> ttnn.Tensor:
        conv = self._native_conv("decoder.conv_in", self.hf.decoder.conv_in, _leading(x))
        return to_nchw(conv.forward(to_nhwc(x)))

    def _dec_tail(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """``conv_norm_out -> SiLU -> conv_out``, the decoder's 128 -> 3 head.

        128 channels do not shard at TP=8, so this is the replicated regime --
        the same one the graduated ``decoder`` stub runs its own tail in.  It is
        also the widest activation in the model: ``(B, 128, H, W)`` at FULL image
        resolution, so it is the norm whose chunk count the batch axis matters
        most for (see ``_group_norm_out_blocks``).
        """
        decoder = self.hf.decoder
        batch = _leading(x)
        norm = self._native_norm("decoder.conv_norm_out", decoder.conv_norm_out, batch, sharded=False)
        conv = self._native_conv("decoder.conv_out", decoder.conv_out, batch)
        h = to_nhwc(x)
        h = norm.forward(h, num_out_blocks=_group_norm_out_blocks(norm, h))
        return to_nchw(conv.forward(to_tile(h)))

    # ------------------------------------------------------------- small helpers

    def quant_conv(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """``AutoencoderKLFlux2.quant_conv`` -- 1x1, 64 -> 64, on the moments.

        Carries the leading batch: ``(B, 64, h, w)`` in, ``(B, 64, h, w)`` out.
        """
        staged = self._stage(x)
        conv = self._native_conv("quant_conv", self.hf.quant_conv, _leading(staged))
        return to_nchw(conv.forward(to_nhwc(staged)))

    def post_quant_conv(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """``AutoencoderKLFlux2.post_quant_conv`` -- 1x1, 32 -> 32, before decode."""
        staged = self._stage(x)
        conv = self._native_conv("post_quant_conv", self.hf.post_quant_conv, _leading(staged))
        return to_nchw(conv.forward(to_nhwc(staged)))

    def moments_to_mode(self, moments: ttnn.Tensor) -> ttnn.Tensor:
        """``DiagonalGaussianDistribution(moments).mode()`` == ``chunk(...,2,dim=1)[0]``.

        The channel axis is dim 1 of an NCHW tensor, so in ROW_MAJOR this is a
        contiguous prefix slice and carries no tile-alignment constraint.  The
        leading bound is ``n`` read off the tensor, never a literal 1: at B=32 a
        literal would keep sample 0 and silently drop the other 31.
        """
        n, c, h, w = (int(d) for d in moments.shape)
        assert c % 2 == 0, f"moments must have an even channel count, got {c}"
        return ttnn.slice(to_row_major(moments), [0, 0, 0, 0], [n, c // 2, h, w])

    # ------------------------------------------------------------ HF module views

    def hf_view(self, half: str):
        """The ``encoder`` / ``decoder`` torch module at THIS stage's depth.

        Weights are shared with ``self.hf``; with ``layers=None`` it is
        ``self.hf.<half>`` itself.  This is what a golden has to be computed from
        when ``layers`` is set.
        """
        if half == "encoder":
            return _stack_view(self.hf.encoder, "down_blocks", self.layers)
        if half == "decoder":
            return _stack_view(self.hf.decoder, "up_blocks", self.layers)
        msg = f"half must be 'encoder' or 'decoder', got {half!r}"
        raise ValueError(msg)

    # --------------------------------------------------------------- route: encode

    def _encoder_composite(self, stub: str, route: str, batch: int):
        key = (f"{route}:{stub}", int(batch))
        port = self._routes.get(key)
        if port is None:
            port = self._port(stub, "encoder", f"{route}/encoder", self.hf_view("encoder"), batch)
            self._routes[key] = port
        return port

    def encode(self, pixel_values) -> ttnn.Tensor:
        """``encoder`` port -> pre-quant-conv moments ``(B, 64, H/8, W/8)``.

        ``B`` independent images in, ``B`` independent moment maps out, run in chunks
        of the width one program carries (see ``_CHUNK_START``) and joined on device.
        Feed the result to ``quant_conv`` and then ``moments_to_mode``.
        """
        return self._chunked(
            "encode",
            pixel_values,
            lambda x: self._encoder_composite("encoder", "encode", _leading(x))(x),
        )

    def encode_alias(self, pixel_values) -> ttnn.Tensor:
        """``encoder_stack`` port -> pre-quant-conv moments.

        ``encoder_stack``'s capture manifest records ``submodule_path = encoder``:
        it is the same ``diffusers.Encoder``, brought up as a second independent
        component.  The multi-reference edit head needs two encoder invocations
        (one per reference image), so both are real work, not a duplicate.
        """
        return self._chunked(
            "encode",
            pixel_values,
            lambda x: self._encoder_composite("encoder_stack", "encode_alias", _leading(x))(x),
        )

    def _ensure_encode_blockwise(self, batch: int = 1):
        key = ("encode_blockwise", int(batch))
        built = self._routes.get(key)
        if built is not None:
            return built
        encoder = self.hf.encoder
        route = "encode_blockwise"

        stem = self._port("patch_embed", "encoder.conv_in", f"{route}/encoder.conv_in", encoder.conv_in, batch)
        blocks = [
            DownBlockPort(
                i,
                f"encoder.down_blocks.{i}",
                [
                    self._port(
                        "down_encoder_block2_d",
                        f"encoder.down_blocks.{i}",
                        f"{route}/encoder.down_blocks.{i}",
                        _block_view(block, self.layers),
                        batch,
                    )
                ],
            )
            for i, block in enumerate(encoder.down_blocks)
        ]
        mid = self._port(
            "u_net_mid_block2_d", "encoder.mid_block", f"{route}/encoder.mid_block", encoder.mid_block, batch
        )
        built = (stem, blocks, mid)
        self._routes[key] = built
        return built

    def encode_blockwise(self, pixel_values) -> ttnn.Tensor:
        """``patch_embed`` -> 4x ``down_encoder_block2_d`` -> ``u_net_mid_block2_d``
        -> native tail -> ``quant_conv``.

        ``patch_embed`` is the scaffold's generic image-stem role, bound to
        ``encoder.conv_in``; it is the only rung-``emit`` component here, so it
        runs replicated over the mesh rather than channel-parallel.
        """

        def body(x):
            stem, blocks, mid = self._ensure_encode_blockwise(_leading(x))
            x = stem(x)
            for block in blocks:
                x = block(x)
            x = mid(x)
            self._consume(x)
            return self._enc_tail(x)

        return self._chunked("encode", pixel_values, body)

    def _ensure_encode_decomposed(self, batch: int = 1):
        key = ("encode_decomposed", int(batch))
        built = self._routes.get(key)
        if built is not None:
            return built
        encoder = self.hf.encoder
        route = "encode_decomposed"

        down0 = encoder.down_blocks[0]
        keep = _clamped(self.layers, len(down0.resnets))
        steps = [
            self._port(
                "resnet_block2_d",
                f"encoder.down_blocks.0.resnets.{j}",
                f"{route}/encoder.down_blocks.0.resnets.{j}",
                down0.resnets[j],
                batch,
            )
            for j in range(keep)
        ]
        steps.append(
            self._port(
                "downsample2_d",
                "encoder.down_blocks.0.downsamplers.0",
                f"{route}/encoder.down_blocks.0.downsamplers.0",
                down0.downsamplers[0],
                batch,
            )
        )
        # fill the ladder `__init__` laid out, in place: the list object the walk
        # already holds is the one this route runs.  A port is batch-specific, so
        # only the FIRST batch to build this route can own the shared ladder; a
        # second batch gets its own equally-shaped list rather than overwriting
        # the steps the first batch's cached route still points at.
        blocks = self.down_blocks if not any(b.steps for b in self.down_blocks) else _fresh_ladder(self.down_blocks)
        blocks[0].steps = steps
        for i, block in enumerate(encoder.down_blocks[1:], start=1):
            blocks[i].steps = [
                self._port(
                    "down_encoder_block2_d",
                    f"encoder.down_blocks.{i}",
                    f"{route}/encoder.down_blocks.{i}",
                    _block_view(block, self.layers),
                    batch,
                )
            ]

        mid = encoder.mid_block
        mid_steps = [
            self._port(
                "resnet_block2_d",
                "encoder.mid_block.resnets.0",
                f"{route}/encoder.mid_block.resnets.0",
                mid.resnets[0],
                batch,
            ),
            self._port(
                "attention",
                "encoder.mid_block.attentions.0",
                f"{route}/encoder.mid_block.attentions.0",
                mid.attentions[0],
                batch,
            ),
            self._port(
                "resnet_block2_d",
                "encoder.mid_block.resnets.1",
                f"{route}/encoder.mid_block.resnets.1",
                mid.resnets[1],
                batch,
            ),
        ]
        built = (blocks, mid_steps)
        self._routes[key] = built
        return built

    def encode_decomposed(self, pixel_values) -> ttnn.Tensor:
        """The Call-4 encode chain: native ``conv_in`` -> ``resnet_block2_d`` x2 +
        ``downsample2_d`` (down 0) -> ``down_encoder_block2_d`` (down 1..3) ->
        ``resnet_block2_d`` + ``attention`` + ``resnet_block2_d`` (mid) -> native
        tail -> ``quant_conv``.
        """

        def body(x):
            blocks, mid_steps = self._ensure_encode_decomposed(_leading(x))
            x = self._enc_conv_in(x)
            for block in blocks:
                x = block(x)
            for step in mid_steps:
                x = step(x)
            self._consume(x)
            return self._enc_tail(x)

        return self._chunked("encode", pixel_values, body)

    # --------------------------------------------------------------- route: decode

    def _decoder_composite(self, stub: str, route: str, batch: int):
        key = (f"{route}:{stub}", int(batch))
        port = self._routes.get(key)
        if port is None:
            port = self._port(stub, "decoder", f"{route}/decoder", self.hf_view("decoder"), batch)
            self._routes[key] = port
        return port

    def decode(self, latents) -> ttnn.Tensor:
        """``decoder`` port -> image ``(B, 3, H, W)``.

        ``latents`` are POST-``post_quant_conv``, and already un-normalised and
        unpatchified by the pipeline -- i.e. exactly what ``diffusers.Decoder``
        itself is called with inside ``AutoencoderKLFlux2._decode``.  ``B``
        independent latents in, ``B`` independent images out, run in chunks of the
        width one program carries (see ``_CHUNK_START``) and joined on device.
        """
        return self._chunked(
            "decode",
            latents,
            lambda x: self._decoder_composite("decoder", "decode", _leading(x))(x),
        )

    def decode_alias(self, latents) -> ttnn.Tensor:
        """``decoder_head`` port, same input contract as ``decode``.

        ``decoder_head``'s capture manifest records ``submodule_path = decoder``:
        the same ``diffusers.Decoder``, brought up as a second component, and the
        edit head's own decode.
        """
        return self._chunked(
            "decode",
            latents,
            lambda x: self._decoder_composite("decoder_head", "decode_alias", _leading(x))(x),
        )

    def _ensure_decode_decomposed(self, batch: int = 1):
        key = ("decode_decomposed", int(batch))
        built = self._routes.get(key)
        if built is not None:
            return built
        decoder = self.hf.decoder
        route = "decode_decomposed"

        mid = decoder.mid_block
        mid_steps = [
            self._port(
                "mlp",
                "decoder.mid_block.resnets.0",
                f"{route}/decoder.mid_block.resnets.0",
                mid.resnets[0],
                batch,
            ),
            self._port(
                "self_attention",
                "decoder.mid_block.attentions.0",
                f"{route}/decoder.mid_block.attentions.0",
                mid.attentions[0],
                batch,
            ),
            self._port(
                "mlp",
                "decoder.mid_block.resnets.1",
                f"{route}/decoder.mid_block.resnets.1",
                mid.resnets[1],
                batch,
            ),
        ]

        up = decoder.up_blocks
        # fill the ladder `__init__` laid out, in place (see _ensure_encode_decomposed)
        blocks = self.up_blocks if not any(b.steps for b in self.up_blocks) else _fresh_ladder(self.up_blocks)
        blocks[0].steps = [
            self._port(
                "up_decoder_block2_d",
                "decoder.up_blocks.0",
                f"{route}/decoder.up_blocks.0",
                _block_view(up[0], self.layers),
                batch,
            )
        ]
        blocks[1].steps = [
            self._port(
                "layer",
                "decoder.up_blocks.1",
                f"{route}/decoder.up_blocks.1",
                _block_view(up[1], self.layers),
                batch,
            )
        ]

        keep = _clamped(self.layers, len(up[2].resnets))
        up2_steps = [
            self._port(
                "mlp",
                f"decoder.up_blocks.2.resnets.{j}",
                f"{route}/decoder.up_blocks.2.resnets.{j}",
                up[2].resnets[j],
                batch,
            )
            for j in range(keep)
        ]
        up2_steps.append(
            self._port(
                "upsample2_d",
                "decoder.up_blocks.2.upsamplers.0",
                f"{route}/decoder.up_blocks.2.upsamplers.0",
                up[2].upsamplers[0],
                batch,
            )
        )
        blocks[2].steps = up2_steps
        blocks[3].steps = [
            self._port(
                "up_decoder_block2_d",
                "decoder.up_blocks.3",
                f"{route}/decoder.up_blocks.3",
                _block_view(up[3], self.layers),
                batch,
            )
        ]

        built = (mid_steps, blocks)
        self._routes[key] = built
        return built

    def decode_decomposed(self, latents) -> ttnn.Tensor:
        """The Call-4 decode chain: native ``conv_in`` ->
        ``mlp`` + ``self_attention`` + ``mlp`` (mid) -> ``up_decoder_block2_d``
        (up 0) -> ``layer`` (up 1) -> ``mlp`` x3 + ``upsample2_d`` (up 2) ->
        ``up_decoder_block2_d`` (up 3) -> native tail.

        ``mlp`` / ``self_attention`` / ``layer`` are the scaffold's generic
        transformer roles; this checkpoint is a convnet, so they are bound to the
        decoder's residual block, its mid attention and its repeating up block
        respectively (see each stub's docstring).
        """

        def body(x):
            mid_steps, blocks = self._ensure_decode_decomposed(_leading(x))
            x = self._dec_conv_in(x)
            for step in mid_steps:
                x = step(x)
            for block in blocks:
                x = block(x)
            self._consume(x)
            return self._dec_tail(x)

        return self._chunked("decode", latents, body)

    # ----------------------------------------------------------- trace: pin / step

    # The trace contract: ``pin_*`` does every scrap of host work up front -- stage
    # the input into a device buffer that stays allocated, build and bind the port,
    # and run ONE full forward so that everything the tt_dit primitives build
    # lazily is already built.  Concretely that first forward is what populates
    # ``Conv2d._prepared_weight`` / ``_prepared_bias`` (``ttnn.prepare_conv_weights``
    # runs on the host), ``CCLManager``'s per-shape all-gather / reduce-scatter
    # ping-pong buffers (each allocated from a ``torch.empty``) and its per-axis
    # semaphores.  All of those are cached by shape, and ``*_step`` re-runs the
    # identical shapes, so the second call touches none of them.
    #
    # ``*_step`` then reads ONLY the resident buffers: no ``ttnn.from_torch``, no
    # per-call ``ttnn.zeros`` / ``ttnn.arange``, and no torch at all.

    def _pin(self, route: str, family: str, x, build, *, layout=ttnn.TILE_LAYOUT) -> dict:
        """Pin ``route`` at the input's batch, in chunks of what one program carries.

        The port is built at the CHUNK width, not at the full batch, for the same
        reason the eager routes chunk (see `_CHUNK_START`): the conv halo a B=32
        program asks for exceeds L1_SMALL.  The pinned chunk width goes into the
        resident dict so ``*_step`` re-runs the identical shapes -- which is what makes
        the step traceable, since the chunk loop is a FIXED, unrolled count.  The port
        itself is built ONCE, here, and its weights stay resident for every step.
        """
        resident = self._stage(x, layout=layout)
        batch = _leading(resident)
        width = self.chunk_width(family, batch)
        while True:
            port = build(width)
            try:
                warm = self._step_chunks(port, resident, width)
                break
            except (RuntimeError, ValueError) as exc:
                narrower = self._narrower(width, batch)
                if narrower is None or not _is_too_wide(exc):
                    raise
                print(
                    f"vae {family}: pinning a chunk of {width} of {batch} did not fit "
                    f"({type(exc).__name__}); re-pinning at {narrower}",
                    flush=True,
                )
                self._drop_built_at(width)
                self._chunk_learned[family] = narrower
                width = narrower
        return {
            "route": route,
            "input": resident,
            "port": port,
            "output": warm,
            "batch": batch,
            "chunk": width,
        }

    def _step_chunks(self, port, x: ttnn.Tensor, width: int) -> ttnn.Tensor:
        """One forward over the whole resident batch, in chunks of ``width``.

        Pure device: ``ttnn.slice`` reads the resident buffer (so overwriting it in
        place between steps still reaches the step) and ``ttnn.concat`` joins the
        results, with no host op and no per-call staging.
        """
        return self._map_over_leading(port, x, width)

    def pin_encode(self, pixel_values) -> dict:
        """Pin the COMPOSITE ``encoder`` route for tracing, at the input's batch.

        ``pixel_values`` may be host torch or an already-staged ttnn NCHW tensor.
        The returned dict holds the persistent input buffer under ``"input"``
        (overwrite it in place between steps), the bound port under ``"port"``,
        the warm-up output under ``"output"`` and the pinned batch under
        ``"batch"``.  The batch is part of what is pinned: the warm-up forward is
        what bakes each conv's prepared weight, and that weight is a function of
        the batch, so a step at another batch is a different program.
        """
        return self._pin("encode", "encode", pixel_values, lambda b: self._encoder_composite("encoder", "encode", b))

    def encode_step(self, resident: dict) -> ttnn.Tensor:
        """One host-op-free ``encoder`` forward -> pre-quant-conv moments ``(B, 64, h, w)``."""
        return self._step_chunks(resident["port"], resident["input"], resident["chunk"])

    def pin_decode(self, latents) -> dict:
        """Pin the COMPOSITE ``decoder`` route for tracing, at the input's batch.

        ``latents`` are POST-``post_quant_conv``, the same contract as ``decode``.
        """
        return self._pin("decode", "decode", latents, lambda b: self._decoder_composite("decoder", "decode", b))

    def decode_step(self, resident: dict) -> ttnn.Tensor:
        """One host-op-free ``decoder`` forward -> image ``(B, 3, H, W)``."""
        return self._step_chunks(resident["port"], resident["input"], resident["chunk"])

    # -------------------------------------------------------------- block handles

    def blocked_group_norms(self, batch: int) -> list[str]:
        """Every stub-owned ``GroupNorm`` this stage cannot hand a chunk count to.

        The diagnosis a batched route reports when it dies in ``ttnn.group_norm``
        -- see ``unchunkable_group_norms`` for what the fix is and where it lives.
        Empty at ``batch == 1``.
        """
        out = []
        for (stub, path, built_batch), port in self._built.items():
            if built_batch != int(batch):
                continue
            out += [f"{stub}@{path}.{n}" for n in unchunkable_group_norms(port, batch)]
        return sorted(out)

    def built_down_blocks(self, batch: int = 1) -> list:
        """The 4 encoder stages of the DECOMPOSED encode route, STAGED, as
        ``DownBlockPort``s (stage 0 is the resnet/resnet/downsampler chain, 1..3
        are single ``down_encoder_block2_d`` ports).  NCHW in, NCHW out.

        At the first batch built, ``self.down_blocks`` is the same list; this
        builds its ports first.
        """
        blocks, _ = self._ensure_encode_decomposed(batch)
        return list(blocks)

    def built_up_blocks(self, batch: int = 1) -> list:
        """The 4 decoder stages of the DECOMPOSED decode route, STAGED, as
        ``UpBlockPort``s (stage 2 is the ``mlp`` x3 + ``upsample2_d`` chain)."""
        _, blocks = self._ensure_decode_decomposed(batch)
        return list(blocks)

    @property
    def down_blocks_blockwise(self) -> list:
        """The 4 ``down_encoder_block2_d`` stages of the BLOCKWISE encode route."""
        _, blocks, _ = self._ensure_encode_blockwise()
        return list(blocks)

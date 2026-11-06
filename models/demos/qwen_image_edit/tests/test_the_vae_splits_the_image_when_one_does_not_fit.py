# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""How the VAE lays a batch of images over the mesh has to follow the image size, not a fixed choice.

batch_parallel="full" gives every chip a WHOLE image and pads a batch smaller than the mesh with zero images,
so its per-chip activation never falls below one image however small the per-program batch is. That is the
fastest layout -- no conv exchanges halos -- and it is right for the 256 x 256 area the PCC gate runs, where a
whole image's first stage is 24 MB. At the 1024 x 1024 area the pipeline targets that same stage is 384 MB and
a full-resolution encode died on it with 2.89 GB free, at every batch size, because no batch makes it smaller.
The image itself has to be partitioned over the mesh there instead.

Host only: `_image_split` reads nothing but the mesh shape, so a stand-in with a shape is device enough.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from models.demos.qwen_image_edit.tt.inputs import DEFAULT_AREA, PIPELINE_AREA
from models.demos.qwen_image_edit.tt.vae import DECODE_MAX_BATCH, ENCODE_MAX_BATCH, STAGE_BYTES_PER_CHIP, _image_split

MESH = SimpleNamespace(shape=(2, 4))  # the T3K the gate and the demo both run
ONE_CHIP = SimpleNamespace(shape=None)  # mesh_shape falls back to 1 x 1
DIM = 96  # base_dim: the channels both stacks carry at the image's own resolution
SCALE = 8  # 2 ** (len(dim_mult) - 1): the decoder partitions a latent grid this much smaller per side
AREAS = (256, 512, 1024, 2048)

# (name, images per program when a whole image fits, grid it partitions) of the two stacks
STACKS = (("encode", ENCODE_MAX_BATCH, 1), ("decode", DECODE_MAX_BATCH, SCALE))

# share of one image each split leaves on a chip, on a 2 x 4 mesh
SHARES = {"full": 1.0, True: 1.0 / 4, False: 1.0 / 8}


def _stage_bytes(area, parallel):
    """Bytes of the image-resolution activation one chip holds under this split, at a square `area`."""
    return area * area * DIM * 4 * SHARES[parallel]


def _split(area, stack, device=MESH):
    _, max_batch, downsample = stack
    return _image_split(device, DIM, (area, area), max_batch, downsample=downsample)


@pytest.mark.parametrize("area", AREAS)
@pytest.mark.parametrize("stack", STACKS, ids=[s[0] for s in STACKS])
def test_the_chosen_split_fits_the_budget_or_is_the_finest_there_is(area, stack):
    parallel, _ = _split(area, stack)
    assert _stage_bytes(area, parallel) <= STAGE_BYTES_PER_CHIP or parallel is False


@pytest.mark.parametrize("area", AREAS)
@pytest.mark.parametrize("stack", STACKS, ids=[s[0] for s in STACKS])
def test_it_never_splits_finer_than_it_has_to(area, stack):
    """Every split that partitions the image costs halo exchanges, so a coarser one that fits must win."""
    parallel, _ = _split(area, stack)
    for p in [p for p, s in SHARES.items() if s > SHARES[parallel]]:
        assert _stage_bytes(area, p) > STAGE_BYTES_PER_CHIP, f"{stack[0]} at {area}: {p!r} fits and is coarser"


@pytest.mark.parametrize("stack", STACKS, ids=[s[0] for s in STACKS])
def test_the_gate_s_own_area_keeps_the_layout_it_measured(stack):
    """The whole point of keeping "full": at DEFAULT_AREA both stacks run the layout their per-program
    batches were measured against, so the gate's timings and its PCC threshold still mean what they meant."""
    assert _split(DEFAULT_AREA, stack) == ("full", stack[1])


@pytest.mark.parametrize("stack", STACKS, ids=[s[0] for s in STACKS])
def test_the_declared_area_partitions_the_image(stack):
    parallel, batch = _split(PIPELINE_AREA, stack)
    assert parallel != "full", "a whole 1024 x 1024 image per chip is the allocation that failed"
    assert batch >= 1


@pytest.mark.parametrize("stack", STACKS, ids=[s[0] for s in STACKS])
def test_no_image_size_is_the_layout_it_has_always_had(stack):
    """A caller that does not say what it will run -- the e2e gate, the perf tests, the stack probes --
    gets exactly the build it got before this choice existed."""
    _, max_batch, downsample = stack
    assert _image_split(MESH, DIM, None, max_batch, downsample=downsample) == ("full", max_batch)


@pytest.mark.parametrize("area", AREAS)
@pytest.mark.parametrize("stack", STACKS, ids=[s[0] for s in STACKS])
def test_one_chip_has_nothing_to_partition(area, stack):
    assert _split(area, stack, device=ONE_CHIP) == ("full", stack[1])


@pytest.mark.parametrize("stack", STACKS, ids=[s[0] for s in STACKS])
def test_a_bigger_image_is_never_given_a_coarser_split(stack):
    """Monotonic in the image size: growing the image can only push the layout finer."""
    chosen = [SHARES[_split(a, stack)[0]] for a in AREAS]
    assert chosen == sorted(chosen, reverse=True), dict(zip(AREAS, chosen))


def test_the_finest_split_runs_one_image_per_program():
    """The Wan causal conv fuses its temporal front pad into the H halo exchange, and that fused
    neighbor_pad only takes B == 1 -- so the split that partitions H cannot batch."""
    big = 1 << 16  # far past anything that fits: forces the finest split
    for stack in STACKS:
        assert _split(big, stack) == (False, 1), stack[0]


def test_a_grid_the_mesh_cannot_divide_is_never_split_along_it():
    """The decoder partitions the LATENT grid. An image's sides are multiples of the VAE's 16-pixel step, so
    the encoder can always take a 2 x 4 mesh, but an eighth of such a side need not divide by 4 -- and
    BatchSplit.scatter asserts on exactly that. The chooser has to see it rather than fail at the first
    conv: 1072 is a width calculate_dimensions produces, and 1072 / 8 = 134 is not a multiple of 4."""
    h, w = 1072, 1072
    _, max_batch, _ = STACKS[1]
    assert (w // SCALE) % MESH.shape[1] != 0, "pick a width whose latent the mesh columns cannot divide"
    parallel, batch = _image_split(MESH, DIM, (h, w), max_batch, downsample=SCALE)
    assert parallel == "full" and batch == max_batch, "it offered a split the latent grid cannot take"
    # the encoder partitions the image itself, whose sides a 2 x 4 mesh always divides, so that same image
    # still gets a split there -- 1072 x 1072 is well past one whole image per chip
    rows, cols = MESH.shape
    parallel, _ = _image_split(MESH, DIM, (h, w), STACKS[0][1])
    assert parallel != "full"
    assert h % rows == 0 and w % cols == 0

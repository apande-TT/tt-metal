# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A batch whose working set does not fit a chip runs as groups of it, one program each.

No stage of this pipeline has a cross-sample term -- each image is edited on its own -- so the batch is the
one dimension that can always be cut. It has to be cut at a full-resolution edit: at 1024 x 1024 every stage
carries roughly 14x what it does at the 256 x 256 area the PCC gate runs, and a batch-32 edit died twice over
before the first denoise step, each time on a buffer near 1.94 GB with a 2.9 GB working budget.

The cut is a SLICE of the encoded batch, not a smaller `encode_inputs` call, because the VL processor pads a
batch of prompts to the longest in that batch: re-encoding a group would hand its samples different inputs
from the ones the whole batch gives them. These tests pin that -- every per-sample field follows the indices,
the images' concatenated patch rows are cut on their own grid boundaries, and nothing shared is disturbed.

Host only: EncodedInputs is a plain dataclass of host tensors.
"""

from __future__ import annotations

import pytest
import torch

from models.demos.qwen_image_edit.tt.inputs import EncodedInputs, sample_groups

GRID = (1, 2, 3)  # (t, h, w): 6 patch rows per image
CHANNELS = 7  # stand-in for the processor's 1176 patch features
E2E_BATCH = 32  # the batch the gate and the demo run


def _encoded(batch, grids=None):
    """A stand-in encoding whose every per-sample value says which sample it came from."""
    grids = grids or [GRID] * batch
    rows = [t * h * w for t, h, w in grids]
    return EncodedInputs(
        batch=batch,
        seeds=[100 + i for i in range(batch)],
        prompts=[f"prompt {i}" for i in range(batch)],
        images=[f"image {i}" for i in range(batch)],
        width=64,
        height=64,
        num_inference_steps=3,
        vl={
            "input_ids": torch.arange(batch * 5).reshape(batch, 5),
            "attention_mask": torch.ones(batch, 5, dtype=torch.long),
            # the images' patch rows CONCATENATED, each row carrying its sample's index
            "pixel_values": torch.cat([torch.full((r, CHANNELS), float(i)) for i, r in enumerate(rows)]),
            "image_grid_thw": torch.tensor(grids),
        },
        vae_image=torch.arange(batch * 3.0).reshape(batch, 3, 1, 1, 1),
        latents=torch.arange(batch * 4.0).reshape(batch, 4, 1),
        timesteps=torch.tensor([1.0, 2.0, 3.0]),
        sigmas=torch.tensor([1.0, 0.5, 0.0]),
        latent_hw=(8, 8),
        img_shapes=[(1, 4, 4), (1, 8, 8)],
    )


# ---- the groups ------------------------------------------------------------------------------------


@pytest.mark.parametrize("per", [None, 0, 1, 2, 3, 5, 8, 31, 32, 64])
def test_every_sample_runs_exactly_once_and_in_order(per):
    assert [i for g in sample_groups(E2E_BATCH, per) for i in g] == list(range(E2E_BATCH))


@pytest.mark.parametrize("per", [1, 2, 3, 5, 8, 31, 32, 64])
def test_no_group_is_larger_than_one_program_carries(per):
    assert all(len(g) <= per for g in sample_groups(E2E_BATCH, per))


@pytest.mark.parametrize("per", [None, 0, E2E_BATCH, E2E_BATCH + 1])
def test_a_batch_that_fits_stays_one_program(per):
    """The gate's own batch is untouched: asking for no cut, or for one at least as wide, runs it as it was."""
    assert sample_groups(E2E_BATCH, per) == [list(range(E2E_BATCH))]


def test_the_last_group_takes_the_remainder():
    assert [len(g) for g in sample_groups(10, 4)] == [4, 4, 2]


# ---- the slice -------------------------------------------------------------------------------------


def test_every_per_sample_field_follows_the_indices():
    e = _encoded(4)
    idx = [2, 0]
    s = e.subset(idx)
    assert s.batch == len(idx)
    assert s.seeds == [e.seeds[i] for i in idx]
    assert s.prompts == [e.prompts[i] for i in idx]
    assert s.images == [e.images[i] for i in idx]
    assert torch.equal(s.vl["input_ids"], e.vl["input_ids"][idx])
    assert torch.equal(s.vl["attention_mask"], e.vl["attention_mask"][idx])
    assert torch.equal(s.vl["image_grid_thw"], e.vl["image_grid_thw"][idx])
    assert torch.equal(s.vae_image, e.vae_image[idx])
    assert torch.equal(s.latents, e.latents[idx])


def test_the_patch_rows_are_cut_on_their_image_boundaries():
    """pixel_values is one row per PATCH, not per sample: a sample's rows are a run of them."""
    e = _encoded(4)
    s = e.subset([3, 1])
    rows = GRID[0] * GRID[1] * GRID[2]
    assert s.vl["pixel_values"].shape == (2 * rows, CHANNELS)
    assert torch.equal(s.vl["pixel_values"][:rows], torch.full((rows, CHANNELS), 3.0))
    assert torch.equal(s.vl["pixel_values"][rows:], torch.full((rows, CHANNELS), 1.0))


def test_the_cut_holds_when_the_images_are_not_on_one_grid():
    """The boundaries come from each image's own grid, so images of different sizes still cut correctly."""
    grids = [(1, 2, 2), (1, 3, 3), (1, 1, 4)]
    e = _encoded(3, grids)
    for i, (t, h, w) in enumerate(grids):
        pix = e.subset([i]).vl["pixel_values"]
        assert pix.shape == (t * h * w, CHANNELS)
        assert bool((pix == float(i)).all())


@pytest.mark.parametrize("field", ["width", "height", "num_inference_steps", "latent_hw", "img_shapes"])
def test_what_the_batch_shares_is_not_disturbed(field):
    e = _encoded(4)
    assert getattr(e.subset([1]), field) == getattr(e, field)


def test_the_schedule_is_not_disturbed():
    e = _encoded(4)
    s = e.subset([1])
    assert torch.equal(s.timesteps, e.timesteps) and torch.equal(s.sigmas, e.sigmas)


def test_slicing_leaves_the_batch_it_came_from_alone():
    e = _encoded(4)
    before = e.vl["pixel_values"].clone()
    e.subset([2, 0])
    assert e.batch == 4 and len(e.seeds) == 4
    assert torch.equal(e.vl["pixel_values"], before)


def test_taking_every_sample_in_order_is_the_batch_itself():
    e = _encoded(4)
    s = e.subset(list(range(4)))
    assert s.batch == e.batch and s.seeds == e.seeds
    for k, t in e.vl.items():
        assert torch.equal(s.vl[k], t)
    assert torch.equal(s.vae_image, e.vae_image) and torch.equal(s.latents, e.latents)


def test_the_groups_of_a_batch_rebuild_it():
    """What the demo relies on: run the groups, concatenate, and you have the batch's own samples in order."""
    e = _encoded(10)
    for per in (1, 3, 4, 10):
        parts = [e.subset(g) for g in sample_groups(e.batch, per)]
        assert sum(p.batch for p in parts) == e.batch
        assert [s for p in parts for s in p.seeds] == e.seeds
        assert torch.equal(torch.cat([p.latents for p in parts]), e.latents)
        assert torch.equal(torch.cat([p.vl["pixel_values"] for p in parts]), e.vl["pixel_values"])

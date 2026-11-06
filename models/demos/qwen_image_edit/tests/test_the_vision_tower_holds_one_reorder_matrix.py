# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The vision tower's token reorder is one matrix, not one per image of the batch.

The tower runs its patches in window-major order, which is a permutation of the input rows. It applies that
permutation as a product with a 0/1 row-selection matrix, and unapplies it on the merged rows the same way.
The order comes from the grid alone -- `_vision_metadata` reads nothing but (t, h, w) -- so every image of a
batch selects by the SAME matrix, and `consts` used to upload one copy of it per image anyway, so that the
product's batch dims lined up. A batched matmul broadcasts a batch-1 first operand, so they need not.

That was free at the area the PCC gate runs and fatal at the one the pipeline targets: the matrix is
s_pad ** 2 entries, and s_pad follows the image. A batch-32 edit at 1024 x 1024 died uploading it, asking for
1,938,817,024 B with the whole chip's spare DRAM an order of magnitude short -- and the live caller
(`TtQwenTextEncoder.prepare`) uploads its pixel rows already in window order and sets `pixels_in_window_order`,
so the forward never even multiplied by the copies it died building.

Host only: nothing here needs a device -- `_one_hot` is numpy, and the rest is a signature and arithmetic.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from models.demos.qwen_image_edit_text_encoder._stubs.attention import pad_to_tile
from models.demos.qwen_image_edit_text_encoder._stubs.encoder_stack import _one_hot
from models.demos.qwen_image_edit_text_encoder._stubs.vision_transformer_pretrained_model import TtVisionTransformer

# The condition image's patch grid at each edit area, and the token count it gives. The VL processor resizes to
# a whole number of merged patches, so the grid -- not the requested area -- is what sets the matrix's side.
GRIDS = {256: (1, 20, 20), 1024: (1, 74, 74)}  # (t, h, w); 1024's is the grid the failing run reported
BATCHES = (1, 2, 8, 16, 32)
FLOAT32 = 4  # the dtype consts uploads the matrix as on the non-precise path, and the one the OOM counted


def _side(grid):
    """The matrix's side at this grid: one row per padded token."""
    t, h, w = grid
    return pad_to_tile(t * h * w)


def _rows(n):
    """A row selection of n rows -- a reversal, which is a permutation like the tower's own."""
    return np.arange(n)[::-1].copy()


# ---- the contract the broadcast rests on -------------------------------------------------------------


def test_the_matrix_selects_the_rows_it_names():
    n = 64
    rows = _rows(n)
    x = np.random.default_rng(0).standard_normal((n, 5)).astype(np.float32)
    got = _one_hot(rows, n)[0, 0] @ x
    assert np.array_equal(got, x[rows])


def test_rows_past_the_selection_stay_zero():
    """s_pad > s: the padded tail selects nothing, so the product leaves it zero rather than duplicating a row."""
    s, s_pad = 50, pad_to_tile(50)
    assert s_pad > s
    x = np.random.default_rng(1).standard_normal((s_pad, 3)).astype(np.float32)
    got = _one_hot(_rows(s), s_pad)[0, 0] @ x
    assert np.array_equal(got[:s], x[_rows(s)])
    assert not got[s:].any()


def test_the_same_matrix_serves_every_image():
    """The claim the single copy rests on: broadcasting one matrix over a batch is what per-image copies did."""
    n, batch = 32, 5
    rows = _rows(n)
    one = _one_hot(rows, n)
    per_image = np.repeat(one, batch, axis=0)
    x = np.random.default_rng(2).standard_normal((batch, 1, n, 4)).astype(np.float32)
    assert np.array_equal(one @ x, per_image @ x)


# ---- the shape and the cost --------------------------------------------------------------------------


def test_the_matrix_is_one_copy():
    for grid in GRIDS.values():
        n = _side(grid)
        assert _one_hot(_rows(n), n).shape == (1, 1, n, n)


def test_the_constants_do_not_take_a_batch():
    """The guard against reintroducing the per-image copies: nothing about them is a function of the batch,
    so `consts` has nothing to tell it what the batch is."""
    assert list(inspect.signature(TtVisionTransformer.consts).parameters) == ["self", "grid"]


@pytest.mark.parametrize("batch", BATCHES)
def test_the_cost_does_not_follow_the_batch(batch):
    n = _side(GRIDS[1024])
    one = _one_hot(_rows(n), n).nbytes
    assert one == n * n * FLOAT32
    assert np.repeat(_one_hot(_rows(n), n), batch, axis=0).nbytes == batch * one


def test_a_per_image_copy_is_what_the_failing_run_asked_for():
    """Ties the arithmetic to the observed failure: the batch-32 edit encodes 16 images per mesh row (the tower
    is data-parallel over the two rows), and 16 copies at that grid is the buffer the allocator refused."""
    n = _side(GRIDS[1024])
    assert 16 * n * n * FLOAT32 == 1938817024


def test_the_gate_s_area_never_paid_enough_to_notice():
    """Why this survived to now: at the area the PCC gate runs, every copy of the batch is a few MB."""
    n = _side(GRIDS[256])
    assert max(BATCHES) * n * n * FLOAT32 < 100 * 1024 * 1024

"""A SegmentPlan must describe exactly the mask it replaces, chunk by chunk.

The vision tower used to upload a [s_pad, s_pad] additive mask per attention kind and add it to scores of
that same shape. Both cost s_pad ** 2: 61 MB for the pair at the 256 x 256 area the PCC gate runs, but
969 MB at the 1024 x 1024 area the pipeline targets -- more than a chip has free once the weights are
resident, so a full-resolution edit died on the FIRST of the two masks, before one score was computed,
and did so at every batch size because neither tensor depends on the batch.

SegmentPlan carries the cu_seqlens those masks are built from and hands the attention row chunks instead.
This checks the replacement is an identity and not an approximation, on the real window geometry that
_vision_metadata produces, which is what makes it safe to run under an unchanged PCC threshold:

  * every chunk bound is a tile multiple, since a window ends wherever the image's last column falls and
    a tiled tensor cannot be sliced anywhere else,
  * the chunks partition the rows -- none computed twice, none skipped,
  * no row loses an unmasked column to the chunk it lands in, which is what would silently change an
    attention result rather than fail,
  * each chunk's mask equals the block the dense matrix holds there, and
  * at the gate's own resolution the whole sequence is one chunk, so that path computes what it did.

Host only: no device and no checkpoint, so it runs wherever the stubs import.
"""

from __future__ import annotations

import numpy as np
import pytest

from models.demos.qwen_image_edit_text_encoder._stubs.attention import TILE, SegmentPlan, block_mask, pad_to_tile
from models.demos.qwen_image_edit_text_encoder._stubs.encoder_stack import _vision_metadata

# Qwen2.5-VL's vision geometry, read off the tower rather than retyped as numbers with no source.
MERGE, WINDOW, PATCH = 2, 112, 14
HEADS = 4  # heads per chip at TP 4; only ever a divisor of the score budget

# (grid_h, grid_w) of the condition image at the 256, 512 and 1024 pixel areas. The first is what the PCC
# gate runs; the last is the area the pipeline declares (inputs.PIPELINE_AREA).
GRIDS = {"256": (18, 18), "512": (36, 36), "1024": (104, 104)}


def _plans(grid):
    """The two plans a vision forward builds for this grid, with the dense mask each one stands for."""
    gh, gw = grid
    _, _, cu_window, cu_full = _vision_metadata(np.array([(1, gh, gw)]), MERGE, WINDOW, PATCH)
    s = gh * gw
    s_pad = pad_to_tile(s)
    for cu in (cu_full, cu_window):
        yield SegmentPlan(cu, s, s_pad), block_mask(cu, s, s_pad)[0, 0], s, s_pad


@pytest.mark.parametrize("area", sorted(GRIDS))
def test_every_chunk_bound_is_a_tile_multiple(area):
    for plan, _, _, _ in _plans(GRIDS[area]):
        for chunk in plan.chunks(HEADS):
            assert all(b % TILE == 0 for b in chunk), "a tiled tensor cannot be sliced at %r" % (chunk,)


@pytest.mark.parametrize("area", sorted(GRIDS))
def test_the_chunks_partition_the_rows(area):
    for plan, _, _, s_pad in _plans(GRIDS[area]):
        seen = np.zeros(s_pad, dtype=bool)
        for r0, r1, _, _ in plan.chunks(HEADS):
            assert not seen[r0:r1].any(), "rows %d..%d computed twice" % (r0, r1)
            seen[r0:r1] = True
        assert seen.all(), "some rows would never be computed"


@pytest.mark.parametrize("area", sorted(GRIDS))
def test_no_row_loses_an_unmasked_column(area):
    """The failure this protects against is silent: a dropped column changes a softmax, it does not raise."""
    for plan, dense, s, _ in _plans(GRIDS[area]):
        for r0, r1, c0, c1 in plan.chunks(HEADS):
            rows = dense[r0 : min(r1, s)]
            if not rows.size:
                continue  # the padded tail, which no real row reads
            open_cols = np.where((rows == 0.0).any(axis=0))[0]
            assert open_cols.min() >= c0 and open_cols.max() < c1, "rows %d..%d lose a column" % (r0, r1)


@pytest.mark.parametrize("area", sorted(GRIDS))
def test_each_chunk_mask_is_the_dense_block(area):
    for plan, dense, s, s_pad in _plans(GRIDS[area]):
        for r0, r1, c0, c1 in plan.chunks(HEADS):
            got = block_mask(plan.cu_seqlens, s, s_pad, rows=(r0, r1), cols=(c0, c1))[0, 0]
            assert np.array_equal(got, dense[r0:r1, c0:c1]), "chunk %d..%d differs from the mask" % (r0, r1)


def test_the_whole_mask_is_still_the_default():
    """No sub-range asked for = the matrix the tower used to upload, unchanged."""
    for plan, dense, s, s_pad in _plans(GRIDS["256"]):
        whole = block_mask(plan.cu_seqlens, s, s_pad)
        assert whole.shape == (1, 1, s_pad, s_pad)
        assert np.array_equal(whole[0, 0], dense)


def test_the_gate_s_own_area_is_one_chunk():
    """Why an unchanged PCC threshold is safe: at the area the gate runs, the plan does not divide anything,
    so the attention performs the same single pass it did before the plan existed."""
    for plan, _, _, s_pad in _plans(GRIDS["256"]):
        chunks = plan.chunks(HEADS)
        assert chunks == [(0, s_pad, 0, s_pad)], "the gate's area should not be chunked: %r" % (chunks,)


def test_the_declared_area_is_divided_far_below_the_dense_cost():
    """And why the plan exists at all: at the pipeline's own area the scores a chunk holds are a small
    fraction of the matrix, which is the difference between running and an out-of-memory abort."""
    for plan, _, _, s_pad in _plans(GRIDS["1024"]):
        chunks = plan.chunks(HEADS)
        assert len(chunks) > 1
        peak = max((r1 - r0) * (c1 - c0) for r0, r1, c0, c1 in chunks)
        assert peak * 8 < s_pad * s_pad, "peak %d is not far below the dense %d" % (peak, s_pad * s_pad)


def test_chunks_of_one_shape_share_one_mask():
    """The masks are keyed by what they contain, not where they sit. Keying them by position would hold
    more bytes of mask than the dense matrix the plan replaces."""
    for plan, _, _, _ in _plans(GRIDS["1024"]):
        chunks = plan.chunks(HEADS)
        assert len({plan._pattern(c) for c in chunks}) < len(chunks)

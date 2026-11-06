# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The VAE mid-block's precise attention must bound the scores it holds, and leave the gate's pass alone.

`_precise.matmul_bt` builds THREE [samples, n, n] float32 products and adds them, where n is H * W of the
latent grid -- and the block all-gathers that grid whatever spatial partition it runs under. At the 256 x 256
area the PCC gate runs n is 1024, so each product is 4 MB; at the 1024 x 1024 area the pipeline targets n is
16384 and each is 1 GB, which is why a full-resolution encode died on a 1,073,741,824 B DRAM buffer with
2.89 GB free. A softmax row reads only its own scores, so the attention runs in row chunks.

Host only: this checks the chunk ARITHMETIC (`_score_rows` and the cut `_row_chunked` makes with it). The
numbers it produces are the e2e PCC gate's business, and the gate is exactly the case that must not be cut
at all -- which is asserted here, because an unchunked pass is what keeps its threshold meaningful.
"""

from __future__ import annotations

import math

import pytest

from models.tt_dit.pipelines.qwen_image_edit_transformer._stubs import _precise
from models.tt_dit.pipelines.qwen_image_edit_vae._stubs.qwen_image_attention_block import _TILE, _score_rows

# The VAE downsamples by 8 in each spatial dimension (dim_mult has four entries, three of them resampling),
# so the attention's sequence is (area / 8) ** 2 tokens. 256 is DEFAULT_AREA, 1024 is PIPELINE_AREA.
AREAS = (256, 512, 1024, 2048)
SCORE_BYTES = 4  # float32
CHANNELS = 384  # base_dim * dim_mult[-1]: the mid block's width, and q's last dim


def _n(area):
    return (area // 8) ** 2


def _chunks(n, samples=1):
    """(start, stop) of every row chunk the attention runs, from _precise's own cut rule on the shape the
    block hands it: q is [samples, n, C] and it asks for the token dim."""
    shape = (samples, n, CHANNELS)
    cut = _precise.row_cuts(shape, _score_rows(n), dim=-2)
    if cut is None:
        return [(0, n)]
    ax, step = cut
    assert ax == 1, f"the token dim of {shape} is axis 1, not {ax}"
    return [(r0, min(n, r0 + step)) for r0 in range(0, n, step)]


@pytest.mark.parametrize("area", AREAS)
def test_a_chunk_never_holds_more_scores_than_the_budget(area):
    n = _n(area)
    for r0, r1 in _chunks(n):
        held = (r1 - r0) * n * SCORE_BYTES
        assert held <= _precise.CHUNK_ROWS_BYTES, f"area {area}: a chunk holds {held} B of scores"


@pytest.mark.parametrize("area", AREAS)
def test_the_chunks_partition_the_rows(area):
    n = _n(area)
    covered = _chunks(n)
    assert covered[0][0] == 0 and covered[-1][1] == n
    for (_, prev_end), (start, _) in zip(covered, covered[1:]):
        assert start == prev_end, f"area {area}: rows {prev_end}..{start} are computed twice or not at all"


@pytest.mark.parametrize("area", AREAS)
def test_every_cut_is_tile_aligned(area):
    """A tiled tensor cannot be sliced anywhere else, and ttnn.slice is how a chunk is taken."""
    n = _n(area)
    for r0, r1 in _chunks(n):
        assert r0 % _TILE == 0, f"area {area}: chunk starts at row {r0}"
        assert r1 % _TILE == 0 or r1 == n, f"area {area}: chunk ends at row {r1}"


def test_the_gate_s_own_area_is_not_chunked():
    """Why the unchanged PCC threshold still means what it meant: at DEFAULT_AREA the whole matrix is one
    chunk, so the block performs the single pass it performed before the chunking existed."""
    assert _chunks(_n(256)) == [(0, _n(256))]


def test_the_declared_area_is_cut_far_below_the_dense_cost():
    n = _n(1024)
    chunks = _chunks(n)
    assert len(chunks) > 1
    peak = max((r1 - r0) for r0, r1 in chunks) * n * SCORE_BYTES
    assert peak * 8 < n * n * SCORE_BYTES, f"peak {peak} B is not far below the dense {n * n * SCORE_BYTES} B"


@pytest.mark.parametrize("area", AREAS)
def test_more_samples_per_program_do_not_widen_a_chunk(area):
    """The scores are per sample, so a chunk of a 4-image program holds four times the rows' scores. The cut
    has to tighten with the sample count, not ignore it (cutting the BATCH would leave whole matrices)."""
    n = _n(area)
    for samples in (1, 2, 4, 8):
        for r0, r1 in _chunks(n, samples):
            held = samples * (r1 - r0) * n * SCORE_BYTES
            assert (
                held <= _precise.CHUNK_ROWS_BYTES or (r1 - r0) == _TILE
            ), f"area {area}, {samples} samples: a chunk holds {held} B of scores"


def test_the_budget_is_the_one_the_exact_linears_use():
    """Not a second number for the same job: the row-chunk budget lives in _precise with the linears."""
    assert _score_rows(_n(1024)) == max(_TILE, _precise.CHUNK_ROWS_BYTES // (SCORE_BYTES * _n(1024)) // _TILE * _TILE)


def test_the_cut_falls_on_the_token_dim_and_not_the_batch():
    """_row_chunked's default cuts whole samples off a batched tensor, which would leave every sample's
    n x n matrix intact. The block asks for the token dim; this is the assertion that it has to."""
    n = _n(1024)
    shape = (8, n, CHANNELS)
    assert _precise.row_cuts(shape, _score_rows(n))[0] == 0, "the default still cuts the batch"
    assert _precise.row_cuts(shape, _score_rows(n), dim=-2)[0] == 1


def test_a_narrower_budget_cuts_the_gate_s_area_too():
    """A guard on the test itself: the assertions above would pass trivially if _score_rows always returned
    more rows than there are. Halving the budget until the gate's own matrix is cut proves they bite."""
    n = _n(256)
    original = _precise.CHUNK_ROWS_BYTES
    try:
        _precise.CHUNK_ROWS_BYTES = n * SCORE_BYTES * _TILE  # exactly one tile of rows fits
        assert len(_chunks(n)) == math.ceil(n / _TILE) > 1
    finally:
        _precise.CHUNK_ROWS_BYTES = original

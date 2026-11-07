# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The pipeline has to be told its image size at BUILD time, and both entry points have to tell it the same.

The VAE lays its stacks out for the image size -- above the default area a whole image per chip stops
fitting and the image is partitioned over the mesh instead -- so `build_pipeline(image_hw=...)` decides it
before any input exists. The demo takes that from its own encoding; the perf entry point has no encoding
yet when it builds, so it asks `condition_size` directly. These pin that the two agree, and that the pair
is handed over as (height, width): swapped, a non-square condition image lays the stacks out for the wrong
axis and nothing in the shapes would complain until a conv halo did.

Host only, but it runs the real processor path once per area (the example image is cached).
"""

from __future__ import annotations

import pytest

from models.demos.qwen_image_edit.tt import inputs as I

AREAS = (I.DEFAULT_AREA, I.PIPELINE_AREA)  # the area the PCC gate runs and the one the pipeline targets


@pytest.fixture(scope="module")
def image():
    return I.load_example_image()


@pytest.mark.parametrize("area", AREAS)
def test_the_size_asked_for_is_the_size_encoded(area, image):
    """What makes it safe to ask before encoding: the answer is the encoding's own width and height."""
    enc = I.encode_inputs(1, area=area, num_inference_steps=2)
    assert I.condition_size(image, area) == (enc.width, enc.height)


@pytest.mark.parametrize("area", AREAS)
def test_the_size_fits_the_latent_stride(area, image):
    """Rounded DOWN to the packed-latent stride, so the latent grid is whole on both axes."""
    w, h = I.condition_size(image, area)
    stride = I.VAE_SCALE * 2
    assert w % stride == 0 and h % stride == 0
    assert (w, h) <= I.calculated_size(image, area)


def test_a_larger_area_never_gives_a_smaller_image(image):
    sizes = [I.condition_size(image, a) for a in (256, 320, 512, 768, 1024)]
    assert all(a[0] <= b[0] and a[1] <= b[1] for a, b in zip(sizes, sizes[1:]))


def test_the_perf_entry_point_hands_over_height_then_width(monkeypatch, image):
    """The order the build takes: image_hw is (height, width), as TtQwenVAE reads it."""
    import models.demos.qwen_image_edit.tests.e2e.test_image_edit_perf as perf

    for area in AREAS:
        monkeypatch.setattr(perf, "PERF_AREA", area)
        w, h = I.condition_size(image, area)
        assert perf._condition_hw() == (h, w)

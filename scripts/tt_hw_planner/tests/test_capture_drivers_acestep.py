"""Tests for ACE-Step capture drivers."""

from __future__ import annotations

from unittest import mock

import pytest


def test_is_acestep_model() -> None:
    from scripts.tt_hw_planner.capture_drivers import is_acestep_model

    m = mock.Mock()
    m.config.model_type = "acestep"
    assert is_acestep_model(m)
    m.config.model_type = "llama"
    assert not is_acestep_model(m)


def test_try_capture_drivers_delegates_to_acestep() -> None:
    from scripts.tt_hw_planner.capture_drivers import try_capture_drivers

    m = mock.Mock()
    m.config.model_type = "acestep"
    with mock.patch(
        "scripts.tt_hw_planner.capture_drivers.acestep.drive_acestep",
    ) as drive:
        ok, lines = try_capture_drivers(m, pixel_values=None)
    assert ok is True
    drive.assert_called_once()
    assert any("acestep" in line for line in lines)


def test_try_capture_drivers_unknown_model() -> None:
    from scripts.tt_hw_planner.capture_drivers import try_capture_drivers

    class _Generic:
        config = mock.Mock(model_type="llama")

        def forward(self, **kwargs):
            raise TypeError("no inputs")

    ok, _lines = try_capture_drivers(_Generic(), pixel_values=mock.Mock())
    assert ok is False

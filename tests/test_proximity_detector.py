"""
Unit tests for Module C — proximity_detector.detect_front_proximity()
======================================================================
Tests gate logic: zone filtering, bonnet rejection, area thresholds.
Skill: tdd-workflow
"""

import pytest
from module_c.proximity_detector import (
    detect_front_proximity,
    FRONT_GATE_Y_FRAC,
    PROXIMITY_CLOSE,
    PROXIMITY_VERY_CLOSE,
)
from module_c.guidance_states import PROX_NONE, PROX_CLOSE, PROX_VERY_CLOSE


# Test constants
W, H = 1920, 1080
ZONE_LEFT_X  = 600.0
ZONE_RIGHT_X = 1320.0


def _make_box(cx, cy, w, h):
    """Helper to create a (cx, cy, w, h) vehicle box tuple."""
    return (cx, cy, w, h)


class TestNoVehicle:
    """No vehicles → PROX_NONE."""

    def test_empty_list(self):
        result = detect_front_proximity([], ZONE_LEFT_X, ZONE_RIGHT_X, W, H)
        assert result == PROX_NONE


class TestOutsideEgoZone:
    """Vehicles outside ego lane boundaries → PROX_NONE."""

    def test_vehicle_far_left(self):
        # Vehicle centre at x=200, well left of zone_left_x=600
        box = _make_box(200, 400, 100, 80)
        result = detect_front_proximity([box], ZONE_LEFT_X, ZONE_RIGHT_X, W, H)
        assert result == PROX_NONE

    def test_vehicle_far_right(self):
        # Vehicle centre at x=1700, well right of zone_right_x=1320
        box = _make_box(1700, 400, 100, 80)
        result = detect_front_proximity([box], ZONE_LEFT_X, ZONE_RIGHT_X, W, H)
        assert result == PROX_NONE


class TestBonnetRejection:
    """Vehicles in the bonnet region (bottom of frame) → PROX_NONE."""

    def test_vehicle_in_bonnet_region(self):
        # Near bottom of frame
        bonnet_y = int(H * 0.95)
        box = _make_box(960, bonnet_y, 300, 200)
        result = detect_front_proximity([box], ZONE_LEFT_X, ZONE_RIGHT_X, W, H)
        assert result == PROX_NONE


class TestProximityThresholds:
    """Test area-based CLOSE and VERY_CLOSE detection."""

    def test_small_vehicle_none(self):
        """A small vehicle should return PROX_NONE (below CLOSE threshold)."""
        # Very small bbox (20x20 = 400 px² → 400 / (1920*1080) ≈ 0.02%)
        box = _make_box(960, 400, 20, 20)
        result = detect_front_proximity([box], ZONE_LEFT_X, ZONE_RIGHT_X, W, H)
        assert result == PROX_NONE

    def test_medium_vehicle_close(self):
        """A medium vehicle should return PROX_CLOSE."""
        # bbox area ~ 2-6% of frame → CLOSE
        # 2% of 1920*1080 = 41472 → sqrt = ~204 → 200x200
        box = _make_box(960, 500, 200, 210)
        result = detect_front_proximity([box], ZONE_LEFT_X, ZONE_RIGHT_X, W, H)
        assert result in (PROX_CLOSE, PROX_VERY_CLOSE)

    def test_large_vehicle_very_close(self):
        """A large vehicle should return PROX_VERY_CLOSE."""
        # bbox area > 6% of frame → VERY_CLOSE
        # 6% of 1920*1080 = 124416 → sqrt = ~353 → 400x350
        box = _make_box(960, 600, 400, 350)
        result = detect_front_proximity([box], ZONE_LEFT_X, ZONE_RIGHT_X, W, H)
        assert result == PROX_VERY_CLOSE


class TestMultipleVehicles:
    """With multiple vehicles, closest (largest area) should win."""

    def test_closest_wins(self):
        small = _make_box(960, 400, 50, 50)     # far away
        large = _make_box(960, 600, 400, 350)   # very close
        result = detect_front_proximity([small, large], ZONE_LEFT_X, ZONE_RIGHT_X, W, H)
        assert result == PROX_VERY_CLOSE

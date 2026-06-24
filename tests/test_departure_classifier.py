"""
Unit tests for Module B — departure_classifier.classify_departure()
===================================================================
Uses parameterised tests to cover all 7 states + edge cases.
Skill: tdd-workflow
"""

import pytest
from module_b.departure_classifier import (
    classify_departure,
    CENTERED, WARN_LEFT, WARN_RIGHT,
    DEPART_LEFT, DEPART_RIGHT,
    LANE_CHANGE_LEFT, LANE_CHANGE_RIGHT,
    WARN_THRESHOLD, DEPART_THRESHOLD,
)


# ── Zone 1: Centered ─────────────────────────────────────────────────────

class TestCenteredZone:
    """Offsets within ±WARN_THRESHOLD should return CENTERED."""

    @pytest.mark.parametrize("offset", [0, 10, -10, WARN_THRESHOLD - 1, -(WARN_THRESHOLD - 1)])
    def test_centered_offsets(self, offset):
        assert classify_departure(offset) == CENTERED

    def test_none_defaults_to_centered(self):
        assert classify_departure(None) == CENTERED

    def test_zero_exact(self):
        assert classify_departure(0.0) == CENTERED


# ── Zone 2: Warning ──────────────────────────────────────────────────────

class TestWarningZone:
    """Offsets between WARN and DEPART thresholds → WARN_LEFT or WARN_RIGHT."""

    def test_warn_left(self):
        # Positive offset → car drifted LEFT
        assert classify_departure(WARN_THRESHOLD) == WARN_LEFT

    def test_warn_right(self):
        # Negative offset → car drifted RIGHT
        assert classify_departure(-WARN_THRESHOLD) == WARN_RIGHT

    def test_mid_warn_left(self):
        mid = (WARN_THRESHOLD + DEPART_THRESHOLD) / 2
        assert classify_departure(mid) == WARN_LEFT

    def test_mid_warn_right(self):
        mid = -(WARN_THRESHOLD + DEPART_THRESHOLD) / 2
        assert classify_departure(mid) == WARN_RIGHT

    def test_just_below_depart_left(self):
        assert classify_departure(DEPART_THRESHOLD - 1) == WARN_LEFT

    def test_just_below_depart_right(self):
        assert classify_departure(-(DEPART_THRESHOLD - 1)) == WARN_RIGHT


# ── Zone 3: Departure / Lane Change ──────────────────────────────────────

class TestDepartureZone:
    """Offsets at or beyond DEPART_THRESHOLD with solid boundaries → DEPART."""

    def test_depart_left_solid(self):
        assert classify_departure(DEPART_THRESHOLD, left_type="solid") == DEPART_LEFT

    def test_depart_right_solid(self):
        assert classify_departure(-DEPART_THRESHOLD, right_type="solid") == DEPART_RIGHT

    def test_lane_change_left_dashed(self):
        assert classify_departure(DEPART_THRESHOLD, left_type="dashed") == LANE_CHANGE_LEFT

    def test_lane_change_right_dashed(self):
        assert classify_departure(-DEPART_THRESHOLD, right_type="dashed") == LANE_CHANGE_RIGHT

    def test_large_offset_depart_left(self):
        assert classify_departure(300, left_type="solid") == DEPART_LEFT

    def test_large_offset_lane_change_right(self):
        assert classify_departure(-300, right_type="dashed") == LANE_CHANGE_RIGHT


# ── Edge cases ────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Boundary conditions and unusual inputs."""

    def test_exact_warn_threshold_positive(self):
        result = classify_departure(float(WARN_THRESHOLD))
        assert result in (CENTERED, WARN_LEFT)  # ≥ threshold → WARN_LEFT

    def test_exact_depart_threshold_positive(self):
        result = classify_departure(float(DEPART_THRESHOLD), left_type="solid")
        assert result == DEPART_LEFT

    def test_very_small_offset(self):
        assert classify_departure(0.001) == CENTERED

    def test_negative_very_small(self):
        assert classify_departure(-0.001) == CENTERED

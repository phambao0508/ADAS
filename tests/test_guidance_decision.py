"""
Unit tests for Module C — guidance_decision.decide_guidance()
=============================================================
Maps to the truth table in the module docstring.
Skill: tdd-workflow
"""

import pytest
from module_c.guidance_states import (
    GUIDE_NONE, GUIDE_LEFT, GUIDE_RIGHT, GUIDE_BOTH,
    GUIDE_SLOW, GUIDE_URGENT,
    PROX_NONE, PROX_CLOSE, PROX_VERY_CLOSE,
)
from module_c.guidance_decision import decide_guidance, reset_hysteresis


@pytest.fixture(autouse=True)
def _reset():
    """Reset hysteresis state between tests."""
    reset_hysteresis()
    yield
    reset_hysteresis()


# ── Priority 1: VERY_CLOSE → always GUIDE_URGENT ─────────────────────────

class TestUrgent:
    def test_urgent_all_clear_dashed(self):
        assert decide_guidance(PROX_VERY_CLOSE, True, True, "dashed", "dashed") == GUIDE_URGENT

    def test_urgent_none_clear_solid(self):
        assert decide_guidance(PROX_VERY_CLOSE, False, False, "solid", "solid") == GUIDE_URGENT

    def test_urgent_overrides_everything(self):
        """URGENT should win regardless of lane clearance or line type."""
        for lc in (True, False):
            for rc in (True, False):
                for lt in ("solid", "dashed"):
                    for rt in ("solid", "dashed"):
                        result = decide_guidance(PROX_VERY_CLOSE, lc, rc, lt, rt)
                        assert result == GUIDE_URGENT


# ── Priority 2: NONE → always GUIDE_NONE ─────────────────────────────────

class TestNoProximity:
    def test_no_vehicle_no_guidance(self):
        assert decide_guidance(PROX_NONE, True, True, "dashed", "dashed") == GUIDE_NONE

    def test_no_vehicle_regardless_of_lanes(self):
        assert decide_guidance(PROX_NONE, False, False, "solid", "solid") == GUIDE_NONE


# ── Priority 3: CLOSE — evaluate lane options ────────────────────────────

class TestClose:
    def test_both_clear_both_dashed(self):
        result = decide_guidance(PROX_CLOSE, True, True, "dashed", "dashed")
        assert result == GUIDE_BOTH

    def test_left_clear_left_dashed(self):
        result = decide_guidance(PROX_CLOSE, True, False, "dashed", "solid")
        assert result == GUIDE_LEFT

    def test_right_clear_right_dashed(self):
        result = decide_guidance(PROX_CLOSE, False, True, "solid", "dashed")
        assert result == GUIDE_RIGHT

    def test_both_clear_left_solid_right_dashed(self):
        """Only right is dashed → can only go right."""
        result = decide_guidance(PROX_CLOSE, True, True, "solid", "dashed")
        assert result == GUIDE_RIGHT

    def test_both_clear_both_solid(self):
        """Both solid → can't change lanes → SLOW."""
        result = decide_guidance(PROX_CLOSE, True, True, "solid", "solid")
        assert result == GUIDE_SLOW

    def test_neither_clear(self):
        result = decide_guidance(PROX_CLOSE, False, False, "dashed", "dashed")
        assert result == GUIDE_SLOW

    def test_left_clear_but_left_solid(self):
        """Clear but can't cross solid → SLOW."""
        result = decide_guidance(PROX_CLOSE, True, False, "solid", "solid")
        assert result == GUIDE_SLOW


# ── Hysteresis ────────────────────────────────────────────────────────────

class TestHysteresis:
    def test_directional_holds_before_slow(self):
        """Once LEFT is decided, it should persist for MIN_HOLD frames
        even if next frame would produce SLOW."""
        # Frame 1: GUIDE_LEFT
        r1 = decide_guidance(PROX_CLOSE, True, False, "dashed", "solid")
        assert r1 == GUIDE_LEFT

        # Frame 2-4: conditions change to SLOW, but hysteresis holds LEFT
        for _ in range(3):
            r = decide_guidance(PROX_CLOSE, False, False, "solid", "solid")
            assert r == GUIDE_LEFT

        # Frame 5: now it should switch to SLOW
        r5 = decide_guidance(PROX_CLOSE, False, False, "solid", "solid")
        assert r5 == GUIDE_SLOW

    def test_urgent_resets_hysteresis(self):
        """URGENT should override any held directional state."""
        # Set up directional state
        decide_guidance(PROX_CLOSE, True, False, "dashed", "solid")
        # URGENT should override
        r = decide_guidance(PROX_VERY_CLOSE, False, False, "solid", "solid")
        assert r == GUIDE_URGENT

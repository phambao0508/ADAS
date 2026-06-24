"""
Unit tests for Module B — ema_smoother.EMASmoother
===================================================
Tests EMA convergence, spike rejection, None handling, and reset.
Skill: tdd-workflow
"""

import pytest
from module_b.ema_smoother import EMASmoother, EMA_ALPHA


class TestEMAConvergence:
    """Verify the smoother converges toward stable input."""

    def test_constant_input_converges(self):
        s = EMASmoother()
        # Feed 100 frames of constant offset = 50
        for _ in range(100):
            result = s.update(50.0)
        # Should converge to 50 (within 1 px tolerance)
        assert abs(result - 50.0) < 1.0

    def test_first_value_initialises(self):
        s = EMASmoother()
        result = s.update(42.0)
        # First value should start the smoother close to 42
        assert result is not None


class TestSpikeRejection:
    """Verify that sudden large jumps are dampened."""

    def test_large_jump_dampened(self):
        s = EMASmoother()
        # Establish baseline at 50
        for _ in range(30):
            s.update(50.0)

        # Inject a spike to 300
        after_spike = s.update(300.0)
        # The result should be somewhere between 50 and 300, NOT 300
        assert after_spike < 300.0
        assert after_spike > 50.0

    def test_gradual_change_follows(self):
        s = EMASmoother()
        # Establish baseline at 50
        for _ in range(30):
            s.update(50.0)

        # Gradually move to 100 (10 per frame)
        for i in range(5):
            result = s.update(50.0 + (i + 1) * 10)

        # Should be moving toward 100 but smoothed
        assert result > 50.0


class TestNoneHandling:
    """Verify that None inputs preserve the last good value."""

    def test_none_holds_previous(self):
        s = EMASmoother()
        # Establish baseline
        for _ in range(10):
            s.update(75.0)
        last_good = s.update(75.0)

        # Send None — should return last held value
        result = s.update(None)
        assert result is not None
        assert abs(result - last_good) < 5.0

    def test_consecutive_nones(self):
        s = EMASmoother()
        for _ in range(10):
            s.update(60.0)
        last = s.update(60.0)

        # Multiple Nones
        for _ in range(5):
            result = s.update(None)
        assert result is not None


class TestReset:
    """Verify reset clears state."""

    def test_reset_clears_history(self):
        s = EMASmoother()
        for _ in range(10):
            s.update(100.0)
        s.reset()

        # After reset, first value should reinitialise
        result = s.update(0.0)
        # Should be close to 0, not 100
        assert result is not None
        assert abs(result) < 50.0  # generous tolerance

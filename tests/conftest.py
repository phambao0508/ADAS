"""
Shared pytest fixtures for ADAS unit tests.
"""

import sys
import os
import pytest
import numpy as np

# Ensure the project root is on sys.path so `states`, `module_a`, etc. resolve
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def dummy_frame():
    """A 1080p black frame for HUD rendering tests."""
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def frame_dims():
    """Standard frame dimensions."""
    return {"width": 1920, "height": 1080}

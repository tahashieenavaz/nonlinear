import pytest
from typing import Optional


def assert_equal(a, b, message: Optional[str] = None):
    assert a == b, message


def assert_almost_equal(a, b, epsilon: float = 1e-6, message: Optional[str] = None):
    assert a == pytest.approx(b, abs=epsilon), message

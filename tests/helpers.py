import pytest
import torch
from typing import Optional


def assert_equal(a, b, message: Optional[str] = None):
    if isinstance(a, torch.Tensor):
        a = a.item()

    if isinstance(b, torch.Tensor):
        b = b.item()

    assert a == b, message


def assert_almost_equal(a, b, epsilon: float = 1e-5, message: Optional[str] = None):
    if isinstance(a, torch.Tensor):
        a = a.item()

    if isinstance(b, torch.Tensor):
        b = b.item()
    assert a == pytest.approx(b, abs=epsilon), message

import torch
from nonlinear.functional import abslu
from helpers import assert_equal


def test_abslu_accuracy():
    x = torch.tensor(0.0)
    expected_y = 0.0
    assert_equal(abslu(x, alpha=0.18), expected_y)

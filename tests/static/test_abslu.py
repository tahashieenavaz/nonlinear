import torch
from nonlinear.functional import abslu
from tests.helpers import assert_equal


def test_abslu_accuracy():
    x = [-1.0, -0.5, 0, 0.5, 1]
    y = [0.18, 0.09, 0, 0.5, 1]
    for a, b in zip(x, y):
        a = torch.tensor(a)
        b = torch.tensor(b)
        assert_equal(abslu(a, alpha=0.18), b)

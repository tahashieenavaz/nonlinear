import torch
from nonlinear.functional import abslu
from nonlinear.static import AbsLU
from tests.helpers import assert_equal


def test_abslu_functional_accuracy():
    x = [-1.0, -0.5, 0, 0.5, 1]
    y = [0.18, 0.09, 0, 0.5, 1]

    for a, b in zip(x, y):
        a = torch.tensor(a)
        b = torch.tensor(b)
        assert_equal(abslu(a, alpha=0.18), b)


def test_abslu_module_accuracy():
    module = AbsLU(alpha=0.18)
    x = [-1.0, -0.5, 0, 0.5, 1]
    y = [0.18, 0.09, 0, 0.5, 1]

    for a, b in zip(x, y):
        a = torch.tensor(a)
        b = torch.tensor(b)
        assert_equal(module(a), b)

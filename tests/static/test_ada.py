import torch
from nonlinear.functional import ada
from nonlinear.static import ADA
from tests.helpers import assert_almost_equal

x = [-1.0, -0.5, 0, 0.5, 1]
y = [-0.36788, -0.30327, 0, 0.5, 1]


def test_ada_functional_accuracy():
    for a, b in zip(x, y):
        a = torch.tensor(a)
        b = torch.tensor(b)
        assert_almost_equal(ada(a), b)


def test_ada_module_accuracy():
    module = ADA()

    for a, b in zip(x, y):
        a = torch.tensor(a)
        b = torch.tensor(b)
        assert_almost_equal(module(a), b)

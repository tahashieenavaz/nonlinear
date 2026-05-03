import torch
from ..ActivationFunction import ActivationFunction
from ..functional import self_arctan


class SelfArctan(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self_arctan(x)

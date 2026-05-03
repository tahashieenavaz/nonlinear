import torch
from ..ActivationFunction import ActivationFunction
from ..functional import selfarctan


class SelfArctan(ActivationFunction):
    def __init__(self, *, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return selfarctan(x, inplace=self.inplace)

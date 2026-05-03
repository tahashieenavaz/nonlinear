import torch
from ..ActivationFunction import ActivationFunction
from ..functional import eanaf


class EANAF(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return eanaf(x)

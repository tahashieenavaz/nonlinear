import torch
from ..activation_function import ActivationFunction
from ..functional import asilu


class ASiLU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return asilu(x=x)

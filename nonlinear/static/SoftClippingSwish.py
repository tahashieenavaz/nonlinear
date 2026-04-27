import torch
from ..ActivationFunction import ActivationFunction


class SoftClippingSwish(ActivationFunction):
    def __init__(self):
        super().__init__()

    def swish(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.swish(x))

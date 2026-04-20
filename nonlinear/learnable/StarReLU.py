import torch
from ..ChannelBasedActivationFunction import ChannelBasedActivationFunction


class StarReLU(ChannelBasedActivationFunction):
    def __init__(self):
        super().__init__()
        self.a = None
        self.b = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.initialize(x, ["a", "b"], [0.8944, -0.4472])
        a = self.a.view(self.parameter_shape(x))
        b = self.b.view(self.parameter_shape(x))
        return a * torch.relu(x).pow(2) + b

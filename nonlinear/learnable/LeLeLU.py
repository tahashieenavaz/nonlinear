import torch
from ..ChannelBasedActivationFunction import ChannelBasedActivationFunction


class LeLeLU(ChannelBasedActivationFunction):
    def __init__(self):
        super().__init__()
        self.a = None

    def forward(self, x: torch.Tensor):
        self.initialize(x, "a")
        a = self.a.view(self.parameter_shape(x))
        return torch.where(x >= 0, a * x, 0.01 * a * x)

import torch
from ..ChannelBasedActivationFunction import ChannelBasedActivationFunction


class RTPReLU(ChannelBasedActivationFunction):
    def __init__(self, channels: int, *, sigma: float = 0.75, a: float = 1.0):
        super().__init__()
        self.sigma = sigma
        self.a = torch.nn.Parameter(torch.full((channels,), float(a)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = self.get_shape(x)
        a_view = self.a.view(shape)
        if self.training:
            b = torch.randn_like(x) * self.sigma
        else:
            b = 0.0
        return torch.where(x + b >= 0, x, x / a_view)

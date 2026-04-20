import torch
from ..ChannelBasedActivationFunction import ChannelBasedActivationFunction


class PairedReLU(ChannelBasedActivationFunction):
    def __init__(
        self,
        channels: int,
        *,
        a: float = 0.5,
        b: float = 0.0,
        c: float = -0.5,
        d: float = 0.0,
    ):
        super().__init__()
        # scale params
        self.a = torch.nn.Parameter(torch.full((channels,), a))
        self.c = torch.nn.Parameter(torch.full((channels,), c))

        # threshold params
        self.b = torch.nn.Parameter(torch.full((channels,), b))
        self.d = torch.nn.Parameter(torch.full((channels,), d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = self.get_shape(x)

        a = self.a.view(shape)
        b = self.b.view(shape)
        c = self.c.view(shape)
        d = self.d.view(shape)

        alef = torch.relu(a * x - b)
        be = torch.relu(c * x - d)

        return torch.cat([alef, be], dim=1)

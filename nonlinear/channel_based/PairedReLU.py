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
        self.a = self.get_parameter(channels=channels, initial_value=a)
        self.c = self.get_parameter(channels=channels, initial_value=c)

        # threshold params
        self.b = self.get_parameter(channels=channels, initial_value=b)
        self.d = self.get_parameter(channels=channels, initial_value=d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = self.get_shape(x)

        a = self.a.view(shape)
        b = self.b.view(shape)
        c = self.c.view(shape)
        d = self.d.view(shape)

        alef = torch.relu(a * x - b)
        be = torch.relu(c * x - d)

        return torch.cat([alef, be], dim=1)

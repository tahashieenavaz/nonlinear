import torch
from ..LearnableActivationFunction import LearnableActivationFunction


class AReLU(LearnableActivationFunction):
    def __init__(
        self,
        *,
        a: float = 0.9,
        b: float = 2.0,
        clamp_min: float = 0.0,
        clamp_max: float = 0.99,
    ):
        super().__init__()
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.a = torch.nn.Parameter(torch.tensor(a))
        self.b = torch.nn.Parameter(torch.tensor(b))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        negative_slope = torch.clamp(self.a, min=self.clamp_min, max=self.clamp_max)
        positive_slope = 1.0 + torch.sigmoid(self.b)
        return torch.where(x >= 0, positive_slope * x, negative_slope * x)

import torch
from ..stochastic_activation_function import StochasticActivationFunction


class BrownianReLU(StochasticActivationFunction):
    def __init__(self, *, alpha: float = 0.01, M: int = 100):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.M = M

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        relu = torch.nn.functional.relu(x)
        if self.training:
            negative_mask = x <= 0
            std = torch.sqrt(torch.abs(x) / self.M)
            noise = torch.normal(mean=0.0, std=std)
            negative_value = torch.where(
                negative_mask, -self.alpha * noise, torch.zeros_like(x)
            )
            return relu + negative_value
        return relu

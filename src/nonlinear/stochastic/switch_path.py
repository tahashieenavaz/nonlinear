import torch
from ..stochastic_activation_function import StochasticActivationFunction


class SwitchPath(StochasticActivationFunction):
    def __init__(self, *, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            mask = torch.empty_like(x).bernoulli_(self.alpha)
            relu_pos = torch.nn.functional.relu(x)
            relu_neg = torch.nn.functional.relu(-x)
            return (1 - mask) * relu_pos + mask * relu_neg
        else:
            return (1 - self.alpha) * F.relu(x) + self.alpha * F.relu(-x)

    def set_alpha(self, alpha: float) -> None:
        self.alpha = alpha

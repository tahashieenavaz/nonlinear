import torch
from ..functional import aoaf
from nonlinear import ActivationFunction

"""
Paper: https://doi.org/10.3390/electronics11223799

Formula: f(z) = ReLU(z - b*a) + c*a
    
Variables:
    - x: Input tensor.
    - a: Channel-wise / Neuron-wise mean of the input.
    - b, c: Fixed hyperparameters (Recommended: b=c=0.17).
"""


class AOAF(ActivationFunction):
    def __init__(self, *, b: float = 0.17, c: float = 0.17):
        super().__init__()
        self.b = b
        self.c = c

    def extra_repr(self) -> str:
        return f"b={self.b}, c={self.c}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return aoaf(x=x, b=self.b, c=self.c)

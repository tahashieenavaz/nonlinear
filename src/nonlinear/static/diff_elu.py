import torch
from nonlinear import ActivationFunction
from nonlinear.functional import diffelu


class DiffELU(ActivationFunction):
    def __init__(self, *, a: float = 0.3, b: float = 0.1):
        super().__init__()
        self.a = a
        self.b = b

    def extra_repr(self) -> str:
        return f"a={self.a}, b={self.b}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return diffelu(x, a=self.a, b=self.b)

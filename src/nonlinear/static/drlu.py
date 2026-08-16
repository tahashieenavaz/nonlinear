import torch
from nonlinear import ActivationFunction
from nonlinear.functional import drlu


class DRLU(ActivationFunction):
    def __init__(self, *, alpha: float = 0.08):
        super().__init__()
        if alpha < 0:
            raise ValueError(
                f"alpha parameter in DRLU must be positive. Authors used [0.06, 0.08, 0.1]."
            )
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drlu(x, alpha=self.alpha)

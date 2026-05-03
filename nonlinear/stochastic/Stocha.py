import torch
from ..StochasticActivationFunction import StochasticActivationFunction


class Stocha(StochasticActivationFunction):
    def __init__(self, *, p: float = 0.05, inference_mode: str = "relu"):
        super().__init__()
        self.p = p
        self.inference_mode = inference_mode.lower()
        if self.inference_mode not in ["relu", "silu", "stocha"]:
            raise ValueError("inference_mode must be 'relu', 'silu', or 'stocha'")

    def extra_repr(self) -> str:
        return f"p={self.p}, inference_mode={self.inference_mode}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training or self.inference_mode == "stocha":
            random_mask = torch.rand_like(x) < self.p
            keep_silu = (x >= 0) | random_mask
            return torch.nn.functional.silu(x) * keep_silu.to(x.dtype)

        if self.inference_mode == "relu":
            return torch.nn.functional.relu(x)

        return torch.nn.functional.silu(x)

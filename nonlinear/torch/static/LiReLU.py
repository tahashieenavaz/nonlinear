import torch


class LiReLU(torch.nn.Module):
    def __init__(self, *, a: float = 0.01):
        super().__init__()
        self.register_buffer("a", torch.tensor(float(a)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a * x + torch.relu(x)

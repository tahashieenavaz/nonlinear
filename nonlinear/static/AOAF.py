import torch
import torch.nn as nn


class AOAF(nn.Module):
    def __init__(self, *, b: float = 0.17, c: float = 0.17):
        super().__init__()
        self.b = b
        self.c = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 1:
            reduce_dims = [0] + list(range(2, x.ndim))
        else:
            reduce_dims = [0]
        a = x.mean(dim=reduce_dims, keepdim=True)
        return torch.relu(x - self.b * a) + self.c * a

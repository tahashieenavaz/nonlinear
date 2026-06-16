import torch
from nonlinear.functional import golu

class GoLU(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return golu(x=x)

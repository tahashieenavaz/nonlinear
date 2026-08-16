import torch
from nonlinear import ActivationFunction


class DeepFourier(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            # 4D Tensor: (Batch, Channels, Height, Width)
            # we must concatenate along the channel dimension (dim=1)
            feature_dim = 1
        else:
            # 2D Tensor: (Batch, Features)
            # we concatenate along the feature dimension (dim=-1 or dim=1)
            feature_dim = -1

        return torch.cat([torch.sin(x), torch.cos(x)], dim=feature_dim)

import torch


def aoaf(x: torch.Tensor, *, b: float = 0.17, c: float = 0.17) -> torch.Tensor:
    reduce_dims = [d for d in range(x.ndim) if d != 1]
    a = x.mean(dim=reduce_dims, keepdim=True)
    return torch.nn.functional.relu(x - b * a) + c * a

import torch


def __g(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x / 2)


def __h(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(x)


def eanaf(x: torch.Tensor) -> torch.Tensor:
    return x * __g(__h(x))

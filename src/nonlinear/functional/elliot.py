import torch


def elliot(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.nn.functional.softsign(x))

import torch
import matplotlib.pyplot as plt


class ActivationFunction(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward()")

    @torch.inference_mode()
    def plot(self, x_range=(-5, 5), steps=200):
        x = torch.linspace(*x_range, steps)
        y = self(x).cpu()
        plt.plot(x, y)
        plt.show()

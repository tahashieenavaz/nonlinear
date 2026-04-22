import torch
import matplotlib.pyplot as plt


class ActivationFunction(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward()")

    @torch.no_grad()
    def plot(self, x_range: tuple = (-5, 5), num_points: int = 1000) -> None:
        self.eval()

        x = torch.linspace(x_range[0], x_range[1], num_points)
        y = self.forward(x)

        x = x.cpu().numpy()
        y = y.cpu().numpy()

        plt.figure(figsize=(6, 4))
        plt.plot(x, y, linewidth=2)
        plt.axhline(0, linewidth=1)
        plt.axvline(0, linewidth=1)

        plt.title(self.__class__.__name__)
        plt.tight_layout()
        plt.show()

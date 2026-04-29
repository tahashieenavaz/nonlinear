import torch
import matplotlib.pyplot as plt


class ActivationFunction(torch.nn.Module):
    @torch.inference_mode()
    def plot(self):
        try:
            fresh_instance = self.__class__()
        except:
            fresh_instance = self.__class__(512)

        x = torch.randn(2, 512)
        y = fresh_instance(x)

        x = x.flatten().cpu()
        y = y.flatten().cpu()

        plt.scatter(x, y, alpha=0.3, marker=".", color="purple")
        plt.title(f"{self.__class__.__name__}")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward()")

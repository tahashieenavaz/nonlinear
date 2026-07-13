import torch
import matplotlib.pyplot as plt


class ActivationFunction(torch.nn.Module):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance._init_args = args
        instance._init_kwargs = kwargs
        return instance

    @torch.inference_mode()
    def plot(self):
        fresh_instance = self.__class__(*self._init_args, **self._init_kwargs)

        x = torch.linspace(-10, 10, 1024).reshape(2, 512)
        y = fresh_instance(x)

        x_plot = x.flatten().cpu()
        y_plot = y.flatten().cpu()

        plt.scatter(x_plot, y_plot, alpha=0.3, marker=".", color="purple")
        plt.title(f"{self.__class__.__name__}")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward()")

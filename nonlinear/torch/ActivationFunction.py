import torch
import matplotlib.pyplot as plt


class ActivationFunction(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward()")

    @torch.no_grad()
    def plot(
        self, x_range: tuple = (-10, 10), num_points: int = 2000, num_channels: int = 1
    ) -> None:
        self.eval()

        x1d = torch.linspace(x_range[0], x_range[1], num_points)
        x_input = x1d.view(1, 1, num_points).expand(1, num_channels, num_points)
        y = self.forward(x_input)

        y_np = y.squeeze(0).cpu().numpy()
        x_np = x1d.cpu().numpy()

        plt.figure(figsize=(8, 5))
        plt.grid(True, linestyle="--", alpha=0.5, zorder=0)
        plt.plot(
            x_np,
            x_np,
            color="gray",
            linestyle=":",
            linewidth=1.5,
            label="Identity (y=x)",
            zorder=1,
        )

        plt.axhline(0, color="black", linewidth=1.2, zorder=2)
        plt.axvline(0, color="black", linewidth=1.2, zorder=2)

        if num_channels == 1:
            y_plot = y_np[0] if y_np.ndim > 1 else y_np
            plt.plot(
                x_np,
                y_plot,
                linewidth=2.5,
                color="#1f77b4",
                label=self.__class__.__name__,
                zorder=3,
            )
        else:
            colors = plt.cm.viridis(torch.linspace(0, 0.9, num_channels).numpy())
            for c in range(num_channels):
                plt.plot(
                    x_np,
                    y_np[c],
                    linewidth=2.5,
                    color=colors[c],
                    label=f"Channel {c+1}",
                    zorder=3,
                )

        plt.title(
            f"{self.__class__.__name__} Activation Function",
            fontsize=14,
            fontweight="bold",
            pad=10,
        )
        plt.xlabel("Input $x$", fontsize=12)
        plt.ylabel("Output $f(x)$", fontsize=12)
        plt.legend(loc="best", fontsize=10, frameon=True, shadow=True)

        plt.tight_layout()
        plt.show()

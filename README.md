# Non-Linear

A hand-curated collections of activations functions for deep learning research. 

## Channel-Based Activation Functions

<div align="center">

| Activation Function | Formula |
|---|---|
| [FPAF](nonlinear/channel_based/FPAF.py) | $f(x) = a \cdot \mu(x)$ if $x \geq 0$, else $b \cdot \nu(x)$ |
| [DPReLU](nonlinear/channel_based/DPReLU.py) | $f(x) = ax$ if $x \geq 0$, else $bx$ |
| [DualLine](nonlinear/channel_based/DualLine.py) | $f(x) = ax + m$ if $x \geq 0$, else $bx + m$ |
| [FReLU](nonlinear/channel_based/FReLU.py) | $f(x) = \text{ReLU}(x) + b$ |
| [LeLeLU](nonlinear/channel_based/LeLeLU.py) | $f(x) = ax$ if $x \geq 0$, else $0.01ax$ |
| [PREU](nonlinear/channel_based/PREU.py) | $f(x) = ax$ if $x \geq 0$, else $ax \exp(bx)$ |
| [ShiLU](nonlinear/channel_based/ShiLU.py) | $f(x) = a \cdot \text{ReLU}(x) + b$ |
| [StarReLU](nonlinear/channel_based/StarReLU.py) | $f(x) = a \cdot \text{ReLU}(x)^2 + b$ |
| [EPReLU](nonlinear/channel_based/EPReLU.py) | $f(x) = kx$ if $x \geq 0$, else $\frac{x}{a}$ (where $k$ updates dynamically) |
| [PairedReLU](nonlinear/channel_based/PairedReLU.py) | $f(x) = \left[\text{ReLU}(ax - b), \text{ReLU}(cx - d)\right]$ (concatenated along channel dimension) |
| [RMAF](nonlinear/channel_based/RMAF.py) | $f(x) = \frac{abx}{(0.25(1 + \exp(-x)) + 0.75)^c}$ |
| [PTELU](nonlinear/channel_based/PTELU.py) | $f(x) = x$ if $x \geq 0$, else $\vert a \vert \tanh(\vert b \vert x)$ |
| [TaLU](nonlinear/channel_based/TaLU.py) | $f(x) = x$ if $x \geq \vert b \vert$; $\tanh(x)$ if $x > \vert a \vert$; else $\tanh(\vert a \vert)$ |
| [TanhLU](nonlinear/channel_based/TanhLU.py) | $f(x) = a \tanh(cx) + bx$ |

</div>
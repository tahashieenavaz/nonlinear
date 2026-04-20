# nonlinear
A hand-curated collections of activations functions for deep learning research. 

## Channel-Based Activation Functions

Here are the formulas for the activation functions implemented in `__init__.py`:

- **FPAF**: $f(x) = \begin{cases} a \cdot \mu(x) & \text{if } x \geq 0 \\ b \cdot \nu(x) & \text{otherwise} \end{cases}$
- **DPReLU**: $f(x) = \begin{cases} a x & \text{if } x \geq 0 \\ b x & \text{otherwise} \end{cases}$
- **DualLine**: $f(x) = \begin{cases} a x + m & \text{if } x \geq 0 \\ b x + m & \text{otherwise} \end{cases}$
- **FReLU**: $f(x) = \text{ReLU}(x) + b$
- **LeLeLU**: $f(x) = \begin{cases} a x & \text{if } x \geq 0 \\ 0.01 a x & \text{otherwise} \end{cases}$
- **PREU**: $f(x) = \begin{cases} a x & \text{if } x \geq 0 \\ a x \exp(b x) & \text{otherwise} \end{cases}$
- **ShiLU**: $f(x) = a \cdot \text{ReLU}(x) + b$
- **StarReLU**: $f(x) = a \cdot \text{ReLU}(x)^2 + b$
- **EPReLU**: $f(x) = \begin{cases} k x & \text{if } x \geq 0 \\ \frac{x}{a} & \text{otherwise} \end{cases}$ (where $k$ updates dynamically during training)
- **PairedReLU**: $f(x) = [\text{ReLU}(a x - b), \text{ReLU}(c x - d)]$ (concatenated along channel dimension)
- **RMAF**: $f(x) = \frac{a b x}{(0.25(1 + \exp(-x)) + 0.75)^c}$
- **PTELU**: $f(x) = \begin{cases} x & \text{if } x \geq 0 \\ |a| \tanh(|b| x) & \text{otherwise} \end{cases}$
- **TaLU**: $f(x) = \begin{cases} x & \text{if } x \geq |b| \\ \tanh(x) & \text{if } x > |a| \\ \tanh(|a|) & \text{otherwise} \end{cases}$
- **TanhLU**: $f(x) = a \tanh(c x) + b x$

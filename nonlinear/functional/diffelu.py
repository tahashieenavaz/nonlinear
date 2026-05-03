import torch


class __DiffELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, a: float, b: float) -> torch.Tensor:
        ctx.a = a
        ctx.b = b

        out = x.clone()
        mask = x < 0
        ctx.save_for_backward(x)
        x_neg = x[mask]
        out[mask] = a * (x_neg * x_neg.exp() - b * torch.exp(b * x_neg))
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        a = ctx.a
        b = ctx.b
        grad_x = grad_output.clone()
        mask = x < 0
        x_neg = x[mask]
        exp_x = x_neg.exp()
        exp_bx = torch.exp(b * x_neg)
        derivative_neg = a * (exp_x * (x_neg + 1) - (b**2) * exp_bx)
        grad_x[mask] = grad_output[mask] * derivative_neg
        return grad_x, None, None


def diffelu(x: torch.Tensor, *, a: float = 0.3, b: float = 0.1) -> torch.Tensor:
    return __DiffELU.apply(x, a, b)

import torch

"""
Paper: https://doi.org/10.3390/electronics11223799

Formula: f(z) = ReLU(z - b*a) + c*a
    
Variables:
    - x: Input tensor.
    - a: Channel-wise / Neuron-wise mean of the input.
    - b, c: Fixed hyperparameters (Recommended: b=c=0.17).
"""


class __AOAF(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, b: float, c: float, inplace: bool
    ) -> torch.Tensor:
        ctx.b = b
        ctx.c = c
        ctx.reduce_dims = [d for d in range(x.ndim) if d != 1]
        a = x.mean(dim=ctx.reduce_dims, keepdim=True)

        if inplace:
            ctx.mark_dirty(x)
            x.sub_(b * a)
            mask = x > 0
            ctx.save_for_backward(mask)
            x.relu_()
            x.add_(c * a)
            return x
        else:
            z = x - b * a
            mask = z > 0
            ctx.save_for_backward(mask)
            return torch.relu(z) + c * a

    @staticmethod
    def backward(ctx, gradients: torch.Tensor):
        (mask,) = ctx.saved_tensors
        b = ctx.b
        c = ctx.c
        reduce_dims = ctx.reduce_dims
        gradient_mask = gradients * mask
        gradient_mean = gradients.mean(dim=reduce_dims, keepdim=True)
        gradient_mask_mean = gradient_mask.mean(dim=reduce_dims, keepdim=True)
        grad_x = gradient_mask + (c * gradient_mean - b * gradient_mask_mean)
        return grad_x, None, None, None


def aoaf(x: torch.Tensor, *, b: float = 0.17, c: float = 0.17, inplace: bool = False):
    return __AOAF.apply(x, b, c, inplace)

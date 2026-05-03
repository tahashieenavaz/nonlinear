import torch


class __Ada(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
        ctx.save_for_backward(x.clone() if inplace else x)
        if inplace:
            ctx.mark_dirty(x)
            negative_mask = x < 0
            x[negative_mask] *= torch.exp(x[negative_mask])
            return x
        return torch.where(x >= 0, x, x * torch.exp(x))

    @staticmethod
    def backward(ctx, gradients: torch.Tensor):
        (x,) = ctx.saved_tensors
        grad_x = gradients.clone()
        negative_mask = x < 0
        x_neg = x[negative_mask]
        grad_x[negative_mask] *= torch.exp(x_neg) * (1 + x_neg)
        return grad_x, None


def ada(x: torch.Tensor, *, inplace: bool = False) -> torch.Tensor:
    return __Ada.apply(x, inplace)

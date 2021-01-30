import torch
import torch.autograd as autograd
import torch.nn.functional as F

__all__ = ['soft_lb', 'soft_ub', 'soft_ub_lb', 'passthrough_lb',
           'passthrough_ub', 'passthrough_ub_lb']


# ---- soft bounds ----

def soft_lb(x, vmin):
    """
    Activation is approximately linear everywhere except near lower boundary,
    where it converges to an asymptote
    """
    return F.softplus(x-vmin) + vmin

def soft_ub(x, vmax):
    """
    Activation is approximately linear everywhere except near upper boundary,
    where it converges to an asymptote
    """
    return -F.softplus(-(x-vmax)) + vmax

def soft_ub_lb(x, vmin, vmax):
    """
    Activation is approximately linear everywhere except near lower and upper
    boundaries, where it converges to an asymptote at each
    """
    assert vmax > vmin
    scale = (vmax-vmin)/2
    return scale*(torch.tanh((x-vmin)/scale - 1) + 1) + vmin


# ---- hard bounds with pass-through gradients ----

class LowerBound(autograd.Function):
    """Lower bounding with selective pass-through gradients

    This function behaves just like `torch.max`, but when inputs < bound, we
    pass forward the gradient if it will take us toward the bound.
    """
    @staticmethod
    def forward(ctx, inputs, bound):
        b = bound * torch.ones_like(inputs)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through = (inputs >= b) | (grad_output < 0)
        grad_output = grad_output.masked_fill(~pass_through, 0.)
        return grad_output, None


class UpperBound(autograd.Function):
    """Lower bounding with selective pass-through gradients

    This function behaves just like `torch.min`, but when inputs > bound, we
    pass forward the gradient if it will take us toward the bound.
    """
    @staticmethod
    def forward(ctx, inputs, bound):
        b = bound * torch.ones_like(inputs)
        ctx.save_for_backward(inputs, b)
        return torch.min(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through = (inputs <= b) | (grad_output > 0)
        grad_output = grad_output.masked_fill(~pass_through, 0.)
        return grad_output, None

def passthrough_lb(x, vmin):
    return LowerBound.apply(x, vmin)

def passthrough_ub(x, vmax):
    return UpperBound.apply(x, vmax)

def passthrough_ub_lb(x, vmin, vmax):
    return UpperBound.apply(LowerBound.apply(x, vmin), vmax)
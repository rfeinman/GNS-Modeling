import torch
import torch.nn.functional as F



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
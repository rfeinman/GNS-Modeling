import numpy as np
import torch



def get_loss_limits(losses, pct=90):
    if torch.is_tensor(losses):
        losses = losses.numpy()
    losses = losses.flatten()
    vmax = np.percentile(losses, pct)
    vmin = np.min(losses)
    vmean = (vmax+vmin)/2
    vmin = vmean - 1.5*(vmax-vmin)/2
    vmax = vmean + 1.5*(vmax-vmin)/2
    return vmin, vmax

def get_intervals(iterations, nint=10):
    intervals = torch.linspace(0,iterations-0.5,nint)
    intervals = intervals.floor().int()
    return intervals
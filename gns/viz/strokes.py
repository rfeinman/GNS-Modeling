import numpy as np
import torch
from pybpl.util.stroke import dist_along_traj

from ..utils import torch_to_numpy, traj_from_spline
from .general import get_color



def space_motor_to_img(x):
    x = torch_to_numpy(x)
    x = np.copy(x)
    x[:,1] = -x[:,1]
    return x

def add_arrow(line, size=15, **kwargs):
    color = line.get_color()
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    delta = np.array([xdata[-1] - xdata[-2], ydata[-1] - ydata[-2]])
    delta /= max(np.linalg.norm(delta), 1e-5)
    line.axes.annotate('',
        xytext=(xdata[-1], ydata[-1]),
        xy=(xdata[-1] + 10*delta[0], ydata[-1] + 10*delta[1]),
        arrowprops=dict(arrowstyle="-|>", color=color),
        size=size, **kwargs
    )

def plot_traj(axis, x, color, lw=2, arrow_size=0, pt_size=0):
    x = torch_to_numpy(x)
    x = space_motor_to_img(x)
    if len(x) > 1 and dist_along_traj(x) > 0.01:
        line, = axis.plot(x[:,0], x[:,1], color=color, linewidth=lw)
        if arrow_size > 0:
            add_arrow(line, arrow_size)
    else:
        axis.plot(x[0,0], x[0,1], color=color, linewidth=lw, marker='.')
    if pt_size > 0:
        axis.scatter(x[0,0], x[0,1], color=color, linewidth=lw, s=pt_size)

def plot_spline(axis, y, color, lw, neval=200, start=None):
    x = traj_from_spline(y, neval)
    if start is not None:
        start = torch_to_numpy(start)
        x = x + start
    plot_traj(axis, x, color, lw)

def plot_motor_to_image(axis, strokes, lw=2, colored=True, arrow_size=0,
                        pt_size=0, imsize=(105,105)):
    ns = len(strokes)
    if colored:
        colors = [get_color(i) for i in range(ns)]
    else:
        colors = ['black' for i in range(ns)]
    if imsize != (105,105):
        scale = torch.tensor(imsize, dtype=torch.float32)/105
        strokes = [scale*stk for stk in strokes]
    for i in range(ns):
        plot_traj(axis, strokes[i], colors[i], lw, arrow_size, pt_size)
    axis.set_xticks([])
    axis.set_yticks([])
    if imsize == (105,105):
        axis.set_xlim(0,105)
        axis.set_ylim(105,0)
    else:
        axis.set_xlim(0,imsize[1])
        axis.set_ylim(imsize[0],0)
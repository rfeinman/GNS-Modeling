import sys
if 'matplotlib' not in sys.modules:
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
from pybpl.util.stroke import dist_along_traj
from pybpl.util.affine import apply_warp
from pybpl.splines import get_stk_from_bspline
from neuralbpl.viz import get_color, plot_image, plot_pimg_over_img

from ...rendering.renderer import Renderer
from .util import get_intervals


__all__ = ['plot_parse_progress_token']

# motor_to_image transformation
M2I = torch.tensor([1., -1.])


def plot_traj(axis, x, color, lw, ls, marker):
    x = x*M2I
    if len(x) > 1 and dist_along_traj(x) > 0.01:
        axis.plot(x[:,0], x[:,1], color=color, linewidth=lw, linestyle=ls)
    else:
        axis.plot(x[0,0], x[0,1], color=color, linewidth=lw, marker=marker)

def plot_motor_to_image(axis, strokes, lw=2., is_type=False):
    ns = len(strokes)
    colors = [get_color(i) for i in range(ns)]
    if is_type:
        ls, marker = '--', '*'
    else:
        ls, marker = '-', '.'
    for i in range(ns):
        plot_traj(axis, strokes[i], colors[i], lw, ls, marker)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlim(0,105)
    axis.set_ylim(105,0)

def get_token_drawing(state):
    splines = state['x']
    splines = [spl + eps for spl,eps in zip(splines, state['loc_noise'])]
    splines = [spl + eps for spl,eps in zip(splines, state['shape_noise'])]
    splines = apply_warp(splines, state['affine'])
    drawing = [get_stk_from_bspline(x) for x in splines]
    return drawing

def plot_parse_progress_token(losses, states, img, nint=10, mode='motor', scale=1.4):
    assert mode in ['motor', 'motor-img', 'pimg', 'heatmap']
    if mode in ['pimg', 'heatmap']:
        renderer = Renderer()
    iters = len(losses)
    intervals = get_intervals(iters, nint)
    img = img.cpu()

    fig, axes = plt.subplots(nrows=1, ncols=nint, figsize=(scale*nint, scale))
    for i, idx in enumerate(intervals):
        state = states[idx]
        drawing_type = [get_stk_from_bspline(spl.cpu()) for spl in state['x']]
        drawing_token = get_token_drawing(state)
        if mode in ['motor', 'motor-img']:
            if mode == 'motor-img':
                plot_image(axes[i], img)
            plot_motor_to_image(axes[i], drawing_token, lw=scale)
            plot_motor_to_image(axes[i], drawing_type, lw=scale, is_type=True)
        elif mode == 'pimg':
            pimg = renderer(drawing_token, state['blur'], state['epsilon'])
            plot_image(axes[i], pimg.cpu())
        else:
            pimg = renderer(drawing_token, state['blur'], state['epsilon'])
            plot_image(axes[i], img)
            plot_pimg_over_img(axes[i], pimg.cpu())
        axes[i].set_title('%i\n%0.2f' % (idx, losses[idx].item()))
    plt.subplots_adjust(wspace=0)
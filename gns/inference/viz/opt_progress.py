import sys
if 'matplotlib' not in sys.modules:
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt
from pybpl.splines import get_stk_from_bspline

from ...viz import plot_motor_to_image, plot_image, plot_pimg_over_img
from ...rendering.renderer import Renderer
from .util import get_intervals, get_loss_limits



# ---- single parse (Parse) ----

def plot_parse_losses(losses, loss_lim=None, nint=10, scale=1.):
    iters = len(losses)
    intervals = get_intervals(iters, nint)
    if loss_lim is None:
        vmin, vmax = get_loss_limits(losses)
    else:
        vmin, vmax = loss_lim
    plt.figure(figsize=(scale*13,scale*4))
    plt.plot(losses)
    for idx in intervals:
        plt.plot([idx, idx],[vmin, vmax], color='gray', linestyle='--')
    plt.xlabel('iteration')
    plt.xticks(intervals)
    plt.title('Loss trajectory', fontsize=14)
    plt.ylim(vmin-20, vmax+20)

def plot_parse_progress(losses, states, img, nint=10, mode='strokes', scale=1.4):
    assert mode in ['strokes', 'pimg', 'heatmap']
    if mode in ['pimg', 'heatmap']:
        renderer = Renderer()
    iters = len(losses)
    intervals = get_intervals(iters, nint)
    img = img.cpu()

    fig, axes = plt.subplots(nrows=1, ncols=nint, figsize=(scale*nint, scale))
    for i, idx in enumerate(intervals):
        splines = states[idx]['x']
        drawing = [get_stk_from_bspline(spl.cpu()) for spl in splines]
        if mode == 'strokes':
            plot_image(axes[i], img)
            plot_motor_to_image(axes[i], drawing, lw=scale)
        elif mode == 'pimg':
            pimg = renderer(drawing, blur_sigma=states[idx]['blur'],
                            epsilon=states[idx]['epsilon'])
            plot_image(axes[i], pimg.cpu())
        else:
            pimg = renderer(drawing, blur_sigma=states[idx]['blur'],
                            epsilon=states[idx]['epsilon'])
            plot_image(axes[i], img)
            plot_pimg_over_img(axes[i], pimg.cpu())
        axes[i].set_title('%i\n%0.2f' % (idx, losses[idx].item()))
    plt.subplots_adjust(wspace=0)



# ---- multiple parses (list[Parse]) ----

def plot_parselist_losses(losses, loss_lim=None, nint=10, scale=1.):
    iters, K = losses.shape
    intervals = get_intervals(iters, nint)
    if loss_lim is None:
        vmin, vmax = get_loss_limits(losses)
    else:
        vmin, vmax = loss_lim
    plt.figure(figsize=(scale*13,scale*4))
    for k in range(K):
        plt.plot(losses[:,k], label='parse %i' % (k+1))
    for idx in intervals:
        plt.plot([idx, idx],[vmin, vmax], color='gray', linestyle='--')
    plt.xlabel('iteration')
    plt.xticks(intervals)
    plt.title('Loss trajectory', fontsize=14)
    plt.ylim(vmin-20, vmax+20)
    plt.legend()

def plot_parselist_progress(losses, states, images, nint=10, mode='strokes', scale=1.4):
    assert mode in ['strokes', 'pimg', 'heatmap']
    if mode in ['pimg', 'heatmap']:
        renderer = Renderer()
    iters, K = losses.shape
    intervals = get_intervals(iters, nint)
    images = images.cpu()
    if images.dim() == 2:
        images = images[None].repeat(K,1,1)
    else:
        assert images.dim() == 3

    fig, axes = plt.subplots(nrows=K, ncols=nint, figsize=(scale*nint, scale*K))
    for i, idx in enumerate(intervals):
        parses = states[idx]
        for j in range(K):
            splines = parses[j]['x']
            drawing = [get_stk_from_bspline(spl.cpu()) for spl in splines]
            if mode == 'strokes':
                plot_image(axes[j,i], images[j])
                plot_motor_to_image(axes[j,i], drawing, lw=scale)
            elif mode == 'pimg':
                pimg = renderer(drawing, blur_sigma=parses[j]['blur'],
                                epsilon=parses[j]['epsilon'])
                plot_image(axes[j,i], pimg.cpu())
            else:
                pimg = renderer(drawing, blur_sigma=parses[j]['blur'],
                                epsilon=parses[j]['epsilon'])
                plot_image(axes[j,i], images[j])
                plot_pimg_over_img(axes[j,i], pimg.cpu())
            if j == 0:
                axes[j,i].set_title('%i\n\n%0.2f' % (idx, losses[idx,j].item()))
            else:
                axes[j,i].set_title('%0.2f' % losses[idx,j].item())
            if i == 0:
                axes[j,i].set_ylabel('parse %i' % (j+1))
    plt.subplots_adjust(wspace=0., hspace=0.02)
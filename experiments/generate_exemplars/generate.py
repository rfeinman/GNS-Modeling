import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.distributions as dist

from pybpl.splines import get_stk_from_bspline
from gns.rendering import Renderer
from gns.token import TokenModel
from gns.inference import optimization as opt
from gns.viz import plot_image

from get_parses import load_image



def load_parses(img_ids):
    parses = {}
    log_probs = {}
    for img_id in img_ids:
        savedir = os.path.join('./parses/img_%0.2i' % (img_id+1))
        # parses
        parse_files = [f for f in os.listdir(savedir) if f.startswith('parse')]
        parses[img_id] = []
        for f in sorted(parse_files):
            state_dict = torch.load(os.path.join(savedir, f), map_location='cpu')
            init_parse = [val for key,val in state_dict.items() if key.startswith('x')]
            parse = opt.ParseWithToken(init_parse)
            parse.load_state_dict(state_dict)
            parses[img_id].append(parse)
        # log probs
        log_probs[img_id] = torch.load(os.path.join(savedir, 'log_probs.pt'))

    return parses, log_probs

class ImageFits:
    def __init__(self, images, img_ids):
        self.train_imgs = images
        self.train_parses, self.train_scores = load_parses(img_ids)

def sample_token(token_model, base_parses, weights):
    pid = dist.Categorical(weights).sample()
    splines = list(base_parses[pid].x)
    splines = token_model.sample(splines)
    return splines

def sample(token_model, fits, trial_id, T):
    parses = fits.train_parses[trial_id]
    scores = fits.train_scores[trial_id]
    # compute parse weights
    if T != 1:
        scores = scores / T
    log_weights = scores - torch.logsumexp(scores, 0)
    weights = torch.exp(log_weights)

    samples = []
    for i in range(9):
        splines = sample_token(token_model, parses, weights)
        strokes = list(map(get_stk_from_bspline, splines))
        samples.append(strokes)

    return samples

def grid_axes(fig, grid, nrow, ncol):
    axes = np.zeros((nrow, ncol), dtype=object)
    for i in range(nrow):
        for j in range(ncol):
            axes[i,j] = fig.add_subplot(grid[i,j])
    return axes

def color_axis(ax, color):
    for key in ['bottom', 'top', 'right', 'left']:
        ax.spines[key].set_color(color)

@torch.no_grad()
def show_subgrid(renderer, fig, subgrid, img, samples):
    # plot
    nrow, ncol = 4, 3
    inner = gridspec.GridSpecFromSubplotSpec(
        nrow, ncol, subplot_spec=subgrid, hspace=0., wspace=0.)
    axes = grid_axes(fig, inner, nrow, ncol)
    plot_image(axes[0,1], img)
    color_axis(axes[0,1], color='red')
    for j in [0,2]:
        axes[0,j].set_axis_off()
    for i in range(nrow-1):
        for j in range(ncol):
            strokes = samples[i*ncol+j]
            pimg = renderer(strokes, blur_sigma=0.)
            plot_image(axes[i+1,j], pimg > 0.5)

def main():
    print('Loading target images...')
    images = np.zeros((50,105,105), dtype=bool)
    for i in range(50):
        images[i] = load_image(os.path.join('./targets/handwritten%i.png' % (i+1)))

    print('Loading model parses...')
    img_ids = np.arange(50)
    fits = ImageFits(images, img_ids)

    print('Generating new exemplars...')
    renderer = Renderer()
    token_model = TokenModel()
    torch.manual_seed(4)

    nrow, ncol = 7, 7
    size = 2
    fig = plt.figure(figsize=(size*ncol*0.75, size*nrow))
    outer = gridspec.GridSpec(nrow, ncol)
    for i in range(nrow*ncol):
        samples_i = sample(token_model, fits, i, T=8)
        show_subgrid(
            renderer, fig, outer[i],
            img=fits.train_imgs[i],
            samples=samples_i
        )
    plt.show()

if __name__ == '__main__':
    main()
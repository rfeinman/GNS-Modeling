import sys
if 'matplotlib' not in sys.modules:
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def emphasize_axis(ax, lw=2., color=None):
    for key in ['bottom', 'top', 'right', 'left']:
        ax.spines[key].set_linewidth(lw)
        if color is not None:
            ax.spines[key].set_color(color)

def plot_image(ax, img, vmin=None, vmax=None):
    ax.imshow(img, cmap=plt.cm.binary, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])

def plot_pimg_over_img(ax, probs, cmap=plt.cm.plasma, vmax=None):
    if vmax is None:
        vmax = probs.max()
    # Normalize the colors b/w 0 and 1, we'll then pass an MxNx4 array to imshow
    colors = Normalize(0., vmax)(probs)
    colors = cmap(colors)
    # Create an alpha channel based on weight values
    alphas = Normalize()(probs)
    # Now set the alpha channel to the one we created above
    colors[..., -1] = alphas
    ax.imshow(colors, vmin=0., vmax=0.)

def get_color(k, more=False):
    if more:
        scol = ['red', 'green', 'blue', 'orange', 'cyan', 'magenta', 'yellow', 'gray']
    else:
        scol = ['r','g','b','m','c']
    ncol = len(scol)
    if k < ncol:
        return scol[k]
    else:
        return scol[k % ncol]

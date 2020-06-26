import sys
import math
if 'matplotlib' not in sys.modules:
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt
from pybpl.splines import get_stk_from_bspline
from neuralbpl.viz import plot_motor_to_image, get_color, plot_image



def plot_parse(ax, parse, cpts_size=10, arrow_size=0):
    strokes = [get_stk_from_bspline(spl, 200) if len(spl) > 1 else spl for spl in parse]
    plot_motor_to_image(ax, strokes, arrow_size=arrow_size)
    for i, spl in enumerate(parse):
        ax.scatter(spl[:,0], -spl[:,1], color=get_color(i), s=cpts_size)

def plot_parses(img, parses, scores=None, n=None, cpts=False, arrows=True, figscale=1.):
    cpts_size = 5 if cpts else 0
    arrow_size = 15 if arrows else 0
    nparse = len(parses)
    if n is None:
        n = math.ceil(nparse/10)
        n = max(n,2)
    m = 10
    if scores is None:
        figsize = (figscale*(m+1), figscale*n)
    else:
        figsize = (figscale*(m+1), 1.2*figscale*n)
    fig, axes = plt.subplots(n,m+1,figsize=figsize)
    # plot the target image
    plot_image(axes[0,0], img)
    axes[0,0].set_title('input')
    for i in range(1,n):
        axes[i,0].set_axis_off()
    # plot samples
    ix = 0
    for i in range(n):
        for j in range(1,m+1):
            if ix >= nparse:
                axes[i,j].set_axis_off()
                ix += 1
                continue
            plot_parse(axes[i,j], parses[ix], cpts_size=cpts_size, arrow_size=arrow_size)
            if scores is not None:
                axes[i,j].set_title("%0.2f" % scores[ix])
            ix += 1
    if scores is None:
        plt.subplots_adjust(hspace=0., wspace=0.)
    else:
        plt.subplots_adjust(hspace=0.4)

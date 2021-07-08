import sys
import multiprocessing as mp
import numpy as np
if 'matplotlib' not in sys.modules:
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
from pybpl.splines import get_stk_from_bspline
from pybpl.rendering import render_image
from pybpl.parameters import Parameters


def torch_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise Exception

def numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.tensor(x)
    elif isinstance(x, torch.Tensor):
        return x
    else:
        raise Exception

def traj_from_spline(y, neval=200):
    if len(y) > 1:
        y = numpy_to_torch(y).float()
        x = get_stk_from_bspline(y, neval)
    else:
        x = y
    x = torch_to_numpy(x)

    return x

def parallel(f, x, starmap=False):
    """
    TODO
    """
    p = mp.Pool()
    if starmap:
        y = p.starmap(f, x)
    else:
        y = p.map(f, x)
    p.close()
    p.join()

    return y

def render_strokes(strokes, blur=0.5, blur_fsize=11):
    assert (blur_fsize in np.arange(1,102)) and (blur_fsize % 2 == 1)
    ps = Parameters()
    ps.fsize = blur_fsize
    pimg, _ = render_image(
        strokes,
        epsilon=0.,
        blur_sigma=blur,
        ps=ps
    )

    return pimg

def render_strokes_small(strokes, blur=0.):
    """
    Use this version for (28,28) images
    """
    # set parameters
    ps = Parameters()
    ps.imsize = (28,28)
    ps.ink_ncon = 1
    ps.ink_a = 0.5
    ps.ink_b = 1.5
    # update strokes
    strokes = [(28/105)*stk for stk in strokes]
    # render image
    pimg, _ = render_image(
        strokes,
        epsilon=0.,
        blur_sigma=blur,
        ps=ps
    )

    return pimg

def show_weights(W, ncol=10):
    nrow = max(int(np.ceil(W.shape[0]/10)), 2)
    fig, axes = plt.subplots(nrow,ncol,figsize=(ncol/2,nrow/2))
    for i in range(nrow):
        for j in range(ncol):
            ix = i*ncol + j
            if ix < W.shape[0]:
                axes[i,j].imshow(W[ix,0], vmin=W.min(), vmax=W.max(), cmap='gray')
                axes[i,j].axis('off')
    plt.show()

def train_test_split(*tensors, test_size, stratify=None, random_state=None):
    from sklearn.model_selection import train_test_split as tt_split
    if stratify is not None:
        assert isinstance(stratify, torch.Tensor)
        stratify = stratify.numpy()
    arrays = [elt.numpy() for elt in tensors]
    arrays_split = tt_split(
        *arrays, test_size=test_size, stratify=stratify,
        random_state=random_state
    )
    tensors_split = [torch.from_numpy(elt) for elt in arrays_split]

    return tensors_split
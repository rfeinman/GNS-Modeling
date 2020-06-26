import sys
if 'matplotlib' not in sys.modules:
    import matplotlib as mpl
    mpl.use('Agg')
from matplotlib import transforms
from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import torch

from .general import get_color
from ..mixnet.losses import gmm_losses
from ..utils.mvn import gauss_norm_to_orig



def gauss2d_motor_to_image(mean, cov):
    """
    Convert Gaussian2D parameters from motor space to image space
    """
    assert mean.shape == (2,)
    assert cov.shape == (2,2)
    assert torch.abs(cov[0,1] - cov[1,0]) < 1e-3
    # convert mean
    mean1 = torch.tensor([mean[0], -mean[1]])
    # convert covariance
    cov1 = cov.clone()
    cov1[0,1] *= -1
    cov1[1,0] *= -1

    return mean1, cov1

def confidence_ellipse(ax, mean, cov, facecolor, edgecolor='black', n_std=2.0,
                       alpha=0.5, **ell_kwargs):
    """
    Plot an ellipse showing the confidence interval for a 2D Gaussian
    """
    assert mean.shape == (2,)
    assert cov.shape == (2,2)
    assert np.abs(cov[0,1] - cov[1,0]) < 1e-3

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # covariance eigenvalues
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        **ell_kwargs
    )

    # Calculating the stdandard deviation of x & y and
    # multiplying with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean[0], mean[1])

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)

def plot_gauss2d(ax, mean, cov, color, alpha=0.5, **ell_kwargs):
    """
    Plot the mean point and confidence ellipse for a 2D Gaussian.
    UNUSED
    """
    confidence_ellipse(ax, mean, cov, facecolor=color, alpha=alpha, **ell_kwargs)
    ax.scatter(mean[0], mean[1], c='red', s=10)

def plot_gmm(ax, mix_probs, means, covs, mean_X=None, std_X=None,
             meancolor=None, over_img=False, zorder=0):
    if over_img:
        # if plotting over an image, set axis accordingly
        ax.set_xlim(0,105)
        ax.set_ylim(105,0)
    # plot GMM
    K = len(mix_probs)
    for i, k in enumerate(torch.argsort(mix_probs)):
        # get color for this mixture
        color_i = get_color(K-i-1, more=True)
        # get params
        mean, cov, weight = means[k], covs[k], mix_probs[k].item()
        # convert from normalized space to original space
        if (mean_X is not None) and (std_X is not None):
            mean, cov = gauss_norm_to_orig(mean, cov, mean_X=mean_X, std_X=std_X)
        # convert from motor space to image space
        if over_img:
            mean, cov = gauss2d_motor_to_image(mean, cov)
        # plot gaussian
        confidence_ellipse(
            ax=ax, mean=mean.numpy(), cov=cov.numpy(), facecolor=color_i,
            alpha=weight, zorder=zorder
        )
        zorder += 1
        mean_color = color_i if (meancolor is None) else meancolor
        ax.scatter(mean[0], mean[1], c=mean_color, s=10, alpha=weight, zorder=zorder)
        zorder += 1

    return zorder

def plot_gmm_heatmap(ax, mix_probs, means, covs, mean_X=None, std_X=None,
                     cmap=plt.cm.plasma, transparency=False, vmax=None):
    # normalize parameters
    K = len(mix_probs)
    means = means.clone()
    covs = covs.clone()
    for k in range(K):
        if (mean_X is not None) and (std_X is not None):
            means[k], covs[k] = gauss_norm_to_orig(means[k], covs[k], mean_X=mean_X, std_X=std_X)
        means[k], covs[k] = gauss2d_motor_to_image(means[k], covs[k])

    # compute probability map
    xi, yi = torch.meshgrid(torch.arange(105), torch.arange(105))
    x_grid = torch.stack([yi, xi], dim=-1)
    neg_log_probs = gmm_losses(
        Y_pred=(mix_probs, means, covs),
        Y=x_grid,
        full_cov=True
    )
    probs = torch.exp(-neg_log_probs)
    probs = probs.numpy()

    if transparency:
        if vmax is None:
            vmax = probs.max()
        # Create an alpha channel based on weight values
        alphas = Normalize()(probs)
        # Normalize the colors b/w 0 and 1, we'll then pass an MxNx4 array to imshow
        colors = Normalize(0, vmax)(probs)
        colors = cmap(colors)
        # Now set the alpha channel to the one we created above
        colors[..., -1] = alphas
        ax.imshow(colors)
    else:
        ax.imshow(probs, cmap=cmap)
        ax.set_xlim(0,105)
        ax.set_ylim(105,0)

    return vmax
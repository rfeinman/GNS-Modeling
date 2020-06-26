import numpy as np
import torch


def set_triu(R, corrs):
    d = R.size(-1)
    x_ix, y_ix = np.triu_indices(d, 1)
    ix = np.stack([x_ix, y_ix])
    ix = torch.from_numpy(ix)
    R[...,ix[0],ix[1]] = corrs # (n,d,d)

def set_tril(R, corrs):
    d = R.size(-1)
    x_ix, y_ix = np.tril_indices(d, -1)
    ix = np.stack([x_ix, y_ix])
    ix = torch.from_numpy(ix)
    R[...,ix[0],ix[1]] = corrs # (n,d,d)

def compute_cov(scales, corrs):
    """
    :param scales: (...,d)
    :param corrs: (...,e)
    :return cov: (...,d,d)
    """
    d = scales.size(-1)
    assert corrs.size(-1) == int(d*(d-1)/2)

    # get correlation matrix
    R = torch.diag_embed(torch.ones_like(scales))
    set_triu(R, corrs)
    set_tril(R, corrs)

    # get covariance matrix
    scales = torch.diag_embed(scales) # (n,d,d)
    Cov = scales @ R @ scales

    return Cov

def compute_cov2d(scales, corrs):
    """
    This version is specific to 2D mvns

    scales: (...,d)
    corrs: (...,)
    """
    # compute covariances
    cov12 = corrs*torch.prod(scales,dim=-1) # (...,)
    covs = torch.diag_embed(scales**2) # (...,d,d)
    I = torch.diag_embed(torch.ones_like(scales)) # (...,d,d)
    covs = covs + cov12.unsqueeze(-1).unsqueeze(-1)*(1.-I)

    return covs

def tikhonov(scales, corrs, alpha):
    """
    scales: (...,d)
    corrs: (...,)
    alpha: float
    """
    scales_ = torch.sqrt(scales**2 + alpha)
    corrs_ = corrs*torch.prod(scales, -1)/torch.prod(scales_, -1)

    return scales_, corrs_

def gauss_norm_to_orig(means, covs, mean_X, std_X):
    """
    Convert MVN parameters from normalized space to original space

    means: (...,d)
    covs: (...,d,d)
    mean_X: (d,)
    std_X: (d,)
    """
    means = means*std_X + mean_X
    covs = torch.diag(std_X) @ covs @ torch.diag(std_X)

    return means, covs
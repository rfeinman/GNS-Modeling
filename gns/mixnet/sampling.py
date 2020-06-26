import math
import torch
from torch.distributions import Normal, Independent, MultivariateNormal
from torch.distributions import Categorical


def sample_xy_diag(mix_probs, means, scales):
    # get MVN means and scales
    mixtures = Categorical(mix_probs).sample() # (n,)
    means_sel = torch.stack([elt[i] for elt, i in zip(means, mixtures)]) # (n,d)
    scales_sel = torch.stack([elt[i] for elt, i in zip(scales, mixtures)]) # (n,d)

    # sample from MVNs
    norm = Normal(means_sel, scales_sel)
    mvn = Independent(norm, 1)
    samples = mvn.sample() # (n,d)

    return samples

def sample_xy_full(mix_probs, means, covs):
    # get MVN means and scales
    mixtures = Categorical(mix_probs).sample() # (n,)
    means_sel = torch.stack([elt[i] for elt, i in zip(means, mixtures)]) # (n,d)
    covs_sel = torch.stack([elt[i] for elt, i in zip(covs, mixtures)]) # (n,d)

    # sample from MVNs
    mvn = MultivariateNormal(means_sel, covs_sel)
    samples = mvn.sample() # (n,d)

    return samples

def truncate_samp(x, end):
    if torch.any(end == 1.):
        length = torch.nonzero(end).min() + 1
        return x[:length]
    else:
        return x


# ---- temperature adjustment code ----

def adjust_categorical(probs, temp):
    probs = torch.log(probs) / temp
    pmax, _ = probs.max(dim=-1, keepdim=True)
    probs = probs - pmax
    probs = torch.exp(probs)
    psum = probs.sum(dim=-1, keepdim=True)
    probs = probs / psum

    return probs

def adjust_bernoulli(p_1, temp):
    p_0 = 1. - p_1
    probs = torch.stack([p_0, p_1], dim=-1)
    probs = adjust_categorical(probs, temp)
    p_1 = probs[...,1]

    return p_1

def adjust_gmm(x_pred, temp):
    mix_probs, means, scales, corrs = x_pred
    mix_probs = adjust_categorical(mix_probs, temp)
    scales = scales * math.sqrt(temp)
    return (mix_probs, means, scales, corrs)
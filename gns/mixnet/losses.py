import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.distributions import Normal, Independent, MultivariateNormal

from ..utils.nn import apply_mask


def mvn_full(x, means, covs):
    x = x.unsqueeze(-2) # (...,1,d)
    mvn_logprobs = MultivariateNormal(means, covs).log_prob(x) # (...,k)

    return mvn_logprobs

def mvn_full_custom(x, means, scales, corrs):
    x_diff = x.unsqueeze(-2) - means # (...,k,d)
    Z1 = torch.sum(x_diff**2/scales**2, -1) # (...,k)
    Z2 = 2*corrs*torch.prod(x_diff,-1)/torch.prod(scales,-1) # (...,k)
    mvn_logprobs1 = -(Z1-Z2)/(2*(1-corrs**2)) # (...,k)
    mvn_logprobs2 = -torch.log(2*np.pi*torch.prod(scales,-1)*torch.sqrt(1-corrs**2)) # (...,k)
    mvn_logprobs = mvn_logprobs1 + mvn_logprobs2 # (...,k)

    return mvn_logprobs

def mvn_diag(x, means, scales):
    x = x.unsqueeze(-2) # (...,1,d)
    mvn_logprobs = Independent(Normal(means, scales), 1).log_prob(x) # (...,k)

    return mvn_logprobs

def gmm_losses(Y_pred, Y, full_cov=False):
    """
    :param Y_pred: tuple
        mix_probs: (...,k)
        means: (...,k,d)
        scales: (...,k,d)
        corrs: (...,k)
    :param Y: (...,d)
    :param full_cov: bool
    :return losses: (...,)
    """
    assert isinstance(Y_pred, tuple)
    # mvn log-probabilities
    if full_cov:
        if len(Y_pred) == 3:
            mix_probs, means, covs = Y_pred
            mvn_logprobs = mvn_full(Y, means, covs) # (...,k)
        else:
            mix_probs, means, scales, corrs = Y_pred
            mvn_logprobs = mvn_full_custom(Y, means, scales, corrs) # (...,k)
    else:
        mix_probs, means, scales = Y_pred
        mvn_logprobs = mvn_diag(Y, means, scales) # (...,k)

    # GMM log-prob
    gmm_logprobs = torch.log(mix_probs) + mvn_logprobs # (...,k)
    logprobs = torch.logsumexp(gmm_logprobs, dim=-1) # (...,)
    losses = -logprobs # (...,)

    return losses

def gmm_losses_seq(Y_pred, Y, mask, full_cov=False):
    """
    :param Y_pred: tuple
    :param Y: (n,s,d)
    :param mask: (n,s)
    :return losses: (n,)
    """
    # compute negative log-probs
    losses_seq = gmm_losses(Y_pred, Y, full_cov) # (n,s)
    # sum over non-padding locations
    losses = torch.sum(apply_mask(losses_seq, mask), dim=-1) # (n,)

    return losses

def end_losses(end_probs, end, logits=False):
    """
    :param end_probs: (...,)
    :param end: (...,)
    :return losses: (...,)
    """
    if logits:
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    else:
        loss_fn = nn.BCELoss(reduction='none')
    losses = loss_fn(end_probs, end)

    return losses

def end_losses_seq(end_probs, end, mask, logits=False):
    """
    :param end_probs: (n,s)
    :param end: (n,s)
    :param mask: (n,s)
    :return losses: (n,)
    """
    # compute negative log-probs
    losses_seq = end_losses(end_probs, end, logits) # (n,s)
    # sum over non-padding locations
    losses = torch.sum(apply_mask(losses_seq, mask), dim=-1) # (n,)

    return losses
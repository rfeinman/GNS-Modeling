import torch
import torch.nn as nn
import torch.distributions as dist
from pybpl.util import apply_warp



class TokenModel(nn.Module):
    def __init__(self, shape_noise=True, loc_noise=True, affine_noise=True):
        super().__init__()
        # control points noise
        self.register_buffer('mu_shape', torch.tensor(0.))
        self.register_buffer('sigma_shape', torch.tensor(3.3593))
        self.shape_noise = shape_noise

        # location noise
        self.register_buffer('mu_loc', torch.zeros(2))
        self.register_buffer('sigma_loc', torch.tensor([1.3536, 1.2056]))
        self.loc_noise = loc_noise

        # affine warp
        mu_affine = torch.tensor([1., 1., 0., 0.])
        cov_affine = torch.zeros(4,4)
        cov_affine[:2,:2] = torch.tensor([[0.0470, 0.0223],[0.0223, 0.0359]])
        cov_affine[2,2] = 5.2658**2
        cov_affine[3,3] = 5.0867**2
        self.register_buffer('mu_affine', mu_affine)
        self.register_buffer('cov_affine', cov_affine)
        self.affine_noise = affine_noise

        # initial score buffer
        self.register_buffer('score_init', torch.tensor(0.))

    @property
    def shape_dist(self):
        return dist.Normal(self.mu_shape, self.sigma_shape)

    @property
    def loc_dist(self):
        return dist.Independent(dist.Normal(self.mu_loc, self.sigma_loc), 1)

    @property
    def affine_dist(self):
        return dist.MultivariateNormal(self.mu_affine, self.cov_affine)

    def sample_shape_noise(self, size):
        return self.shape_dist.sample(size)

    def score_shape_noise(self, eps_shape):
        return self.shape_dist.log_prob(eps_shape).sum()

    def sample_loc_noise(self):
        return self.loc_dist.sample()

    def score_loc_noise(self, eps_loc):
        return self.loc_dist.log_prob(eps_loc)

    def sample_affine_warp(self):
        return self.affine_dist.sample()

    def score_affine_warp(self, affine_warp):
        return self.affine_dist.log_prob(affine_warp)

    def sample(self, splines, return_noises=False):
        noises = []

        # sample shape noise
        if self.shape_noise:
            eps_shape = [self.sample_shape_noise(spl.size()) for spl in splines]
            splines = [spl + eps for spl,eps in zip(splines, eps_shape)]
            noises.append(eps_shape)

        # sample location noise
        if self.loc_noise:
            eps_loc = [self.sample_loc_noise() for spl in splines]
            splines = [spl + eps for spl,eps in zip(splines, eps_loc)]
            noises.append(eps_loc)

        # sample affine warp
        if self.affine_noise:
            affine_warp = self.sample_affine_warp()
            splines = apply_warp(splines, affine_warp)
            noises.append(affine_warp)

        if return_noises:
            return splines, noises
        else:
            return splines

    def log_prob(self, eps_shape=None, eps_loc=None, affine_warp=None):
        """
        Parameters
        ----------
        eps_shape : list[torch.Tensor]
            list of (ncpts,2) location noise tensors, one per stroke
        eps_loc : list[torch.Tensor]
            list of (2,) location noise tensors, one per stroke
        affine_warp : torch.Tensor
            (4,) affine warp tensor

        Returns
        -------
        score : torch.Tensor
            (scalar) total log-probability
        """
        score = self.score_init

        # score shape noise
        if eps_shape is not None:
            eps_shape = torch.cat([eps for eps in eps_shape])
            score = score + self.score_shape_noise(eps_shape)

        # score location noise
        if eps_loc is not None:
            eps_loc = torch.stack([eps for eps in eps_loc])
            score = score + self.score_loc_noise(eps_loc).sum()

        if affine_warp is not None:
            score = score + self.score_affine_warp(affine_warp)

        return score

    def log_prob_multi(self, eps_shape, eps_loc, affine_warp):
        """
        Parameters
        ----------
        eps_shape : list[list[torch.Tensor]]
        eps_loc : list[list[torch.Tensor]]
        affine_warp : list[torch.Tensor]

        Returns
        -------
        scores : torch.Tensor
            (n,) tensor of log-probabilities per character
        """
        ncpts_tot = [sum([len(elt) for elt in plist]) for plist in eps_shape]
        nstk_tot = [len(plist) for plist in eps_loc]
        eps_shape = torch.cat([elt for plist in eps_shape for elt in plist]) # (n,m,2)
        eps_loc = torch.stack([elt for plist in eps_loc for elt in plist])
        affine_warp = torch.stack(affine_warp)

        scores_shape = self.shape_dist.log_prob(eps_shape).sum(-1)
        scores_shape = torch.split(scores_shape, ncpts_tot, 0)
        scores_shape = torch.stack([score.sum() for score in scores_shape])

        scores_loc = self.score_loc_noise(eps_loc)
        scores_loc = torch.split(scores_loc, nstk_tot, 0)
        scores_loc = torch.stack([score.sum() for score in scores_loc])

        scores_affine = self.score_affine_warp(affine_warp)

        scores = scores_shape + scores_loc + scores_affine
        return scores


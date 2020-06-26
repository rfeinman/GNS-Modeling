import torch
import torch.distributions as dist

from pybpl.util import nested_map



def to_cuda(x):
    return x.cuda()

class FullModel:
    """
    Full graphical model with the following joint distribution:
        P(type, token, image) = P(type) * P(token | type) * P(image | token)
    """
    def __init__(self, renderer, type_model=None, token_model=None,
                 drawings_to_type=True, denormalize=False, gpu_if_avail=True):
        self.renderer = renderer
        self.type_model = type_model
        self.token_model = token_model
        self.drawings_to_type = drawings_to_type
        self.denormalize = denormalize
        self.gpu = gpu_if_avail and torch.cuda.is_available()

    def likelihood_losses_fn(self, parses, images, drawings=None):
        if drawings is None:
            drawings = [p.drawing for p in parses]
        n = len(parses)
        pimgs = [self.renderer(drawings[i], parses[i].blur, parses[i].epsilon) for i in range(n)]
        pimgs = torch.stack(pimgs)
        loglike = dist.Bernoulli(pimgs).log_prob(images)
        losses = -torch.sum(loglike, dim=(1,2))
        return losses

    def type_losses_fn(self, parses, drawings):
        splines_list = [list(parse.x) for parse in parses]
        if self.gpu:
            drawings = nested_map(to_cuda, drawings)
            splines_list = nested_map(to_cuda, splines_list)
        if self.drawings_to_type:
            losses = self.type_model.losses_fn(
                splines_list, drawings, denormalize=self.denormalize)
        else:
            losses = self.type_model.losses_fn(
                splines_list, denormalize=self.denormalize)
        return losses

    def token_losses_fn(self, parses):
        eps_shape = [list(parse.shape_noise) for parse in parses]
        eps_loc = [list(parse.loc_noise) for parse in parses]
        affine = [parse.affine for parse in parses]
        if self.gpu:
            eps_shape = nested_map(to_cuda, eps_shape)
            eps_loc = nested_map(to_cuda, eps_loc)
            affine = nested_map(to_cuda, affine)
        losses = -self.token_model.log_prob_multi(eps_shape, eps_loc, affine)
        return losses

    def losses_fn(self, parses, images):
        drawings = [p.drawing for p in parses]
        losses = self.likelihood_losses_fn(parses, images, drawings)
        if self.type_model is not None:
            losses = losses + self.type_losses_fn(parses, drawings)
        if self.token_model is not None:
            losses = losses + self.token_losses_fn(parses)
        return losses

    def loss_fn(self, parse, image):
        losses = self.losses_fn([parse], image[None])
        return losses[0]

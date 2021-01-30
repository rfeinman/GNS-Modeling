import torch
import torch.nn as nn
from pybpl.splines import get_stk_from_bspline
from pybpl.util.affine import apply_warp

from . import soft_constraints as C



def param_state(p):
    return p.detach().cpu().clone()


class Parse(nn.Module):
    def __init__(self, init_parse, init_blur=16., init_epsilon=0.5, bound_mode='soft'):
        super().__init__()
        if bound_mode not in ['soft', 'passthrough']:
            raise ValueError("bound_mode must be either 'soft' or 'passthrough'")
        self.bound_mode = bound_mode
        self.x = nn.ParameterList([nn.Parameter(spl.clone()) for spl in init_parse])
        self.blur_base = nn.Parameter(torch.tensor(init_blur, dtype=torch.float))
        self.epsilon_base = nn.Parameter(torch.tensor(init_epsilon, dtype=torch.float))

    @property
    def drawing(self):
        return [get_stk_from_bspline(x) for x in self.x]

    @property
    def blur(self):
        if self.bound_mode == 'soft':
            return C.soft_ub_lb(self.blur_base, 0.5, 16.)
        elif self.bound_mode == 'passthrough':
            return C.passthrough_ub_lb(self.blur_base, 0.5, 16.)
        else:
            raise RuntimeError('invalid bound_mode encountered.')

    @property
    def epsilon(self):
        if self.bound_mode == 'soft':
            return C.soft_ub_lb(self.epsilon_base, 1e-4, 0.5)
        elif self.bound_mode == 'passthrough':
            return C.passthrough_ub_lb(self.epsilon_base, 1e-4, 0.5)
        else:
            raise RuntimeError('invalid bound_mode encountered.')

    @property
    def state(self):
        return {
            'x' : list(map(param_state, self.x)),
            'blur' : param_state(self.blur),
            'epsilon' : param_state(self.epsilon)
        }

    @property
    def render_params(self):
        return [self.blur_base, self.epsilon_base]

    @property
    def stroke_params(self):
        return list(self.x.parameters())


class ParseWithToken(Parse):
    def __init__(self, init_parse, init_blur=16., init_epsilon=0.5, bound_mode='soft'):
        # super
        super().__init__(init_parse, init_blur, init_epsilon, bound_mode)

        # stroke params
        loc_noise = [nn.Parameter(torch.zeros(2)) for spl in init_parse]
        shape_noise = [nn.Parameter(torch.zeros_like(spl)) for spl in init_parse]
        affine = torch.tensor([1., 1., 0., 0.])
        self.loc_noise = nn.ParameterList(loc_noise)
        self.shape_noise = nn.ParameterList(shape_noise)
        self.affine = nn.Parameter(affine)

    @property
    def drawing(self):
        splines = self.x
        splines = [spl + eps for spl,eps in zip(splines, self.loc_noise)]
        splines = [spl + eps for spl,eps in zip(splines, self.shape_noise)]
        splines = apply_warp(splines, self.affine)
        drawing = [get_stk_from_bspline(x) for x in splines]
        return drawing

    @property
    def state(self):
        state = super().state
        state['loc_noise'] = list(map(param_state, self.loc_noise))
        state['shape_noise'] = list(map(param_state, self.shape_noise))
        state['affine'] = param_state(self.affine)
        return state

    @property
    def stroke_params(self):
        p = super().stroke_params
        p = p + list(self.loc_noise.parameters())
        p = p + list(self.shape_noise.parameters())
        p = p + [self.affine]
        return p


def parse_list(init_parses, with_token=False, **kwargs):
    if with_token:
        init_fn = lambda x : ParseWithToken(x, **kwargs)
    else:
        init_fn = lambda x : Parse(x, **kwargs)
    return list(map(init_fn, init_parses))

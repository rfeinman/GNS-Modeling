import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
from pybpl.splines import get_stk_from_bspline
from gns import MODEL_SAVE_PATH
from gns.rendering import Renderer

from .terminate_model import CNNTerminate
from .location_model import CNNStart
from .stroke_model import LSTMConditioned



def load_terminate_model(save_dir, downsize):
    model = CNNTerminate(
        dropout=0.5,
        cnn_dropout=0.5,
        downsize=downsize
    )
    save_file = os.path.join(save_dir, 'terminate_model.pt')
    model.load_state_dict(torch.load(save_file, map_location='cpu'))
    return model

def load_location_model(save_dir, downsize):
    model = CNNStart(
        k=20,
        dropout=0.5,
        cnn_dropout=0.3,
        reg_cov=0.001,
        alpha=0.,
        downsize=downsize
    )
    save_file = os.path.join(save_dir, 'location_model.pt')
    model.load_state_dict(torch.load(save_file, map_location='cpu'))
    return model

def load_stroke_model(save_dir, downsize):
    model = LSTMConditioned(
        k=20,
        d=2,
        hidden_dim=256,
        combine_dim=256,
        attention_mode='sat',
        attention_dim=256,
        num_layers=2,
        dropout=0.3,
        recurrent_dropout=0.2,
        start_dropout=0.,
        downsize=downsize
    )
    save_file = os.path.join(save_dir, 'stroke_model.pt')
    model.load_state_dict(torch.load(save_file, map_location='cpu'))
    return model

class TypeModel(nn.Module):
    def __init__(self, save_dir=None, downsize=True):
        super().__init__()

        if save_dir is None:
            save_dir = MODEL_SAVE_PATH
        assert os.path.exists(save_dir)

        self.loc = load_location_model(save_dir, downsize).requires_grad_(False)
        self.stk = load_stroke_model(save_dir, downsize).requires_grad_(False)
        self.term = load_terminate_model(save_dir, downsize).requires_grad_(False)

        self.renderer = Renderer(blur_sigma=0.5)

        self.register_buffer('space_mean', torch.tensor([50., -50.]))
        self.register_buffer('space_scale', torch.tensor([20., 20.]))

    def cuda(self, device=None):
        super().cuda(device)
        self.renderer.painter = self.renderer.painter.cpu()
        return self

    def normalize(self, x, inverse=False):
        if inverse:
            return x*self.space_scale + self.space_mean
        else:
            return (x - self.space_mean)/self.space_scale

    def loss_fn(self, splines, drawing=None, filter_small=False,
                denormalize=False):
        if drawing is None: # get strokes
            drawing = splines_to_strokes(splines)
        if filter_small: # filter small strokes
            keep_ix = find_keepers(splines)
            splines = [splines[i] for i in keep_ix]
            drawing = [drawing[i] for i in keep_ix]
        ns = len(splines)
        device = splines[0].device
        pad_val = self.stk.pad_val

        # compute partial canvas renders
        canvases = self.renderer.forward_partial(drawing) # (ns+1,105,105)
        canvases = canvases.unsqueeze(1)

        # normalize spatial coordinates
        splines = [self.normalize(x) for x in splines]
        drawing = [self.normalize(x) for x in drawing]

        # collect model inputs
        prevs = get_input_prev(drawing, device) # (ns+1, 2)
        locs = get_input_loc(drawing, device) # (ns+1, 2)
        trajs = get_input_traj(splines, pad_val=pad_val) # (ns+1, T, 2)
        trajs = rnn.pad_sequence(trajs, batch_first=True, padding_value=pad_val)
        terms = get_input_term(ns, device) # (ns+1,)

        # compute losses
        if denormalize:
            losses_loc = self.loc.losses_fn(
                x_canv=canvases[:-1], prev=prevs[:-1], start=locs[:-1],
                mean_X=self.space_mean, std_X=self.space_scale)
            losses_stk = self.stk.losses_fn(
                x_canv=canvases[:-1], start=locs[:-1], x=trajs[:-1],
                std_X=self.space_scale)
        else:
            losses_loc = self.loc.losses_fn(x_canv=canvases[:-1], prev=prevs[:-1], start=locs[:-1])
            losses_stk = self.stk.losses_fn(x_canv=canvases[:-1], start=locs[:-1], x=trajs[:-1])
        losses_term = self.term.losses_fn(x=canvases, y=terms)
        loss = losses_loc.sum() + losses_stk.sum() + losses_term.sum()
        return loss

    def _losses_fn(self, splines_list, drawing_list=None, filter_small=False,
                   denormalize=False):
        assert isinstance(splines_list[0], list)
        if drawing_list is None:
            drawing_list = [splines_to_strokes(splines) for splines in splines_list]
        if filter_small: # filter small strokes
            keep_ix_list = [find_keepers(splines) for splines in splines_list]
            splines_list = [[splines[i] for i in keep_ix] for
                            splines, keep_ix in zip(splines_list, keep_ix_list)]
            drawing_list = [[drawing[i] for i in keep_ix] for
                            drawing, keep_ix in zip(drawing_list, keep_ix_list)]
        nsamp = len(splines_list)
        ns_list = [len(splines) for splines in splines_list]
        device = splines_list[0][0].device
        pad_val = self.stk.pad_val

        # compute partial canvas renders; returns list of (ns+1,105,105)
        canvases = self.renderer.forward_partial(drawing_list, concat=True)
        canvases = canvases.unsqueeze(1) # (ntot, 1, 105, 105)

        # normalize spatial coordinates
        splines_list = [[self.normalize(x) for x in splines] for splines in splines_list]
        drawing_list = [[self.normalize(x) for x in drawing] for drawing in drawing_list]

        # collect model inputs
        prevs = torch.cat([get_input_prev(drawing, device) for drawing in drawing_list])
        locs = torch.cat([get_input_loc(drawing, device) for drawing in drawing_list])
        trajs = [elt for splines in splines_list for elt in get_input_traj(splines, pad_val)]
        trajs = rnn.pad_sequence(trajs, batch_first=True, padding_value=pad_val)
        terms = torch.cat([get_input_term(ns, device) for ns in ns_list])

        # compute losses
        sizes = [ns+1 for ns in ns_list]
        if denormalize:
            losses_loc = self.loc.losses_fn(
                x_canv=canvases, prev=prevs, start=locs,
                mean_X=self.space_mean, std_X=self.space_scale)
            losses_stk = self.stk.losses_fn(
                x_canv=canvases, start=locs, x=trajs,
                std_X=self.space_scale
            )
        else:
            losses_loc = self.loc.losses_fn(x_canv=canvases, prev=prevs, start=locs)
            losses_stk = self.stk.losses_fn(x_canv=canvases, start=locs, x=trajs)
        losses_term = self.term.losses_fn(x=canvases, y=terms)
        losses_loc = torch.split(losses_loc, sizes, 0)
        losses_stk = torch.split(losses_stk, sizes, 0)
        losses_term = torch.split(losses_term, sizes, 0)
        losses = torch.zeros(nsamp, device=device)
        for i, (ll,ls,lt) in enumerate(zip(losses_loc, losses_stk, losses_term)):
            losses[i] = ll[:-1].sum() + ls[:-1].sum() + lt.sum()
        return losses

    def losses_fn(self, splines_list, drawing_list=None, filter_small=False,
                  denormalize=False, max_size=400):
        n = len(splines_list)
        if n <= max_size:
            return self._losses_fn(
                splines_list, drawing_list, filter_small, denormalize)
        losses = torch.zeros(n, device=splines_list[0][0].device)
        nbatch = math.ceil(n/max_size)
        for i in range(nbatch):
            start = i*max_size
            end = (i+1)*max_size
            splines_batch = splines_list[start:end]
            drawings_batch = None if drawing_list is None else drawing_list[start:end]
            losses[start:end] = self._losses_fn(
                splines_batch, drawings_batch, filter_small, denormalize)
        return losses


def get_input_prev(drawing, device): # returns (ns+1, 2)
    init = torch.tensor([-2.5, 2.5], device=device)
    prevs = torch.stack([init] + [stk[-1] for stk in drawing])
    return prevs

def get_input_loc(drawing, device): # returns (ns+1, 2)
    pad = torch.zeros(2, device=device)
    locs = [stk[0] for stk in drawing] + [pad]
    locs = torch.stack(locs)
    return locs

def get_input_traj(splines, pad_val=1000.): # returns (ns+1, T, 2)
    trajs = [x[1:]-x[:-1] for x in splines]
    trajs.append(torch.randn_like(trajs[-1]))
    trajs = [F.pad(x, [0,0,0,1], value=pad_val) for x in trajs]
    return trajs

def get_input_term(ns, device):
    terms = torch.zeros(ns+1, device=device)
    terms[-1] = 1.
    return terms

def find_keepers(splines):
    keep_ix = [i for i,spl in enumerate(splines) if len(spl) > 1]
    return keep_ix

def splines_to_strokes(splines):
    return [get_stk_from_bspline(spl) for spl in splines]

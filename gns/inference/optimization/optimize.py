from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .parse import Parse, ParseWithToken
from .early_stopping import EarlyStopper



def optimize_parse(parse, image, loss_fn, iterations=3000, optimizer=None,
                   tune_blur=True, tune_fn=None, stopper=None, clip_grad=None,
                   progbar=True):
    # check inputs
    assert isinstance(parse, Parse)
    assert (stopper is None) or isinstance(stopper, EarlyStopper)
    if optimizer is None:
        if isinstance(parse, ParseWithToken):
            lr_render, lr_stroke = (0.306983, 0.044114)
        else:
            lr_render, lr_stroke = (0.05, 0.1)
        param_groups = [
            {'params': parse.render_params, 'lr': lr_render},
            {'params': parse.stroke_params, 'lr': lr_stroke}
        ]
        optimizer = optim.Adam(param_groups)

    # run optimizer
    if tune_blur:
        losses = torch.zeros(iterations+1)
    else:
        losses = torch.zeros(iterations)
    states = []
    iterator = tqdm(range(iterations)) if progbar else range(iterations)
    for i in iterator:
        states.append(parse.state)
        optimizer.zero_grad()
        loss = loss_fn(parse, image)
        loss.backward()
        if clip_grad is not None:
            nn.utils.clip_grad_value_(parse.render_params, clip_grad)
        optimizer.step()
        losses[i] = loss.item()
        if progbar:
            iterator.set_postfix(loss=loss.item())
        if (stopper is not None) and stopper(loss.item()):
            if progbar: iterator.close()
            break
    if not tune_blur:
        return losses, states
    # perform final tuning of blur parameters
    final_loss, final_state = blur_tuning_single(
        parse, image, loss_fn=loss_fn, tune_fn=tune_fn)
    losses[i+1] = final_loss.item()
    states.append(final_state)

    return losses, states

def optimize_parselist(parse_list, image, loss_fn, iterations=3000,
                       optimizer=None, tune_blur=True, tune_fn=None,
                       clip_grad=None, progbar=True):
    # check inputs
    assert isinstance(parse_list, list)
    assert all([isinstance(elt, Parse) for elt in parse_list])
    K = len(parse_list)
    render_params = [p for parse in parse_list for p in parse.render_params]
    stroke_params = [p for parse in parse_list for p in parse.stroke_params]
    if optimizer is None:
        if isinstance(parse_list[0], ParseWithToken):
            lr_render, lr_stroke = (0.306983, 0.044114)
        else:
            lr_render, lr_stroke = (0.05, 0.1)
        param_groups = [
            {'params': render_params, 'lr': lr_render},
            {'params': stroke_params, 'lr': lr_stroke}
        ]
        optimizer = optim.Adam(param_groups)
    assert image.dim() in [2,3]
    if image.dim() == 3:
        assert image.size(0) == K

    # run optimizer
    if tune_blur:
        loss_vals = torch.zeros(iterations+1, K)
    else:
        loss_vals = torch.zeros(iterations, K)
    states = []
    iterator = tqdm(range(iterations)) if progbar else range(iterations)
    for i in iterator:
        states.append([parse.state for parse in parse_list])
        optimizer.zero_grad()
        losses = loss_fn(parse_list, image)
        loss = torch.sum(losses)
        loss.backward()
        if clip_grad is not None:
            nn.utils.clip_grad_value_(render_params, clip_grad)
        optimizer.step()
        loss_vals[i] = losses.detach().cpu()
    # return if we are done
    if not tune_blur:
        return loss_vals, states
    # perform final tuning of blur parameters
    final_losses, final_states = blur_tuning_multi(
        parse_list, image, loss_fn=loss_fn, tune_fn=tune_fn)
    loss_vals[-1] = final_losses.cpu()
    states.append(final_states)

    return loss_vals, states

@torch.no_grad()
def blur_tuning_single(parse, img, loss_fn, tune_fn=None, nbins=20):
    if tune_fn is None:
        tune_fn = loss_fn
    pblur = parse.blur_base.item()
    blur_grid = np.linspace(-50,pblur,nbins)[::-1]
    losses = torch.zeros(nbins)
    for i, pblur_ in enumerate(blur_grid):
        parse.blur_base.data = torch.tensor(pblur_, device=parse.blur_base.device)
        losses[i] = tune_fn(parse, img)
    best_loss, best_ix = torch.min(losses, dim=0)
    parse.blur_base.data = torch.tensor(
        blur_grid[best_ix.item()], device=parse.blur_base.device)
    best_state = parse.state
    if loss_fn != tune_fn:
        best_loss = loss_fn(parse, img)
    return best_loss, best_state

@torch.no_grad()
def blur_tuning_multi(parse_list, img, loss_fn, tune_fn=None, nbins=20):
    if tune_fn is None:
        tune_fn = loss_fn
    nparses = len(parse_list)
    blur_params = [parse.blur_base.item() for parse in parse_list]
    blur_grids = [np.linspace(-50,pblur,nbins)[::-1] for pblur in blur_params]
    losses = torch.zeros(nbins, nparses)
    for i, pblur_vals in enumerate(zip(*blur_grids)):
        for parse, pblur in zip(parse_list, pblur_vals):
            parse.blur_base.data = torch.tensor(pblur, device=parse.blur_base.device)
        losses[i] = tune_fn(parse_list, img)
    best_losses, best_idx = torch.min(losses, dim=0)
    for parse, ix, grid in zip(parse_list, best_idx, blur_grids):
        parse.blur_base.data = torch.tensor(grid[ix], device=parse.blur_base.device)
    best_states = [parse.state for parse in parse_list]
    if loss_fn != tune_fn:
        best_losses = loss_fn(parse_list, img)
    return best_losses, best_states

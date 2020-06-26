import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .parse import Parse



def optimize_parselist_mt(parse_list, image, loss_fn, iterations=3000,
                          optimizer_init=None, stopper_init=None,
                          clip_grad=None):
    """
    Parallel version of ParseList optimizer that optimizes each parse in a
    separate thread. Works with CUDA backend.
    """
    assert isinstance(parse_list, list)
    assert all([isinstance(elt, Parse) for elt in parse_list])
    K = len(parse_list)
    loss_vals = torch.zeros((iterations, K), dtype=torch.float)
    states = np.zeros((iterations, K), dtype=object)
    if image.dim() == 2:
        image = image[None].repeat(K,1,1)
    assert image.dim() == 3

    if optimizer_init is None:
        def optimizer_init(parse):
            param_groups = [
                {'params': parse.render_params, 'lr': 0.05},
                {'params': parse.stroke_params, 'lr': 0.1}
            ]
            return optim.Adam(param_groups)

    if stopper_init is None:
        #stopper_init = lambda: EarlyStopper(delta=1., patience=50)
        stopper_init = lambda: None


    def optimize(ix):
        parse = parse_list[ix]
        img = image[ix]
        optimizer = optimizer_init(parse)
        stopper = stopper_init()
        def closure():
            optimizer.zero_grad()
            loss = loss_fn(parse, img)
            loss.backward()
            if clip_grad is not None:
                nn.utils.clip_grad_value_(parse.render_params, clip_grad)
            return loss

        for t in range(iterations):
            states[t,ix] = parse.state
            if t == iterations-1: # last step
                loss = loss_fn(parse, img)
                loss_vals[t,ix] = loss.item()
                continue
            loss = optimizer.step(closure)
            loss_vals[t,ix] = loss.item()
            if (stopper is not None) and stopper(loss.item()):
                break

    threads = []
    for ix in range(K):
        t = torch.jit._fork(optimize, ix)
        threads.append(t)
    for t in threads:
        torch.jit._wait(t)

    return loss_vals, states
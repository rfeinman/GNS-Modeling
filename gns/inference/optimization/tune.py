import numpy as np
import torch



def get_param_grid(pblur, peps, nbins_blur, nbins_eps):
    blur_grid = np.linspace(-50, pblur, nbins_blur)
    eps_grid = np.linspace(-2, 1, nbins_eps-1)
    eps_grid = np.append(eps_grid, peps)
    param_grid = np.meshgrid(blur_grid, eps_grid)
    param_grid = np.stack(param_grid, axis=-1).reshape(-1,2)
    return param_grid

@torch.no_grad()
def render_tuning_multi(parse_list, img, tune_fn, nbins_blur=20, nbins_eps=40):
    K = len(parse_list)
    drawing_list = [parse.drawing for parse in parse_list]
    blur_params = [parse.blur_base.item() for parse in parse_list]
    eps_params = [parse.epsilon_base.item() for parse in parse_list]
    param_grids = [get_param_grid(blur_params[k], eps_params[k], nbins_blur, nbins_eps)
                   for k in range(K)]
    losses = torch.zeros(nbins_blur*nbins_eps, K)
    for i, param_vals in enumerate(zip(*param_grids)):
        for parse, (pblur, peps) in zip(parse_list, param_vals):
            parse.blur_base.data = torch.tensor(pblur, device=parse.blur_base.device)
            parse.epsilon_base.data = torch.tensor(peps, device=parse.epsilon_base.device)
        losses[i] = tune_fn(parse_list, drawing_list, img)
    best_losses, best_idx = torch.min(losses, dim=0)
    for parse, ix, grid in zip(parse_list, best_idx, param_grids):
        pblur, peps = grid[ix]
        parse.blur_base.data = torch.tensor(pblur, device=parse.blur_base.device)
        parse.epsilon_base.data = torch.tensor(peps, device=parse.epsilon_base.device)
    best_states = [parse.state for parse in parse_list]
    return best_losses, best_states
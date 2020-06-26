import warnings
import math
import itertools
import numpy as np
import torch
from pybpl.data import unif_space
from pybpl.matlab.bottomup import generate_random_parses
from neuralbpl.omniglot.minimal_splines import fit_minimal_spline

from .util import sample_from_generator



def process_parse(parse, device=None):
    parse_ = []
    for stk in parse:
        # for ntraj = 1, set spline as the original stroke
        if len(stk) == 1:
            spl = torch.tensor(stk, dtype=torch.float, device=device)
            parse_.append(spl)
            continue
        # for ntraj > 1, fit minimal spline to the stroke
        stk = unif_space(stk)
        stk = torch.tensor(stk, dtype=torch.float, device=device)
        spl = fit_minimal_spline(stk, thresh=0.7, max_nland=50)
        parse_.append(spl)
    return parse_

def apply_config(parse, config):
    ns = len(parse)
    order, direction = config
    parse_ = [parse[order[i]] for i in range(ns)]
    parse_ = [parse_[i].flip(dims=[0]) if direction[i] else parse_[i] for i in range(ns)]
    return parse_

def search_parse(parse, score_fn, configs_per=100, trials_per=800):
    assert trials_per >= configs_per
    ns = len(parse)
    if ns > 9:
        warnings.warn('parse searching not yet implemented for '
                      'large characters with ns > 9.')
        return [], []
    
    nconfigs = math.factorial(ns) * 2**ns

    # get all ordering & direction configurations (as generators)
    ordering_configs = itertools.permutations(range(ns))
    direction_configs = itertools.product([False,True], repeat=ns)
    configs = itertools.product(ordering_configs, direction_configs)

    # if we have too many configurations, sample subset
    if nconfigs > trials_per:
        configs = sample_from_generator(
            configs, nelt=nconfigs, nsamp=trials_per, replace=ns>7)

    # score configurations and take top-(configs_per)
    parses = [apply_config(parse, c) for c in configs]
    log_probs = score_fn(parses)
    log_probs, idx = torch.sort(log_probs, descending=True)
    parses = [parses[i] for i in idx]
    return parses[:configs_per], log_probs[:configs_per]

def get_topK_parses(img, k, score_fn, configs_per=100, trials_per=800,
                    device=None, seed=3, **grp_kwargs):
    # generate random walks (the "base parses")
    base_parses = generate_random_parses(I=img, seed=seed, **grp_kwargs)
    # convert strokes to minimal splines
    base_parses = [process_parse(parse, device) for parse in base_parses]

    # search for best stroke ordering & stroke direction configurations
    np.random.seed(seed)
    n = len(base_parses)
    parses = []; log_probs = []
    for i in range(n):
        parses_i, log_probs_i = search_parse(
            base_parses[i], score_fn, configs_per, trials_per)
        parses.extend(parses_i)
        log_probs.append(log_probs_i)
    log_probs = torch.cat(log_probs)

    # refine to unique & sort
    log_probs, idx = np.unique(log_probs.cpu().numpy(), return_index=True)
    log_probs = torch.from_numpy(log_probs).flip(dims=[0])
    idx = torch.from_numpy(idx).flip(dims=[0])
    parses = [parses[i] for i in idx]

    return parses[:k], log_probs[:k]
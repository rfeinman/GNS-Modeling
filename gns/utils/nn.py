import torch
import torch.nn as nn


def load_submodule(f, submod_name, cpu=False):
    if cpu:
        mod_state_dict = torch.load(f, map_location='cpu')
    else:
        mod_state_dict = torch.load(f)
    submod_state_dict = {}
    for mod_key,val in mod_state_dict.items():
        if mod_key.startswith(submod_name):
            submod_key = '.'.join(mod_key.split('.')[1:])
            submod_state_dict[submod_key] = val

    return submod_state_dict

def cnn_features(pretrained_cnn_path=None, dropout=0., dropout2d=0.):
    assert dropout == 0 or dropout2d == 0, 'Only 1 of 2 dropout modes allowed'
    features = nn.Sequential(
        nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        nn.Dropout(dropout),
        nn.Dropout2d(dropout2d),
        nn.Conv2d(20, 40, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        nn.Dropout(dropout),
        nn.Dropout2d(dropout2d),
        nn.Conv2d(40, 80, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        nn.Dropout(dropout),
        nn.Dropout2d(dropout2d)
    )
    fmap_size = (80,13,13)
    if pretrained_cnn_path is not None:
        features.load_state_dict(load_submodule(pretrained_cnn_path, 'features'))
        for param in features.parameters():
            param.requires_grad = False

    return features, fmap_size

def compute_cov2d(scales, corrs):
    """
    scales: (...,k,d)
    corrs: (...,k)
    """
    # compute covariances
    cov12 = corrs*torch.prod(scales,dim=-1) # (...,k)
    covs = torch.diag_embed(scales**2) # (...,k,d,d)
    I = torch.diag_embed(torch.ones_like(scales)) # (...,k,d,d)
    covs = covs + cov12.unsqueeze(-1).unsqueeze(-1)*(1.-I)

    return covs

def tikhonov(scales, corrs, alpha):
    """
    scales: (...,k,d)
    corrs: (...,k)
    alpha: float
    """
    scales1 = torch.sqrt(scales**2 + alpha)
    corrs1 = corrs*torch.prod(scales, -1)/torch.prod(scales1, -1)

    return scales1, corrs1

def seq_from_offset(off, mean_off=None, std_off=None):
    s, d = off.shape
    if mean_off is None:
        mean_off = torch.tensor([4.0701, -5.2096])
    if std_off is None:
        std_off = torch.tensor([17.3200, 17.9916])
    off = off*std_off + mean_off
    # get sequence from offsets
    seq = torch.zeros(s+1,d)
    seq[1:] = torch.cumsum(off, dim=0)

    return seq

def zscore(x, mean, std, inverse=False):
    if inverse:
        x = x*std + mean
    else:
        x = (x-mean)/std
    return x

def zscore_images(x, inverse=False):
    mean = 0.0779
    std = 0.2681
    return zscore(x, mean, std, inverse)

def zscore_starts(x, inverse=False):
    mean = torch.tensor([42.8051, -36.6407], dtype=torch.float)
    std = torch.tensor([18.4673, 18.6131], dtype=torch.float)
    return zscore(x, mean, std, inverse)

def apply_mask(x, mask):
    if mask.dtype != torch.bool:
        mask = mask.bool()
    return torch.where(mask, x, torch.zeros_like(x))
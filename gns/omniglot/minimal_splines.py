import torch
from pybpl.splines import fit_bspline_to_traj


def fit_minimal_spline(stroke, thresh, max_nland=100, normalize=True):
    assert isinstance(stroke, torch.Tensor)
    ntraj = stroke.size(0)

    # determine num control points
    for nland in range(1, min(ntraj+1, max_nland)):
        spline, residuals = fit_bspline_to_traj(stroke, nland, include_resid=True)
        loss = torch.sum(residuals).item()
        if normalize:
            loss = loss/float(ntraj)
        if loss < thresh:
            return spline

    return spline
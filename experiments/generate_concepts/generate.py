from tqdm.notebook import tqdm
import matplotlib.pylab as plt
import torch
from pybpl.splines import get_stk_from_bspline
from gns.type import TypeModel
from gns.utils import render_strokes
from gns.viz import plot_image



MEAN = torch.tensor([50., -50.])
SCALE = torch.tensor([20., 20.])

def convert_space(x, inverse=False):
    if inverse:
        return x*SCALE + MEAN
    else:
        return (x - MEAN)/SCALE

def transform_spline(x, start):
    """
    x : (t,2) spline in normalized space, starting at 0
    """
    assert len(x.shape) == 2 and x.size(1) == 2
    assert start.shape == torch.Size([2])
    assert torch.sum(torch.abs(x[0])) < 1e-5
    x = torch.cumsum(x, dim=0)
    x = get_stk_from_bspline(x, neval=200)
    x = x - x[0] + start
    x = convert_space(x, inverse=True)
    return x

@torch.no_grad()
def draw_sample(model, max_strokes=10, temp=1.):
    canv = torch.zeros(1,105,105) # initialize blank image canvas
    trajs = []
    loc = torch.zeros(2)
    for i in range(max_strokes):
        loc_z = convert_space(loc) # (2,) nn space
        # sample stroke location (starting point)
        start_z = model.loc.sample(canv=canv, prev=loc_z, temp=temp)[0] # (2,) nn space
        # sample stroke trajectory
        spline_off_z = model.stk.sample(canv, start_z, temp=temp) # (m,2) nn space, offsets
        # evaluate the stroke and update the image canvas
        traj = transform_spline(spline_off_z, start_z)
        trajs.append(traj)
        canv = render_strokes(trajs).unsqueeze(0)
        loc = traj[-1]
        # sample termination indicator
        end = model.term.sample(x=canv, temp=temp).item()
        if end == 1.:
            break

    # return final canvas (the sampled image)
    return canv, trajs

def main():
    # load model
    model = TypeModel().eval()

    # draw 100 samples
    torch.manual_seed(4)
    samples = torch.zeros(100,105,105)
    for i in tqdm(range(100)):
        samples[i], _ = draw_sample(model, temp=0.5)

    # plot
    fig, axes = plt.subplots(10,10,figsize=(10, 10))
    axes = axes.ravel()
    for i in range(100):
        plot_image(axes[i], samples[i])
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.show()

if __name__ == '__main__':
    main()